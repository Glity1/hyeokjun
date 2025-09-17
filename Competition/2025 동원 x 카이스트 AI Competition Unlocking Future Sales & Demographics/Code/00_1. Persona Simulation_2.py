# -*- coding: utf-8 -*-
"""
통합 파이프라인 v9: 보정(베이스라인) + 페르소나 + 몬테카를로 시뮬레이션 + 제출파일 생성

설명
- 이벤트/프로모션 등은 LLM 페르소나 JSON의 월패턴에만 반영(코드는 중복 반영 금지).
- 출시월 고정: 2024-07 → 12개월(2024-07~2025-06)
- 제품명: 화이트리스트(정확 일치) 강제
- 베이스라인: POS 시즌성 + (있으면) 닐슨 분기 블렌드 + 런치커브 + 가격탄력성 + μ(카테고리 점유) + SKU caps/mx
- 페르소나: purchase_probability 가중 → 제품별 월별 ‘모양’(부스트, 평균=1) → 베이스라인에 곱(총량 보존/반영 토글)
- 시뮬레이션: Poisson / Negative Binomial로 월별 불확실성 샘플링, 평균치로 제출
- 제출: sample_submission.csv의 헤더/순서 그대로 저장 (행이 없으면 형식 동일한 새 행 추가)
"""

from __future__ import annotations
import json, re, time, random, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42); random.seed(42)

# =========================
# 0) 경로 & 토글 / 하이퍼
# =========================
# 데이터/출력 경로
DATA_DIR = Path("./_data/dacon/dongwon")
POS_DIR  = DATA_DIR / "pos_data"
SAVE_DIR = Path("./_save/v9_integrated"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 필수 입력
PRODUCTS_PATH       = DATA_DIR / "product_info.csv"
POS_PATH            = POS_DIR / "marketlink_POS_master.xlsx"
PRICE_TABLE_PATH    = POS_DIR / "신제품_시세_가격표.xlsx"
PERSONA_JSON_PATH   = DATA_DIR / "personas.json"                 # 페르소나 JSON (프롬프트 산출물)
SAMPLE_SUB_PATH     = DATA_DIR / "sample_submission.csv"            # 제출 샘플 파일 경로(컬럼/순서 매칭용)
FINAL_SUB_PATH      = SAVE_DIR / "submission_v9_mc.csv"          # 최종 제출 파일

# 선택 파일
NIELSEN_Q_PATH      = POS_DIR / "닐슨코리아_2011_2019_분기별마스터.xlsx"
CORR_MONTHLY_PATH   = DATA_DIR / "보정계수_월별.xlsx"

# 설정(경제/스케일)
PRICE_FLOOR_WON     = 1500.0
MU_SCALE            = 1.00   # 전체 μ 스케일(0.95~1.10 등 실험)

# 제출/후처리 옵션
ROUND_SUBMISSION    = False
CLIP_MIN            = 1.0    # 0 근처 SMAPE 안정화
PERSONA_STRENGTH    = 0.80   # 1.0=그대로, <1 완화, >1 증폭
PERSONA_TOTAL_MODE  = "preserve"  # "preserve"(총량 보존) | "scale"(총량 변화 허용)

# 시즌성 관련
USE_NIELSEN_IF_AVAILABLE = True
SEASONALITY_BLEND        = 0.60
SOURCE_BLEND_NIELSEN_POS = 0.70
ALLOW_MULTI_PEAK         = True
MA_WINDOW_FOR_POST       = 2

# POS 윈도우(학습용 과거 구간)
POS_WINDOWS: List[Tuple[str,str]] = [
    ("2020-07", "2021-06"),
    ("2021-07", "2022-06"),
    ("2022-07", "2023-06"),
]

# 출시월 & 제출 대상 월
LAUNCH_START_YM = "2024-07"
ID_COL = "product_name"
MONTH_COLS = [f"months_since_launch_{i}" for i in range(1,13)]

# mx/caps/priors 파일
SKU_CAPS_OUT_PATH   = DATA_DIR / "sku_caps.csv"
MU_BY_CAT_OUT_PATH  = DATA_DIR / "mu_by_category.csv"
SKU_MX_CSV_PATH     = DATA_DIR / "sku_mx.csv"
CAT_MX_CSV_PATH     = DATA_DIR / "cat_mx.csv"
CURVE_PRIORS_CSV    = DATA_DIR / "category_curve_priors.csv"
AUTO_CAP_FROM_DEBUG = True
AUTO_CAP_FACTOR     = 1.10
DEBUG_CAP_PATH      = SAVE_DIR / "_debug_sku_max.csv"

# -------- 화이트리스트(정확 일치) --------
CANONICAL_PRODUCTS = [
    "덴마크 하이그릭요거트 400g",
    "동원맛참 고소참기름 135g",
    "동원맛참 고소참기름 90g",
    "동원맛참 매콤참기름 135g",
    "동원맛참 매콤참기름 90g",
    "동원참치액 순 500g",
    "동원참치액 순 900g",
    "동원참치액 진 500g",
    "동원참치액 진 900g",
    "리챔 오믈레햄 200g",
    "리챔 오믈레햄 340g",
    "소화가 잘되는 우유로 만든 바닐라라떼 250mL",
    "소화가 잘되는 우유로 만든 카페라떼 250mL",
    "프리미엄 동원참치액 500g",
    "프리미엄 동원참치액 900g",
]
CATEGORY_MAP: Dict[str, List[str]] = {
    "발효유": ["덴마크 하이그릭요거트 400g"],
    "커피": ["소화가 잘되는 우유로 만든 카페라떼 250mL", "소화가 잘되는 우유로 만든 바닐라라떼 250mL"],
    "전통기름": ["동원맛참 고소참기름 90g", "동원맛참 고소참기름 135g", "동원맛참 매콤참기름 90g", "동원맛참 매콤참기름 135g"],
    "조미료": ["동원참치액 순 500g", "동원참치액 순 900g", "동원참치액 진 500g", "동원참치액 진 900g", "프리미엄 동원참치액 500g", "프리미엄 동원참치액 900g"],
    "어육가공품": ["리챔 오믈레햄 200g", "리챔 오믈레햄 340g"],
}

# 하드캡 시드(옵션)
SKU_MONTHLY_HARD_CAP: Dict[str, int] = {
    "덴마크 하이그릭요거트 400g": 118_962,
}

# 제품 prior 예시(옵션)
PRODUCT_PRIORS: Dict[str, float] = {
    "소화가 잘되는 우유로 만든 바닐라라떼 250mL": 1.05,
    "소화가 잘되는 우유로 만든 카페라떼 250mL": 0.95,
    "동원맛참 고소참기름 90g": 1.03,
    "동원맛참 고소참기름 135g": 1.03,
    "동원맛참 매콤참기름 90g": 0.97,
    "동원맛참 매콤참기름 135g": 0.97,
    "동원참치액 순 500g": 1.02,
    "동원참치액 순 900g": 1.02,
    "동원참치액 진 500g": 0.98,
    "동원참치액 진 900g": 0.98,
}

# 커브 priors(카테고리)
CATEGORY_CURVE_PRIORS_DEFAULT = {
    "발효유":   {"early_peak":0.25, "mid_peak":0.55, "flat":0.20, "strength":0.50},
    "커피":    {"early_peak":0.50, "mid_peak":0.35, "flat":0.15, "strength":0.50},
    "전통기름": {"early_peak":0.20, "mid_peak":0.30, "flat":0.50, "strength":0.50},
    "조미료":   {"early_peak":0.20, "mid_peak":0.35, "flat":0.45, "strength":0.50},
    "어육가공품":{"early_peak":0.25, "mid_peak":0.45, "flat":0.30, "strength":0.50},
}
CURVE_PRIORS_RUNTIME = CATEGORY_CURVE_PRIORS_DEFAULT.copy()

# 배치 조합 (max_share_per_sku, elasticity, size_exponent)
SUBMISSION_COMBOS = [
    (0.050, -0.20,  0.70),
    (0.065, -0.20,  0.70),
    (0.050, -0.25,  0.70),
    (0.050, -0.20, -0.30),
]

# 런타임 컨테이너
SKU_CAPS_RUNTIME: Dict[str, int] = {}
SKU_MX_RUNTIME: Dict[str, float]  = {}
CAT_MX_RUNTIME: Dict[str, float]  = {}
SEASONALITY_BY_CAT: Dict[str, np.ndarray] = {}

# =========================
# 1) 유틸/파서
# =========================
def norm_name(s: str) -> str:
    return str(s).strip().replace("\u00A0"," ").replace("\u2009"," ").replace("  "," ")

CANONICAL_MAP = { norm_name(x): x for x in CANONICAL_PRODUCTS }

def canonicalize_product_name(name: str) -> Optional[str]:
    if name is None: return None
    key = norm_name(str(name))
    return CANONICAL_MAP.get(key)

def launch_months(start_ym: str, n: int = 12) -> List[str]:
    idx = pd.period_range(start=start_ym, periods=n, freq="M")
    return [str(p) for p in idx]

LAUNCH_MONTHS = launch_months(LAUNCH_START_YM)

_size_pat = re.compile(r"(\d+(?:\.\d+)?)\s*(mL|ml|ML|l|L|g|G)")
def parse_size_from_name(name: str) -> float:
    m = _size_pat.findall(str(name))
    if not m: return np.nan
    val, unit = m[-1]
    x = float(val); unit = unit.lower()
    if unit in ("ml",): return x
    if unit in ("l",):  return x * 1000.0
    if unit in ("g",):  return x
    return np.nan

def find_col(cands: List[str], columns: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in cands:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in sorted(cands, key=len, reverse=True):
        for col in columns:
            if cand.lower() in col.lower():
                return col
    return None

def months_span(start_ym: str, end_ym: str) -> List[str]:
    idx = pd.period_range(start=start_ym, end=end_ym, freq='M')
    return [str(p) for p in idx]

def cols_in_windows(columns: List[str], windows: List[Tuple[str,str]]) -> List[str]:
    allowed = set()
    for s,e in windows:
        allowed.update(months_span(s,e))
    return [c for c in sorted(columns) if c in allowed]

def map_category(name: str) -> str:
    for cat, items in CATEGORY_MAP.items():
        if name in items: return cat
    return "기타"

# =========================
# 2) 데이터 로드
# =========================
def load_products() -> pd.DataFrame:
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"필수 파일 없음: {PRODUCTS_PATH}")
    df = pd.read_csv(PRODUCTS_PATH)
    if ID_COL not in df.columns:
        raise RuntimeError(f"{PRODUCTS_PATH}에 '{ID_COL}' 컬럼 필요")

    # product_info의 '대표 판매가'는 무시 → 시세표로 덮어씀
    if "대표 판매가" in df.columns:
        df.drop(columns=["대표 판매가"], inplace=True)

    if not PRICE_TABLE_PATH.exists():
        raise FileNotFoundError(f"가격 시세표 없음: {PRICE_TABLE_PATH}")

    price_df = pd.read_excel(PRICE_TABLE_PATH)
    name_col  = find_col(["제품명","상품명","품명","product_name"], list(price_df.columns)) or "제품명"
    price_col = find_col(["대표 판매가","대표판매가","price","가격","판매가"], list(price_df.columns))
    if not price_col:
        raise RuntimeError(f"{PRICE_TABLE_PATH.name}에서 가격 컬럼을 찾을 수 없음")

    price_df = price_df[[name_col, price_col]].rename(columns={name_col: ID_COL, price_col: "대표 판매가"})

    # 정규화 + 병합
    df[ID_COL]       = df[ID_COL].map(norm_name).astype(str)
    price_df[ID_COL] = price_df[ID_COL].map(norm_name).astype(str)
    df = df.merge(price_df, on=ID_COL, how="left")

    # 숫자화 + 하한선 + 검증
    df["대표 판매가"] = pd.to_numeric(df["대표 판매가"], errors="coerce").clip(lower=PRICE_FLOOR_WON)
    matched = int(df["대표 판매가"].notna().sum()); total = int(len(df))
    print(f"[PRICE] matched {matched}/{total} from '{PRICE_TABLE_PATH.name}'")
    if matched != total:
        missing = df.loc[df["대표 판매가"].isna(), ID_COL].tolist()
        raise RuntimeError(f"[PRICE] 가격 미매칭 {total - matched}개: {missing}")

    # 화이트리스트 강제 + 파생
    df["product_name"] = df["product_name"].map(canonicalize_product_name)
    if df["product_name"].isna().any():
        bad = df.loc[df["product_name"].isna()]
        raise RuntimeError(f"[CHECK] 화이트리스트 불일치 제품 존재: {bad}")
    df["category"] = df[ID_COL].apply(map_category)
    df["size_parsed"] = df[ID_COL].apply(parse_size_from_name)
    return df

def load_pos() -> pd.DataFrame:
    if not POS_PATH.exists():
        raise FileNotFoundError(f"필수 파일 없음: {POS_PATH}")
    pos = pd.read_excel(POS_PATH)
    need = ["연도","월","구분","카테고리","매출액(백만원)"]
    for c in need:
        if c not in pos.columns: raise RuntimeError(f"{POS_PATH}에 '{c}' 컬럼 필요")
    pos["연월"] = pos["연도"].astype(str) + "-" + pos["월"].astype(str).str.zfill(2)
    total = (pos[pos["구분"]=="총매출"].groupby(["카테고리","연월"])["매출액(백만원)"].sum().reset_index())
    pvt = total.pivot(index="카테고리", columns="연월", values="매출액(백만원)").fillna(0)
    pvt = pvt.reindex(sorted(pvt.columns), axis=1)
    keep_cols = cols_in_windows(list(pvt.columns), POS_WINDOWS)
    pvt = pvt[keep_cols]
    print(f"[CHECK] POS months (windows): {keep_cols[0]} → {keep_cols[-1]} | cols={len(keep_cols)} | rows={pvt.shape[0]}")
    return pvt

# =========================
# 2-1) 닐슨 & 보정계수 → 분기 기반 시즌성
# =========================
def _read_excel_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(path)
    except Exception as e:
        print(f"[WARN] read fail: {path.name} -> {e}")
        return None

def _pick_value_column(df: pd.DataFrame) -> str:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise RuntimeError("수치형 값 컬럼을 찾지 못함")
    return max(num_cols, key=lambda c: float(df[c].fillna(0).abs().sum()))

def load_monthly_corrections() -> Optional[pd.Series]:
    if not CORR_MONTHLY_PATH.exists(): return None
    df = _read_excel_safe(CORR_MONTHLY_PATH)
    if df is None: return None
    m_col = find_col(["월","month"], list(df.columns)) or "월"
    w_col = find_col(["계수","보정","weight"], list(df.columns)) or None
    try:
        if w_col is None: w_col = _pick_value_column(df)
        s = (df.set_index(m_col)[w_col]).astype(float)
        s = s[[i for i in s.index if str(i).isdigit() and 1<=int(i)<=12]]
        s.index = pd.Index([int(i) for i in s.index], name="월")
        s = s.reindex(range(1,13)).fillna(1.0)
        s = s / s.sum() * 12.0
        print("[INFO] 월별 보정계수 로드 OK")
        return s
    except Exception as e:
        print(f"[WARN] 월별 보정계수 파싱 실패: {e}")
        return None

def moy_from_pivot(pvt: pd.DataFrame) -> Dict[str, np.ndarray]:
    def _m(ym): return int(str(ym).split("-")[1])
    months = np.array([_m(c) for c in pvt.columns])
    out = {}
    for cat in pvt.index:
        vals = np.maximum(pvt.loc[cat].to_numpy(float), 0.0)
        sums = np.zeros(12); cnts = np.zeros(12)
        for v,m in zip(vals, months):
            idx = m-1
            sums[idx]+=v; cnts[idx]+=1
        s = np.divide(sums, np.maximum(cnts,1.0))
        s = np.maximum(s, 0.0)
        s = s / (s.sum() if s.sum()>0 else 1.0)
        out[cat] = s
    return out

def build_category_seasonality_map(products: pd.DataFrame, pos_pvt_windowed: pd.DataFrame) -> Dict[str, np.ndarray]:
    pos_moy = moy_from_pivot(pos_pvt_windowed)
    if not USE_NIELSEN_IF_AVAILABLE:
        print("[SEASON] Nielsen 비사용 → POS 시즌성 사용")
        return pos_moy

    dfq = _read_excel_safe(NIELSEN_Q_PATH)
    if dfq is None:
        print("[SEASON] Nielsen 분기 파일 없음 → POS 시즌성 사용")
        return pos_moy

    cat_col = find_col(["카테고리","category"], list(dfq.columns)) or "카테고리"
    q_col   = find_col(["분기","quarter","Q"], list(dfq.columns)) or "분기"
    v_col   = find_col(["매출","금액","판매액","value","액"], list(dfq.columns)) or _pick_value_column(dfq)

    use = dfq[[cat_col, q_col, v_col]].copy()
    qshare = (use.groupby([cat_col, q_col])[v_col]
                 .sum()
                 .unstack(q_col, fill_value=0)
                 .reindex(columns=[1,2,3,4], fill_value=0))
    denom = qshare.sum(axis=1).replace(0, np.nan)
    qshare = qshare.div(denom, axis=1).fillna(0.25)

    month_corr = load_monthly_corrections()
    q_to_months = {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}
    nielsen_moy: Dict[str, np.ndarray] = {}
    for cat, row in qshare.iterrows():
        s = np.zeros(12, float)
        for qi, wq in zip([1,2,3,4], row.values.tolist()):
            months = q_to_months[qi]
            if month_corr is None: w = np.ones(3, float)
            else: w = month_corr.reindex(months).astype(float).to_numpy()
            w = w / (w.sum() if w.sum()>0 else 3.0)
            for m, wm in zip(months, w):
                s[m-1] += wq * wm
        s = s / (s.sum() if s.sum()>0 else 1.0)
        nielsen_moy[str(cat)] = s

    alpha = SOURCE_BLEND_NIELSEN_POS
    out = {}
    for cat in products["category"].unique():
        s_pos = pos_moy.get(cat, np.ones(12)/12)
        s_nil = nielsen_moy.get(cat, s_pos)
        s = (1-alpha)*np.asarray(s_pos,float) + alpha*np.asarray(s_nil,float)
        s = s / (s.sum() if s.sum()>0 else 1.0)
        out[cat] = s
    print(f"[SEASON] 분기 기반 시즌성(Q→12M)  |  blend(Nielsen:POS)={alpha:.2f}:{1-alpha:.2f}")
    return out

# =========================
# 2-2) LLM 페르소나 JSON → 제품별 부스트 벡터
# =========================
def persona_pattern_from_item(item: dict) -> Optional[np.ndarray]:
    # 권장: monthly_by_launch (길이 12)
    if "monthly_by_launch" in item and isinstance(item["monthly_by_launch"], list) and len(item["monthly_by_launch"]) == 12:
        return np.array(item["monthly_by_launch"], float)
    # 대안: monthly_by_calendar {"YYYY-MM": v}
    if "monthly_by_calendar" in item and isinstance(item["monthly_by_calendar"], dict):
        arr = []
        for ym in LAUNCH_MONTHS:
            v = item["monthly_by_calendar"].get(ym, None)
            arr.append(np.nan if v is None else float(v))
        a = np.array(arr, float)
        if np.isnan(a).any():
            s = pd.Series(a).interpolate(limit_direction="both").fillna(0.0)
            a = s.to_numpy(float)
        return a
    return None

def _agg_for_product(personas: List[Dict[str,Any]]) -> Optional[np.ndarray]:
    if not personas: return None
    pp, pat = [], []
    for p in personas:
        arr = persona_pattern_from_item(p)
        if arr is None: 
            continue
        prob = float(p.get("purchase_probability", 0.0))
        pp.append(max(0.0, min(100.0, prob)) / 100.0)
        pat.append(np.array(arr, dtype=float))
    if len(pp) == 0: return None
    w = np.array(pp, float)
    w = w / (w.sum() if w.sum()>0 else 1.0)
    P = np.stack(pat, axis=0)
    agg = (w[:,None] * P).sum(axis=0)  # 길이 12
    return agg

def aggregate_persona_patterns(persona_data: Any) -> Dict[str, np.ndarray]:
    """제품명 화이트리스트 강제 + 가중 평균 패턴(길이12) 반환"""
    out: Dict[str, np.ndarray] = {}
    def _add(prod_name: str, personas: List[dict]):
        canon = canonicalize_product_name(prod_name)
        if not canon:
            print(f"[WARN] persona product 미일치 → 스킵: {prod_name}")
            return
        agg = _agg_for_product(personas)
        if agg is not None:
            out[canon] = agg
    if isinstance(persona_data, dict):
        for prod, lst in persona_data.items():
            _add(prod, lst if isinstance(lst, list) else [])
    elif isinstance(persona_data, list):
        buckets: Dict[str, List[dict]] = {}
        for item in persona_data:
            prod = item.get("product") or item.get("제품명") or item.get("product_name")
            if not prod:
                print("[WARN] persona item에 product 필드 없음 → 스킵")
                continue
            buckets.setdefault(prod, []).append(item)
        for prod, lst in buckets.items():
            _add(prod, lst)
    else:
        raise ValueError("지원하지 않는 페르소나 JSON 구조")
    print(f"[PERSONA] aggregated for {len(out)} products (whitelist enforced)")
    return out

def persona_map_to_dataframe(persona_map: Dict[str, np.ndarray]) -> pd.DataFrame:
    """제품별 길이12 벡터 → boost_m01..12 (평균 1로 정규화)"""
    rows = []
    for name, vec in persona_map.items():
        v = np.array(vec, float)
        mean = float(v.mean()) if v.size>0 else 1.0
        boost = (v / (mean if mean>0 else 1.0))  # 평균 1.0
        rows.append([name] + boost.tolist())
    cols = ["product_name"] + [f"boost_m{i:02d}" for i in range(1,13)]
    return pd.DataFrame(rows, columns=cols)

def load_persona_json(path: Path = PERSONA_JSON_PATH) -> Optional[Any]:
    if not path.exists():
        print("[PERSONA] JSON 파일 없음 → 미적용")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[PERSONA] JSON 파싱 실패: {e}")
        return None

# =========================
# 3) caps/mx/priors
# =========================
def load_sku_caps(products: pd.DataFrame) -> Dict[str, int]:
    caps: Dict[str, int] = {}
    for k, v in SKU_MONTHLY_HARD_CAP.items():
        if pd.notna(v):
            caps[str(k).strip()] = int(v)
    if SKU_CAPS_OUT_PATH.exists():
        try:
            df = pd.read_csv(SKU_CAPS_OUT_PATH)
            if {"product_name","cap"}.issubset(df.columns):
                for _, r in df.iterrows():
                    caps[str(r["product_name"]).strip()] = int(r["cap"])
        except Exception as e:
            print(f"[CAPS] read {SKU_CAPS_OUT_PATH.name} failed: {e}")
    if AUTO_CAP_FROM_DEBUG and DEBUG_CAP_PATH.exists():
        try:
            s = pd.read_csv(DEBUG_CAP_PATH, header=None, index_col=0, names=["max_month"])
            for name, mx in s["max_month"].items():
                key = str(name).strip()
                if key not in caps:
                    caps[key] = int(round(float(mx) * float(AUTO_CAP_FACTOR)))
            print(f"[CAPS] auto from debug: {len(s)} rows × factor {AUTO_CAP_FACTOR:.2f}")
        except Exception as e:
            print(f"[CAPS] failed to read debug caps: {e}")
    valid = set(products[ID_COL].astype(str))
    caps = {k: v for k, v in caps.items() if k in valid}
    out = pd.DataFrame({"product_name": list(caps.keys()), "cap": list(caps.values())}).sort_values("product_name")
    out.to_csv(SKU_CAPS_OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[CAPS] saved → {SKU_CAPS_OUT_PATH.as_posix()}")
    return caps

def sku_hard_cap(name: str, default: float = np.inf) -> float:
    return float(SKU_CAPS_RUNTIME.get(str(name), SKU_MONTHLY_HARD_CAP.get(str(name), default)))

def load_mx_maps(products: pd.DataFrame):
    global SKU_MX_RUNTIME, CAT_MX_RUNTIME
    SKU_MX_RUNTIME, CAT_MX_RUNTIME = {}, {}
    if CAT_MX_CSV_PATH.exists():
        try:
            dfc = pd.read_csv(CAT_MX_CSV_PATH)
            if {"category","mx"}.issubset(dfc.columns):
                for _, r in dfc.iterrows():
                    c = str(r["category"]).strip()
                    v = float(r["mx"])
                    if np.isfinite(v) and 0 < v <= 1:
                        CAT_MX_RUNTIME[c] = v
            print(f"[MX] cat_mx loaded: {len(CAT_MX_RUNTIME)} cats")
        except Exception as e:
            print(f"[MX] read cat_mx.csv failed: {e}")
    if SKU_MX_CSV_PATH.exists():
        try:
            dfs = pd.read_csv(SKU_MX_CSV_PATH)
            if {"product_name","mx"}.issubset(dfs.columns):
                valid = set(products[ID_COL].astype(str))
                for _, r in dfs.iterrows():
                    k = str(r["product_name"]).strip()
                    v = float(r["mx"])
                    if k in valid and np.isfinite(v) and 0 < v <= 1:
                        SKU_MX_RUNTIME[k] = v
            print(f"[MX] sku_mx loaded: {len(SKU_MX_RUNTIME)} SKUs")
        except Exception as e:
            print(f"[MX] read sku_mx.csv failed: {e}")

def sku_max_share_limit(name: str, category: str, global_mx: float) -> float:
    if str(name) in SKU_MX_RUNTIME: return float(SKU_MX_RUNTIME[str(name)])
    if str(category) in CAT_MX_RUNTIME: return float(CAT_MX_RUNTIME[str(category)])
    return float(global_mx)

# =========================
# 4) 런치커브 템플릿 & priors 블렌딩
# =========================
LAUNCH_TEMPLATES: Dict[str, np.ndarray] = {
    "early_peak": np.array([0.12,0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03], float),
    "mid_peak"  : np.array([0.06,0.09,0.12,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.05], float),
    "flat"      : np.array([0.08,0.09,0.10,0.10,0.10,0.09,0.09,0.09,0.08,0.07,0.06,0.05], float),
}
for k in LAUNCH_TEMPLATES:
    LAUNCH_TEMPLATES[k] = LAUNCH_TEMPLATES[k] / LAUNCH_TEMPLATES[k].sum()

def product_heuristic_curve_weights(name: str, price: float) -> Dict[str, float]:
    n = str(name)
    w = {"early_peak":0.33, "mid_peak":0.34, "flat":0.33}
    if "요거트" in n or "그릭" in n:
        w["mid_peak"] += 0.10; w["flat"] -= 0.10
    if "라떼" in n or "카페" in n or "바닐라" in n:
        w["early_peak"] += 0.05; w["flat"] -= 0.05
    if "프리미엄" in n:
        w["flat"] += 0.05; w["early_peak"] -= 0.05
    if price >= 5000:
        w["flat"] += 0.03; w["early_peak"] -= 0.03
    s = sum(max(0.0, v) for v in w.values())
    for k in w: w[k] = max(0.0, w[k]) / (s if s>0 else 1.0)
    return w

def load_category_curve_priors():
    global CURVE_PRIORS_RUNTIME
    CURVE_PRIORS_RUNTIME = {k: v.copy() for k, v in CATEGORY_CURVE_PRIORS_DEFAULT.items()}
    if CURVE_PRIORS_CSV.exists():
        try:
            df = pd.read_csv(CURVE_PRIORS_CSV)
            need = {"category","early_peak","mid_peak","flat"}
            if not need.issubset(df.columns):
                print(f"[CURVE PRIORS] CSV missing columns → using defaults")
                return
            for _, r in df.iterrows():
                cat = str(r["category"])
                ep = float(r["early_peak"]); mp = float(r["mid_peak"]); ft = float(r["flat"])
                st = float(r.get("strength", CURVE_PRIORS_RUNTIME.get(cat,{}).get("strength", 0.5)))
                s = max(1e-9, ep+mp+ft)
                CURVE_PRIORS_RUNTIME[cat] = {"early_peak": ep/s, "mid_peak": mp/s, "flat": ft/s, "strength": float(np.clip(st,0,1))}
            print(f"[CURVE PRIORS] loaded for {len(CURVE_PRIORS_RUNTIME)} cats")
        except Exception as e:
            print(f"[CURVE PRIORS] read failed: {e}")

def blended_curve_weights(name: str, category: str, price: float) -> Dict[str, float]:
    base = product_heuristic_curve_weights(name, price)
    pri  = CURVE_PRIORS_RUNTIME.get(category, CATEGORY_CURVE_PRIORS_DEFAULT.get(category, {"early_peak":1/3,"mid_peak":1/3,"flat":1/3,"strength":0.0}))
    beta = float(pri.get("strength", 0.5))
    w = {k: (1.0 - beta) * base.get(k,0.0) + beta * float(pri.get(k,0.0)) for k in ["early_peak","mid_peak","flat"]}
    s = sum(w.values()); s = s if s>0 else 1.0
    return {k: v/s for k, v in w.items()}

def make_launch_curve(weights: Dict[str,float]) -> np.ndarray:
    curve = np.zeros(12, float)
    for k, a in weights.items():
        curve += LAUNCH_TEMPLATES[k] * float(a)
    curve = np.maximum(curve, 0.0)
    return curve / max(curve.sum(), 1e-12)

def blend_curve_with_seasonality(curve: np.ndarray, seasonality: np.ndarray, beta: float) -> np.ndarray:
    a = np.asarray(curve,float); b = np.asarray(seasonality,float)
    m = min(a.size, b.size)
    out = (1.0 - beta)*a[:m] + beta*b[:m]
    out = np.maximum(out, 0.0)
    return out / (out.sum() if out.sum()>0 else 1.0)

# =========================
# 5) 파라미터/후처리
# =========================
@dataclass
class ModelParams:
    mu_cat: float = 0.0030
    max_share_per_sku: float = 0.05
    elasticity: float = -0.20
    up_rate_limit: float = 2.0
    down_rate_limit: float = 0.55
    ma_window: int = MA_WINDOW_FOR_POST

def moving_average(arr: np.ndarray, win: int = 3) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    n = x.size
    if n == 0 or win <= 1: return x.astype(float)
    win = int(max(1, min(win, n)))
    kernel = np.ones(win, dtype=float) / win
    left_pad  = (win - 1) // 2
    right_pad = win // 2
    a = np.pad(x, (left_pad, right_pad), mode="edge")
    out = np.convolve(a, kernel, mode="valid")
    return out.astype(float)

def postprocess_monthly(m: np.ndarray,
                        up_limit: float = 1.8,
                        down_limit: float = 0.6,
                        ma_window: int = 3,
                        hard_cap: float = np.inf,
                        mono_peak: bool = False) -> np.ndarray:
    x = np.maximum(m.astype(float), 0.0)
    if mono_peak:
        peak = int(np.argmax(x))
        for i in range(1, peak+1):
            x[i] = min(max(x[i], x[i-1]), x[i-1]*up_limit)
        for i in range(peak+1, len(x)):
            x[i] = min(x[i-1], max(x[i-1]*down_limit, x[i]))
    else:
        for i in range(1, len(x)):
            hi = x[i-1]*up_limit; lo = x[i-1]*down_limit
            x[i] = min(max(x[i], lo), hi)
    x = moving_average(x, win=ma_window)
    if np.isfinite(hard_cap): x = np.minimum(x, hard_cap)
    x = np.nan_to_num(x, nan=0.0, posinf=hard_cap, neginf=0.0)
    return x

# =========================
# 6) 예측 (베이스라인)
# =========================
def compute_size_scalar(g: pd.DataFrame, exp: float) -> np.ndarray:
    if exp == 0.0: return np.ones(len(g), float)
    s = g["size_parsed"].astype(float).to_numpy()
    if np.all(np.isnan(s)): return np.ones(len(g), float)
    med = np.nanmedian(s)
    if not np.isfinite(med) or med <= 0: return np.ones(len(g), float)
    base = np.where(np.isfinite(s) & (s>0), s / med, 1.0)
    base = np.clip(base, 1e-6, 1e6)
    return np.power(base, float(exp))

def build_prices(products: pd.DataFrame) -> Tuple[pd.Series, float]:
    if products["대표 판매가"].isna().any():
        missing = products.loc[products["대표 판매가"].isna(), ID_COL].tolist()
        raise RuntimeError(f"[PRICE] 대표 판매가 NaN 존재: {missing}")
    med_cat = products.groupby("category")["대표 판매가"].median().clip(lower=PRICE_FLOOR_WON)
    global_med = float(med_cat.median())
    return med_cat, global_med

def make_cat_monthly_sales_won(pos_pvt_windowed: pd.DataFrame) -> pd.Series:
    if pos_pvt_windowed.shape[1] == 0:
        return pd.Series(0.0, index=pos_pvt_windowed.index)
    return (pos_pvt_windowed.mean(axis=1) * 1e6)

def build_predictions(products: pd.DataFrame,
                      pos_pvt_windowed: pd.DataFrame,
                      params: ModelParams,
                      mu_by_cat: Dict[str, float],
                      size_exponent: float = 0.0) -> pd.DataFrame:
    pro = products.copy()
    cat_avg_prices, global_price = build_prices(pro)
    cat_month_won = make_cat_monthly_sales_won(pos_pvt_windowed)

    rows = []
    for cat, g in pro.groupby("category"):
        n = len(g)
        if n == 0: continue

        cat_price = float(cat_avg_prices.get(cat, global_price))
        cat_price = max(cat_price, PRICE_FLOOR_WON)
        cat_units_month = float(cat_month_won.get(cat, 0.0)) / cat_price

        mu_cat = float(mu_by_cat.get(cat, params.mu_cat))

        price = g["대표 판매가"].astype(float).clip(lower=PRICE_FLOOR_WON).to_numpy()
        invp = 1.0 / price
        kw = g[ID_COL].astype(str).str.contains("프리미엄|그릭|요거트|라떼|카페|바닐라|참기름|리챔|참치액", regex=True).astype(float).to_numpy()
        w_size = compute_size_scalar(g, size_exponent)
        prior = g[ID_COL].map(PRODUCT_PRIORS).fillna(1.0).to_numpy()

        weight = invp * (1.0 + 0.10*kw) * w_size * prior
        weight = weight / weight.sum() if weight.sum()>0 else np.ones(n)/n

        desired_share = weight * mu_cat
        caps = np.array([sku_max_share_limit(g.iloc[i][ID_COL], cat, params.max_share_per_sku) for i in range(n)], dtype=float)
        share = np.minimum(desired_share, caps)
        for _ in range(6):
            deficit = mu_cat - share.sum()
            if deficit <= 1e-12: break
            room = np.maximum(0.0, caps - share)
            total_room = room.sum()
            if total_room <= 1e-12: break
            add = room / total_room * deficit
            share = np.minimum(share + add, caps)

        sku_month = share * cat_units_month

        rel = price / cat_price
        elas = params.elasticity
        price_mult = np.clip(1.0 + elas * (rel - 1.0), 0.85, 1.15)
        sku_month *= price_mult

        season = np.asarray(SEASONALITY_BY_CAT.get(cat, np.ones(12)/12), float)

        for i, (_, r) in enumerate(g.iterrows()):
            name = r[ID_COL]
            p = float(r["대표 판매가"])
            w = blended_curve_weights(name, cat, p)
            curve = make_launch_curve(w)                       # 합=1
            curve = blend_curve_with_seasonality(curve, season, beta=SEASONALITY_BLEND)
            monthly = curve * (sku_month[i] * 12.0)
            monthly = postprocess_monthly(
                monthly,
                up_limit=params.up_rate_limit,
                down_limit=params.down_rate_limit,
                ma_window=params.ma_window,
                hard_cap=sku_hard_cap(name, default=np.inf),
                mono_peak=(not ALLOW_MULTI_PEAK),
            )
            rows.append([name] + monthly.tolist())

    out = pd.DataFrame(rows, columns=[ID_COL] + MONTH_COLS)
    return products[[ID_COL]].merge(out, on=ID_COL, how="left").fillna(0.0)

def apply_persona_boosts(pred: pd.DataFrame, persona_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """월별 예측에 페르소나 부스트 곱. PERSONA_TOTAL_MODE='preserve'이면 총량 보존."""
    if persona_df is None or persona_df.empty:
        return pred

    mcols = [f"months_since_launch_{i}" for i in range(1, 13)]
    bcols = [f"boost_m{i:02d}" for i in range(1, 13)]

    merged = pred.merge(persona_df, on="product_name", how="left")
    for i in range(12):
        bc = bcols[i]; mc = mcols[i]
        boost = merged[bc].fillna(1.0) ** float(PERSONA_STRENGTH)
        merged[mc] = merged[mc].astype(float) * boost.astype(float)

    if PERSONA_TOTAL_MODE == "preserve":
        total_before = pred[mcols].sum(axis=1).replace(0, np.nan).to_numpy()
        total_after  = merged[mcols].sum(axis=1).replace(0, np.nan).to_numpy()
        scale = np.where(np.isfinite(total_before) & np.isfinite(total_after) & (total_after > 0),
                         total_before / total_after, 1.0)
        for mc in mcols:
            merged[mc] = merged[mc].to_numpy() * scale

    return merged[pred.columns]

# =========================
# 7) μ_suggest
# =========================
def suggest_mu_by_category(products: pd.DataFrame,
                           pos_pvt_windowed: pd.DataFrame,
                           params: ModelParams,
                           size_exponent: float = 0.0) -> Dict[str, float]:
    pro = products.copy()
    cat_avg_prices, global_price = build_prices(pro)
    cat_month_won = make_cat_monthly_sales_won(pos_pvt_windowed)

    mu_suggest: Dict[str, float] = {}
    for cat, g in pro.groupby("category"):
        if g.empty: continue
        cat_price = float(cat_avg_prices.get(cat, global_price))
        cat_price = max(cat_price, PRICE_FLOOR_WON)
        cat_units_month = float(cat_month_won.get(cat, 0.0)) / cat_price

        price = g["대표 판매가"].astype(float).clip(lower=PRICE_FLOOR_WON).to_numpy()
        invp = 1.0 / price
        kw = g[ID_COL].astype(str).str.contains("프리미엄|그릭|요거트|라떼|카페|바닐라|참기름|리챔|참치액", regex=True).astype(float).to_numpy()
        w_size = compute_size_scalar(g, size_exponent)
        prior = g[ID_COL].map(PRODUCT_PRIORS).fillna(1.0).to_numpy()
        weight = invp * (1.0 + 0.10*kw) * w_size * prior
        weight = weight / weight.sum() if weight.sum()>0 else np.ones(len(g))/len(g)

        rel = price / cat_price
        elas = params.elasticity
        price_mult = np.clip(1.0 + elas * (rel - 1.0), 0.85, 1.15)

        bounds = []
        for i, (_, r) in enumerate(g.iterrows()):
            name = r[ID_COL]
            cap = sku_hard_cap(name, default=np.inf)
            if not np.isfinite(cap): 
                continue
            p = float(r["대표 판매가"])
            curve = make_launch_curve(blended_curve_weights(name, cat, p))  # 이벤트 없음
            peak_frac = float(np.max(curve))
            denom = cat_units_month * 12.0 * peak_frac * weight[i] * price_mult[i]
            if denom > 0:
                bounds.append(float(cap) / denom)

        mu_suggest[cat] = float(min(bounds))*0.95 if bounds else None
    return mu_suggest

# =========================
# 8) 로그/세이브 (베이스라인 산출)
# =========================
def sanity_and_log(pred: pd.DataFrame,
                   products: pd.DataFrame,
                   pos_pvt_windowed: pd.DataFrame,
                   out_path: Path,
                   params: ModelParams,
                   mu_by_cat: Dict[str, float],
                   size_exponent: float = 0.0):
    cat_avg_prices, global_price = build_prices(products)
    cat_month_won = make_cat_monthly_sales_won(pos_pvt_windowed)
    meta = products[[ID_COL, "category"]].merge(pred[[ID_COL]+MONTH_COLS], on=ID_COL, how="left")
    meta["sum_12m_units"] = meta[MONTH_COLS].sum(axis=1)
    realized = (meta.groupby("category")["sum_12m_units"].sum()).to_frame("units_12m")
    denom = (cat_month_won / cat_avg_prices.reindex(cat_month_won.index).fillna(global_price)).astype(float) * 12.0
    realized["mu_realized"] = realized["units_12m"] / denom
    print("\n[Sanity] Category-level realized μ (≈ target)")
    print(realized["mu_realized"].round(4).sort_index())

    # 디버그: SKU 월 피크
    sku_max_series = pred.set_index(ID_COL)[MONTH_COLS].max(axis=1).astype(int)
    debug_path = SAVE_DIR / "_debug_sku_max.csv"
    try:
        sku_max_series.to_csv(debug_path, encoding="utf-8-sig")
    except PermissionError:
        alt = SAVE_DIR / f"_debug_sku_max_{int(time.time())}.csv"
        sku_max_series.to_csv(alt, encoding="utf-8-sig")
        print(f"[WARN] '{debug_path.name}' 열림 → '{alt.name}' 저장")

    # 제출 파일 저장(중간 산출물)
    tmp = pred.copy()
    if CLIP_MIN is not None:
        tmp[MONTH_COLS] = tmp[MONTH_COLS].clip(lower=CLIP_MIN)
    if ROUND_SUBMISSION:
        tmp[MONTH_COLS] = np.round(tmp[MONTH_COLS])
    tmp[[ID_COL] + MONTH_COLS].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_path}")

# =========================
# 9) 몬테카를로 시뮬레이션
# =========================
MC_MODEL     = "poisson"  # "poisson" | "nb"
NB_K         = 10.0       # Negative Binomial k(크면 포아송에 수렴)
N_SIM        = 5000
PRESERVE_LAM_TOTAL = True # 부스트 후 베이스라인 총량 보존(=lam 스케일)
MC_SEED      = 42

def compute_lambda_from_baseline_and_persona(baseline_df: pd.DataFrame,
                                             persona_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """베이스라인에 페르소나 부스트 적용 → 최종 λ(월별 기대값) DataFrame 반환"""
    if persona_df is not None and not persona_df.empty:
        pred = apply_persona_boosts(baseline_df, persona_df)
    else:
        pred = baseline_df.copy()

    if PRESERVE_LAM_TOTAL and persona_df is not None and not persona_df.empty:
        # apply_persona_boosts가 이미 preserve 총량을 처리함(PERSONA_TOTAL_MODE="preserve")
        pass

    return pred[[ID_COL] + MONTH_COLS].copy()

def sample_counts_poisson(lam: np.ndarray) -> np.ndarray:
    return np.random.poisson(np.maximum(lam, 0.0))

def sample_counts_nb(lam: np.ndarray, k: float) -> np.ndarray:
    # NegBin(r=k, p=k/(k+lam)) → mean=lam, var=lam + lam^2/k
    k = float(max(1e-6, k))
    p = k / (k + np.maximum(lam, 1e-12))
    return np.random.negative_binomial(k, p)

def run_mc_for_all(lam_df: pd.DataFrame,
                   model: str = "poisson",
                   nb_k: float = 10.0,
                   n_sim: int = 5000,
                   hard_cap_map: Optional[Dict[str, float]] = None,
                   seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """
    lam_df: [product_name, m1..m12]
    반환:
      - mean_df: 제품별 월별 평균(제출용)
      - dist_map: 제품별 요약분포(예: quantiles) {name: {"mean":arr,"p2.5":arr,"p97.5":arr}}
    """
    rng = np.random.default_rng(seed)
    names = lam_df[ID_COL].tolist()
    lam_mat = lam_df[MONTH_COLS].to_numpy(dtype=float)  # [N,12]

    means = np.zeros_like(lam_mat)
    q025  = np.zeros_like(lam_mat)
    q975  = np.zeros_like(lam_mat)

    dist_map: Dict[str, Dict[str, np.ndarray]] = {}

    for i, name in enumerate(names):
        lam = lam_mat[i]
        sims = np.zeros((n_sim, 12), dtype=float)
        for t in range(n_sim):
            if model == "poisson":
                s = rng.poisson(lam)
            elif model == "nb":
                s = sample_counts_nb(lam, k=nb_k)
            else:
                raise ValueError("model ∈ {'poisson','nb'}")
            if hard_cap_map:
                cap = hard_cap_map.get(name, np.inf)
                s = np.minimum(s, cap)
            sims[t,:] = s
        means[i,:] = sims.mean(axis=0)
        q025[i,:]  = np.percentile(sims,  2.5, axis=0)
        q975[i,:]  = np.percentile(sims, 97.5, axis=0)
        dist_map[name] = {"mean": means[i].copy(), "p2.5": q025[i].copy(), "p97.5": q975[i].copy()}

    mean_df = pd.DataFrame({ID_COL: names})
    for j, mc in enumerate(MONTH_COLS):
        mean_df[mc] = means[:, j]
    return mean_df, dist_map

# =========================
# 10) 제출 파일: 샘플 형식 맞추기
# =========================
def detect_submission_columns_keep_order(sample_df: pd.DataFrame) -> Tuple[str, List[str]]:
    """샘플 제출 파일에서 ID 컬럼(첫 컬럼)과 월 컬럼(12개)을 원래 순서 그대로 추출"""
    cols = list(sample_df.columns)
    id_col = cols[0]
    # 우선 months_since_launch_1..12 패턴으로 탐색
    pat = re.compile(r"^months_since_launch_(\d{1,2})$")
    month_cols = [c for c in cols if pat.match(c)]
    if len(month_cols) == 12:
        month_cols = sorted(month_cols, key=lambda c: int(pat.match(c).group(1)))
    else:
        # fallback: 첫 컬럼 제외 모두 월 컬럼으로 간주
        month_cols = [c for c in cols if c != id_col]
        if len(month_cols) != 12:
            raise RuntimeError("샘플에서 월 컬럼(12개) 탐지 실패: " + ",".join(cols))
    return id_col, month_cols

def fill_submission_like_sample(sample_df: pd.DataFrame,
                                id_col: str,
                                month_cols: List[str],
                                result_df: pd.DataFrame,
                                round_int: bool = True,
                                clip_min: Optional[float] = 1.0) -> pd.DataFrame:
    """
    sample_df 형식/순서를 그대로 유지하여 result_df의 예측을 채움.
    result_df: [product_name, months_since_launch_1..12]
    """
    out = sample_df.copy()
    res = result_df.set_index(ID_COL)

    for name, row in res.iterrows():
        vals = row[MONTH_COLS].astype(float).to_numpy()
        if clip_min is not None:
            vals = np.clip(vals, clip_min, None)
        if round_int:
            vals = np.round(vals).astype(int)
            vals = np.maximum(vals, 1)
        if (out[id_col] == name).any():
            out.loc[out[id_col] == name, month_cols] = vals
        else:
            new_row = {id_col: name}
            new_row.update({c: v for c, v in zip(month_cols, vals)})
            append_df = pd.DataFrame([new_row], columns=list(out.columns))
            out = pd.concat([out, append_df], axis=0, ignore_index=True)

    # 안전: 타입/NaN 정리
    for c in month_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(1)
        if round_int:
            out[c] = out[c].astype(int)
    return out

# =========================
# 11) MAIN
# =========================
def main():
    # 1) 데이터 로드
    products = load_products()
    pos_pivot_windowed = load_pos()
    SKU_CAPS_RUNTIME.update(load_sku_caps(products))
    load_category_curve_priors()
    load_mx_maps(products)
    global SEASONALITY_BY_CAT
    SEASONALITY_BY_CAT = build_category_seasonality_map(products, pos_pivot_windowed)

    # 2) μ_suggest 계산 → 스케일 적용
    size_exp_for_mu = SUBMISSION_COMBOS[0][2]
    mu_suggest = suggest_mu_by_category(products, pos_pivot_windowed, params=ModelParams(), size_exponent=size_exp_for_mu)
    print("\n[Suggest μ by category from CAPS]")
    mu_fixed = {}
    for cat in sorted(products["category"].unique()):
        v = mu_suggest.get(cat, None)
        mu_fixed[cat] = float(v) if v is not None else 0.0030
        print(f"  - {cat}: μ = {mu_fixed[cat]:.4f} ({'suggest' if v is not None else 'fallback'})")
    if MU_SCALE != 1.00:
        mu_fixed = {k: max(1e-6, float(v) * float(MU_SCALE)) for k, v in mu_fixed.items()}
        print(f"[MU] scaled by {MU_SCALE:.3f}")
    (pd.DataFrame({"category": list(mu_fixed.keys()),
                   "mu": [mu_fixed[c] for c in mu_fixed]})
        .sort_values("category")
        .to_csv(MU_BY_CAT_OUT_PATH, index=False, encoding="utf-8-sig"))
    print(f"[MU] saved → {MU_BY_CAT_OUT_PATH.as_posix()}")

    # 3) 페르소나 JSON → 부스트 DF
    persona_json = load_persona_json(PERSONA_JSON_PATH)
    if persona_json is not None:
        persona_map = aggregate_persona_patterns(persona_json)  # {product: len12}
        persona_df = persona_map_to_dataframe(persona_map)      # [product, boost_m01..12]
    else:
        persona_df = None

    # 4) 배치 예측(베이스라인) + (옵션)페르소나 부스트 적용 → 중간 산출 저장
    print(f"[BATCH] prepare {len(SUBMISSION_COMBOS)} submissions  |  season_blend={SEASONALITY_BLEND}  multi_peak={ALLOW_MULTI_PEAK}")
    pred_mats: List[np.ndarray] = []
    base_ids = products[ID_COL].tolist()

    for (mx_global, el, szexp) in SUBMISSION_COMBOS:
        params = ModelParams(max_share_per_sku=mx_global, elasticity=el)
        pred = build_predictions(products, pos_pivot_windowed, params, mu_fixed, size_exponent=szexp)

        # 페르소나 분포 부스트 (모양만 반영, 총량 보존/스케일은 옵션)
        if persona_df is not None:
            pred = apply_persona_boosts(pred, persona_df)

        fname = f"submission_v9_baseline_mu{params.mu_cat:.4f}_mx{mx_global:.3f}_el{el:+.2f}_size{szexp:+.2f}_QSEASON.csv"
        out_path = SAVE_DIR / fname
        sanity_and_log(pred, products, pos_pivot_windowed, out_path, params, mu_fixed, size_exponent=szexp)

        pm = (pred.set_index(ID_COL)
                  .reindex(base_ids)[MONTH_COLS]
                  .to_numpy())
        pred_mats.append(pm)

    # 5) 앙상블(가중 평균)
    if pred_mats:
        stack = np.stack(pred_mats, axis=0)  # [K, N, 12]
        base_w = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)
        w = base_w[:stack.shape[0]] if stack.shape[0] <= len(base_w) else np.ones(stack.shape[0], float)
        w = w / w.sum()
        ens = (w[:,None,None] * stack).sum(axis=0)
        ens_df = products[[ID_COL]].copy()
        ens_df[MONTH_COLS] = ens
        if CLIP_MIN is not None:
            ens_df[MONTH_COLS] = ens_df[MONTH_COLS].clip(lower=CLIP_MIN)
        ens_path = SAVE_DIR / "submission_v9_ENSEMBLED_baseline.csv"
        ens_df.to_csv(ens_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {ens_path}  (ensemble of {len(pred_mats)} members with weights {w.tolist()})")
    else:
        raise RuntimeError("예측 행렬이 비어 있습니다.")

    # 6) (선택) 페르소나 부스트를 앙상블 후 한 번 더 적용하고 싶다면 아래 주석 해제
    # if persona_df is not None:
    #     ens_df = apply_persona_boosts(ens_df, persona_df)

    # 7) 몬테카를로: λ = 앙상블(이미 부스트 반영됨) → Poisson/NB 샘플링
    lam_df = compute_lambda_from_baseline_and_persona(ens_df, persona_df=None)  # ens_df가 이미 부스트 반영이라면 None
    hard_cap_map = {k: float(v) for k, v in SKU_CAPS_RUNTIME.items()} if SKU_CAPS_RUNTIME else None
    mean_df, dist_map = run_mc_for_all(
        lam_df,
        model=MC_MODEL,
        nb_k=NB_K,
        n_sim=N_SIM,
        hard_cap_map=hard_cap_map,
        seed=MC_SEED
    )

    # 월별 요약 통계(옵션 저장)
    q025_df = mean_df.copy()
    q975_df = mean_df.copy()
    for name, q in dist_map.items():
        q025_df.loc[q025_df[ID_COL]==name, MONTH_COLS] = q["p2.5"]
        q975_df.loc[q975_df[ID_COL]==name, MONTH_COLS] = q["p97.5"]
    mean_df.to_csv(SAVE_DIR / "mc_monthly_mean.csv", index=False, encoding="utf-8-sig")
    q025_df.to_csv(SAVE_DIR / "mc_monthly_p2.5.csv", index=False, encoding="utf-8-sig")
    q975_df.to_csv(SAVE_DIR / "mc_monthly_p97.5.csv", index=False, encoding="utf-8-sig")
    print("[SAVE] mc_monthly_mean.csv / mc_monthly_p2.5.csv / mc_monthly_p97.5.csv")

    # 8) 제출 파일: 샘플 형식 유지하여 채우기
    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(f"샘플 제출 파일을 찾을 수 없습니다: {SAMPLE_SUB_PATH}")
    sample_df = pd.read_csv(SAMPLE_SUB_PATH)
    id_col, month_cols = detect_submission_columns_keep_order(sample_df)

    submission_df = fill_submission_like_sample(
        sample_df=sample_df,
        id_col=id_col,
        month_cols=month_cols,
        result_df=mean_df,          # 시뮬레이션 평균으로 제출
        round_int=True,
        clip_min=CLIP_MIN
    )
    if ROUND_SUBMISSION:
        submission_df[month_cols] = np.round(submission_df[month_cols])

    submission_df.to_csv(FINAL_SUB_PATH, index=False, encoding="utf-8-sig")
    print(f"[SUBMIT] {FINAL_SUB_PATH}  | rows={len(submission_df)}  id_col='{id_col}'")

    print("\n✅ Done: 베이스라인(현실 앵커) + 페르소나(모양) + MC(불확실성) → 제출 파일 생성 완료.")

if __name__ == "__main__":
    main()
