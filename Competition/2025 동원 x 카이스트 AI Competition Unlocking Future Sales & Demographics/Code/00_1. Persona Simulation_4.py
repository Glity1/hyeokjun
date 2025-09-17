# -*- coding: utf-8 -*-
"""
Dongwon — L1 Anchor Pipeline (카테고리 앵커 고정 + (옵션) 바이어스 캘리브레이션 + 월·SKU 분배)
전략 고정:
  1) 카테고리별 앵커 방법
     method_by_cat = { "발효유": "ets", "어육가공품": "yoy", "전통기름": "yoy", "조미료": "yoy", "커피": "yoy_ets_ensemble" }
     - 'yoy_ets_ensemble'은 YoY 0.8 + ETS 0.2
  2) (선택) 2024H2 실측으로 바이어스 캘리브레이션
     bias = Actual_2024H2 / Pred_2024H2(method)
     Adj_2025H1 = Pred_2025H1 ** (1-λ) * (Pred_2025H1 * bias) ** λ  (λ ∈ [0.3, 0.5])
  3) 월·SKU 분배
     - POS 2023 월 시즌으로 반기 앵커 → 월(반기 내부 6개월 정규화)
     - 카테고리→SKU: s_p,t = α_p * w_p,t (α_p는 SKU 정적 가중(기본 균등), w_p,t는 페르소나 월형상 혼합)
       → share_p,t = s_p,t / Σ_p s_p,t → qty_p,t = cat_qty_t * share_p,t
  4) 제출 파일(wide): sample_submission.csv 기준으로 저장

필요 파일 경로:
- 닐슨 분기: /mnt/data/닐슨코리아_2011_2019_분기별 매출액.xlsx
- 닐슨 반기: /mnt/data/닐슨코리아_2020_2024 반기별 매출액.xlsx
- POS 월:    /mnt/data/marketlink_POS_master.xlsx  (없으면 균등 시즌)
- 제품 정보: /mnt/data/product_info.csv           (없으면 균등 α)
- 페르소나:  ./_data/dacon/dongwon/personas.json
- 샘플 제출: ./_data/dacon/dongwon/sample_submission.csv
"""

from __future__ import annotations
import warnings, re, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# =========================
# 경로 & 고정 설정
# =========================
# (필요 시 경로만 수정)
NIELSEN_Q_FP = Path("./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2019_분기별 매출액.xlsx")
NIELSEN_H_FP = Path("./_data/dacon/dongwon/pos_data/닐슨코리아_2020_2024 반기별 매출액.xlsx")
POS_FP       = Path("./_data/dacon/dongwon/pos_data/marketlink_POS_master.xlsx")    # 2023 월 시즌. 없으면 균등
PRODUCT_INFO_FP = Path("./_data/dacon/dongwon/product_info.csv")            # α 가중에 활용(없으면 균등)
DATA_DIR     = Path("./_data/dacon/dongwon")
PERSONA_FP   = DATA_DIR / "personas.json"
SAMPLE_FP    = DATA_DIR / "sample_submission.csv"
SAVE_DIR     = Path("./_save/anchor_pipeline"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 기간/월
EVAL_START = "2024-07"; EVAL_END = "2025-06"
MONTHS = pd.period_range(EVAL_START, EVAL_END, freq="M").strftime("%Y-%m").tolist()
HALF_OF = {m: ("2024H2" if m[:4]=="2024" and int(m[5:])>=7 else "2025H1") for m in MONTHS}
HALF_MONTHS = {"2024H2":[f"2024-{mm:02d}" for mm in range(7,13)],
               "2025H1":[f"2025-{mm:02d}" for mm in range(1,7)]}

# 카테고리 고정 맵(화이트리스트 15개)
CATEGORY_OF = {
    "덴마크 하이그릭요거트 400g": "발효유",
    "소화가 잘되는 우유로 만든 바닐라라떼 250mL": "커피",
    "소화가 잘되는 우유로 만든 카페라떼 250mL": "커피",
    "동원맛참 고소참기름 90g": "전통기름",
    "동원맛참 고소참기름 135g": "전통기름",
    "동원맛참 매콤참기름 90g": "전통기름",
    "동원맛참 매콤참기름 135g": "전통기름",
    "동원참치액 순 500g": "조미료",
    "동원참치액 순 900g": "조미료",
    "동원참치액 진 500g": "조미료",
    "동원참치액 진 900g": "조미료",
    "프리미엄 동원참치액 500g": "조미료",
    "프리미엄 동원참치액 900g": "조미료",
    "리챔 오믈레햄 200g": "어육가공품",
    "리챔 오믈레햄 340g": "어육가공품",
}

# 카테고리별 앵커 방법 고정
METHOD_BY_CAT = {
    "발효유": "ets",
    "어육가공품": "yoy",
    "전통기름": "yoy",
    "조미료": "yoy",
    "커피": "yoy_ets_ensemble",  # 0.8*YoY + 0.2*ETS
}
ENSEMBLE_W = {"YoY": 0.8, "ETS": 0.2}  # 커피용

# 바이어스 보정 파라미터
USE_BIAS_CAL = True     # 2024H2 실측으로 2025H1 부분 보정
BIAS_LAMBDA  = 0.4      # λ in [0.3, 0.5]
USE_ACTUAL_2024H2_AS_ANCHOR = False  # True면 2024H2 앵커는 실측으로 대체(검증 관점)

# 페르소나 섞기 파라미터
TAU = 0.7               # softmax temperature

# =========================
# 유틸
# =========================
def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    return ex / s if s>0 else np.ones_like(ex)/len(ex)

def _fmt(x):
    return "nan" if (x is None or (isinstance(x,float) and not np.isfinite(x))) else f"{x:,.0f}"

# =========================
# 닐슨 로더 (분기/반기 → 반기 통합)
# =========================
def read_nielsen_quarter_kor(fp: Path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(fp, sheet_name=sheet_name)
    need = ['연도','분기','카테고리','매출액(백만원)']
    for c in need:
        if c not in df.columns:
            raise ValueError(f"분기 파일 컬럼 누락: {c} / 실제: {list(df.columns)}")
    out = df[need].copy()
    out.columns = ["year","quarter","category","amount"]
    out["quarter"] = out["quarter"].astype(str).str.extract(r"(\d+)")[0].astype(int).clip(1,4)
    out["amount"]  = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out["category"] = out["category"].astype(str).str.strip()
    return out

def read_nielsen_half_kor(fp: Path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(fp, sheet_name=sheet_name)
    half_col = "반기" if "반기" in df.columns else ("분기" if "분기" in df.columns else None)
    if half_col is None:
        raise ValueError(f"반기 파일에 '반기' 또는 '분기' 컬럼이 없습니다. 실제: {list(df.columns)}")
    need = ['연도', half_col, '카테고리', '매출액(백만원)']
    for c in need:
        if c not in df.columns:
            raise ValueError(f"반기 파일 컬럼 누락: {c} / 실제: {list(df.columns)}")
    out = df[need].copy()
    out.columns = ["year","half_raw","category","amount"]
    s = out["half_raw"].astype(str).str.upper()
    out["half"] = np.where(s.str.contains("2"), "H2", "H1")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out["category"] = out["category"].astype(str).str.strip()
    return out[["category","year","half","amount"]]

def quarter_to_half(qdf: pd.DataFrame) -> pd.DataFrame:
    qq = qdf.copy()
    qq["half"] = np.where(qq["quarter"].isin([1,2]), "H1", "H2")
    hh = qq.groupby(["category","year","half"], as_index=False)["amount"].sum()
    return hh

def concat_half_series(h_from_q: pd.DataFrame, h_direct: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([h_from_q, h_direct], ignore_index=True)
    # 혹시 중복 있으면 합산
    df = df.groupby(["category","year","half"], as_index=False)["amount"].sum()
    order = {"H1":1, "H2":2}
    df["_ho"] = df["half"].map(order)
    df = df.sort_values(["category","year","_ho"]).drop(columns="_ho").reset_index(drop=True)
    return df

# =========================
# 앵커 예측 (YoY/ETS + ensemble + 바이어스)
# =========================
def forecast_yoy_for_cat(train_half_df_cat: pd.DataFrame) -> dict:
    """train_half_df_cat: 2011~2024H1 (2024H2 제외)"""
    g = (train_half_df_cat.groupby(["year","half"], as_index=False)["amount"].sum()
                         .sort_values(["year","half"]))
    piv = g.pivot(index="year", columns="half", values="amount").sort_index()
    a_2023h1 = float(piv.loc[2023, "H1"]) if (2023 in piv.index and "H1" in piv.columns) else np.nan
    a_2023h2 = float(piv.loc[2023, "H2"]) if (2023 in piv.index and "H2" in piv.columns) else np.nan
    a_2024h1 = float(piv.loc[2024, "H1"]) if (2024 in piv.index and "H1" in piv.columns) else np.nan

    if np.isfinite(a_2023h2) and np.isfinite(a_2023h1) and np.isfinite(a_2024h1) and a_2023h1>0:
        r1 = a_2024h1 / a_2023h1
        y_2024h2 = a_2023h2 * r1
    else:
        r1 = np.nan; y_2024h2 = a_2023h2 if np.isfinite(a_2023h2) else np.nan

    if np.isfinite(y_2024h2) and np.isfinite(a_2023h2) and np.isfinite(a_2024h1) and a_2023h2>0:
        r2 = y_2024h2 / a_2023h2
        y_2025h1 = a_2024h1 * r2
    else:
        r2 = np.nan; y_2025h1 = a_2024h1 if np.isfinite(a_2024h1) else np.nan

    return {"2024H2": y_2024h2, "2025H1": y_2025h1, "r1": r1, "r2": r2}

def forecast_ets_for_cat(train_half_df_cat: pd.DataFrame) -> dict:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception:
        return {"2024H2": np.nan, "2025H1": np.nan}
    g = (train_half_df_cat.groupby(["year","half"], as_index=False)["amount"].sum()
                         .sort_values(["year","half"]))
    y = g["amount"].astype(float).values
    if len(y) < 6:
        return {"2024H2": np.nan, "2025H1": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        f = fit.forecast(2)  # next 2 halves
    return {"2024H2": float(f[0]), "2025H1": float(f[1])}

def decide_anchor_for_category(cat: str, train_df_cat: pd.DataFrame, actual_2024H2: float|None) -> dict:
    """카테고리별 방법 적용 + (옵션) 바이어스 보정까지 반영한 앵커 반환"""
    method = METHOD_BY_CAT.get(cat, "yoy")
    yoy = forecast_yoy_for_cat(train_df_cat)
    ets = forecast_ets_for_cat(train_df_cat)

    if method == "yoy":
        h2_pred, h1_pred = yoy["2024H2"], yoy["2025H1"]
    elif method == "ets":
        h2_pred, h1_pred = ets["2024H2"], ets["2025H1"]
    elif method == "yoy_ets_ensemble":
        # 0.8*YoY + 0.2*ETS
        h2_pred = ENSEMBLE_W["YoY"] * yoy["2024H2"] + ENSEMBLE_W["ETS"] * ets["2024H2"]
        h1_pred = ENSEMBLE_W["YoY"] * yoy["2025H1"] + ENSEMBLE_W["ETS"] * ets["2025H1"]
    else:
        h2_pred, h1_pred = yoy["2024H2"], yoy["2025H1"]

    # (옵션) 2024H2 실측으로 바이어스 → 2025H1 부분 보정
    if USE_BIAS_CAL and actual_2024H2 is not None and np.isfinite(actual_2024H2) and np.isfinite(h2_pred) and h2_pred>0:
        bias = actual_2024H2 / h2_pred
        # 혼합 지수 방식
        h1_adj = (h1_pred ** (1.0 - BIAS_LAMBDA)) * ((h1_pred * bias) ** (BIAS_LAMBDA))
    else:
        bias = np.nan
        h1_adj = h1_pred

    # (옵션) 2024H2 앵커를 실측으로 대체
    h2_final = actual_2024H2 if (USE_ACTUAL_2024H2_AS_ANCHOR and actual_2024H2 is not None and np.isfinite(actual_2024H2)) else h2_pred
    h1_final = h1_adj

    return {"method": method, "2024H2": h2_final, "2025H1": h1_final, "pred_raw_2024H2": h2_pred, "bias": bias}

# =========================
# POS 2023 월 시즌 로더 (없으면 균등)
# =========================
def load_pos_monthly_season_2023(fp: Path) -> dict[str, dict[int, float]]:
    """
    반환: {category: {1: wJan, ..., 12: wDec}}, 각 카테고리 합계=12 (월평균=1) 형태
    입력 파일 컬럼이 다양할 수 있어 유연 매핑. 실패 시 빈 dict 반환.
    """
    if not fp.exists():
        return {}
    try:
        df = pd.read_excel(fp)
    except Exception:
        try:
            df = pd.read_csv(fp)
        except Exception:
            return {}

    # 컬럼 추론
    cols = {c:str(c).lower() for c in df.columns}
    cat_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["카테고리","category","cat"])), None)
    y_col   = next((c for c in df.columns if any(k in str(c).lower() for k in ["연도","year"])), None)
    m_col   = next((c for c in df.columns if any(k in str(c).lower() for k in ["월","month","mm"])), None)
    date_col= next((c for c in df.columns if any(k in str(c).lower() for k in ["일자","date"])), None)
    val_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["수량","매출","판매","units","amount","qty","value"])), None)

    if cat_col is None or val_col is None:
        return {}

    tmp = df.copy()
    tmp[cat_col] = tmp[cat_col].astype(str).str.strip()

    # 연/월 만들기
    if date_col is not None and date_col in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp["year"] = tmp["date"].dt.year
        tmp["month"]= tmp["date"].dt.month
    else:
        if y_col is None or m_col is None:
            return {}
        tmp["year"]  = pd.to_numeric(tmp[y_col], errors="coerce")
        tmp["month"] = pd.to_numeric(tmp[m_col], errors="coerce")

    tmp[val_col] = pd.to_numeric(tmp[val_col], errors="coerce").fillna(0.0)
    tmp = tmp.dropna(subset=["year","month"])
    tmp["year"] = tmp["year"].astype(int)
    tmp["month"]= tmp["month"].astype(int)

    # 2023년만
    tmp = tmp[tmp["year"]==2023]
    if tmp.empty:
        return {}

    season = {}
    for cat, g in tmp.groupby(tmp[cat_col]):
        s = g.groupby("month")[val_col].sum()
        # 1~12 월 모두 생성, 결측은 0
        s = s.reindex(range(1,13), fill_value=0.0)
        # 월평균=1 정규화(합=12)
        if s.sum() > 0:
            w = (s / s.mean()).to_dict()
        else:
            w = {m:1.0 for m in range(1,13)}
        season[str(cat)] = w
    return season

def get_half_norm_weights_from_season(season12: dict[int,float], half_key: str) -> np.ndarray:
    """season12: {1..12: w}, half_key: '2024H2' or '2025H1' -> 길이 6, 합=1"""
    if half_key == "2024H2":
        months = [7,8,9,10,11,12]
    else:
        months = [1,2,3,4,5,6]
    arr = np.array([season12.get(m, 1.0) for m in months], dtype=float)
    s = arr.sum()
    if s <= 0:
        return np.ones(6)/6.0
    return arr / s

# =========================
# 페르소나 로더 & 월형상
# =========================
def load_personas(fp: Path) -> dict:
    d = pd.read_json(fp, typ="dict")
    return d

def persona_month_shape(per: dict) -> np.ndarray:
    """12개월 배열(평균=1) 반환"""
    if "monthly_by_launch" in per:
        w = np.array(per["monthly_by_launch"], float)
    else:
        d = per["monthly_by_calendar"]
        # 키가 정확히 12개(2024-07~2025-06)라고 가정
        order = MONTHS
        w = np.array([d[m] for m in order], float)
    return w / w.mean()

def mix_persona_shapes(product_personas: list[dict]) -> np.ndarray:
    """purchase_probability 기반 softmax 가중 혼합 → 12개월 합=1"""
    probs = np.array([p["purchase_probability"]/100.0 for p in product_personas], float)
    s = softmax(probs / TAU)
    W = np.vstack([persona_month_shape(p) for p in product_personas])  # (k,12)
    u = (s[:,None]*W).sum(axis=0)  # 평균=1 정도, 아래서 합=1로
    u = u / u.sum()
    return u  # length=12, sum=1

# =========================
# 제품 α (정적 가중; 기본 균등)
# =========================
def compute_alpha_per_category(products_by_cat: dict[str, list[str]],
                               personas: dict,
                               product_info_fp: Path|None=None,
                               gamma: float = 1.1) -> dict[str, dict[str,float]]:
    """
    반환: {cat: {product: α}}  (카테고리별 Σα = 1)
    - 커피 카테고리: 페르소나 구매확률 기반 + (있으면) product_info의 hint 가중
    - 나머지 카테고리: 기본 균등 (원하면 동일 로직 확장 가능)
    gamma: 페르소나 기반 차이를 강조하는 지수(>1이면 차이 확대)
    허용되는 hint 컬럼명 예: ['alpha_hint','baseline_share','alpha','share_hint','가중치']
    """
    # 1) product_info 로드 (옵션)
    hints = {}
    if product_info_fp is not None and product_info_fp.exists():
        try:
            # 확장자에 따라 읽기
            if product_info_fp.suffix.lower() in [".xlsx", ".xls"]:
                pdf = pd.read_excel(product_info_fp)
            else:
                pdf = pd.read_csv(product_info_fp)
            # 제품명 컬럼 찾기
            name_col = next((c for c in pdf.columns if str(c).lower() in ["product_name","제품명","product","sku"]), None)
            if name_col is None:
                name_col = pdf.columns[0]
            # 힌트 컬럼 찾기
            cand = ["alpha_hint","baseline_share","alpha","share_hint","가중치"]
            hint_col = next((c for c in pdf.columns if str(c).lower() in cand), None)
            if hint_col is not None:
                tmp = pdf[[name_col, hint_col]].copy()
                tmp.columns = ["product","hint"]
                tmp["hint"] = pd.to_numeric(tmp["hint"], errors="coerce")
                hints = {r.product: float(r.hint) for r in tmp.itertuples(index=False) if pd.notna(r.hint) and r.hint>0}
        except Exception:
            hints = {}

    alpha: dict[str, dict[str,float]] = {}
    for c, prods in products_by_cat.items():
        if not prods:
            alpha[c] = {}
            continue

        # 기본: 균등
        w_raw = {p: 1.0 for p in prods}

        if c == "커피":
            # 2) 페르소나 기반 스코어
            for p in prods:
                per_arr = personas[p]
                # purchase_probability 평균 (0~100) → 0~1 스케일 → 지수 gamma로 강조
                probs = [float(per.get("purchase_probability", 0)) for per in per_arr]
                persona_score = max(1e-9, np.mean(probs) / 100.0) ** gamma

                # 3) product_info 힌트가 있으면 결합(기하 평균 느낌: 60% 힌트, 40% 페르소나)
                hint = hints.get(p, None)
                if hint is not None and np.isfinite(hint) and hint > 0:
                    # hint는 스케일 자유(0~1이든 0~100이든) → 양수로만 사용, 상대 비교만 중요
                    hint_score = float(hint)
                    w_raw[p] = (persona_score ** 0.4) * (hint_score ** 0.6)
                else:
                    w_raw[p] = persona_score

        # 4) 정규화(Σα=1)
        tot = sum(w_raw.values())
        if tot <= 0:
            alpha[c] = {p: 1.0/len(prods) for p in prods}
        else:
            alpha[c] = {p: w_raw[p]/tot for p in prods}

    return alpha

# =========================
# 메인 파이프라인
# =========================
if __name__ == "__main__":
    # 0) 데이터 로드
    # 닐슨 반기 통합
    qdf = read_nielsen_quarter_kor(NIELSEN_Q_FP)
    hdf = read_nielsen_half_kor(NIELSEN_H_FP)
    h_from_q = quarter_to_half(qdf)
    half_all = concat_half_series(h_from_q, hdf)  # category, year, half, amount

    # ============================================================
    # Nielsen 분기+반기 → 반기 단일 통합 (정리 + 가드 + 저장)
    # - category 클린업 / 연도 가드 / 중복 시 half_direct 우선
    # - RAW/AGG/PIVOT 저장 (엑셀 엔진 자동 폴백)
    # - 이후 파이프라인 입력: half_all = AGG
    # ============================================================
    import pandas as pd
    import numpy as np
    from pathlib import Path

    COMBINE_DIR = SAVE_DIR / "nielsen_combined"
    COMBINE_DIR.mkdir(parents=True, exist_ok=True)

    # 0) 소스 라벨 부여
    q_half_raw = h_from_q.copy()
    q_half_raw["source"] = "from_quarter"   # 2011~2019 기대
    h_half_raw = hdf.copy()
    h_half_raw["source"] = "half_direct"    # 2020~2024 기대

    # 1) 이어붙인 뒤 category 클린업
    raw = pd.concat([q_half_raw, h_half_raw], ignore_index=True)
    raw["category"] = raw["category"].astype(str).str.strip()
    valid_mask = raw["category"].notna() & (raw["category"].str.len() > 0) & (raw["category"].str.lower() != "nan")
    dropped_cat = int((~valid_mask).sum())
    raw = raw[valid_mask].copy()

    # 2) 연도 가드: from_quarter ≤2019, half_direct ≥2020
    before_n = len(raw)
    raw = raw[~((raw["source"] == "from_quarter") & (raw["year"] >= 2020))].copy()
    raw = raw[~((raw["source"] == "half_direct") & (raw["year"] <= 2019))].copy()
    dropped_year_guard = before_n - len(raw)

    # 3) 중복 키에서 half_direct 우선
    #    (category, year, half)가 half_direct에 있으면 from_quarter 동일 키는 제거
    keys_half = set(h_half_raw.assign(source="half_direct")[["category","year","half"]].apply(tuple, axis=1))
    from_mask = (raw["source"] == "from_quarter")
    dup_mask = from_mask & raw[["category","year","half"]].apply(tuple, axis=1).isin(keys_half)
    dropped_overlap = int(dup_mask.sum())
    raw = raw[~dup_mask].copy()

    # 4) 정렬
    order_half = {"H1": 1, "H2": 2}
    raw["_ho"] = raw["half"].map(order_half)
    raw = raw.sort_values(["category","year","_ho","source"]).drop(columns="_ho").reset_index(drop=True)

    # 5) AGG 집계(모델 입력용) & PIVOT(검수용)
    agg = (raw.groupby(["category","year","half"], as_index=False)["amount"]
            .sum()
            .sort_values(["category","year","half"])
            .reset_index(drop=True))

    pivot = (agg.pivot_table(index=["category","year"], columns="half", values="amount", fill_value=0)
                .reset_index()
                .sort_values(["category","year"]))

    # 6) 저장 (CSV + XLSX; openpyxl 고정)
    raw_csv   = COMBINE_DIR / "nielsen_half_combined_raw.csv"
    agg_csv   = COMBINE_DIR / "nielsen_half_combined_agg.csv"
    pivot_csv = COMBINE_DIR / "nielsen_half_combined_pivot.csv"
    raw_xlsx   = COMBINE_DIR / "nielsen_half_combined_raw.xlsx"
    agg_xlsx   = COMBINE_DIR / "nielsen_half_combined_agg.xlsx"
    pivot_xlsx = COMBINE_DIR / "nielsen_half_combined_pivot.xlsx"

    # CSV는 항상 저장
    raw.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    agg.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    pivot.to_csv(pivot_csv, index=False, encoding="utf-8-sig")

    # 엑셀 저장 (openpyxl 고정)
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("⚠️ 'openpyxl'이 설치되어 있지 않습니다. 엑셀(xlsx) 저장을 건너뜁니다. "
            "다음 명령으로 설치 후 다시 실행하세요: pip install openpyxl")
    else:
        with pd.ExcelWriter(raw_xlsx, engine="openpyxl") as w:
            raw.to_excel(w, index=False, sheet_name="raw")
        with pd.ExcelWriter(agg_xlsx, engine="openpyxl") as w:
            agg.to_excel(w, index=False, sheet_name="agg")
        with pd.ExcelWriter(pivot_xlsx, engine="openpyxl") as w:
            pivot.to_excel(w, index=False, sheet_name="pivot")
        print("[Excel 저장 완료] openpyxl 엔진으로 RAW/AGG/PIVOT xlsx 저장")

    # 7) 무결성/범위 요약 출력
    yr_tbl = raw.groupby("source")["year"].agg(["min","max","nunique"]).reset_index()
    print("\n[정리 요약]")
    print(f" - category 클린업 드롭: {dropped_cat}행")
    print(f" - 연도 가드 드롭     : {dropped_year_guard}행")
    print(f" - 중복키 드롭(from_quarter → half_direct 우선): {dropped_overlap}행")
    print("\n[출처별 연도 범위]")
    print(yr_tbl.to_string(index=False))

    # 8) 이후 파이프라인 입력도 클린 AGG로 교체
    half_all = agg.copy()

    # 카테고리 목록 & 2024H2 실측(있으면 사용)
    cats = sorted(half_all["category"].unique())
    actual_2024H2_by_cat = {}
    for c in cats:
        row = half_all[(half_all["category"]==c) & (half_all["year"]==2024) & (half_all["half"]=="H2")]
        actual_2024H2_by_cat[c] = float(row["amount"].sum()) if not row.empty else None

    # POS 2023 월 시즌 (없으면 빈 dict)
    season_pos = load_pos_monthly_season_2023(POS_FP)

    # 페르소나 로드
    personas = pd.read_json(PERSONA_FP, typ="dict")
    # 제품 리스트/카테고리 매핑
    sku_list = list(personas.keys())
    for p in sku_list:
        if p not in CATEGORY_OF:
            raise ValueError(f"화이트리스트 외 제품: {p}")
    products_by_cat = defaultdict(list)
    for p in sku_list:
        products_by_cat[CATEGORY_OF[p]].append(p)

    # 제품별 페르소나 월형상(12개월, 합=1)
    Wp = {}  # product -> 12-array
    for p in sku_list:
        Wp[p] = mix_persona_shapes(personas[p])  # sum=1

    # 제품 α(정적 가중) — 기본 균등
    alpha = compute_alpha_per_category(products_by_cat, personas, PRODUCT_INFO_FP, gamma=1.15)

    # 1) 카테고리 앵커 산출
    anchors = {}  # cat -> {"2024H2": x, "2025H1": y, "method": m, "bias": b}
    for c in cats:
        g_all = half_all[half_all["category"]==c]
        # 학습은 2024H2 제외
        train = g_all[~((g_all["year"]==2024) & (g_all["half"]=="H2"))]
        act = actual_2024H2_by_cat.get(c, None)
        anchors[c] = decide_anchor_for_category(c, train, act)

    # 2) 카테고리 반기 앵커 → 월 분배 (POS 2023 시즌 사용, 반기 내부 6개월 정규화)
    #    cat_month_qty: {(cat, 'YYYY-MM'): value}
    cat_month_qty = {}
    for c in cats:
        # 시즌: 없으면 균등 1
        season12 = season_pos.get(c, {m:1.0 for m in range(1,13)})
        for half_key in ["2024H2","2025H1"]:
            anchor = anchors[c].get(half_key, np.nan)
            if not np.isfinite(anchor) or anchor<=0:
                # 앵커 없음 → 0
                for m in HALF_MONTHS[half_key]:
                    cat_month_qty[(c, m)] = 0.0
                continue
            w6 = get_half_norm_weights_from_season(season12, half_key)  # 길이6 합=1
            for m, w in zip(HALF_MONTHS[half_key], w6):
                cat_month_qty[(c, m)] = float(anchor * w)

    # 3) 카테고리→SKU 분배 (α와 페르소나 월형상 결합)
    #    s_p,t = α_p * w_p,t  → share_p,t = s_p,t / Σ_p s_p,t  → qty_p,t = cat_qty_t * share_p,t
    pred_rows = []
    for m in MONTHS:
        for c in cats:
            total_cm = cat_month_qty.get((c, m), 0.0)
            prods = products_by_cat.get(c, [])
            if not prods or total_cm <= 0:
                for p in prods:
                    pred_rows.append({"product": p, "month": m, "qty": 0.0})
                continue
            # 분자: α * w_p,t
            scores = {}
            for p in prods:
                alpha_p = alpha[c].get(p, 0.0)
                # 페르소나 월형상에서 해당 월 index
                idx = MONTHS.index(m)
                w_pm = Wp[p][idx]  # 전체 12개월 합은 1
                scores[p] = max(1e-12, alpha_p * w_pm)
            denom = sum(scores.values())
            if denom <= 0:
                # 모든 점수가 0이면 균등
                for p in prods:
                    q = total_cm / len(prods)
                    pred_rows.append({"product": p, "month": m, "qty": q})
            else:
                for p in prods:
                    share = scores[p]/denom
                    q = total_cm * share
                    pred_rows.append({"product": p, "month": m, "qty": q})

    df_pred = pd.DataFrame(pred_rows)

    # 4) 제출 스키마(wide)로 저장 (샘플 기준)
    sub_path = SAVE_DIR / "submission_anchor.csv"
    sample = pd.read_csv(SAMPLE_FP)
    req_cols = list(sample.columns)
    assert req_cols and req_cols[0]=="product_name", "샘플 첫 컬럼이 product_name 이어야 합니다."

    # 제품명 검증
    pred_products = set(df_pred["product"].unique())
    sample_products = set(sample["product_name"].unique())
    miss = sample_products - pred_products
    extra = pred_products - sample_products
    if miss:
        raise ValueError(f"예측에 없는 제품이 샘플에 있습니다: {sorted(miss)}")
    if extra:
        raise ValueError(f"샘플에 없는 제품을 예측했습니다: {sorted(extra)}")

    # 월 → launch index 매핑
    month_to_idx = {m:i+1 for i,m in enumerate(MONTHS)}  # 2024-07 → 1 ... 2025-06 → 12
    wide_map = {}
    for i,m in enumerate(MONTHS, start=1):
        s = (df_pred[df_pred["month"]==m].set_index("product")["qty"])
        wide_map[f"months_since_launch_{i}"] = s

    sub = sample.copy()
    for col, s in wide_map.items():
        sub[col] = sub["product_name"].map(s).fillna(0.0)

    # 반올림 → int, 음수 방지
    val_cols = [c for c in sub.columns if c != "product_name"]
    sub[val_cols] = (
    sub[val_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
    .round(0)          # 또는 np.rint(...)
    .clip(lower=0)     # <-- lower 사용
    .astype("int64")
)

    sub = sub[req_cols]
    sub.to_csv(sub_path, index=False, encoding="utf-8-sig")

    # 5) 리포트(요약 앵커 테이블 저장)
    rep_rows = []
    for c in cats:
        rep_rows.append({
            "category": c,
            "method": anchors[c]["method"],
            "anchor_2024H2": anchors[c]["2024H2"],
            "anchor_2025H1": anchors[c]["2025H1"],
            "pred_raw_2024H2": anchors[c]["pred_raw_2024H2"],
            "bias_2024H2": anchors[c]["bias"],
            "use_actual_2024H2": USE_ACTUAL_2024H2_AS_ANCHOR
        })
    df_rep = pd.DataFrame(rep_rows)
    df_rep.to_csv(SAVE_DIR/"anchor_summary.csv", index=False, encoding="utf-8-sig")

    # === 반기 합계가 앵커(반올림 정수)와 정확히 일치하도록 미세 보정 ===
    
    anch = pd.read_csv(SAVE_DIR/"anchor_summary.csv", encoding="utf-8")
    sub  = sub.copy()  # 기존 DataFrame 사용

    CATEGORY_OF = {
        "덴마크 하이그릭요거트 400g": "발효유",
        "소화가 잘되는 우유로 만든 바닐라라떼 250mL": "커피",
        "소화가 잘되는 우유로 만든 카페라떼 250mL": "커피",
        "동원맛참 고소참기름 90g": "전통기름",
        "동원맛참 고소참기름 135g": "전통기름",
        "동원맛참 매콤참기름 90g": "전통기름",
        "동원맛참 매콤참기름 135g": "전통기름",
        "동원참치액 순 500g": "조미료",
        "동원참치액 순 900g": "조미료",
        "동원참치액 진 500g": "조미료",
        "동원참치액 진 900g": "조미료",
        "프리미엄 동원참치액 500g": "조미료",
        "프리미엄 동원참치액 900g": "조미료",
        "리챔 오믈레햄 200g": "어육가공품",
        "리챔 오믈레햄 340g": "어육가공품",
    }

    sub["cat"] = sub["product_name"].map(CATEGORY_OF)
    anch = anch.set_index("category")

    h2_cols = [f"months_since_launch_{i}" for i in range(1,7)]   # 2024-07~12
    h1_cols = [f"months_since_launch_{i}" for i in range(7,13)]  # 2025-01~06

    def apply_half_adjustment(sub_df: pd.DataFrame, cols: list[str], anchor_col: str) -> pd.DataFrame:
        # 반기 타깃 앵커를 정수로(반올림) 맞춘 뒤, 카테고리별로 차액을 마지막 달 한 칸에만 반영
        for c, g in sub_df.groupby("cat"):
            if c not in anch.index: 
                continue
            target = int(round(float(anch.loc[c, anchor_col])))              # 정수 타깃
            current = int(sub_df.loc[g.index, cols].values.sum())            # 현재 정수 합
            dz = target - current                                            # 필요한 보정량(±1~2 예상)
            if dz == 0:
                continue
            last_col = cols[-1]                                              # 반기의 마지막 달
            # 마지막 달에서 가장 큰 값을 가진 SKU를 찾아 보정량을 흡수(음수 방지)
            block = sub_df.loc[g.index, [last_col]]
            i_target = block[last_col].idxmax() if dz >= 0 else block[last_col].idxmax()
            new_val = sub_df.at[i_target, last_col] + dz
            if new_val < 0:
                # 아주 드문 케이스: 음수가 되면, 두 번째로 큰 칸에 나눠서 반영
                sub_df.at[i_target, last_col] = 0
                dz_rest = new_val  # 음수(절대값만큼 더 빼야 함)
                # 다른 SKU에 분산 (간단히 역순 정렬로 분배)
                others = block.drop(index=i_target).sort_values(last_col, ascending=False).index.tolist()
                for idx in others:
                    if dz_rest == 0: break
                    take = min(sub_df.at[idx, last_col], abs(dz_rest))
                    sub_df.at[idx, last_col] = sub_df.at[idx, last_col] - take
                    dz_rest += take  # dz_rest는 음수 → 0으로 수렴
            else:
                sub_df.at[i_target, last_col] = new_val
        return sub_df

    sub = apply_half_adjustment(sub, h2_cols, "anchor_2024H2")
    sub = apply_half_adjustment(sub, h1_cols, "anchor_2025H1")

    # 타입 원복 및 저장
    val_cols = [c for c in sub.columns if c.startswith("months_since_launch_")]
    sub[val_cols] = sub[val_cols].round(0).clip(lower=0).astype("int64")
    sub = sub.drop(columns=["cat"])
    sub.to_csv(SAVE_DIR/"submission_anchor_adjusted.csv", index=False, encoding="utf-8-sig")
    print("✔︎ 반기 합계 = 앵커(정수) 정확히 일치: submission_anchor_adjusted.csv 저장")


    print("[완료] 제출 저장:", sub_path.resolve())
    print("[요약] 앵커/방법:", (SAVE_DIR/"anchor_summary.csv").resolve())
