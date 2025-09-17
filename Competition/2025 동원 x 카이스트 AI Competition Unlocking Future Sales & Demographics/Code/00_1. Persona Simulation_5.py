# -*- coding: utf-8 -*-
"""
Dongwon — L1 Anchor Pipeline (카테고리 총량 앵커 + (옵션) 바이어스 보정 + 월·SKU 분배 + 반기합 정합 보정 + QC)
- Nielsen 분기/반기 → 반기 단일 시계열 통합(클린/저장)
- 카테고리별 앵커 산출: YoY / ETS / (커피) YoY-ETS 앙상블
- (옵션) 2024H2 실측으로 2025H1 바이어스 부분 보정(λ)
- POS 2023 월 시즌으로 반기→월 분배
- α(정적 가중)×페르소나 월형상으로 카테고리→SKU 분배
- 제출(wide) 생성 + 반기합=앵커(정수) 정확히 맞추는 미세 보정
- QC 요약 저장

필요 파일 경로:
- 닐슨 분기: ./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2019_분기별 매출액.xlsx
- 닐슨 반기: ./_data/dacon/dongwon/pos_data/닐슨코리아_2020_2024 반기별 매출액.xlsx
- POS 월:    ./_data/dacon/dongwon/pos_data/marketlink_POS_master.xlsx (없으면 균등 시즌)
- 제품 정보: ./_data/dacon/dongwon/product_info.csv (없으면 α 균등)
- 페르소나:  ./_data/dacon/dongwon/personas.json
- 샘플 제출: ./_data/dacon/dongwon/sample_submission.csv
"""

from __future__ import annotations
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# =========================
# 경로 & 고정 설정
# =========================
NIELSEN_Q_FP = Path("./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2019_분기별 매출액.xlsx")
NIELSEN_H_FP = Path("./_data/dacon/dongwon/pos_data/닐슨코리아_2020_2024 반기별 매출액.xlsx")
POS_FP       = Path("./_data/dacon/dongwon/pos_data/marketlink_POS_master.xlsx")
PRODUCT_INFO_FP = Path("./_data/dacon/dongwon/product_info.csv")
DATA_DIR     = Path("./_data/dacon/dongwon")
PERSONA_FP   = DATA_DIR / "personas.json"
SAMPLE_FP    = DATA_DIR / "sample_submission.csv"
SAVE_DIR     = Path("./_save/anchor_pipeline"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 기간/월
EVAL_START = "2024-07"; EVAL_END = "2025-06"
MONTHS = pd.period_range(EVAL_START, EVAL_END, freq="M").strftime("%Y-%m").tolist()
HALF_MONTHS = {
    "2024H2": [f"2024-{mm:02d}" for mm in range(7,13)],
    "2025H1": [f"2025-{mm:02d}" for mm in range(1,7)],
}

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
USE_BIAS_CAL = True      # 2024H2 실측으로 2025H1 부분 보정
BIAS_LAMBDA  = 0.4       # λ in [0.3, 0.5]
USE_ACTUAL_2024H2_AS_ANCHOR = False  # True면 2024H2 앵커를 실측으로 대체

# 페르소나 섞기 파라미터
TAU = 0.7  # softmax temperature

# =========================
# 유틸
# =========================
def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    return ex / s if s > 0 else np.ones_like(ex) / len(ex)

def _fmt(x):
    return "nan" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:,.0f}"

# =========================
# 닐슨 로더 (분기/반기 → 반기 통합)
# =========================
def read_nielsen_quarter_kor(fp: Path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(fp, sheet_name=sheet_name)
    need = ['연도', '분기', '카테고리', '매출액(백만원)']
    for c in need:
        if c not in df.columns:
            raise ValueError(f"분기 파일 컬럼 누락: {c} / 실제: {list(df.columns)}")
    out = df[need].copy()
    out.columns = ["year", "quarter", "category", "amount"]
    out["quarter"] = out["quarter"].astype(str).str.extract(r"(\d+)")[0].astype(int).clip(1, 4)
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
    out.columns = ["year", "half_raw", "category", "amount"]
    s = out["half_raw"].astype(str).str.upper()
    out["half"] = np.where(s.str.contains("2"), "H2", "H1")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out["category"] = out["category"].astype(str).str.strip()
    return out[["category", "year", "half", "amount"]]

def quarter_to_half(qdf: pd.DataFrame) -> pd.DataFrame:
    qq = qdf.copy()
    qq["half"] = np.where(qq["quarter"].isin([1, 2]), "H1", "H2")
    hh = qq.groupby(["category", "year", "half"], as_index=False)["amount"].sum()
    return hh

# === 통합 + 정리 + 저장 ===
def combine_and_save_halves(h_from_q: pd.DataFrame, h_direct: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    q_half_raw = h_from_q.copy()
    q_half_raw["source"] = "from_quarter"   # 2011~2019 기대
    h_half_raw = h_direct.copy()
    h_half_raw["source"] = "half_direct"    # 2020~2024 기대

    # 이어붙이고 category 클린업
    raw = pd.concat([q_half_raw, h_half_raw], ignore_index=True)
    raw["category"] = raw["category"].astype(str).str.strip()
    valid_mask = raw["category"].notna() & (raw["category"].str.len() > 0) & (raw["category"].str.lower() != "nan")
    dropped_cat = int((~valid_mask).sum())
    raw = raw[valid_mask].copy()

    # 연도 가드
    before_n = len(raw)
    raw = raw[~((raw["source"] == "from_quarter") & (raw["year"] >= 2020))].copy()
    raw = raw[~((raw["source"] == "half_direct") & (raw["year"] <= 2019))].copy()
    dropped_year_guard = before_n - len(raw)

    # 중복 키에서 half_direct 우선
    keys_half = set(h_half_raw.assign(source="half_direct")[["category", "year", "half"]].apply(tuple, axis=1))
    from_mask = (raw["source"] == "from_quarter")
    dup_mask = from_mask & raw[["category", "year", "half"]].apply(tuple, axis=1).isin(keys_half)
    dropped_overlap = int(dup_mask.sum())
    raw = raw[~dup_mask].copy()

    # 정렬
    order_half = {"H1": 1, "H2": 2}
    raw["_ho"] = raw["half"].map(order_half)
    raw = raw.sort_values(["category", "year", "_ho", "source"]).drop(columns="_ho").reset_index(drop=True)

    # AGG/PIVOT
    agg = (raw.groupby(["category", "year", "half"], as_index=False)["amount"]
           .sum()
           .sort_values(["category", "year", "half"])
           .reset_index(drop=True))
    pivot = (agg.pivot_table(index=["category", "year"], columns="half", values="amount", fill_value=0)
             .reset_index()
             .sort_values(["category", "year"]))

    # 저장 (CSV + XLSX; openpyxl 사용)
    raw_csv   = out_dir / "nielsen_half_combined_raw.csv"
    agg_csv   = out_dir / "nielsen_half_combined_agg.csv"
    pivot_csv = out_dir / "nielsen_half_combined_pivot.csv"
    raw_xlsx   = out_dir / "nielsen_half_combined_raw.xlsx"
    agg_xlsx   = out_dir / "nielsen_half_combined_agg.xlsx"
    pivot_xlsx = out_dir / "nielsen_half_combined_pivot.xlsx"

    raw.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    agg.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    pivot.to_csv(pivot_csv, index=False, encoding="utf-8-sig")

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("⚠️ openpyxl 미설치: xlsx 저장 생략( pip install openpyxl )")
    else:
        with pd.ExcelWriter(raw_xlsx, engine="openpyxl") as w:
            raw.to_excel(w, index=False, sheet_name="raw")
        with pd.ExcelWriter(agg_xlsx, engine="openpyxl") as w:
            agg.to_excel(w, index=False, sheet_name="agg")
        with pd.ExcelWriter(pivot_xlsx, engine="openpyxl") as w:
            pivot.to_excel(w, index=False, sheet_name="pivot")
        print("[Excel 저장 완료] openpyxl 엔진으로 RAW/AGG/PIVOT xlsx 저장")

    # 요약 로그
    yr_tbl = raw.groupby("source")["year"].agg(["min", "max", "nunique"]).reset_index()
    print("\n[정리 요약]")
    print(f" - category 클린업 드롭: {dropped_cat}행")
    print(f" - 연도 가드 드롭     : {dropped_year_guard}행")
    print(f" - 중복키 드롭(from_quarter → half_direct 우선): {dropped_overlap}행")
    print("\n[출처별 연도 범위]")
    print(yr_tbl.to_string(index=False))

    return agg  # 클린 AGG 반환(모델 입력용)

# =========================
# 앵커 예측 (YoY/ETS + ensemble + 바이어스)
# =========================
def forecast_yoy_for_cat(train_half_df_cat: pd.DataFrame) -> dict:
    """train_half_df_cat: 2011~2024H1 (2024H2 제외)"""
    g = (train_half_df_cat.groupby(["year", "half"], as_index=False)["amount"].sum()
         .sort_values(["year", "half"]))
    piv = g.pivot(index="year", columns="half", values="amount").sort_index()
    a_2023h1 = float(piv.loc[2023, "H1"]) if (2023 in piv.index and "H1" in piv.columns) else np.nan
    a_2023h2 = float(piv.loc[2023, "H2"]) if (2023 in piv.index and "H2" in piv.columns) else np.nan
    a_2024h1 = float(piv.loc[2024, "H1"]) if (2024 in piv.index and "H1" in piv.columns) else np.nan

    if np.isfinite(a_2023h2) and np.isfinite(a_2023h1) and np.isfinite(a_2024h1) and a_2023h1 > 0:
        r1 = a_2024h1 / a_2023h1
        y_2024h2 = a_2023h2 * r1
    else:
        r1 = np.nan; y_2024h2 = a_2023h2 if np.isfinite(a_2023h2) else np.nan

    if np.isfinite(y_2024h2) and np.isfinite(a_2023h2) and np.isfinite(a_2024h1) and a_2023h2 > 0:
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
    g = (train_half_df_cat.groupby(["year", "half"], as_index=False)["amount"].sum()
         .sort_values(["year", "half"]))
    y = g["amount"].astype(float).values
    if len(y) < 6:
        return {"2024H2": np.nan, "2025H1": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        f = fit.forecast(2)  # next 2 halves
    return {"2024H2": float(f[0]), "2025H1": float(f[1])}

def decide_anchor_for_category(cat: str,
                               train_df_cat: pd.DataFrame,
                               actual_2024H2: float | None,
                               bias_lambda: float | None = None) -> dict:
    """카테고리별 방법 적용 + (옵션) 바이어스 보정까지 반영한 앵커 반환"""
    if bias_lambda is None:
        bias_lambda = BIAS_LAMBDA

    method = METHOD_BY_CAT.get(cat, "yoy")
    yoy = forecast_yoy_for_cat(train_df_cat)
    ets = forecast_ets_for_cat(train_df_cat)

    if method == "yoy":
        h2_pred, h1_pred = yoy["2024H2"], yoy["2025H1"]
    elif method == "ets":
        h2_pred, h1_pred = ets["2024H2"], ets["2025H1"]
    elif method == "yoy_ets_ensemble":
        h2_pred = ENSEMBLE_W["YoY"] * yoy["2024H2"] + ENSEMBLE_W["ETS"] * ets["2024H2"]
        h1_pred = ENSEMBLE_W["YoY"] * yoy["2025H1"] + ENSEMBLE_W["ETS"] * ets["2025H1"]
    else:
        h2_pred, h1_pred = yoy["2024H2"], yoy["2025H1"]

    # (옵션) 2024H2 실측 → 2025H1 부분 보정
    if USE_BIAS_CAL and actual_2024H2 is not None and np.isfinite(actual_2024H2) and np.isfinite(h2_pred) and h2_pred > 0:
        bias = actual_2024H2 / h2_pred
        h1_adj = (h1_pred ** (1.0 - bias_lambda)) * ((h1_pred * bias) ** (bias_lambda))
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

    cat_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["카테고리","category","cat"])), None)
    y_col   = next((c for c in df.columns if any(k in str(c).lower() for k in ["연도","year"])), None)
    m_col   = next((c for c in df.columns if any(k in str(c).lower() for k in ["월","month","mm"])), None)
    date_col= next((c for c in df.columns if any(k in str(c).lower() for k in ["일자","date"])), None)
    val_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["수량","매출","판매","units","amount","qty","value"])), None)
    if cat_col is None or val_col is None:
        return {}

    tmp = df.copy()
    tmp[cat_col] = tmp[cat_col].astype(str).str.strip()

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

    tmp = tmp[tmp["year"] == 2023]
    if tmp.empty:
        return {}

    season = {}
    for cat, g in tmp.groupby(tmp[cat_col]):
        s = g.groupby("month")[val_col].sum()
        s = s.reindex(range(1,13), fill_value=0.0)
        w = (s / s.mean()).to_dict() if s.sum() > 0 else {m:1.0 for m in range(1,13)}
        season[str(cat)] = w
    return season

def get_half_norm_weights_from_season(season12: dict[int, float], half_key: str) -> np.ndarray:
    """season12: {1..12: w}, half_key: '2024H2' or '2025H1' -> 길이 6, 합=1"""
    months = [7,8,9,10,11,12] if half_key == "2024H2" else [1,2,3,4,5,6]
    arr = np.array([season12.get(m, 1.0) for m in months], dtype=float)
    s = arr.sum()
    return (arr / s) if s > 0 else np.ones(6) / 6.0

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
        w = np.array([d[m] for m in MONTHS], float)
    return w / w.mean()

def mix_persona_shapes(product_personas: list[dict]) -> np.ndarray:
    """purchase_probability 기반 softmax 가중 혼합 → 12개월 합=1"""
    probs = np.array([p.get("purchase_probability", 0)/100.0 for p in product_personas], float)
    s = softmax(probs / TAU)
    W = np.vstack([persona_month_shape(p) for p in product_personas])  # (k,12)
    u = (s[:, None] * W).sum(axis=0)
    return u / u.sum()  # 길이 12, 합=1

# =========================
# 제품 α (정적 가중; 기본 균등, 커피는 페르소나+힌트)
# =========================
def compute_alpha_per_category(products_by_cat: dict[str, list[str]],
                               personas: dict,
                               product_info_fp: Path | None = None,
                               gamma: float = 1.15) -> dict[str, dict[str, float]]:
    """
    반환: {cat: {product: α}} (카테고리별 Σα = 1)
    - 커피: 페르소나 구매확률 평균^gamma × (있으면) product_info 힌트(60%) 결합
    - 기타: 균등
    """
    hints = {}
    if product_info_fp is not None and product_info_fp.exists():
        try:
            pdf = pd.read_excel(product_info_fp) if product_info_fp.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(product_info_fp)
            name_col = next((c for c in pdf.columns if str(c).lower() in ["product_name","제품명","product","sku"]), None) or pdf.columns[0]
            cand = ["alpha_hint","baseline_share","alpha","share_hint","가중치"]
            hint_col = next((c for c in pdf.columns if str(c).lower() in [x.lower() for x in cand]), None)
            if hint_col is not None:
                tmp = pdf[[name_col, hint_col]].copy()
                tmp.columns = ["product", "hint"]
                tmp["hint"] = pd.to_numeric(tmp["hint"], errors="coerce")
                hints = {r.product: float(r.hint) for r in tmp.itertuples(index=False) if pd.notna(r.hint) and r.hint > 0}
        except Exception:
            hints = {}

    alpha: dict[str, dict[str, float]] = {}
    for c, prods in products_by_cat.items():
        if not prods:
            alpha[c] = {}
            continue
        w_raw = {p: 1.0 for p in prods}
        if c == "커피":
            for p in prods:
                per_arr = personas[p]
                probs = [float(per.get("purchase_probability", 0)) for per in per_arr]
                persona_score = max(1e-9, np.mean(probs) / 100.0) ** gamma
                hint = hints.get(p, None)
                if hint is not None and np.isfinite(hint) and hint > 0:
                    w_raw[p] = (persona_score ** 0.4) * (float(hint) ** 0.6)
                else:
                    w_raw[p] = persona_score
        tot = sum(w_raw.values())
        alpha[c] = ({p: w_raw[p]/tot for p in prods} if tot > 0 else {p: 1.0/len(prods) for p in prods})
    return alpha

# =========================
# 반기합 정합 보정
# =========================
def apply_half_adjustment(sub_df: pd.DataFrame,
                          anch_df: pd.DataFrame,
                          category_of: dict[str, str],
                          half_cols_map: dict[str, list[str]]) -> pd.DataFrame:
    """
    각 카테고리 반기 합계가 anchor_summary의 반올림 정수와 정확히 일치하도록
    반기 마지막 달의 한 칸만 미세 보정(음수 방지).
    """
    sub = sub_df.copy()
    sub["cat"] = sub["product_name"].map(category_of)
    anch = anch_df.set_index("category")

    def adjust_one_half(cols: list[str], anchor_col: str):
        nonlocal sub
        for c, g in sub.groupby("cat"):
            if c not in anch.index:
                continue
            target = int(round(float(anch.loc[c, anchor_col])))
            current = int(sub.loc[g.index, cols].values.sum())
            dz = target - current
            if dz == 0:
                continue
            last_col = cols[-1]
            block = sub.loc[g.index, [last_col]]
            # dz가 +든 -든, 값이 가장 큰 SKU에 반영(음수 위험 최소화)
            i_target = block[last_col].idxmax()
            new_val = int(sub.at[i_target, last_col]) + dz
            if new_val < 0:
                # 드문 케이스: 음수 방지(0으로 만들고 잔여 차액을 다른 SKU에 분산)
                dz_rest = new_val  # 음수
                sub.at[i_target, last_col] = 0
                others = block.drop(index=i_target).sort_values(last_col, ascending=False).index.tolist()
                for idx in others:
                    if dz_rest == 0:
                        break
                    take = min(sub.at[idx, last_col], abs(dz_rest))
                    sub.at[idx, last_col] = int(sub.at[idx, last_col]) - int(take)
                    dz_rest += int(take)
            else:
                sub.at[i_target, last_col] = int(new_val)

    # H2, H1 각각 조정
    adjust_one_half(half_cols_map["2024H2"], "anchor_2024H2")
    adjust_one_half(half_cols_map["2025H1"], "anchor_2025H1")

    # 타입/음수 가드
    val_cols = [c for c in sub.columns if c.startswith("months_since_launch_")]
    sub[val_cols] = sub[val_cols].round(0).clip(lower=0).astype("int64")
    return sub.drop(columns=["cat"])

# =========================
# QC 요약 저장
# =========================
def save_qc_summary(sub_wide: pd.DataFrame, anch_df: pd.DataFrame, out_csv: Path, category_of: dict[str, str]) -> None:
    sub = sub_wide.copy()
    sub["category"] = sub["product_name"].map(category_of)
    h2_cols = [f"months_since_launch_{i}" for i in range(1,7)]
    h1_cols = [f"months_since_launch_{i}" for i in range(7,13)]

    qc = (sub.assign(sum_h2=sub[h2_cols].sum(axis=1),
                     sum_h1=sub[h1_cols].sum(axis=1))
              .groupby("category")[["sum_h2","sum_h1"]].sum()
              .reset_index())

    anch2 = anch_df[["category","anchor_2024H2","anchor_2025H1"]].copy()
    anch2["anchor_h2_int"] = anch2["anchor_2024H2"].round().astype(int)
    anch2["anchor_h1_int"] = anch2["anchor_2025H1"].round().astype(int)

    out = qc.merge(anch2[["category","anchor_h2_int","anchor_h1_int"]], on="category", how="left")
    out["diff_h2"] = out["sum_h2"] - out["anchor_h2_int"]
    out["diff_h1"] = out["sum_h1"] - out["anchor_h1_int"]

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n=== 산출물 QC — 반기 합계 vs 앵커(정수) ===")
    print(out.to_string(index=False))
    print(f"✅ QC 통과 & 저장: {out_csv}")

# =========================
# 메인 파이프라인
# =========================
if __name__ == "__main__":
    # 0) 닐슨 통합
    qdf = read_nielsen_quarter_kor(NIELSEN_Q_FP)
    hdf = read_nielsen_half_kor(NIELSEN_H_FP)
    h_from_q = quarter_to_half(qdf)
    COMBINE_DIR = SAVE_DIR / "nielsen_combined"
    half_all = combine_and_save_halves(h_from_q, hdf, COMBINE_DIR)   # category, year, half, amount

    # 1) 카테고리 목록 & 2024H2 실측(있으면 사용)
    cats = sorted(half_all["category"].unique())
    actual_2024H2_by_cat = {}
    for c in cats:
        row = half_all[(half_all["category"] == c) & (half_all["year"] == 2024) & (half_all["half"] == "H2")]
        actual_2024H2_by_cat[c] = float(row["amount"].sum()) if not row.empty else None

    # 2) POS 2023 월 시즌 (없으면 균등)
    season_pos = load_pos_monthly_season_2023(POS_FP)

    # 3) 페르소나 로드 및 검증
    personas = load_personas(PERSONA_FP)
    sku_list = list(personas.keys())
    for p in sku_list:
        if p not in CATEGORY_OF:
            raise ValueError(f"화이트리스트 외 제품: {p}")
    products_by_cat = defaultdict(list)
    for p in sku_list:
        products_by_cat[CATEGORY_OF[p]].append(p)

    # 제품별 페르소나 월형상(12개월, 합=1)
    Wp = {p: mix_persona_shapes(personas[p]) for p in sku_list}

    # 4) 제품 α(정적 가중)
    alpha = compute_alpha_per_category(products_by_cat, personas, PRODUCT_INFO_FP, gamma=1.15)

    # 5) 카테고리 앵커 산출
    anchors = {}
    for c in cats:
        g_all = half_all[half_all["category"] == c]
        train = g_all[~((g_all["year"] == 2024) & (g_all["half"] == "H2"))]  # 2024H2 제외
        act = actual_2024H2_by_cat.get(c, None)
        anchors[c] = decide_anchor_for_category(c, train, act, bias_lambda=BIAS_LAMBDA)

    # 6) 카테고리 반기 앵커 → 월 분배
    cat_month_qty: dict[tuple[str, str], float] = {}
    for c in cats:
        season12 = season_pos.get(c, {m: 1.0 for m in range(1, 13)})
        for half_key in ["2024H2", "2025H1"]:
            anchor = anchors[c].get(half_key, np.nan)
            if not np.isfinite(anchor) or anchor <= 0:
                for m in HALF_MONTHS[half_key]:
                    cat_month_qty[(c, m)] = 0.0
                continue
            w6 = get_half_norm_weights_from_season(season12, half_key)  # 길이6, 합=1
            for m, w in zip(HALF_MONTHS[half_key], w6):
                cat_month_qty[(c, m)] = float(anchor * w)

    # 7) 카테고리→SKU 분배 (α × 페르소나 월형상)
    pred_rows = []
    for m_idx, m in enumerate(MONTHS):
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
                w_pm = Wp[p][m_idx]  # 12개월 합=1
                scores[p] = max(1e-12, alpha_p * w_pm)
            denom = sum(scores.values())
            if denom <= 0:
                for p in prods:
                    q = total_cm / len(prods)
                    pred_rows.append({"product": p, "month": m, "qty": q})
            else:
                for p in prods:
                    share = scores[p] / denom
                    q = total_cm * share
                    pred_rows.append({"product": p, "month": m, "qty": q})

    df_pred = pd.DataFrame(pred_rows)

    # 8) 제출 스키마(wide) 저장 (초안)
    sub_path = SAVE_DIR / "submission_anchor.csv"
    sample = pd.read_csv(SAMPLE_FP)
    req_cols = list(sample.columns)
    assert req_cols and req_cols[0] == "product_name", "샘플 첫 컬럼이 product_name 이어야 합니다."

    # 제품명 검증
    pred_products = set(df_pred["product"].unique())
    sample_products = set(sample["product_name"].unique())
    miss = sample_products - pred_products
    extra = pred_products - sample_products
    if miss:
        raise ValueError(f"예측에 없는 제품이 샘플에 있습니다: {sorted(miss)}")
    if extra:
        raise ValueError(f"샘플에 없는 제품을 예측했습니다: {sorted(extra)}")

    # 월 → wide 매핑
    wide_map = {}
    for i, m in enumerate(MONTHS, start=1):
        s = df_pred[df_pred["month"] == m].set_index("product")["qty"]
        wide_map[f"months_since_launch_{i}"] = s

    sub = sample.copy()
    for col, s in wide_map.items():
        sub[col] = sub["product_name"].map(s).fillna(0.0)

    # 반올림 → int, 음수 방지
    val_cols = [c for c in sub.columns if c != "product_name"]
    sub[val_cols] = (sub[val_cols]
                     .apply(pd.to_numeric, errors="coerce")
                     .fillna(0.0)
                     .round(0)
                     .clip(lower=0)
                     .astype("int64"))
    sub = sub[req_cols]
    sub.to_csv(sub_path, index=False, encoding="utf-8-sig")
    print("[1/3] 제출 초안 저장:", sub_path.resolve())

    # 9) 앵커 요약 저장
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
    anch_path = SAVE_DIR / "anchor_summary.csv"
    df_rep.to_csv(anch_path, index=False, encoding="utf-8-sig")
    print("[2/3] 앵커 요약 저장:", anch_path.resolve())

    # 10) 반기합 정합 보정본 생성 & 저장
    half_cols_map = {
        "2024H2": [f"months_since_launch_{i}" for i in range(1, 7)],
        "2025H1": [f"months_since_launch_{i}" for i in range(7, 13)],
    }
    sub_adj = apply_half_adjustment(sub_df=sub,
                                    anch_df=df_rep,
                                    category_of=CATEGORY_OF,
                                    half_cols_map=half_cols_map)
    sub_adj_path = SAVE_DIR / "submission_anchor_adjusted.csv"
    sub_adj.to_csv(sub_adj_path, index=False, encoding="utf-8-sig")
    print("[3/3] 반기합 정합 보정본 저장:", sub_adj_path.resolve())

    # 11) QC 요약 저장(조정본 기준)
    qc_path = SAVE_DIR / "qc_summary.csv"
    save_qc_summary(sub_wide=sub_adj, anch_df=df_rep, out_csv=qc_path, category_of=CATEGORY_OF)

    print("\n✅ 파이프라인 완료")
    print(" - 제출 파일(초안):", sub_path.resolve())
    print(" - 제출 파일(조정):", sub_adj_path.resolve())
    print(" - 앵커 요약:", anch_path.resolve())
    print(" - QC 요약:", qc_path.resolve())