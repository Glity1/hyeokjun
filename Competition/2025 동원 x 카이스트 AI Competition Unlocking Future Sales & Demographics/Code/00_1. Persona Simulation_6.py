# -*- coding: utf-8 -*-
"""
Dongwon — L1 Anchor Pipeline (Advanced)
- 앵커 자동 선택/가중 (YoY vs ETS)
- 바이어스 캘리브레이션 λ 자동화
- α(제품 정적 가중) = 페르소나 + 단위가격(가격/용량) + alpha_hint 결합
- 카테고리별 τ(softmax temperature) 오버라이드
- 반기 앵커→월 분배(POS 시즌) + SKU 분배(페르소나) + 반기합계=앵커 정합

필요 파일:
- 닐슨 분기: ./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2019_분기별 매출액.xlsx
- 닐슨 반기: ./_data/dacon/dongwon/pos_data/닐슨코리아_2020_2024 반기별 매출액.xlsx
- POS 월:    ./_data/dacon/dongwon/pos_data/marketlink_POS_master.xlsx (없으면 균등)
- 제품 정보: ./_data/dacon/dongwon/product_info.csv (없으면 힌트/단위가격 미사용)
- 페르소나:  ./_data/dacon/dongwon/personas.json
- 샘플 제출: ./_data/dacon/dongwon/sample_submission.csv
"""

from __future__ import annotations
import warnings, re, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# =========================
# 경로 & 설정
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
HALF_MONTHS = {"2024H2":[f"2024-{mm:02d}" for mm in range(7,13)],
               "2025H1":[f"2025-{mm:02d}" for mm in range(1,7)]}

# 화이트리스트 SKU→카테고리
CATEGORY_OF = {
    "덴마크 하이그릭요거트 400g": "발효유",
    "소화가 잘되는 우유로 만든 바닐라라떼 250mL": "조제커피",
    "소화가 잘되는 우유로 만든 카페라떼 250mL": "조제커피",
    "동원맛참 고소참기름 90g": "참치캔",
    "동원맛참 고소참기름 135g": "참치캔",
    "동원맛참 매콤참기름 90g": "참치캔",
    "동원맛참 매콤참기름 135g": "참치캔",
    "동원참치액 순 500g": "조미료",
    "동원참치액 순 900g": "조미료",
    "동원참치액 진 500g": "조미료",
    "동원참치액 진 900g": "조미료",
    "프리미엄 동원참치액 500g": "조미료",
    "프리미엄 동원참치액 900g": "조미료",
    "리챔 오믈레햄 200g": "식육가공품",
    "리챔 오믈레햄 340g": "식육가공품",
}

# -------------------------
# 고도화 스위치
# -------------------------
AUTO_SELECT_METHOD   = True   # 2024H2 실측으로 YoY/ETS 자동 선택(또는 가중 앙상블)
ADAPTIVE_BIAS        = True   # bias 크기에 따라 λ 자동 조절
USE_ACTUAL_2024H2_AS_ANCHOR = False  # 2024H2 앵커를 실측으로 대체(검증 관점)
BASE_METHOD_BY_CAT = {        # 실측이 없을 때/폴백
    "발효유": "ets",
    "어육가공품": "yoy",
    "전통기름": "yoy",
    "조미료": "yoy",
    "커피": "yoy_ets_ensemble",  # 기본 0.8/0.2
}
ENSEMBLE_W_DEFAULT = {"YoY": 0.8, "ETS": 0.2}

# 페르소나 혼합 온도 τ (카테고리별 오버라이드)
TAU = 0.7
TAU_BY_CAT = {
    # "커피": 0.6,  # 필요하면 켜기
}

# α 계산 파라미터(가중 기하 평균 비율)
ALPHA_WEIGHTS = {
    "기본": {"persona": 0.5, "cheapness": 0.3, "hint": 0.2},
    "커피": {"persona": 0.6, "cheapness": 0.25, "hint": 0.15},
}

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

# 반기 통합(정합 가드 + 저장)
def combine_nielsen_half(h_from_q: pd.DataFrame, h_direct: pd.DataFrame) -> pd.DataFrame:
    COMBINE_DIR = SAVE_DIR / "nielsen_combined"
    COMBINE_DIR.mkdir(parents=True, exist_ok=True)

    q_half_raw = h_from_q.copy(); q_half_raw["source"] = "from_quarter"
    h_half_raw = h_direct.copy(); h_half_raw["source"] = "half_direct"

    raw = pd.concat([q_half_raw, h_half_raw], ignore_index=True)
    raw["category"] = raw["category"].astype(str).str.strip()
    valid_mask = raw["category"].notna() & (raw["category"].str.len()>0) & (raw["category"].str.lower()!="nan")
    raw = raw[valid_mask].copy()

    # 연도 가드
    raw = raw[~((raw["source"]=="from_quarter") & (raw["year"]>=2020))].copy()
    raw = raw[~((raw["source"]=="half_direct") & (raw["year"]<=2019))].copy()

    # 중복키는 half_direct 우선
    keys_half = set(h_half_raw.assign(source="half_direct")[["category","year","half"]].apply(tuple, axis=1))
    dup_mask = (raw["source"]=="from_quarter") & raw[["category","year","half"]].apply(tuple, axis=1).isin(keys_half)
    raw = raw[~dup_mask].copy()

    order_half = {"H1":1,"H2":2}
    raw["_ho"]=raw["half"].map(order_half)
    raw = raw.sort_values(["category","year","_ho","source"]).drop(columns="_ho").reset_index(drop=True)

    agg = (raw.groupby(["category","year","half"], as_index=False)["amount"]
             .sum().sort_values(["category","year","half"]).reset_index(drop=True))
    pivot = (agg.pivot_table(index=["category","year"], columns="half", values="amount", fill_value=0)
               .reset_index().sort_values(["category","year"]))

    # 저장
    for fn,df in [("raw",raw),("agg",agg),("pivot",pivot)]:
        (COMBINE_DIR/f"nielsen_half_combined_{fn}.csv").write_text(
            df.to_csv(index=False, encoding="utf-8-sig"))
        try:
            import openpyxl  # noqa
            with pd.ExcelWriter(COMBINE_DIR/f"nielsen_half_combined_{fn}.xlsx", engine="openpyxl") as w:
                df.to_excel(w, index=False, sheet_name=fn)
        except Exception:
            pass
    return agg

# =========================
# 앵커 예측 (YoY / ETS / 자동선택 & 가중)
# =========================
def forecast_yoy_for_cat(train_half_df_cat: pd.DataFrame) -> dict:
    g = (train_half_df_cat.groupby(["year","half"], as_index=False)["amount"].sum()
                         .sort_values(["year","half"]))
    piv = g.pivot(index="year", columns="half", values="amount").sort_index()
    a23h1 = float(piv.loc[2023, "H1"]) if (2023 in piv.index and "H1" in piv.columns) else np.nan
    a23h2 = float(piv.loc[2023, "H2"]) if (2023 in piv.index and "H2" in piv.columns) else np.nan
    a24h1 = float(piv.loc[2024, "H1"]) if (2024 in piv.index and "H1" in piv.columns) else np.nan

    if np.isfinite(a23h2) and np.isfinite(a23h1) and np.isfinite(a24h1) and a23h1>0:
        r1 = a24h1 / a23h1
        y_2024h2 = a23h2 * r1
    else:
        r1 = np.nan; y_2024h2 = a23h2 if np.isfinite(a23h2) else np.nan

    if np.isfinite(y_2024h2) and np.isfinite(a23h2) and np.isfinite(a24h1) and a23h2>0:
        r2 = y_2024h2 / a23h2
        y_2025h1 = a24h1 * r2
    else:
        r2 = np.nan; y_2025h1 = a24h1 if np.isfinite(a24h1) else np.nan

    return {"2024H2": y_2024h2, "2025H1": y_2025h1}

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
        f = fit.forecast(2)  # next two halves
    return {"2024H2": float(f[0]), "2025H1": float(f[1])}

def decide_anchor_for_category(cat: str, train_df_cat: pd.DataFrame, actual_2024H2: float|None) -> dict:
    yoy = forecast_yoy_for_cat(train_df_cat)
    ets = forecast_ets_for_cat(train_df_cat)

    # 기본(폴백) 방법
    base_method = BASE_METHOD_BY_CAT.get(cat, "yoy")
    wY, wE = ENSEMBLE_W_DEFAULT["YoY"], ENSEMBLE_W_DEFAULT["ETS"]

    # 자동 선택/가중
    if AUTO_SELECT_METHOD and actual_2024H2 is not None and np.isfinite(actual_2024H2):
        ey = abs((yoy["2024H2"] or np.nan) - actual_2024H2)
        ee = abs((ets["2024H2"] or np.nan) - actual_2024H2)
        if np.isfinite(ey) and np.isfinite(ee):
            # 오차 역비례 가중
            eps = 1e-6
            wy = 1.0 / (ey + eps)
            we = 1.0 / (ee + eps)
            wsum = wy + we
            wY, wE = wy/wsum, we/wsum
            method = f"auto_ensemble(yoy={wY:.2f}, ets={wE:.2f})"
            h2_pred = wY*yoy["2024H2"] + wE*ets["2024H2"]
            h1_pred = wY*yoy["2025H1"] + wE*ets["2025H1"]
        else:
            # 둘 중 하나만 유효하면 그걸 사용
            if np.isfinite(ey):
                method = "auto_yoy"
                h2_pred, h1_pred = yoy["2024H2"], yoy["2025H1"]
            elif np.isfinite(ee):
                method = "auto_ets"
                h2_pred, h1_pred = ets["2024H2"], ets["2025H1"]
            else:
                method = base_method
                h2_pred, h1_pred = yoy["2024H2"], yoy["2025H1"]
    else:
        # 수동 고정
        if base_method == "yoy":
            method = "yoy"
            h2_pred, h1_pred = yoy["2024H2"], yoy["2025H1"]
        elif base_method == "ets":
            method = "ets"
            h2_pred, h1_pred = ets["2024H2"], ets["2025H1"]
        else:
            method = "yoy_ets_ensemble"
            h2_pred = ENSEMBLE_W_DEFAULT["YoY"]*yoy["2024H2"] + ENSEMBLE_W_DEFAULT["ETS"]*ets["2024H2"]
            h1_pred = ENSEMBLE_W_DEFAULT["YoY"]*yoy["2025H1"] + ENSEMBLE_W_DEFAULT["ETS"]*ets["2025H1"]

    # 바이어스 보정 (λ 자동화 선택)
    bias = np.nan
    h1_adj = h1_pred
    if actual_2024H2 is not None and np.isfinite(actual_2024H2) and np.isfinite(h2_pred) and h2_pred>0:
        bias = actual_2024H2 / h2_pred
        if ADAPTIVE_BIAS:
            # |log(bias)|가 클수록 λ를 0.5 쪽으로 키움(과적합 방지 위해 상한 0.5)
            lam = np.clip(0.3 + 0.2*min(1.0, abs(np.log(max(bias,1e-6)))), 0.3, 0.5)
        else:
            lam = 0.4
        h1_adj = (h1_pred ** (1.0 - lam)) * ((h1_pred * bias) ** lam)

    h2_final = actual_2024H2 if (USE_ACTUAL_2024H2_AS_ANCHOR and actual_2024H2 is not None and np.isfinite(actual_2024H2)) else h2_pred
    return {"method": method, "2024H2": h2_final, "2025H1": h1_adj, "pred_raw_2024H2": h2_pred, "bias": bias}

# =========================
# POS 2023 월 시즌 (없으면 균등)
# =========================
def load_pos_monthly_season_2023(fp: Path) -> dict[str, dict[int,float]]:
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
    cat_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["카테고리","category","cat"])), None)
    y_col   = next((c for c in df.columns if any(k in str(c).lower() for k in ["연도","year"])), None)
    m_col   = next((c for c in df.columns if any(k in str(c).lower() for k in ["월","month","mm"])), None)
    date_col= next((c for c in df.columns if any(k in str(c).lower() for k in ["일자","date"])), None)
    val_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["수량","매출","판매","units","amount","qty","value"])), None)
    if cat_col is None or val_col is None:
        return {}

    tmp = df.copy()
    tmp[cat_col] = tmp[cat_col].astype(str).str.strip()

    if date_col and date_col in tmp.columns:
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
    tmp["year"] = tmp["year"].astype(int); tmp["month"]= tmp["month"].astype(int)
    tmp = tmp[tmp["year"]==2023]
    if tmp.empty:
        return {}

    season = {}
    for cat, g in tmp.groupby(tmp[cat_col]):
        s = g.groupby("month")[val_col].sum().reindex(range(1,13), fill_value=0.0)
        w = (s/s.mean()).to_dict() if s.sum()>0 else {m:1.0 for m in range(1,13)}
        season[str(cat)] = w
    return season

def get_half_norm_weights_from_season(season12: dict[int,float], half_key: str) -> np.ndarray:
    months = [7,8,9,10,11,12] if half_key=="2024H2" else [1,2,3,4,5,6]
    arr = np.array([season12.get(m,1.0) for m in months], dtype=float)
    s = arr.sum()
    return arr/s if s>0 else np.ones(6)/6.0

# =========================
# 페르소나 월형상
# =========================
def persona_month_shape(per: dict, order: list[str]) -> np.ndarray:
    if "monthly_by_launch" in per:
        w = np.array(per["monthly_by_launch"], float)
    else:
        d = per["monthly_by_calendar"]
        w = np.array([d[m] for m in order], float)
    return w / w.mean()

def mix_persona_shapes_for_product(product: str, personas: dict, tau: float) -> np.ndarray:
    arr = personas[product]
    probs = np.array([p.get("purchase_probability",0)/100.0 for p in arr], float)
    s = softmax(probs / max(1e-6, tau))
    W = np.vstack([persona_month_shape(p, MONTHS) for p in arr])  # (k,12)
    u = (s[:,None]*W).sum(axis=0)
    return u / u.sum()

# =========================
# α(제품 정적 가중) 고도화
# =========================
def _parse_numeric(x):
    try:
        # "1,234ml" "900 g" 등에서 숫자만 추출
        if isinstance(x,str):
            m = re.findall(r"[0-9]+\.?[0-9]*", x.replace(",",""))
            return float(m[0]) if m else np.nan
        return float(x)
    except Exception:
        return np.nan

def compute_alpha_per_category(products_by_cat: dict[str, list[str]],
                               personas: dict,
                               product_info_fp: Path|None=None,
                               persona_gamma: float = 1.15) -> dict[str, dict[str,float]]:
    hints = {}
    size_map = {}
    price_map = {}
    if product_info_fp is not None and product_info_fp.exists():
        try:
            pdf = pd.read_excel(product_info_fp) if product_info_fp.suffix.lower() in [".xlsx",".xls"] else pd.read_csv(product_info_fp)
            # 이름 컬럼 찾기
            name_col = next((c for c in pdf.columns if str(c).lower() in ["product_name","제품명","product","sku"]), None) or pdf.columns[0]
            # 힌트/가격/용량 후보
            hint_col = next((c for c in pdf.columns if str(c).lower() in ["alpha_hint","baseline_share","alpha","share_hint","가중치"]), None)
            price_col = next((c for c in pdf.columns if any(k in str(c).lower() for k in ["가격","price","cost"])), None)
            size_col  = next((c for c in pdf.columns if any(k in str(c).lower() for k in ["용량","중량","size","용량(ml)","중량(g)","ml","g"])), None)

            tmp = pdf.copy()
            tmp[name_col] = tmp[name_col].astype(str).str.strip()
            if hint_col is not None:
                tmp["hint"] = pd.to_numeric(tmp[hint_col], errors="coerce")
                hints = {r[name_col]: float(r["hint"]) for _,r in tmp[[name_col,"hint"]].dropna().iterrows() if r["hint"]>0}

            if size_col is not None:
                tmp["size_n"] = tmp[size_col].apply(_parse_numeric)
                size_map = {r[name_col]: float(r["size_n"]) for _,r in tmp[[name_col,"size_n"]].dropna().iterrows() if r["size_n"]>0}
            if price_col is not None:
                tmp["price_n"] = tmp[price_col].apply(_parse_numeric)
                price_map = {r[name_col]: float(r["price_n"]) for _,r in tmp[[name_col,"price_n"]].dropna().iterrows() if r["price_n"]>0}
        except Exception:
            pass

    alpha: dict[str, dict[str,float]] = {}
    for c, prods in products_by_cat.items():
        if not prods:
            alpha[c] = {}
            continue

        # 1) 페르소나 기반 스코어
        persona_score = {}
        for p in prods:
            probs = [float(per.get("purchase_probability",0)) for per in personas[p]]
            persona_score[p] = max(1e-9, np.mean(probs)/100.0) ** persona_gamma

        # 2) 단위가격 기반 cheapness (저렴할수록 ↑)
        cheap_raw = {}
        for p in prods:
            sz = size_map.get(p, np.nan)
            pr = price_map.get(p, np.nan)
            if np.isfinite(sz) and sz>0 and np.isfinite(pr) and pr>0:
                unit_price = pr / sz
                cheap_raw[p] = 1.0 / unit_price
        # 카테고리 내부 min-max 정규화
        if cheap_raw:
            vals = np.array(list(cheap_raw.values()), float)
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            for p in prods:
                if p in cheap_raw:
                    cheap_raw[p] = (cheap_raw[p]-vmin)/(vmax-vmin + 1e-12) + 1e-6  # 0~1 → 0~1+ε
                else:
                    cheap_raw[p] = 1e-6
        else:
            cheap_raw = {p: 1.0 for p in prods}

        # 3) hint
        hint_raw = {p: max(1e-6, float(hints.get(p, 1.0))) for p in prods}

        # 4) 가중 기하 평균 결합(카테고리별 비율)
        wcfg = ALPHA_WEIGHTS.get(c, ALPHA_WEIGHTS["기본"])
        w_per, w_chp, w_hint = wcfg["persona"], wcfg["cheapness"], wcfg["hint"]

        score = {}
        for p in prods:
            # 기하 평균: a^w1 * b^w2 * c^w3
            score[p] = (persona_score[p] ** w_per) * (cheap_raw[p] ** w_chp) * (hint_raw[p] ** w_hint)

        tot = sum(score.values())
        alpha[c] = {p: (score[p]/tot if tot>0 else 1.0/len(prods)) for p in prods}

    return alpha

# =========================
# 메인
# =========================
if __name__ == "__main__":
    # 0) 닐슨 반기 통합
    qdf = read_nielsen_quarter_kor(NIELSEN_Q_FP)
    hdf = read_nielsen_half_kor(NIELSEN_H_FP)
    h_from_q = quarter_to_half(qdf)
    half_all = combine_nielsen_half(h_from_q, hdf)  # category, year, half, amount

    cats = sorted(half_all["category"].unique())
    actual_2024H2_by_cat = {}
    for c in cats:
        row = half_all[(half_all["category"]==c) & (half_all["year"]==2024) & (half_all["half"]=="H2")]
        actual_2024H2_by_cat[c] = float(row["amount"].sum()) if not row.empty else None

    # 1) POS 2023 시즌
    season_pos = load_pos_monthly_season_2023(POS_FP)

    # 2) 페르소나 & SKU 세팅
    personas = pd.read_json(PERSONA_FP, typ="dict")
    sku_list = list(personas.keys())
    for p in sku_list:
        if p not in CATEGORY_OF:
            raise ValueError(f"화이트리스트 외 제품: {p}")
    products_by_cat = defaultdict(list)
    for p in sku_list:
        products_by_cat[CATEGORY_OF[p]].append(p)

    # 3) 제품별 페르소나 월형상(합=1), 카테고리별 τ 오버라이드 지원
    Wp = {}
    for p in sku_list:
        cat = CATEGORY_OF[p]
        tau_eff = TAU_BY_CAT.get(cat, TAU)
        Wp[p] = mix_persona_shapes_for_product(p, personas, tau_eff)  # len=12, sum=1

    # 4) α(정적 가중) — 페르소나 + 단위가격 + hint
    alpha = compute_alpha_per_category(products_by_cat, personas, PRODUCT_INFO_FP, persona_gamma=1.15)

    # 5) 카테고리 앵커 추정(자동 선택/가중 + 바이어스 λ 자동)
    anchors = {}
    for c in cats:
        g_all = half_all[half_all["category"]==c]
        train = g_all[~((g_all["year"]==2024) & (g_all["half"]=="H2"))]
        act = actual_2024H2_by_cat.get(c, None)
        anchors[c] = decide_anchor_for_category(c, train, act)

    # 6) 카테고리 반기 앵커 → 월 분배(반기 내부 6개월 정규화)
    cat_month_qty = {}
    for c in cats:
        season12 = season_pos.get(c, {m:1.0 for m in range(1,13)})
        for half_key in ["2024H2","2025H1"]:
            anchor = anchors[c].get(half_key, np.nan)
            if not np.isfinite(anchor) or anchor<=0:
                for m in HALF_MONTHS[half_key]: cat_month_qty[(c,m)] = 0.0
                continue
            w6 = get_half_norm_weights_from_season(season12, half_key)
            for m, w in zip(HALF_MONTHS[half_key], w6):
                cat_month_qty[(c, m)] = float(anchor * w)

    # 7) 카테고리→SKU 분배: score_p,t = α_p * Wp[p][t]
    pred_rows = []
    for m in MONTHS:
        for c in cats:
            total_cm = cat_month_qty.get((c, m), 0.0)
            prods = products_by_cat.get(c, [])
            if not prods or total_cm <= 0:
                for p in prods:
                    pred_rows.append({"product": p, "month": m, "qty": 0.0})
                continue
            idx = MONTHS.index(m)
            scores = {p: max(1e-12, alpha[c].get(p,0.0) * Wp[p][idx]) for p in prods}
            denom = sum(scores.values())
            if denom<=0:
                for p in prods:
                    pred_rows.append({"product": p, "month": m, "qty": total_cm/len(prods)})
            else:
                for p in prods:
                    pred_rows.append({"product": p, "month": m, "qty": total_cm * (scores[p]/denom)})

    df_pred = pd.DataFrame(pred_rows)

    # 8) 제출(wide)
    sub_path = SAVE_DIR / "submission_anchor.csv"
    sample = pd.read_csv(SAMPLE_FP)
    req_cols = list(sample.columns)
    assert req_cols and req_cols[0]=="product_name"
    pred_products = set(df_pred["product"].unique())
    sample_products = set(sample["product_name"].unique())
    miss = sample_products - pred_products
    extra = pred_products - sample_products
    if miss:  raise ValueError(f"예측에 없는 제품이 샘플에 있습니다: {sorted(miss)}")
    if extra: raise ValueError(f"샘플에 없는 제품을 예측했습니다: {sorted(extra)}")

    wide_map = {}
    for i,m in enumerate(MONTHS, start=1):
        s = (df_pred[df_pred["month"]==m].set_index("product")["qty"])
        wide_map[f"months_since_launch_{i}"] = s

    sub = sample.copy()
    for col, s in wide_map.items():
        sub[col] = sub["product_name"].map(s).fillna(0.0)

    # 반올림·비음수·정수
    val_cols = [c for c in sub.columns if c != "product_name"]
    sub[val_cols] = (sub[val_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).round(0).clip(lower=0).astype("int64"))
    sub = sub[req_cols]
    sub.to_csv(sub_path, index=False, encoding="utf-8-sig")

    # 9) 앵커 요약 저장
    df_rep = pd.DataFrame([{
        "category": c,
        "method": anchors[c]["method"],
        "anchor_2024H2": anchors[c]["2024H2"],
        "anchor_2025H1": anchors[c]["2025H1"],
        "pred_raw_2024H2": anchors[c]["pred_raw_2024H2"],
        "bias_2024H2": anchors[c]["bias"],
        "use_actual_2024H2": USE_ACTUAL_2024H2_AS_ANCHOR
    } for c in cats])
    df_rep.to_csv(SAVE_DIR/"anchor_summary.csv", index=False, encoding="utf-8-sig")

    # 10) 반기 합계 = 앵커(정수) 정합 보정
    anch = pd.read_csv(SAVE_DIR/"anchor_summary.csv", encoding="utf-8").set_index("category")
    sub["cat"] = sub["product_name"].map(CATEGORY_OF)

    h2_cols = [f"months_since_launch_{i}" for i in range(1,7)]
    h1_cols = [f"months_since_launch_{i}" for i in range(7,13)]

    def apply_half_adjustment(sub_df: pd.DataFrame, cols: list[str], anchor_col: str) -> pd.DataFrame:
        for c, g in sub_df.groupby("cat"):
            if c not in anch.index: continue
            target = int(round(float(anch.loc[c, anchor_col])))
            current = int(sub_df.loc[g.index, cols].values.sum())
            dz = target - current
            if dz == 0: continue
            last_col = cols[-1]
            block = sub_df.loc[g.index, [last_col]]
            i_target = block[last_col].idxmax()
            new_val = sub_df.at[i_target, last_col] + dz
            if new_val < 0:
                sub_df.at[i_target, last_col] = 0
                dz_rest = new_val
                others = block.drop(index=i_target).sort_values(last_col, ascending=False).index.tolist()
                for idx in others:
                    if dz_rest == 0: break
                    take = min(sub_df.at[idx, last_col], abs(dz_rest))
                    sub_df.at[idx, last_col] = sub_df.at[idx, last_col] - take
                    dz_rest += take
            else:
                sub_df.at[i_target, last_col] = new_val
        return sub_df

    sub = apply_half_adjustment(sub, h2_cols, "anchor_2024H2")
    sub = apply_half_adjustment(sub, h1_cols, "anchor_2025H1")

    # 타입·비음수 보장 후 조정본 저장
    val_cols2 = [c for c in sub.columns if c.startswith("months_since_launch_")]
    sub[val_cols2] = sub[val_cols2].round(0).clip(lower=0).astype("int64")
    sub = sub.drop(columns=["cat"])
    sub_adj_path = SAVE_DIR/"submission_anchor_adjusted.csv"
    sub.to_csv(sub_adj_path, index=False, encoding="utf-8-sig")

    print("[완료] 제출 저장:", sub_path.resolve())
    print("[완료] 조정본 저장:", sub_adj_path.resolve())
    print("[요약] 앵커/방법:", (SAVE_DIR/"anchor_summary.csv").resolve())
