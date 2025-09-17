# -*- coding: utf-8 -*-
"""
Persona-based monthly forecasting (cleaned) + optional custom month-share injection.

- Minimal diagnostics only.
- POS/CLICK-based month-share (category level) OR user-provided custom month-share
  (product or category level) with blending.
- Yearly total preserved; only monthly shape redistributed.

Author: you :)
"""

from __future__ import annotations
import json, math, re, unicodedata, warnings, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# -------- Korean font (optional) --------
try:
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\malgun.ttf")
    rcParams["font.family"] = "Malgun Gothic"
    rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# =====================
# Small utilities
# =====================
def P(msg: str = ""):
    print(msg)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _norm_txt(x: object) -> str:
    s = unicodedata.normalize("NFKC", str(x)).replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def savefig_safe(fig, out_path: Path):
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.stem + ".part" + out_path.suffix)
        fig.tight_layout()
        fig.savefig(tmp, dpi=200, bbox_inches="tight")
        tmp.replace(out_path)
        P(f"[VIZ] saved: {out_path.resolve()}")
    except Exception as e:
        P(f"[VIZ][ERROR] save failed: {out_path} — {e}")

# =====================
# CONFIG
# =====================
class CONFIG:
    # Paths
    DATA_DIR = Path("./_data/dacon/dongwon")
    SAVE_DIR = Path("./_save/persona_sim"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

    PATH_PERSONA2 = DATA_DIR / "personas2_withPOS.json"
    PATH_PERSONA1 = DATA_DIR / "personas.json"
    PATH_PERSONA  = PATH_PERSONA2 if PATH_PERSONA2.exists() else PATH_PERSONA1

    PATH_SEARCH  = DATA_DIR / "naver/search_trend_all.csv"
    PATH_CLICK   = DATA_DIR / "naver/click_trend_all.csv"
    PATH_SAMPLE  = DATA_DIR / "sample_submission.csv"

    # POS(month) for optional POS share
    PATH_POS_TOTAL_MONTH = DATA_DIR / "pos_data/marketlink_POS_2020_2023_월별 매출액.xlsx"
    PATH_POS_DW_MONTH    = DATA_DIR / "pos_data/marketlink_POS_2020_2023_동원 F&B_매출액.xlsx"
    PATH_POS_SEG_MONTH   = DATA_DIR / "pos_data/marketlink_POS_2020_2023_세분시장_매출액.xlsx"

    # Evaluation horizon
    PRED_MONTHS = pd.date_range("2024-07-01", "2025-06-01", freq="MS")

    # Global defaults
    EMA_SPAN_DEFAULT        = 6
    MOMENTUM_COEF_DEFAULT   = 0.60
    CLICK_TO_SALES_RATE_DEF = 85
    CALIB_MULT_DEFAULT      = 8.00

    # Calibration mode: "FULL12" | "PUBLIC6" | "PIECEWISE"
    CALIB_MODE = "FULL12"

    # Early-month floor
    USE_EARLY_FLOOR = True
    EARLY_FLOOR_N_MONTHS = 3
    EARLY_FLOOR_FRAC = 0.015

    # --- POS/CLICK month-share option (category-level) ---
    USE_POS_MONTH_SHARE = True          # ← 커스텀 입력을 쓸 땐 False로 자동 전환됨
    POS_SEASON_START = 7
    POS_LAM_UNI      = 0.20
    POS_MIN_MSHARE   = 0.005

    # --- NEW: Custom month-share direct injection ---
    USE_CUSTOM_MONTH_SHARE = True      # True면 커스텀 입력 사용
    CUSTOM_SHARE_WEIGHT    = 1.00       # 1.0=완전 치환, 0.5=절충(모델:커스텀=50:50)
    CUSTOM_MONTH_SHARE_PATH = ""        # CSV 경로(long: key,ym,share | wide: key,m1..m12). 비우면 딕셔너리 사용

    # Seasonalization for search/click templates
    USE_SEASONAL_SC  = True
    SC_SEASON_START  = 7
    SC_LAM_UNI       = 0.10
    SC_DECAY         = 1.00

# =====================
# Metadata
# =====================
CAT2PRODS: Dict[str, List[str]] = {
    "발효유": ["덴마크 하이그릭요거트 400g"],
    "조제커피": ["소화가 잘되는 우유로 만든 카페라떼 250mL",
               "소화가 잘되는 우유로 만든 바닐라라떼 250mL"],
    "조미료": ["동원참치액 순 500g","동원참치액 순 900g",
             "동원참치액 진 500g","동원참치액 진 900g",
             "프리미엄 동원참치액 500g","프리미엄 동원참치액 900g"],
    "식육가공품": ["리챔 오믈레햄 200g","리챔 오믈레햄 340g"],
    "참치캔": ["동원맛참 고소참기름 90g","동원맛참 고소참기름 135g",
             "동원맛참 매콤참기름 90g","동원맛참 매콤참기름 135g"],
}
PROD2CAT = {p: c for c, prods in CAT2PRODS.items() for p in prods}

# 카테고리별 POS 신뢰도(0~1)
MU_BY_CAT = {
    "발효유": 0.85,
    "식육가공품": 0.80,
    "조제커피": 0.80,
    "참치캔": 0.85,
    "조미료": 0.80,  # 21~22 DW 이상치 고려
}

PRICE_PER_UNIT: Dict[str, float] = {
    "덴마크 하이그릭요거트 400g": 4700,
    "동원맛참 고소참기름 135g": 2500,
    "동원맛참 고소참기름 90g": 1800,
    "동원맛참 매콤참기름 135g": 2500,
    "동원맛참 매콤참기름 90g": 1800,
    "동원참치액 순 500g": 5980,
    "동원참치액 순 900g": 9980,
    "동원참치액 진 500g": 5980,
    "동원참치액 진 900g": 9980,
    "프리미엄 동원참치액 500g": 11480,
    "프리미엄 동원참치액 900g": 17980,
    "리챔 오믈레햄 200g": 3980,
    "리챔 오믈레햄 340g": 4780,
    "소화가 잘되는 우유로 만든 바닐라라떼 250mL": 2680,
    "소화가 잘되는 우유로 만든 카페라떼 250mL": 2680,
}

POS_CAT_MAP = {
    "발효유":"발효유",
    "조제커피":"조제커피",
    "조미료":"조미료",
    "식육가공품":"식육가공품",
    "참치캔":"참치캔",
}

# ============== CUSTOM MONTH-SHARE (manual input) ==============
# key를 "카테고리" 또는 "제품명"으로 하고, 값은 길이 12 리스트 (합≈1)
CUSTOM_MONTH_SHARE: Dict[str, List[float]] = {
   
    "발효유": [0.0783,0.0801,0.1024,0.0731,0.0684,0.1067,0.0738,0.0703,0.0974,0.0762,0.0764,0.0969],
    "참치캔": [0.079,0.089,0.085,0.069,0.073,0.101,0.076,0.073,0.097,0.076,0.079,0.103,],
    "조제커피": [0.056,0.057,0.082,0.073,0.083,0.101,0.082,0.078,0.132,0.087,0.075,0.094,],
    "식육가공품": [0.079,0.085,0.091,0.073,0.07,0.101,0.077,0.073,0.095,0.082,0.074,0.1],
    "조미료": [0.025,0.037,0.062,0.049,0.050,0.083,0.074,0.080,0.134,0.113,0.126,0.167,],
}

# =====================
# Loaders
# =====================
def load_personas(path: Path) -> Dict[str, List[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_trends() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    search = pd.read_csv(CONFIG.PATH_SEARCH, parse_dates=["date"]).copy()
    click  = pd.read_csv(CONFIG.PATH_CLICK,  parse_dates=["date"]).copy()
    sample = pd.read_csv(CONFIG.PATH_SAMPLE).copy()
    need_cols = ["product_name","date","gender","age"]
    for col in need_cols:
        if col not in search.columns: raise ValueError(f"[search] missing column: {col}")
        if col not in click.columns:  raise ValueError(f"[click]  missing column: {col}")
    if "search_index" not in search.columns: raise ValueError("search_trend_all.csv requires 'search_index'")
    if "clicks" not in click.columns:       raise ValueError("click_trend_all.csv requires 'clicks'")
    return search, click, sample

def _prepare_pos_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_txt(c) for c in df.columns]
    REN = {"연도":"year","월":"month","구분":"group","카테고리":"category","매출액(백만원)":"sales_mm","비고":"note"}
    df = df.rename(columns={k:v for k,v in REN.items() if k in df.columns})
    if "group" in df.columns:
        df["group_norm"] = df["group"].astype(str).apply(_norm_txt)
        df["is_dw"] = df["group_norm"].str.contains(r"동원\s*F\s*&\s*B", regex=True)
    else:
        df["is_dw"] = False
    if "category" in df.columns:
        df["category_std"] = df["category"].astype(str).apply(_norm_txt).map(POS_CAT_MAP).fillna(df["category"].astype(str))
    if "note" in df.columns:
        df["note_norm"] = df["note"].astype(str).apply(_norm_txt)
        df = df[~df["note_norm"].str.contains("etc", case=False, na=False)]
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "month" in df.columns:
        df["month"] = df["month"].astype(str).str.extract(r"(\d+)").astype(float).astype("Int64")
    if "sales_mm" in df.columns:
        df["sales_mm"] = pd.to_numeric(df["sales_mm"], errors="coerce")
    keep = [c for c in ["year","month","group","category","sales_mm","note","is_dw","category_std"] if c in df.columns]
    return df[keep]

def _safe_read_excel(path: Path) -> Optional[pd.DataFrame]:
    try:
        xls = pd.ExcelFile(path)
        best, best_cols = None, -1
        for s in xls.sheet_names:
            hdr = pd.read_excel(path, sheet_name=s, nrows=0)
            if hdr.shape[1] > best_cols:
                best, best_cols = s, hdr.shape[1]
        raw = pd.read_excel(path, sheet_name=best)
        return _prepare_pos_month(raw)
    except Exception:
        return None

def load_pos_months() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    tot = _safe_read_excel(CONFIG.PATH_POS_TOTAL_MONTH)
    dw  = _safe_read_excel(CONFIG.PATH_POS_DW_MONTH)
    seg = _safe_read_excel(CONFIG.PATH_POS_SEG_MONTH)
    return tot, dw, seg

# =====================
# Personas helpers (lightweight)
# =====================
def get_vw(d: dict, keys: List[str], default_value="", default_weight=0.6) -> Tuple[str, float]:
    for k in keys:
        if k in d:
            obj = d[k]
            if isinstance(obj, dict):
                v = obj.get("value", obj.get("val", default_value))
                w = safe_float(obj.get("weight", default_weight), default_weight)
                return v, w
            return obj, default_weight
    return default_value, default_weight

def age_to_bucket(a: object) -> str:
    s = str(a); nums = re.findall(r"\d+", s)
    if nums: v = int(nums[0]); return f"{(v//10)*10}s"
    for k in ["10","20","30","40","50","60"]:
        if k in s: return f"{k}s"
    return "ETC"

def gender_key(g: object) -> str:
    s = str(g)
    return "F" if ("여" in s or s.upper().startswith("F")) else "M"

def label_level(val: object) -> float:
    s = str(val)
    if "매우" in s: return 1.0
    if "높"  in s: return 0.8
    if "중"  in s: return 0.6
    if "낮"  in s: return 0.4
    return 0.5

def family_mult(v: object) -> float:
    s = str(v)
    if "대가족" in s or "5" in s or "6" in s: return 1.30
    if "4" in s: return 1.20
    if "3" in s: return 1.10
    if "1인" in s: return 0.85
    return 1.00

def loyalty_mult(v: object) -> float:
    s = str(v)
    if "높" in s: return 1.15
    if "중" in s: return 1.05
    return 1.00

def income_mult(v: object) -> float:
    s = str(v); nums = list(map(int, re.findall(r"\d+", s)))
    mean_income = np.mean(nums) if nums else 300
    if mean_income >= 500: return 1.06
    if mean_income >= 350: return 1.03
    if mean_income >= 250: return 1.00
    return 0.97

def health_mult(product_name: str, v: object) -> float:
    s = str(v); base = 1.00
    if "높" in s: base = 1.08
    if "유당" in s and "소화가 잘되는 우유" in product_name: base = 1.15
    return base

def promo_mult(v: object) -> float:
    lv = label_level(v)
    return 0.95 + 0.20*lv

def trend_response_to_kappa(tr_val: object) -> float:
    lv = label_level(tr_val)
    return 0.5 + 1.5*lv

def lifestyle_month_boost(month_int: int, lifestyle_text: object) -> float:
    s = str(lifestyle_text); m = month_int; mult = 1.00
    if any(k in s for k in ["명절","선물","집들이","손님"]):
        if m in [1,2,9,10]: mult *= 1.15
    if any(k in s for k in ["캠핑","야외","피크닉","도시락"]):
        if m in [4,5,6,9,10]: mult *= 1.08
    if any(k in s for k in ["운동","자기관리"]):
        if m in [1,2,3,5]: mult *= 1.06
    if any(k in s for k in ["홈파티","홈쿡"]):
        if m in [11,12,1]: mult *= 1.06
    if any(k in s for k in ["매운","야식"]):
        if m in [11,12,1,2]: mult *= 1.05
    if any(k in s for k in ["국","찌개","김장"]):
        if m in [11,12,1,2]: mult *= 1.12
        if m == 11: mult *= 1.15
    if any(k in s for k in ["오피스","출근","출퇴근"]):
        if m in [3,4,5,9,10]: mult *= 1.04
    return mult

def smooth_event_boost(month_int: int, drivers: object) -> float:
    mult = 1.0
    if drivers is None: return mult
    if isinstance(drivers, str): ds = re.split(r"[,\s/·]+", drivers)
    else: ds = list(drivers)
    norm = [str(d) for d in ds]

    def gauss(center: int, m: int, sigma=1.0, peak=1.15):
        return 1.0 + (peak-1.0)*math.exp(-0.5*((m-center)/sigma)**2)

    m = month_int
    if any("설" in s for s in norm):
        mult *= gauss(1, m, sigma=1.0, peak=1.12); mult *= gauss(2, m, sigma=1.0, peak=1.10)
    if any("추석" in s for s in norm):
        mult *= gauss(9, m, sigma=1.0, peak=1.12); mult *= gauss(10, m, sigma=1.0, peak=1.10)
    if any("김장" in s for s in norm):   mult *= gauss(11, m, sigma=0.8, peak=1.18)
    if any(k in s for k in ["여름","냉음","아이스"] for s in norm): mult *= gauss(7, m, sigma=1.0, peak=1.08)
    if any("연말" in s or "크리스마스" in s for s in norm):          mult *= gauss(12, m, sigma=1.0, peak=1.06)
    return mult

# =====================
# Trend normalization & persona-weighted series
# =====================
def normalize_per_product(df: pd.DataFrame, val_col: str, out_name: str) -> pd.DataFrame:
    df = df.copy()
    df["__min"] = df.groupby("product_name", group_keys=False)[val_col].transform("min")
    df["__max"] = df.groupby("product_name", group_keys=False)[val_col].transform("max")
    df[out_name] = (df[val_col] - df["__min"]) / (df["__max"] - df["__min"] + 1e-6)
    return df.drop(columns=["__min","__max"])

def to_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_bucket"] = df["age"].apply(age_to_bucket)
    df["gender_key"] = df["gender"].apply(gender_key)
    return df

def build_segment_weights(personas: Dict[str, List[dict]]) -> Dict[str, Dict[Tuple[str,str], float]]:
    seg_weights: Dict[str, Dict[Tuple[str,str], float]] = {}
    for product, plist in personas.items():
        seg: Dict[Tuple[str,str], float] = {}
        for p in plist:
            gender_v, gender_w = get_vw(p, ["성별","gender"])
            age_v, age_w       = get_vw(p, ["연령","age"])
            g = gender_key(gender_v); ab = age_to_bucket(age_v)
            purchase_prob = safe_float(p.get("purchase_probability", p.get("purchase_prob", 60)), 60)/100.0
            w = (safe_float(gender_w,0) + safe_float(age_w,0))/2.0
            w *= purchase_prob
            seg[(g,ab)] = seg.get((g,ab), 0.0) + w
        ssum = sum(seg.values()) or 1.0
        seg_weights[product] = {k: v/ssum for k,v in seg.items()}
    return seg_weights

def weighted_by_persona(df: pd.DataFrame, val_col_norm: str, seg_w: Dict[str, Dict[Tuple[str,str], float]]) -> pd.DataFrame:
    out = []
    for prod, g in df.groupby("product_name", group_keys=False):
        wmap = seg_w.get(prod, {})
        if g.empty: continue
        gg = g.copy()
        gg["w"] = gg.apply(lambda r: wmap.get((r["gender_key"], r["age_bucket"]), 0.0), axis=1)
        gg["num"] = gg[val_col_norm] * gg["w"]
        agg = (gg.groupby(["product_name","date"], as_index=False)
                 .agg(wsum=("w","sum"), numsum=("num","sum"), meanval=(val_col_norm,"mean")))
        agg[f"{val_col_norm}_w"] = np.where(agg["wsum"] > 0,
                                            agg["numsum"]/(agg["wsum"]+1e-9),
                                            agg["meanval"])
        out.append(agg[["product_name","date", f"{val_col_norm}_w"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["product_name","date",f"{val_col_norm}_w"])

def _seasonalize_to_pred(df: pd.DataFrame, val_col: str,
                         pred_months: pd.DatetimeIndex,
                         season_start: int = 7,
                         lam_uni: float = 0.10,
                         decay: float = 1.00) -> pd.DataFrame:
    """다년 시즌 템플릿(12) → 예측 구간으로 맵핑"""
    if df.empty: return df
    t = df.copy().dropna(subset=["date"])
    t["month"] = t["date"].dt.month
    t["season_id"] = np.where(t["month"] >= season_start, t["date"].dt.year, t["date"].dt.year - 1)
    t["mos"] = (t["month"] - season_start) % 12
    t["w"] = 1.0 if decay == 1.0 else decay ** (t["season_id"].max() - t["season_id"])
    t["v"] = t[val_col].astype(float) * t["w"].astype(float)
    tpl = (t.groupby(["product_name","mos"]).agg(vsum=("v","sum"), wsum=("w","sum")))
    tpl["wa"] = tpl["vsum"] / tpl["wsum"].replace(0, np.nan)
    tpl = (tpl["wa"].unstack("mos").reindex(columns=range(12)))

    rows = []
    for prod, row in tpl.iterrows():
        vec = row.values.astype(float)
        if np.all(np.isnan(vec)): continue
        m = np.nanmean(vec); vec = np.where(np.isnan(vec), m, vec)
        # 균등 혼합(피크 완화)
        vec = (1-lam_uni)*vec + lam_uni*(vec.mean())
        for i, d in enumerate(pred_months):
            rows.append({"product_name": prod, "date": d, val_col: float(vec[i % 12])})
    return pd.DataFrame(rows)

# =====================
# Product params + predict + calibrate
# =====================
@dataclass
class AlphaBetaKappa:
    alpha: float
    beta: float
    kappa: float
    promo_avg: float

def derive_alpha_beta_kappa(personas: Dict[str, List[dict]]) -> Tuple[Dict[str, AlphaBetaKappa], Dict[str, float]]:
    abk: Dict[str, AlphaBetaKappa] = {}
    for product, plist in personas.items():
        if not plist:
            abk[product] = AlphaBetaKappa(0.5, 0.5, 1.0, 0.6); continue
        kappas, promos = [], []
        for p in plist:
            kappas.append(trend_response_to_kappa(get_vw(p, ["트렌드 반응도","trend_responsiveness"])[0]))
            promos.append(label_level(get_vw(p, ["프로모션 민감도","promotion_sensitivity"])[0]))
        abk[product] = AlphaBetaKappa(0.5, 0.5, float(np.mean(kappas) if kappas else 1.0),
                                      float(np.mean(promos) if promos else 0.6))
    return abk, {}

def build_product_params(personas: Dict[str, List[dict]], click_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    params: Dict[str, Dict[str, float]] = {}
    for product, plist in personas.items():
        if not plist:
            params[product] = dict(ema_span=CONFIG.EMA_SPAN_DEFAULT,
                                   mom_coef=CONFIG.MOMENTUM_COEF_DEFAULT,
                                   click_to_sales_rate=CONFIG.CLICK_TO_SALES_RATE_DEF,
                                   calib_mult=CONFIG.CALIB_MULT_DEFAULT)
            continue

        purchase_probs = [safe_float(p.get("purchase_probability", p.get("purchase_prob", 60)), 60) for p in plist]
        prob_mean = np.mean(purchase_probs) if purchase_probs else 60.0
        promo_elastic_vals = [label_level(get_vw(p, ["프로모션 탄력","promotion_elasticity"])[0]) for p in plist]
        promo_elastic_avg = np.mean(promo_elastic_vals) if promo_elastic_vals else 0.6
        kappas = [trend_response_to_kappa(get_vw(p, ["트렌드 반응도","trend_responsiveness"])[0]) for p in plist]
        kappa = np.mean(kappas) if kappas else 1.0
        loyalty_vals = [loyalty_mult(get_vw(p, ["브랜드 충성도","loyalty"])[0]) for p in plist]
        loyalty_factor = np.mean(loyalty_vals) if loyalty_vals else 1.0
        fam_vals = [family_mult(get_vw(p, ["가족 구성","family"])[0]) for p in plist]
        family_factor = np.mean(fam_vals) if fam_vals else 1.0
        income_vals = [income_mult(get_vw(p, ["소득 구간","income"])[0]) for p in plist]
        income_factor = np.mean(income_vals) if income_vals else 1.0
        health_vals = [health_mult(product, get_vw(p, ["건강 관심도","health"])[0]) for p in plist]
        health_factor = np.mean(health_vals) if health_vals else 1.0

        g = click_df[click_df["product_name"]==product].sort_values("date")
        volatility = g["clicks"].astype(float).diff().abs().rolling(2).mean().mean() if len(g)>=3 else 0.1
        ema_span = int(np.clip(round(6 + 4*volatility - 2*(kappa-1)), 3, 12))
        mom_coef = float(np.clip(0.12 + 0.45 * promo_elastic_avg * kappa * (1.0/loyalty_factor), 0.15, 0.7))

        base_rate = CONFIG.CLICK_TO_SALES_RATE_DEF
        click_to_sales_rate = float(base_rate * (prob_mean/60.0)**1.2 * loyalty_factor * family_factor * income_factor * health_factor)
        params[product] = dict(ema_span=ema_span, mom_coef=mom_coef,
                               click_to_sales_rate=click_to_sales_rate,
                               calib_mult=CONFIG.CALIB_MULT_DEFAULT)
    return params

def add_ema_and_momentum(trend_w: pd.DataFrame, product_params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for prod, g in trend_w.groupby("product_name", group_keys=False):
        span = product_params.get(prod, {}).get("ema_span", CONFIG.EMA_SPAN_DEFAULT)
        gg = g.sort_values("date").copy()
        gg["s_sm"] = gg["s"].ewm(span=span, adjust=False).mean()
        gg["c_sm"] = gg["c"].ewm(span=span, adjust=False).mean()
        gg["s_mom"] = gg["s_sm"].diff().fillna(0.0)
        gg["c_mom"] = gg["c_sm"].diff().fillna(0.0)
        rows.append(gg)
    return pd.concat(rows, ignore_index=True) if rows else trend_w

def persona_base_for_month(p: dict, product_name: str, month_idx: int, month_int: int) -> float:
    monthly = p.get("monthly_by_launch", p.get("monthly", [4]*12))
    month_shape = monthly[month_idx % 12] if isinstance(monthly, list) and len(monthly)>=12 else 4.0
    purchase_prob = safe_float(p.get("purchase_probability", p.get("purchase_prob", 60)), 60) / 100.0
    base = purchase_prob * (month_shape ** 1.2)

    fam  = family_mult(get_vw(p, ["가족 구성","family"])[0])
    loy  = loyalty_mult(get_vw(p, ["브랜드 충성도","loyalty"])[0])
    inc  = income_mult(get_vw(p, ["소득 구간","income"])[0])
    hea  = health_mult(product_name, get_vw(p, ["건강 관심도","health"])[0])
    pro  = promo_mult(get_vw(p, ["프로모션 민감도","promotion_sensitivity"])[0])
    life = lifestyle_month_boost(month_int, get_vw(p, ["라이프스타일","lifestyle"])[0])

    drivers = p.get("drivers") or p.get("드라이버") or p.get("이벤트")
    eventf  = smooth_event_boost(month_int, drivers)

    rep_v, _ = get_vw(p, ["재구매 주기","repurchase_cycle"])
    rep = 1.05 if ("주" in str(rep_v)) else 1.0

    mult = fam * loy * inc * hea * pro * life * rep * eventf
    return base * mult

def predict(personas: Dict[str, List[dict]], trend_w_sm: pd.DataFrame,
            product_params: Dict[str, Dict[str, float]], pred_months: pd.DatetimeIndex) -> pd.DataFrame:
    abk, _ = derive_alpha_beta_kappa(personas)
    rows = []
    for product, plist in personas.items():
        alpha = abk.get(product).alpha if product in abk else 0.5
        beta  = abk.get(product).beta  if product in abk else 0.5
        kappa = abk.get(product).kappa if product in abk else 1.0
        promo_avg = abk.get(product).promo_avg if product in abk else 0.6
        mom_coef_prod = product_params.get(product, {}).get("mom_coef", CONFIG.MOMENTUM_COEF_DEFAULT)
        for i, d in enumerate(pred_months):
            m_int = d.month
            base_sum = sum(persona_base_for_month(p, product, i, m_int) for p in plist)
            tr = trend_w_sm[(trend_w_sm["product_name"]==product) & (trend_w_sm["date"]==d)]
            s_sm=c_sm=s_mom=c_mom=0.0
            if not tr.empty:
                s_sm = float(tr["s_sm"].iloc[0]); c_sm = float(tr["c_sm"].iloc[0])
                s_mom= float(tr["s_mom"].iloc[0]); c_mom= float(tr["c_mom"].iloc[0])
            boost_trend = 1.0 + alpha*np.log1p(kappa*s_sm) + beta*np.log1p(kappa*c_sm)
            mom = max(0.0, alpha*s_mom + beta*c_mom)
            boost_mom = 1.0 + mom_coef_prod * promo_avg * mom
            final_qty = base_sum * boost_trend * boost_mom
            rows.append({"product_name": product, "date": d, "raw": base_sum, "quantity": max(0.0, final_qty)})
    return pd.DataFrame(rows)

def calibrate(pred_df: pd.DataFrame, click: pd.DataFrame, product_params: Dict[str, Dict[str, float]],
              mode: str = CONFIG.CALIB_MODE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    click_pm_raw = click.groupby(["product_name","date"], as_index=False, group_keys=False)["clicks"].mean()
    Y12  = CONFIG.PRED_MONTHS
    PUB6 = pd.date_range("2024-07-01","2024-12-01", freq="MS")
    PRI6 = pd.date_range("2025-01-01","2025-06-01", freq="MS")

    raw_12 = (click_pm_raw[click_pm_raw["date"].isin(Y12)].groupby("product_name", group_keys=False)["clicks"].sum().rename("raw12").reset_index())
    raw_6  = (click_pm_raw[click_pm_raw["date"].isin(PUB6)].groupby("product_name", group_keys=False)["clicks"].sum().rename("raw6").reset_index())

    pred_12 = (pred_df[pred_df["date"].isin(Y12)].groupby("product_name", group_keys=False)["quantity"].sum().rename("pred12").reset_index())
    pred_6  = (pred_df[pred_df["date"].isin(PUB6)].groupby("product_name", group_keys=False)["quantity"].sum().rename("pred6").reset_index())
    pred_p6 = (pred_df[pred_df["date"].isin(PRI6)].groupby("product_name", group_keys=False)["quantity"].sum().rename("pred_priv6").reset_index())

    pp_df = pd.DataFrame.from_dict(product_params, orient="index").reset_index().rename(columns={"index":"product_name"})

    cal = (raw_12.merge(raw_6, on="product_name", how="outer")
               .merge(pred_12, on="product_name", how="outer")
               .merge(pred_6,  on="product_name", how="outer")
               .merge(pred_p6, on="product_name", how="outer")
               .merge(pp_df[["product_name","click_to_sales_rate","calib_mult"]], on="product_name", how="left")
               .fillna(0.0))

    eps = 1e-6
    out = pred_df.copy()

    if mode == "FULL12":
        cal["gamma"] = (cal["click_to_sales_rate"]*cal["raw12"] + eps) / (cal["pred12"] + eps)
        cal["gamma"] *= cal["calib_mult"].replace(0,1.0)
        out = out.merge(cal[["product_name","gamma"]], on="product_name", how="left")
        out["quantity_calib"] = (out["quantity"] * out["gamma"].replace(0,np.nan).fillna(1.0)).clip(lower=0.0)
    elif mode == "PUBLIC6":
        cal["gamma_pub"] = (cal["click_to_sales_rate"]*cal["raw6"] + eps) / (cal["pred6"] + eps)
        cal["gamma_pub"] *= cal["calib_mult"].replace(0,1.0)
        out = out.merge(cal[["product_name","gamma_pub"]], on="product_name", how="left")
        out["quantity_calib"] = (out["quantity"] * out["gamma_pub"].replace(0,np.nan).fillna(1.0)).clip(lower=0.0)
    elif mode == "PIECEWISE":
        cal["gamma_pub"] = (cal["click_to_sales_rate"]*cal["raw6"] + eps) / (cal["pred6"] + eps)
        target12 = cal["click_to_sales_rate"]*cal["raw12"]
        cal["gamma_priv"] = (target12 - cal["gamma_pub"]*cal["pred6"]) / (cal["pred_priv6"] + eps)
        cal["gamma_pub"]  = (cal["gamma_pub"]  * cal["calib_mult"].replace(0,1.0)).clip(lower=0.0)
        cal["gamma_priv"] = (cal["gamma_priv"] * cal["calib_mult"].replace(0,1.0)).clip(lower=0.0)
        out = out.merge(cal[["product_name","gamma_pub","gamma_priv"]], on="product_name", how="left")
        out["gamma_row"] = np.where(out["date"].isin(PUB6), out["gamma_pub"], out["gamma_priv"])
        out["quantity_calib"] = (out["quantity"] * out["gamma_row"].replace(0,np.nan).fillna(1.0)).clip(lower=0.0)
    else:
        raise ValueError(f"Unknown CALIB_MODE: {mode}")

    # Early-month floor
    if CONFIG.USE_EARLY_FLOOR and CONFIG.EARLY_FLOOR_N_MONTHS > 0:
        cal_target12 = (cal[["product_name"]].copy())
        cal_target12["target12"] = cal["click_to_sales_rate"] * cal["raw12"]
        avg_map = (cal_target12.set_index("product_name")["target12"] / 12.0).to_dict()
        out = out.sort_values(["product_name","date"]).copy()
        out["__rn"] = out.groupby("product_name").cumcount() + 1
        def floor_row(r):
            if r["__rn"] <= CONFIG.EARLY_FLOOR_N_MONTHS:
                avg = avg_map.get(r["product_name"], 0.0)
                floor_val = CONFIG.EARLY_FLOOR_FRAC * avg
                return max(r["quantity_calib"], floor_val)
            return r["quantity_calib"]
        out["quantity_calib"] = out.apply(floor_row, axis=1)
        out = out.drop(columns=["__rn"])

    return out, cal

# =====================
# Month-share (POS/CLICK or CUSTOM) + blending
# =====================
def build_pos_month_share(df_pos: pd.DataFrame, category: str, pred_months: pd.DatetimeIndex,
                          lam_uni=0.2, min_mshare=0.005, season_start=7) -> Optional[pd.Series]:
    if df_pos is None or df_pos.empty:
        return None
    cat_norm = _norm_txt(category)
    t = df_pos.copy()
    if "category_std" in t.columns:
        t = t[t["category_std"] == cat_norm]
    elif "category" in t.columns:
        t = t[t["category"].astype(str).apply(_norm_txt) == cat_norm]
    else:
        return None
    if "sales_mm" not in t.columns: return None
    t = t[(t["month"].notna()) & (t["sales_mm"].fillna(0) > 0)]

    month_order = list(range(season_start, 13)) + list(range(1, season_start))
    season_vectors = []
    if "year" in t.columns and t["year"].notna().any():
        years = sorted(t["year"].dropna().unique().astype(int))
        for y in years:
            mask = ((t["year"] == y) & (t["month"].between(season_start, 12))) | \
                   ((t["year"] == y+1) & (t["month"].between(1, season_start-1)))
            w = t[mask]
            if w.empty: continue
            by_m = w.groupby("month")["sales_mm"].sum().astype(float)
            by_m = by_m.reindex(month_order).fillna(0.0)
            if by_m.sum() <= 0: continue
            season_vectors.append((by_m / by_m.sum()).values)
    if not season_vectors:
        by_m_all = t.groupby("month")["sales_mm"].sum().astype(float)
        by_m_all = by_m_all.reindex(month_order).fillna(0.0)
        if by_m_all.sum() <= 0: return None
        season_template = (by_m_all / by_m_all.sum()).values
    else:
        season_template = np.mean(season_vectors, axis=0)

    season_template = (1 - lam_uni) * season_template + lam_uni * (1.0/12)
    season_template = np.clip(season_template, min_mshare, None)
    season_template = season_template / season_template.sum()

    s = pd.Series(index=pred_months, dtype=float)
    for i, d in enumerate(pred_months):
        s.loc[d] = float(season_template[i % 12])
    return s

def build_month_share_by_category(click: pd.DataFrame,
                                  df_total: Optional[pd.DataFrame],
                                  df_seg: Optional[pd.DataFrame]) -> Dict[str, pd.Series]:
    """POS 우선, 없으면 CLICK합으로 카테고리 월비중 생성"""
    pred_months = CONFIG.PRED_MONTHS
    shares: Dict[str, pd.Series] = {}

    if CONFIG.USE_POS_MONTH_SHARE:
        for c in CAT2PRODS.keys():
            s = None
            if df_total is not None:
                s = build_pos_month_share(df_total, c, pred_months,
                                          lam_uni=CONFIG.POS_LAM_UNI, min_mshare=CONFIG.POS_MIN_MSHARE,
                                          season_start=CONFIG.POS_SEASON_START)
            if s is None and df_seg is not None:
                s = build_pos_month_share(df_seg, c, pred_months,
                                          lam_uni=CONFIG.POS_LAM_UNI, min_mshare=CONFIG.POS_MIN_MSHARE,
                                          season_start=CONFIG.POS_SEASON_START)
            if s is not None:
                shares[c] = s

    # Fallback: CLICK 분포
    click_pm = click.groupby(["product_name","date"], as_index=False)["clicks"].mean()
    eps = 1e-9
    for c, prods in CAT2PRODS.items():
        if c in shares: continue
        cm = (click_pm[(click_pm["product_name"].isin(prods)) & (click_pm["date"].isin(pred_months))]
              .groupby("date")["clicks"].sum()).reindex(pred_months, fill_value=0.0) + eps
        shares[c] = (cm / cm.sum()) if cm.sum() > 0 else pd.Series(1.0/len(pred_months), index=pred_months)
    return shares

# ---------- NEW: custom month-share helpers ----------
def _series_from_12(vec12, pred_months: pd.DatetimeIndex) -> pd.Series:
    v = pd.Series(pd.to_numeric(list(vec12), errors="coerce"), dtype=float)
    if len(v) != 12:
        raise ValueError("month-share must have length 12")
    v = v.clip(lower=0)
    ssum = float(v.sum())
    if ssum <= 0: v[:] = 1/12.0
    else:        v = v / ssum
    rep = np.array([v[i % 12] for i in range(len(pred_months))], dtype=float)
    return pd.Series(rep, index=pred_months)

def _load_custom_monthshare_csv(path: str, pred_months: pd.DatetimeIndex) -> dict:
    if not path: return {}
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    out = {}
    # wide: key,m1..m12
    if any(c.startswith("m") for c in cols) and len(cols) >= 13:
        key_col = df.columns[0]; month_cols = df.columns[1:13]
        for _, r in df.iterrows():
            key = str(r[key_col]).strip()
            vec12 = [r[c] for c in month_cols]
            out[key] = _series_from_12(vec12, pred_months)
        return out
    # long: key,ym,share
    need = {"key","ym","share"}
    if need.issubset(set(c.lower() for c in df.columns)):
        df["key"] = df[[c for c in df.columns if c.lower()=="key"][0]].astype(str).str.strip()
        df["ym"]  = pd.to_datetime(df[[c for c in df.columns if c.lower()=="ym"][0]]).dt.to_period("M").dt.to_timestamp()
        df["share"] = pd.to_numeric(df[[c for c in df.columns if c.lower()=="share"][0]], errors="coerce")
        for k, g in df.groupby("key"):
            s = g.set_index("ym")["share"].reindex(pred_months)
            s = s.fillna(s.mean())
            s = s.clip(lower=0)
            den = float(s.sum())
            s = s/den if den>0 else pd.Series(1/len(pred_months), index=pred_months)
            out[str(k).strip()] = s
        return out
    raise ValueError("CUSTOM_MONTH_SHARE CSV 형식 오류: (long: key,ym,share) 또는 (wide: key,m1..m12)")

def build_custom_monthshare_map(pred_months: pd.DatetimeIndex) -> dict:
    if CONFIG.CUSTOM_MONTH_SHARE_PATH:
        return _load_custom_monthshare_csv(CONFIG.CUSTOM_MONTH_SHARE_PATH, pred_months)
    out = {}
    for k, vec12 in CUSTOM_MONTH_SHARE.items():
        out[str(k).strip()] = _series_from_12(vec12, pred_months)
    return out

def _blend_shape(shape_pred: np.ndarray, shape_cat: np.ndarray, mu: float) -> np.ndarray:
    sp = np.asarray(shape_pred, float); sc = np.asarray(shape_cat, float)
    sp = sp / (sp.sum() + 1e-12); sc = sc / (sc.sum() + 1e-12)
    out = (1.0 - mu) * sp + mu * sc
    out = np.clip(out, 0, None)
    s = out.sum()
    return out / s if s > 0 else np.full_like(out, 1.0 / len(out))

def apply_monthshare_blend(
    pred_df: pd.DataFrame,
    month_share_map: dict,                # {제품명 or 카테고리: Series(index=pred_months, sum=1)}
    pred_months: pd.DatetimeIndex,
    mu_by_cat: Dict[str,float] | float,   # 카테고리별 신뢰도 or 스칼라(커스텀은 1.0 권장)
    out_col: str = "quantity_final"
) -> pd.DataFrame:
    if not month_share_map:
        pdf = pred_df.copy()
        pdf[out_col] = pdf.get("quantity_calib", pdf["quantity"])
        return pdf

    if not isinstance(mu_by_cat, dict):
        mu_by_cat = {c: float(mu_by_cat) for c in CAT2PRODS.keys()}

    out_rows = []
    base_col = "quantity_calib" if "quantity_calib" in pred_df.columns else "quantity"
    for prod, g in pred_df.groupby("product_name", group_keys=False):
        cat = PROD2CAT.get(prod, "")
        mu  = float(mu_by_cat.get(cat, 1.0))
        base = (g.set_index("date")[base_col].reindex(pred_months).fillna(0.0))
        total = float(base.sum())
        if total <= 0:
            new = base
        else:
            # 우선순위: 제품 키 → 카테고리 키
            s = month_share_map.get(prod)
            if s is None:
                s = month_share_map.get(cat)
            if s is None:
                new = base
            else:
                s = pd.Series(s).reindex(pred_months)
                s = s.fillna(s.mean()).clip(lower=0)
                den = float(s.sum()); s = s/den if den>0 else pd.Series(1/len(pred_months), index=pred_months)
                if mu >= 1.0 - 1e-9:
                    shape = s.values
                elif mu <= 1e-9:
                    shape = (base / (total+1e-12)).values
                else:
                    shape = _blend_shape((base / (total+1e-12)).values, s.values, mu)
                new = total * pd.Series(shape, index=pred_months)
        gg = g.copy()
        gg[out_col] = gg["date"].map(new).values
        out_rows.append(gg)
    return pd.concat(out_rows, ignore_index=True)

# =====================
# Minimal diagnostics
# =====================
def diag_target_vs_pred(cal_df: pd.DataFrame, pred_df: pd.DataFrame,
                        mode: str = "FULL12", topN: int = 12,
                        save_dir: Path = CONFIG.SAVE_DIR):
    if mode == "FULL12":
        target = (cal_df[["product_name","click_to_sales_rate","raw12"]]
                  .assign(target=lambda x: x["click_to_sales_rate"]*x["raw12"]))
        agg = (pred_df.groupby("product_name", as_index=False)["quantity_calib"].sum()
               .rename(columns={"quantity_calib":"pred_after"}))
    elif mode == "PUBLIC6":
        pub6 = pd.date_range("2024-07-01","2024-12-01", freq="MS")
        target = (cal_df[["product_name","click_to_sales_rate","raw6"]]
                  .assign(target=lambda x: x["click_to_sales_rate"]*x["raw6"]))
        agg = (pred_df[pred_df["date"].isin(pub6)]
               .groupby("product_name", as_index=False)["quantity_calib"].sum()
               .rename(columns={"quantity_calib":"pred_after"}))
    else:
        return diag_target_vs_pred(cal_df, pred_df, mode="FULL12", topN=topN, save_dir=save_dir)

    show = (target.merge(agg, on="product_name", how="left").fillna({"pred_after":0.0}))
    show["gap"] = show["pred_after"] - show["target"]
    show = show.sort_values("target", ascending=False).head(min(topN, len(show)))

    fig, ax = plt.subplots(figsize=(12, max(3, 0.35*len(show))))
    idx = np.arange(len(show)); barw = 0.35
    ax.barh(idx - barw/2, show["target"].values, height=barw, label="target (anchor)")
    ax.barh(idx + barw/2, show["pred_after"].values, height=barw, label="pred (after calib)")
    ax.set_yticks(idx); ax.set_yticklabels(show["product_name"])
    ax.invert_yaxis()
    ax.set_xlabel("units"); ax.set_title(f"Target vs Predicted Sum ({mode})")
    ax.legend()
    out = save_dir / f"diag_target_vs_pred_{mode}.png"
    savefig_safe(fig, out); plt.close(fig)

def diag_gamma_hist(cal_df: pd.DataFrame, save_dir: Path = CONFIG.SAVE_DIR):
    cols = [c for c in ["gamma","gamma_pub","gamma_priv"] if c in cal_df.columns]
    if not cols: return
    for c in cols:
        v = cal_df[c].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(v)==0: continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(v, bins=30)
        ax.set_title(f"Distribution of {c} (n={len(v)})")
        ax.set_xlabel(c); ax.set_ylabel("count")
        ax.axvline(np.median(v), linestyle="--")
        out = CONFIG.SAVE_DIR / f"diag_{c}_hist.png"
        savefig_safe(fig, out); plt.close(fig)

# =====================
# Main
# =====================
def run():
    P("[LOAD] personas, trends, POS(optional)")
    personas = load_personas(CONFIG.PATH_PERSONA)
    search, click, sample = load_trends()
    pos_total, _, pos_seg = load_pos_months()

    # Normalize + segment weighting
    search = normalize_per_product(search, "search_index", "search_norm")
    click_n = normalize_per_product(click,  "clicks",        "click_norm")
    search = to_buckets(search); click_b = to_buckets(click_n)
    seg_w = build_segment_weights(personas)
    sw = weighted_by_persona(search, "search_norm", seg_w)
    cw = weighted_by_persona(click_b, "click_norm", seg_w)

    # Seasonalize to prediction horizon
    if CONFIG.USE_SEASONAL_SC:
        sw = _seasonalize_to_pred(sw[["product_name","date","search_norm_w"]], "search_norm_w",
                                  CONFIG.PRED_MONTHS, season_start=CONFIG.SC_SEASON_START,
                                  lam_uni=CONFIG.SC_LAM_UNI, decay=CONFIG.SC_DECAY).rename(columns={"search_norm_w":"s"})
        cw = _seasonalize_to_pred(cw[["product_name","date","click_norm_w"]], "click_norm_w",
                                  CONFIG.PRED_MONTHS, season_start=CONFIG.SC_SEASON_START,
                                  lam_uni=CONFIG.SC_LAM_UNI, decay=CONFIG.SC_DECAY).rename(columns={"click_norm_w":"c"})
    else:
        sw = sw[sw["date"].isin(CONFIG.PRED_MONTHS)].rename(columns={"search_norm_w":"s"})
        cw = cw[cw["date"].isin(CONFIG.PRED_MONTHS)].rename(columns={"click_norm_w":"c"})

    trend_w = pd.merge(sw, cw, on=["product_name","date"], how="outer").fillna(0.0)

    # Product params
    product_params = build_product_params(personas, click)

    # Smooth trend + momentum
    trend_w_sm = add_ema_and_momentum(trend_w, product_params)

    # Predict
    pred_df = predict(personas, trend_w_sm, product_params, CONFIG.PRED_MONTHS)

    # Calibrate scale (gamma)
    pred_df, cal = calibrate(pred_df, click, product_params, CONFIG.CALIB_MODE)

    # ===== Month-share application =====
    # ① 커스텀 입력 우선
    if CONFIG.USE_CUSTOM_MONTH_SHARE or CONFIG.CUSTOM_MONTH_SHARE_PATH or CUSTOM_MONTH_SHARE:
        CONFIG.USE_CUSTOM_MONTH_SHARE = True
        custom_map = build_custom_monthshare_map(CONFIG.PRED_MONTHS)
        pred_df = apply_monthshare_blend(
            pred_df=pred_df,
            month_share_map=custom_map,                    # 제품/카테고리 키 모두 허용
            pred_months=CONFIG.PRED_MONTHS,
            mu_by_cat=CONFIG.CUSTOM_SHARE_WEIGHT,          # 보통 1.0 (완전치환)
            out_col="quantity_final"
        )
    else:
        # ② POS/CLICK 기반 카테고리 월비중
        month_share = build_month_share_by_category(click, pos_total, pos_seg)
        pred_df = apply_monthshare_blend(
            pred_df=pred_df,
            month_share_map=month_share,                   # 카테고리 키
            pred_months=CONFIG.PRED_MONTHS,
            mu_by_cat=MU_BY_CAT,                           # 카테고리별 신뢰도
            out_col="quantity_final"
        )

    # Build submission (quantity_final 사용)
    products_order = sample["product_name"].tolist()
    pivot_pred = (pred_df.pivot(index="product_name", columns="date", values="quantity_final")
                          .reindex(products_order)
                          .reindex(columns=CONFIG.PRED_MONTHS)
                          .fillna(0.0))
    pivot_pred.columns = [f"months_since_launch_{i+1}" for i in range(len(CONFIG.PRED_MONTHS))]
    pivot_pred.reset_index(inplace=True)

    out_path = CONFIG.SAVE_DIR / "submission_persona_customshare.csv"
    pivot_pred.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Quick sanity
    try:
        tmp = pred_df.copy()
        tmp["price"] = tmp["product_name"].map(PRICE_PER_UNIT).astype(float)
        tmp["rev"] = tmp["quantity_final"].astype(float) * tmp["price"].fillna(1.0)
        rev_by_month = (tmp.groupby("date")["rev"].sum().reindex(CONFIG.PRED_MONTHS).fillna(0.0))
        head6 = float(rev_by_month.iloc[:6].sum()); tail6 = float(rev_by_month.iloc[6:].sum())
        P(f"[DONE] submission saved → {out_path}")
        P(f"        head6_rev={head6:,.0f} | tail6_rev={tail6:,.0f} | tail6_share={(tail6/(head6+tail6+1e-9)):.3f}")
    except Exception:
        P(f"[DONE] submission saved → {out_path}")

    # Minimal diagnostics
    try:
        diag_target_vs_pred(cal, pred_df, mode=CONFIG.CALIB_MODE, topN=12)
        diag_gamma_hist(cal)
    except Exception as e:
        P(f"[VIZ] diagnostics skipped: {e}")

if __name__ == "__main__":
    run()
