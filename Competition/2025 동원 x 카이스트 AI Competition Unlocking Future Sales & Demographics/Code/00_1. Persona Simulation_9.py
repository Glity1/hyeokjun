# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# 0) 경로/파라미터
# =========================================================
DATA_DIR = Path("./_data/dacon/dongwon")
SAVE_DIR = Path("./_save/persona_sim")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PATH_PERSONA = DATA_DIR / "personas.json"
PATH_SEARCH  = DATA_DIR / "naver/search_trend_all.csv"
PATH_CLICK   = DATA_DIR / "naver/click_trend_all.csv"
PATH_SAMPLE  = DATA_DIR / "sample_submission.csv"

# 예측/제출 구간(대회 평가): 2024-07 ~ 2025-06
PRED_MONTHS = pd.date_range("2024-07-01", "2025-06-01", freq="MS")

# 글로벌 기본값(제품별 파라미터의 fallback)
EMA_SPAN_DEFAULT          = 4       # 기본 스무딩 길이
MOMENTUM_COEF_DEFAULT     = 0.40    # 기본 모멘텀 강도
CLICK_TO_SALES_RATE_DEF   = 5    # 기본 클릭→판매 전환율
CALIB_MULT_DEFAULT        = 210.00    # 기본 미세 보정배수

# 캘리브레이션 모드: "FULL12"(권장) | "PUBLIC6" | "PIECEWISE"
CALIB_MODE = "FULL12"

# =========================================================
# 1) 데이터 로드 & 기본 체크
# =========================================================
with open(PATH_PERSONA, "r", encoding="utf-8") as f:
    personas = json.load(f)

search = pd.read_csv(PATH_SEARCH, parse_dates=["date"])
click  = pd.read_csv(PATH_CLICK,  parse_dates=["date"])
sample = pd.read_csv(PATH_SAMPLE)

need_cols = ["product_name","date","gender","age"]
for col in need_cols:
    if col not in search.columns:
        raise ValueError(f"[search] 컬럼 누락: {col}")
    if col not in click.columns:
        raise ValueError(f"[click]  컬럼 누락: {col}")
if "search_index" not in search.columns:
    raise ValueError("search_trend_all.csv에 'search_index' 필요")
if "clicks" not in click.columns:
    raise ValueError("click_trend_all.csv에 'clicks' 필요")

# =========================================================
# 2) 유틸(파서/맵핑)
# =========================================================
def safe_float(x, default=0.0):
    try: return float(x)
    except: return default

def age_to_bucket_korean(a):
    s = str(a)
    nums = re.findall(r"\d+", s)
    if nums:
        v = int(nums[0]); return f"{(v//10)*10}s"
    if "10" in s: return "10s"
    if "20" in s: return "20s"
    if "30" in s: return "30s"
    if "40" in s: return "40s"
    if "50" in s: return "50s"
    if "60" in s: return "60s"
    return "ETC"

def gender_key_korean(g):
    s = str(g)
    return "F" if ("여" in s or str(s).upper().startswith("F")) else "M"

def label_level(val):
    s = str(val)
    if "매우" in s: return 1.0
    if "높"  in s: return 0.8
    if "중"  in s: return 0.6
    if "낮"  in s: return 0.4
    return 0.5

def apply_weighted(mult, w):
    w = safe_float(w, 0.0)
    return 1.0 + w*(mult - 1.0)

def family_mult(v):
    s = str(v)
    if "대가족" in s or "5" in s or "6" in s: return 1.30
    if "4"  in s: return 1.20
    if "3"  in s: return 1.10
    if "1인" in s: return 0.85
    return 1.00

def loyalty_mult(v):
    s = str(v)
    if "높" in s: return 1.15
    if "중" in s: return 1.05
    return 1.00

def income_mult(v):
    s = str(v)
    nums = list(map(int, re.findall(r"\d+", s)))
    mean_income = np.mean(nums) if nums else 300
    if mean_income >= 500: return 1.06
    if mean_income >= 350: return 1.03
    if mean_income >= 250: return 1.00
    return 0.97

def health_mult(product_name, v):
    s = str(v); base = 1.00
    if "높" in s: base = 1.08
    if "유당" in s and "소화가 잘되는 우유" in str(product_name): base = 1.15
    return base

def channel_bias(p):
    ch = str(p.get("구매 채널",{}).get("value",""))
    if any(k in ch for k in ["온라인","새벽","프리미엄","백화점"]): return 1.20
    if any(k in ch for k in ["편의점","오피스","도매","창고형"]):   return 0.85
    return 1.00

def region_bias(v):
    s = str(v)
    if any(k in s for k in ["서울","성남","분당","용인","일산","수원","부산","대구","인천","대전"]): return 1.10
    return 1.00

def job_bias(v):
    s = str(v)
    if any(k in s for k in ["개발","디자","마케","회사원","사무","PM","팀장","임원"]): return 1.10
    if any(k in s for k in ["전업주부","외식","자영업","교사","생산직"]):               return 0.95
    return 1.00

def lifestyle_month_boost(month_int, lifestyle_text):
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

def promo_mult(v):
    lv = label_level(v)
    return 0.95 + 0.20*lv  # 0.95~1.15

def trend_response_to_kappa(tr_val):
    lv = label_level(tr_val)
    return 0.5 + 1.5*lv     # 0.5~2.0

# =========================================================
# 3) 트렌드 정규화(패턴) + (성별×연령) 페르소나 가중 평균
# =========================================================
def normalize_per_product(df, val_col, out_name):
    df = df.copy()
    df["__min"] = df.groupby("product_name", group_keys=False)[val_col].transform("min")
    df["__max"] = df.groupby("product_name", group_keys=False)[val_col].transform("max")
    df[out_name] = (df[val_col] - df["__min"]) / (df["__max"] - df["__min"] + 1e-6)
    return df.drop(columns=["__min","__max"])

search = normalize_per_product(search, "search_index", "search_norm")
click  = normalize_per_product(click,  "clicks",       "click_norm")

def build_segment_weights(personas_dict):
    seg_weights = {}
    for product, plist in personas_dict.items():
        seg = {}
        for p in plist:
            g  = gender_key_korean(p["성별"]["value"])
            ab = age_to_bucket_korean(p["연령"]["value"])
            w = (safe_float(p["성별"]["weight"],0)+safe_float(p["연령"]["weight"],0))/2.0
            w *= (safe_float(p["purchase_probability"],0)/100.0)
            seg[(g,ab)] = seg.get((g,ab), 0.0) + w
        ssum = sum(seg.values()) or 1.0
        seg = {k: v/ssum for k,v in seg.items()}
        seg_weights[product] = seg
    return seg_weights

seg_w = build_segment_weights(personas)

def to_bucket_in_trend(df):
    df = df.copy()
    df["age_bucket"] = df["age"].apply(age_to_bucket_korean)
    df["gender_key"] = df["gender"].apply(gender_key_korean)
    return df

search = to_bucket_in_trend(search)
click  = to_bucket_in_trend(click)

def weighted_by_persona(df, val_col_norm):
    """페르소나 분포(seg_w)로 (product_name, date) 가중 평균 계산"""
    out = []
    for prod, g in df.groupby("product_name", group_keys=False):
        wmap = seg_w.get(prod, {})
        if g.empty:
            continue
        gg = g.copy()
        gg["w"] = gg.apply(lambda r: wmap.get((r["gender_key"], r["age_bucket"]), 0.0), axis=1)
        s = gg.groupby(["product_name","date"], group_keys=False).apply(
            lambda x: (x[val_col_norm]*x["w"]).sum()/(x["w"].sum()+1e-9) if x["w"].sum()>0 else x[val_col_norm].mean()
        )
        agg = s.to_frame(name=f"{val_col_norm}_w").reset_index()
        out.append(agg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["product_name","date",f"{val_col_norm}_w"])

# 정규화된 search/click → 페르소나 가중
search_w = weighted_by_persona(search, "search_norm")
click_w  = weighted_by_persona(click,  "click_norm")

# 예측 구간만 추출 & 병합(정규화 패턴용)
sw = search_w[search_w["date"].isin(PRED_MONTHS)].rename(columns={"search_norm_w":"s"})
cw = click_w [click_w ["date"].isin(PRED_MONTHS)].rename(columns={"click_norm_w":"c"})
trend_w = pd.merge(sw, cw, on=["product_name","date"], how="outer").fillna(0.0)

# =========================================================
# 3-1) 제품별 파라미터 생성 (페르소나 + 클릭 변동성 기반)
# =========================================================
def channel_factor_from_text(txt):
    s = str(txt)
    if any(k in s for k in ["온라인","프리미엄","백화점","새벽"]): return 1.12
    if any(k in s for k in ["편의점","오피스"]): return 0.95
    if any(k in s for k in ["창고형","도매"]):   return 0.98
    return 1.00

def income_factor_from_text(txt):
    nums = list(map(int, re.findall(r"\d+", str(txt))))
    mean_income = np.mean(nums) if nums else 300
    if mean_income >= 600: return 1.06
    if mean_income >= 450: return 1.03
    return 1.00

def loyalty_factor_from_text(txt):
    s = str(txt)
    if "높" in s: return 1.12
    if "중" in s: return 1.05
    return 1.00

def family_factor_from_text(txt):
    s = str(txt)
    if "대가족" in s or "5" in s or "6" in s: return 1.20
    if "4" in s: return 1.12
    if "3" in s: return 1.08
    if "1인" in s: return 0.95
    return 1.00

def health_fit_factor(product_name, health_text):
    s = str(health_text)
    if "유당" in s and "소화가 잘되는 우유" in str(product_name): return 1.12
    if "높" in s: return 1.03
    return 1.00

def build_product_params(personas_dict, click_df_with_norm):
    params = {}
    # click_df_with_norm: product_name, date, click_norm 포함
    for product, plist in personas_dict.items():
        if not plist:
            params[product] = dict(
                ema_span=EMA_SPAN_DEFAULT, mom_coef=MOMENTUM_COEF_DEFAULT,
                click_to_sales_rate=CLICK_TO_SALES_RATE_DEF, calib_mult=CALIB_MULT_DEFAULT
            )
            continue

        # --- 페르소나 요약 ---
        purchase_probs = [safe_float(p.get("purchase_probability",0),0) for p in plist]
        prob_mean = np.mean(purchase_probs) if purchase_probs else 60.0

        promo_levels = [label_level(p.get("프로모션 민감도",{}).get("value","중간")) for p in plist]
        promo_avg = np.mean(promo_levels) if promo_levels else 0.6

        kappa_vals = [trend_response_to_kappa(p.get("트렌드 반응도",{}).get("value","중간")) for p in plist]
        kappa = np.mean(kappa_vals) if kappa_vals else 1.0

        loyalty_fs = [loyalty_factor_from_text(p.get("브랜드 충성도",{}).get("value","중간")) for p in plist]
        loyalty_factor = np.mean(loyalty_fs) if loyalty_fs else 1.0

        ch_fs = [channel_factor_from_text(p.get("구매 채널",{}).get("value","")) for p in plist]
        channel_factor = np.mean(ch_fs) if ch_fs else 1.0

        fam_fs = [family_factor_from_text(p.get("가족 구성",{}).get("value","")) for p in plist]
        family_factor = np.mean(fam_fs) if fam_fs else 1.0

        inc_fs = [income_factor_from_text(p.get("소득 구간",{}).get("value","")) for p in plist]
        income_factor = np.mean(inc_fs) if inc_fs else 1.0

        health_fs = [health_fit_factor(product, p.get("건강 관심도",{}).get("value","")) for p in plist]
        health_factor = np.mean(health_fs) if health_fs else 1.0

        # --- 클릭 변동성(제품별) ---
        g = click_df_with_norm[click_df_with_norm["product_name"]==product].sort_values("date")
        if len(g) >= 3:
            volatility = g["click_norm"].diff().abs().rolling(2).mean().mean()
        else:
            volatility = 0.1

        # --- 파라미터 산출 ---
        ema_span = int(np.clip(round(6 + 4*volatility - 2*(kappa-1)), 3, 12))
        mom_coef = float(np.clip(0.15 + 0.35 * promo_avg * kappa * (1.0/loyalty_factor), 0.15, 0.6))

        base_rate = CLICK_TO_SALES_RATE_DEF
        click_to_sales_rate = float(
            base_rate * (prob_mean/60.0)**1.2 * loyalty_factor * channel_factor *
            family_factor * income_factor * health_factor
        )

        params[product] = dict(
            ema_span=ema_span,
            mom_coef=mom_coef,
            click_to_sales_rate=click_to_sales_rate,
            calib_mult=CALIB_MULT_DEFAULT  # 필요시 제품별 미세보정
        )
    return params

product_params = build_product_params(personas, click)

# =========================================================
# 3-2) EMA 스무딩 + 모멘텀 (제품별 파라미터 적용)
# =========================================================
def add_ema_and_momentum_per_product(df, params):
    rows = []
    for prod, g in df.groupby("product_name", group_keys=False):
        span = params.get(prod, {}).get("ema_span", EMA_SPAN_DEFAULT)
        gg = g.sort_values("date").copy()
        gg["s_sm"] = gg["s"].ewm(span=span, adjust=False).mean()
        gg["c_sm"] = gg["c"].ewm(span=span, adjust=False).mean()
        gg["s_mom"] = gg["s_sm"].diff().fillna(0.0)
        gg["c_mom"] = gg["c_sm"].diff().fillna(0.0)
        rows.append(gg)
    return pd.concat(rows, ignore_index=True) if rows else df

trend_w = add_ema_and_momentum_per_product(trend_w, product_params)

# =========================================================
# 4) 제품별 α·β·κ, 프로모션 감도(페르소나에서 추정)
# =========================================================
product_alpha_beta = {}
product_kappa = {}
product_promo_avg = {}
for product, plist in personas.items():
    if not plist:
        product_alpha_beta[product] = (0.5, 0.5)
        product_kappa[product] = 1.0
        product_promo_avg[product] = 0.6
        continue
    beta_boosts, kappas, promos = [], [], []
    for p in plist:
        b_bias = channel_bias(p) * region_bias(p.get("거주 지역",{}).get("value","")) * \
                 job_bias(p.get("직업",{}).get("value",""))
        beta_boosts.append(apply_weighted(b_bias, 0.5))
        kappas.append(trend_response_to_kappa(p.get("트렌드 반응도",{}).get("value","중간")))
        promos.append(label_level(p.get("프로모션 민감도",{}).get("value","중간")))
    beta_boost = np.mean(beta_boosts) if beta_boosts else 1.0
    alpha, beta = 0.5, 0.5*beta_boost
    ssum = alpha + beta
    alpha, beta = alpha/ssum, beta/ssum
    product_alpha_beta[product] = (float(alpha), float(beta))
    product_kappa[product] = float(np.mean(kappas)) if kappas else 1.0
    product_promo_avg[product] = float(np.mean(promos)) if promos else 0.6

# =========================================================
# 5) 페르소나 ‘모든 필드’ 반영 기본 수요 + 트렌드/모멘텀 부스트
# =========================================================
def persona_base_for_month(p, product_name, month_idx, month_int):
    base = (safe_float(p.get("purchase_probability",0),0)/100.0) * p["monthly_by_launch"][month_idx % 12]
    fam  = apply_weighted(family_mult(p["가족 구성"]["value"]),                 p["가족 구성"]["weight"])
    loy  = apply_weighted(loyalty_mult(p["브랜드 충성도"]["value"]),           p["브랜드 충성도"]["weight"])
    inc  = apply_weighted(income_mult(p["소득 구간"]["value"]),                p["소득 구간"]["weight"])
    hea  = apply_weighted(health_mult(product_name, p["건강 관심도"]["value"]), p["건강 관심도"]["weight"])
    pro  = apply_weighted(promo_mult(p["프로모션 민감도"]["value"]),           p["프로모션 민감도"]["weight"])
    life = apply_weighted(lifestyle_month_boost(month_int, p["라이프스타일"]["value"]), p["라이프스타일"]["weight"])
    agew = apply_weighted(1.02, p["연령"]["weight"])
    genw = apply_weighted(1.02, p["성별"]["weight"])
    jobw = apply_weighted(1.01, p["직업"]["weight"])
    regw = apply_weighted(1.01, p["거주 지역"]["weight"])
    mult = fam*loy*inc*hea*pro*life*agew*genw*jobw*regw
    return base * mult

rows = []
for product, plist in personas.items():
    a, b = product_alpha_beta.get(product,(0.5,0.5))
    kappa = product_kappa.get(product, 1.0)
    promo_avg = product_promo_avg.get(product, 0.6)
    mom_coef_prod = product_params.get(product, {}).get("mom_coef", MOMENTUM_COEF_DEFAULT)
    for i, d in enumerate(PRED_MONTHS):
        m_int = d.month
        base_sum = sum(persona_base_for_month(p, product, i, m_int) for p in plist)

        tr = trend_w[(trend_w["product_name"]==product) & (trend_w["date"]==d)]
        if tr.empty:
            s_sm=c_sm=s_mom=c_mom=0.0
        else:
            s_sm = float(tr["s_sm"].iloc[0]); c_sm = float(tr["c_sm"].iloc[0])
            s_mom= float(tr["s_mom"].iloc[0]); c_mom= float(tr["c_mom"].iloc[0])

        # 트렌드(정규화) 부스트 + 모멘텀(제품별)
        boost_trend = 1.0 + a*np.log1p(kappa*s_sm) + b*np.log1p(kappa*c_sm)
        mom = max(0.0, a*s_mom + b*c_mom)
        boost_mom = 1.0 + mom_coef_prod * promo_avg * mom

        final_qty = base_sum * boost_trend * boost_mom
        rows.append({"product_name": product, "date": d, "raw": base_sum, "quantity": max(0.0, final_qty)})

pred_df = pd.DataFrame(rows)

# =========================================================
# 6) 스케일 캘리브레이션 (원시 클릭 기반, 제품별 전환율/보정배수 적용)
# =========================================================
click_pm_raw = click.groupby(["product_name","date"], as_index=False, group_keys=False)["clicks"].mean()

Y12   = pd.date_range("2024-07-01","2025-06-01", freq="MS")
PUB6  = pd.date_range("2024-07-01","2024-12-01", freq="MS")
PRIV6 = pd.date_range("2025-01-01","2025-06-01", freq="MS")

raw_12 = (click_pm_raw[click_pm_raw["date"].isin(Y12)]
          .groupby("product_name", group_keys=False)["clicks"].sum()
          .rename("raw12").reset_index())
raw_6  = (click_pm_raw[click_pm_raw["date"].isin(PUB6)]
          .groupby("product_name", group_keys=False)["clicks"].sum()
          .rename("raw6").reset_index())

pred_12 = (pred_df[pred_df["date"].isin(Y12)]
           .groupby("product_name", group_keys=False)["quantity"].sum()
           .rename("pred12").reset_index())
pred_6  = (pred_df[pred_df["date"].isin(PUB6)]
           .groupby("product_name", group_keys=False)["quantity"].sum()
           .rename("pred6").reset_index())
pred_priv6 = (pred_df[pred_df["date"].isin(PRIV6)]
              .groupby("product_name", group_keys=False)["quantity"].sum()
              .rename("pred_priv6").reset_index())

# 제품별 파라미터 DF로 변환
pp_df = (pd.DataFrame.from_dict(product_params, orient="index")
         .reset_index().rename(columns={"index":"product_name"}))

cal = (raw_12.merge(raw_6, on="product_name", how="outer")
             .merge(pred_12, on="product_name", how="outer")
             .merge(pred_6, on="product_name", how="outer")
             .merge(pred_priv6, on="product_name", how="outer")
             .merge(pp_df[["product_name","click_to_sales_rate","calib_mult"]], on="product_name", how="left")
             .fillna(0.0))

eps = 1e-6

if CALIB_MODE == "FULL12":
    # 12개월 합 앵커(원시 클릭 12m × 제품별 전환율)
    cal["gamma"] = (cal["click_to_sales_rate"]*cal["raw12"] + eps) / (cal["pred12"] + eps)
    cal["gamma"] *= cal["calib_mult"].replace(0,1.0)
    pred_df = pred_df.merge(cal[["product_name","gamma"]], on="product_name", how="left")
    pred_df["quantity_calib"] = (pred_df["quantity"] * pred_df["gamma"].replace(0,np.nan).fillna(1.0)).clip(lower=0.0)

elif CALIB_MODE == "PUBLIC6":
    cal["gamma_pub"] = (cal["click_to_sales_rate"]*cal["raw6"] + eps) / (cal["pred6"] + eps)
    cal["gamma_pub"] *= cal["calib_mult"].replace(0,1.0)
    pred_df = pred_df.merge(cal[["product_name","gamma_pub"]], on="product_name", how="left")
    pred_df["quantity_calib"] = (pred_df["quantity"] * pred_df["gamma_pub"].replace(0,np.nan).fillna(1.0)).clip(lower=0.0)

elif CALIB_MODE == "PIECEWISE":
    cal["gamma_pub"] = (cal["click_to_sales_rate"]*cal["raw6"] + eps) / (cal["pred6"] + eps)
    target12 = cal["click_to_sales_rate"]*cal["raw12"]
    cal["gamma_priv"] = (target12 - cal["gamma_pub"]*cal["pred6"]) / (cal["pred_priv6"] + eps)
    cal["gamma_pub"]  = (cal["gamma_pub"]  * cal["calib_mult"].replace(0,1.0)).clip(lower=0.0)
    cal["gamma_priv"] = (cal["gamma_priv"] * cal["calib_mult"].replace(0,1.0)).clip(lower=0.0)
    pred_df = pred_df.merge(cal[["product_name","gamma_pub","gamma_priv"]], on="product_name", how="left")
    pred_df["gamma_row"] = np.where(pred_df["date"].isin(PUB6), pred_df["gamma_pub"], pred_df["gamma_priv"])
    pred_df["quantity_calib"] = (pred_df["quantity"] * pred_df["gamma_row"].replace(0,np.nan).fillna(1.0)).clip(lower=0.0)

else:
    raise ValueError(f"Unknown CALIB_MODE: {CALIB_MODE}")

# (선택) 점검 로그
chk_pub = (pred_df[pred_df["date"].isin(PUB6)]
           .groupby("product_name", as_index=False)["quantity_calib"].sum()
           .rename(columns={"quantity_calib":"pred_pub6_after"}))
chk_full = (pred_df[pred_df["date"].isin(Y12)]
            .groupby("product_name", as_index=False)["quantity_calib"].sum()
            .rename(columns={"quantity_calib":"pred_12m_after"}))
dbg = cal.merge(chk_pub, on="product_name", how="left").merge(chk_full, on="product_name", how="left")
print("\n[Calibration check head]")
print(dbg.head(5).to_string(index=False))

# =========================================================
# 7) 제출 포맷으로 변환 (sample_submission과 동일)
# =========================================================
products_order = sample["product_name"].tolist()
pivot_pred = pred_df.pivot(index="product_name", columns="date", values="quantity_calib").reindex(products_order)
pivot_pred = pivot_pred.reindex(columns=PRED_MONTHS).fillna(0.0)
pivot_pred.columns = [f"months_since_launch_{i+1}" for i in range(len(PRED_MONTHS))]
pivot_pred.reset_index(inplace=True)

out_path = SAVE_DIR / "submission_persona_trend_check.csv"
pivot_pred.to_csv(out_path, index=False, encoding="utf-8-sig")

print("="*80)
print("예측/제출 파일 생성 완료")
print("="*80)
print(f"- 제품 수: {pivot_pred['product_name'].nunique()}")
print(f"- 기간: {PRED_MONTHS.min().date()} ~ {PRED_MONTHS.max().date()} (12개월)")
print(f"- 저장: {out_path}")
print("- 샘플 미리보기:")
print(pivot_pred.head(3).to_string(index=False))

# (참고) 제품별 파라미터 일부 확인
pp_show = pd.DataFrame.from_dict(product_params, orient="index").reset_index().rename(columns={"index":"product_name"})
print("\n[product_params preview]")
print(pp_show.head(8).to_string(index=False))
