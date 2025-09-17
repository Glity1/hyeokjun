# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm

# ✅ 윈도우: 맑은 고딕 / 맥: AppleGothic / 리눅스: NanumGothic
plt.rc("font", family="Malgun Gothic")   # 윈도우 기준
mpl.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

DEBUG = True

def P(msg=""):
    if DEBUG:
        print(msg)

def show_df_info(name, df, n=5):
    if not DEBUG: 
        return
    print(f"\n[DF] {name}: shape={getattr(df, 'shape', None)}")
    try:
        print(df.head(n).to_string(index=False))
    except Exception:
        print(type(df), df)

def show_series_stats(name, s):
    if not DEBUG: return
    s = s.astype(float)
    print(f"[SERIES] {name}: sum={s.sum():.6f}, min={s.min():.6f}, max={s.max():.6f}, len={len(s)}")


# =========================================================
# 0) 경로/파라미터
# =========================================================
DATA_DIR = Path("./_data/dacon/dongwon")
SAVE_DIR = Path("./_save/persona_sim")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# personas2.json 우선 사용, 없으면 personas.json fallback
PATH_PERSONA2 = DATA_DIR / "personas2.json"
PATH_PERSONA1 = DATA_DIR / "personas.json"
PATH_PERSONA  = PATH_PERSONA2 if PATH_PERSONA2.exists() else PATH_PERSONA1

PATH_SEARCH  = DATA_DIR / "naver/search_trend_all.csv"
PATH_CLICK   = DATA_DIR / "naver/click_trend_all.csv"
PATH_SAMPLE  = DATA_DIR / "sample_submission.csv"

# 예측/제출 구간(대회 평가): 2024-07 ~ 2025-06
PRED_MONTHS = pd.date_range("2024-07-01", "2025-06-01", freq="MS")

# 글로벌 기본값(제품별 파라미터의 fallback)
EMA_SPAN_DEFAULT          = 6       # 기본 스무딩 길이
MOMENTUM_COEF_DEFAULT     = 0.30    # 기본 모멘텀 강도
CLICK_TO_SALES_RATE_DEF   = 0.05    # 기본 클릭→판매 전환율
CALIB_MULT_DEFAULT        = 1.00    # 기본 미세 보정배수

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

P("\n=== [STEP1] LOAD CHECK ===")
P(f"PERSONA FILE: {PATH_PERSONA}")
show_df_info("search(raw)", search)
show_df_info("click(raw)", click)
show_df_info("sample", sample)

# date 범위/중복/결측 기본 점검
P(f"- search date range: {search['date'].min()} ~ {search['date'].max()}")
P(f"- click  date range: {click['date'].min()} ~ {click['date'].max()}")
P(f"- PRED_MONTHS: {PRED_MONTHS.min()} ~ {PRED_MONTHS.max()}  (len={len(PRED_MONTHS)})")


# =========================================================
# 2) 유틸(파서/맵핑) : personas2/1 모두 커버
# =========================================================
def safe_float(x, default=0.0):
    try: return float(x)
    except: return default

def get_vw(d, key_candidates, default_value="", default_weight=0.6):
    """
    페르소나 항목 접근기:
      - key_candidates: ["성별", "gender"] 같이 후보 키 리스트
      - 반환: (value, weight) 튜플
      - 값이 없는 경우 (default_value, default_weight)
    """
    for k in key_candidates:
        if k in d:
            obj = d[k]
            if isinstance(obj, dict):
                v = obj.get("value", obj.get("val", default_value))
                w = safe_float(obj.get("weight", default_weight), default_weight)
                return v, w
            else:
                # 단일 값만 있는 경우
                return obj, default_weight
    return default_value, default_weight

def get_scalar(d, key_candidates, default=None):
    """단일 스칼라(리스트/문자/숫자) 필드 가져오기"""
    for k in key_candidates:
        if k in d: return d[k]
    return default

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

def channel_bias_from_text(txt):
    s = str(txt)
    if any(k in s for k in ["온라인","프리미엄","백화점","새벽"]): return 1.12
    if any(k in s for k in ["편의점","오피스"]): return 0.95
    if any(k in s for k in ["창고형","도매"]):   return 0.98
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
    lv = label_level(v); return 0.95 + 0.20*lv  # 0.95~1.15

def trend_response_to_kappa(tr_val):
    lv = label_level(tr_val); return 0.5 + 1.5*lv  # 0.5~2.0

# ---- personas2 전용(있으면 사용) : 이벤트/채널/용량/재구매/가격 ----
def event_month_boost(month_int, drivers):
    """
    drivers 예: ["설", "추석", "신학기", "가정의달", "김장", "여름 냉음", "연말"] 등
    문자열 한 덩어리여도 대응.
    """
    mult = 1.00
    if drivers is None: return mult
    if isinstance(drivers, str):
        ds = re.split(r"[,\s/·]+", drivers)
    else:
        ds = list(drivers)

    m = month_int
    norm = [str(d) for d in ds]

    def has(*keys): return any(any(k in s for k in keys) for s in norm)

    # 설/추석
    if has("설"):     # 설=1~2월
        if m in [1,2]: mult *= 1.12
    if has("추석"):   # 추석=9~10월
        if m in [9,10]: mult *= 1.12
    # 신학기, 가정의 달, 김장, 여름 냉음, 연말
    if has("신학기"):   # 3, 9월
        if m in [3,9]: mult *= 1.06
    if has("가정의달","어버이","스승"):  # 5월
        if m == 5: mult *= 1.08
    if has("김장"):
        if m == 11: mult *= 1.18
        if m in [12,1]: mult *= 1.10
    if has("여름","냉음","아이스"):
        if m in [6,7,8]: mult *= 1.06
    if has("연말","크리스마스"):
        if m == 12: mult *= 1.06

    return mult

def channel_mix_factor(primary, secondary):
    p = 1.00
    if primary:
        p *= channel_bias_from_text(primary)
    if secondary:
        # 보조 채널은 영향 30%만 반영
        p *= (1.0 + 0.3*(channel_bias_from_text(secondary)-1.0))
    return p

def package_fit_factor(product_name, fit_text):
    """
    '가구-용량 적합도' 같은 필드가 있으면,
    - 소용량 선호 → 90g/500g/250mL 등에 +,
    - 대용량 선호 → 135g/900g/340g 등에 +
    """
    s = str(fit_text)
    name = str(product_name)
    is_small = any(k in name for k in ["90g","500g","250mL","200g","400g"])
    is_large = any(k in name for k in ["135g","900g","340g"])

    if "소용량" in s or "1-2인" in s or "1~2인" in s:
        return 1.08 if is_small else 0.97
    if "대용량" in s or "3인" in s or "4인" in s or "가족" in s:
        return 1.08 if is_large else 0.97
    return 1.00

def repurchase_factor(rep_text):
    """
    '재구매 주기'를 월 빈도로 대략 환산하여 소폭 가산.
    예: '주 2-3회' → 월 10~12회 → +5~8% 수준
    """
    s = str(rep_text)
    # 주 N회
    m = re.search(r"주\s*([0-9]+)", s)
    if m:
        per_week = int(m.group(1))
        per_month = per_week * 4
    else:
        m2 = re.search(r"월\s*([0-9]+)", s)
        per_month = int(m2.group(1)) if m2 else 0

    if per_month >= 12: return 1.10
    if per_month >= 8:  return 1.08
    if per_month >= 4:  return 1.05
    if per_month >= 2:  return 1.03
    return 1.00

def price_sensitivity_factor(price_text):
    s = str(price_text)
    if any(k in s for k in ["가성비","저가"]): return 1.02
    if any(k in s for k in ["프리미엄","고가","고품질"]): return 1.03
    return 1.00

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

# (성별×연령) 세그먼트 가중치
def build_segment_weights(personas_dict):
    seg_weights = {}
    for product, plist in personas_dict.items():
        seg = {}
        for p in plist:
            # 성별/연령 value, weight 추출 (양 스키마 모두 커버)
            gender_v, gender_w = get_vw(p, ["성별","gender"])
            age_v, age_w       = get_vw(p, ["연령","age"])

            g  = gender_key_korean(gender_v)
            ab = age_to_bucket_korean(age_v)

            # 구매확률(%)도 포함
            purchase_prob = safe_float(p.get("purchase_probability", p.get("purchase_prob", 60)), 60)/100.0
            w = (safe_float(gender_w,0)+safe_float(age_w,0))/2.0
            w *= purchase_prob

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

P("\n=== [STEP2] NORMALIZE & PERSONA WEIGHT ===")
show_df_info("search(normalized)", search.head(10))
show_df_info("click(normalized)",  click.head(10))
show_df_info("search_w(persona)", search_w.head(10))
show_df_info("click_w(persona)",  click_w.head(10))
show_df_info("trend_w(s,c merged)", trend_w.head(12))

# 예측구간 커버리지
cov = (trend_w[trend_w["date"].isin(PRED_MONTHS)]
       .groupby("product_name")["date"].nunique()
       .sort_values(ascending=True))
P(f"- coverage of months in trend_w (min={cov.min()}, max={cov.max()}, expected={len(PRED_MONTHS)})")
P(cov.head(10))


# =========================================================
# 3-1) 제품별 파라미터 생성 (페르소나 + 클릭 변동성 기반, personas2 필드 반영)
# =========================================================
def build_product_params(personas_dict, click_df_with_norm):
    params = {}
    for product, plist in personas_dict.items():
        if not plist:
            params[product] = dict(
                ema_span=EMA_SPAN_DEFAULT, mom_coef=MOMENTUM_COEF_DEFAULT,
                click_to_sales_rate=CLICK_TO_SALES_RATE_DEF, calib_mult=CALIB_MULT_DEFAULT
            )
            continue

        # --- 페르소나 요약 ---
        purchase_probs = [safe_float(p.get("purchase_probability", p.get("purchase_prob", 60)), 60) for p in plist]
        prob_mean = np.mean(purchase_probs) if purchase_probs else 60.0

        promo_levels = [label_level(get_vw(p, ["프로모션 민감도","promotion_sensitivity"])[0]) for p in plist]
        promo_avg = np.mean(promo_levels) if promo_levels else 0.6

        promo_elastic_vals = [label_level(get_vw(p, ["프로모션 탄력","promotion_elasticity"])[0]) for p in plist]
        promo_elastic_avg = np.mean(promo_elastic_vals) if promo_elastic_vals else promo_avg

        kappa_vals = [trend_response_to_kappa(get_vw(p, ["트렌드 반응도","trend_responsiveness"])[0]) for p in plist]
        kappa = np.mean(kappa_vals) if kappa_vals else 1.0

        loyalty_fs = [loyalty_mult(get_vw(p, ["브랜드 충성도","loyalty"])[0]) for p in plist]
        loyalty_factor = np.mean(loyalty_fs) if loyalty_fs else 1.0

        # ✅ 주요/보조 채널은 페르소나별로 factor 계산 후 평균
        ch_fs = []
        for p in plist:
            primary_ch   = get_vw(p, ["주요 채널","primary_channel","구매 채널"])[0]
            secondary_ch = get_vw(p, ["보조 채널","secondary_channel"])[0]
            ch_fs.append(channel_mix_factor(primary_ch, secondary_ch))
        channel_factor = np.mean(ch_fs) if ch_fs else 1.0

        fam_fs = [family_mult(get_vw(p, ["가족 구성","family"])[0]) for p in plist]
        family_factor = np.mean(fam_fs) if fam_fs else 1.0

        inc_fs = [income_mult(get_vw(p, ["소득 구간","income"])[0]) for p in plist]
        income_factor = np.mean(inc_fs) if inc_fs else 1.0

        health_fs = [health_mult(product, get_vw(p, ["건강 관심도","health"])[0]) for p in plist]
        health_factor = np.mean(health_fs) if health_fs else 1.0

        rep_fs = [repurchase_factor(get_vw(p, ["재구매 주기","repurchase_cycle"])[0]) for p in plist]
        rep_factor = np.mean(rep_fs) if rep_fs else 1.0

        price_fs = [price_sensitivity_factor(get_vw(p, ["가격 성향","price_tier"])[0]) for p in plist]
        price_factor = np.mean(price_fs) if price_fs else 1.0

        # --- 클릭 변동성(제품별) → EMA 길이 조정
        g = click_df_with_norm[click_df_with_norm["product_name"]==product].sort_values("date")
        if len(g) >= 3:
            volatility = g["click_norm"].diff().abs().rolling(2).mean().mean()
        else:
            volatility = 0.1

        # --- 파라미터 산출 ---
        ema_span = int(np.clip(round(6 + 4*volatility - 2*(kappa-1)), 3, 12))
        mom_coef = float(np.clip(0.12 + 0.45 * promo_elastic_avg * kappa * (1.0/loyalty_factor), 0.15, 0.7))

        base_rate = CLICK_TO_SALES_RATE_DEF
        click_to_sales_rate = float(
            base_rate * (prob_mean/60.0)**1.2 * loyalty_factor * channel_factor *
            family_factor * income_factor * health_factor * rep_factor * price_factor
        )

        params[product] = dict(
            ema_span=ema_span,
            mom_coef=mom_coef,
            click_to_sales_rate=click_to_sales_rate,
            calib_mult=CALIB_MULT_DEFAULT
        )
    return params


product_params = build_product_params(personas, click)

P("\n=== [STEP3] PRODUCT PARAMS ===")
pp_show = pd.DataFrame.from_dict(product_params, orient="index")
pp_show.index.name = "product_name"
pp_show = pp_show.reset_index()
show_df_info("product_params(sample 10)", pp_show.head(10))

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
        _, _ = get_vw(p, ["거주 지역","region"])  # 지역 weight도 고려하려면 확장 가능
        b_bias = channel_bias_from_text(get_vw(p, ["구매 채널","primary_channel","주요 채널"])[0]) \
                 * region_bias(get_vw(p, ["거주 지역","region"])[0]) \
                 * job_bias(get_vw(p, ["직업","job"])[0])
        beta_boosts.append(apply_weighted(b_bias, 0.5))
        kappas.append(trend_response_to_kappa(get_vw(p, ["트렌드 반응도","trend_responsiveness"])[0]))
        promos.append(label_level(get_vw(p, ["프로모션 민감도","promotion_sensitivity"])[0]))
    beta_boost = np.mean(beta_boosts) if beta_boosts else 1.0
    alpha, beta = 0.5, 0.5*beta_boost
    ssum = alpha + beta
    alpha, beta = alpha/ssum, beta/ssum
    product_alpha_beta[product] = (float(alpha), float(beta))
    product_kappa[product] = float(np.mean(kappas)) if kappas else 1.0
    product_promo_avg[product] = float(np.mean(promos)) if promos else 0.6

# =========================================================
# 5) 페르소나 ‘모든 필드’ 반영 기본 수요 + 이벤트/트렌드/모멘텀 부스트
# =========================================================
def persona_base_for_month(p, product_name, month_idx, month_int):
    # core
    monthly = p.get("monthly_by_launch", p.get("monthly", [4]*12))
    if isinstance(monthly, list) and len(monthly) >= 12:
        month_shape = monthly[month_idx % 12]
    else:
        month_shape = 4.0

    purchase_prob = safe_float(p.get("purchase_probability", p.get("purchase_prob", 60)), 60)/100.0
    base = purchase_prob * month_shape

    # 공통(기존)
    fam_v, fam_w = get_vw(p, ["가족 구성","family"])
    loy_v, loy_w = get_vw(p, ["브랜드 충성도","loyalty"])
    inc_v, inc_w = get_vw(p, ["소득 구간","income"])
    hea_v, hea_w = get_vw(p, ["건강 관심도","health"])
    pro_v, pro_w = get_vw(p, ["프로모션 민감도","promotion_sensitivity"])
    life_v, life_w = get_vw(p, ["라이프스타일","lifestyle"])
    age_v,  age_w  = get_vw(p, ["연령","age"])
    gen_v,  gen_w  = get_vw(p, ["성별","gender"])
    job_v,  job_w  = get_vw(p, ["직업","job"])
    reg_v,  reg_w  = get_vw(p, ["거주 지역","region"])

    fam  = apply_weighted(family_mult(fam_v),  fam_w)
    loy  = apply_weighted(loyalty_mult(loy_v), loy_w)
    inc  = apply_weighted(income_mult(inc_v),  inc_w)
    hea  = apply_weighted(health_mult(product_name, hea_v), hea_w)
    pro  = apply_weighted(promo_mult(pro_v),   pro_w)
    life = apply_weighted(lifestyle_month_boost(month_int, life_v), life_w)
    agew = apply_weighted(1.02, age_w)
    genw = apply_weighted(1.02, gen_w)
    jobw = apply_weighted(1.01, job_w)
    regw = apply_weighted(1.01, reg_w)

    # personas2 확장 항목(있으면 사용)
    primary_ch,  pch_w  = get_vw(p, ["주요 채널","primary_channel","구매 채널"])
    secondary_ch, sch_w = get_vw(p, ["보조 채널","secondary_channel"])
    ch_mix = apply_weighted(channel_mix_factor(primary_ch, secondary_ch), max(pch_w, sch_w))

    fit_v, fit_w = get_vw(p, ["가구-용량 적합도","package_fit"])
    fit = apply_weighted(package_fit_factor(product_name, fit_v), fit_w)

    rep_v, rep_w = get_vw(p, ["재구매 주기","repurchase_cycle"])
    rep = apply_weighted(repurchase_factor(rep_v), rep_w)

    price_v, price_w = get_vw(p, ["가격 성향","price_tier"])
    pricef = apply_weighted(price_sensitivity_factor(price_v), price_w)

    drivers = get_scalar(p, ["drivers","드라이버","이벤트"])
    eventf  = event_month_boost(month_int, drivers)
    # 이벤트는 별도 가중치가 명시되지 않을 수 있으므로 직접 곱
    mult = fam*loy*inc*hea*pro*life*agew*genw*jobw*regw*ch_mix*fit*rep*pricef*eventf
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

P("\n=== [STEP4] PRED (pre-calibration) ===")
show_df_info("pred_df (raw)", pred_df.head(12))

# 앞6/뒤6 비중 제품별 체크(수량 기준 1차)
def head_tail_ratio(df):
    out = []
    for p, g in df.groupby("product_name"):
        g = g.sort_values("date")
        vals = g["quantity"].astype(float).values
        if len(vals) != len(PRED_MONTHS): 
            continue
        head6 = vals[:6].sum(); tail6 = vals[6:].sum(); tot = head6+tail6+1e-9
        out.append([p, head6/tot, tail6/tot])
    return pd.DataFrame(out, columns=["product_name","head6_ratio","tail6_ratio"]).sort_values("tail6_ratio")

ht_raw = head_tail_ratio(pred_df)
show_df_info("head/tail ratio before calibration (bottom 10 by tail)", ht_raw.head(10))


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

P("\n=== [STEP5] CALIBRATION ===")
show_df_info("cal summary (gamma etc.)", cal.head(12))

# 보정 후 헤드/테일 비중(가격 반영 X)
def head_tail_ratio_after(df):
    out = []
    for p, g in df.groupby("product_name"):
        g = g.sort_values("date")
        vals = g["quantity_calib"].astype(float).values
        if len(vals) != len(PRED_MONTHS): 
            continue
        head6 = vals[:6].sum(); tail6 = vals[6:].sum(); tot = head6+tail6+1e-9
        out.append([p, head6/tot, tail6/tot])
    return pd.DataFrame(out, columns=["product_name","head6_ratio","tail6_ratio"]).sort_values("tail6_ratio")

ht_cal = head_tail_ratio_after(pred_df)
show_df_info("head/tail ratio after calibration (bottom 10 by tail)", ht_cal.head(10))

# =========================================================
# 8) 시각화 유틸 & 예시 호출
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _month_key(x):
    """Timestamp/Period/str 무엇이 와도 'YYYY-MM' 문자열로 변환"""
    if isinstance(x, pd.Period):
        return f"{x.year:04d}-{x.month:02d}"
    return pd.Timestamp(x).strftime("%Y-%m")

def visualize_anchor_granularity(
    pred_df,
    PRED_MONTHS,
    products_order=None,
    VAL_COL="quantity_calib",
    monthly_totals=None,      # (선택) DF["date","total_units"]
    product_totals=None,      # (선택) DF["product_name","total_units_12m"]
    cell_anchors=None,        # (선택) DF["product_name","date","units"]
    title_prefix="[Anchor Map]"
):
    """
    앵커 촘촘함 히트맵 (0~4 단계)
      0 = 앵커 없음
      1 = 월 앵커만
      2 = 제품 앵커만
      3 = 월+제품(교차 제약)
      4 = 셀(제품×월) 앵커
    """
    if products_order is None:
        products_order = sorted(pred_df["product_name"].unique().tolist())
    months_order = list(PRED_MONTHS)

    level = np.zeros((len(products_order), len(months_order)), dtype=int)

    has_month = set()
    if monthly_totals is not None and len(monthly_totals):
        # [변경 1] Period.astype(str) 대신 안전한 문자열 변환
        has_month = set(pd.to_datetime(monthly_totals["date"]).dt.strftime("%Y-%m").tolist())

    has_prod = set()
    if product_totals is not None and len(product_totals):
        has_prod = set(product_totals["product_name"].tolist())

    has_cell = set()
    if cell_anchors is not None and len(cell_anchors):
        tmp = cell_anchors.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        # [변경 2] Period.astype(str) 대신 안전한 문자열 변환
        has_cell = set(zip(tmp["product_name"], tmp["date"].dt.strftime("%Y-%m")))

    for i,p in enumerate(products_order):
        for j,d in enumerate(months_order):
            # [변경 3] d.to_period("M").astype(str) → _month_key(d)
            km = _month_key(d)
            if (p, km) in has_cell:
                level[i,j] = 4
            else:
                m_flag = km in has_month
                p_flag = p in has_prod
                if m_flag and p_flag:
                    level[i,j] = 3
                elif m_flag:
                    level[i,j] = 1
                elif p_flag:
                    level[i,j] = 2
                else:
                    level[i,j] = 0

    # 커버리지 통계 & 자유도 근사 출력
    vals, cnts = np.unique(level, return_counts=True)
    total_cells = level.size
    stats = {int(v): int(c) for v,c in zip(vals, cnts)}
    print("\n[Anchor coverage]")
    for v in [0,1,2,3,4]:
        c = stats.get(v, 0)
        print(f" - level {v}: {c} cells ({c/total_cells:.1%})")
    M = len(months_order)
    P = len(products_order)
    C_m = len(monthly_totals) if monthly_totals is not None else 0
    C_p = len(product_totals) if product_totals is not None else 0
    C_cell = stats.get(4, 0)
    approx_constraints = max(0, (min(M, C_m) + min(P, C_p) - 1)) + C_cell
    approx_dof = max(0, total_cells - approx_constraints)
    print(f" - approx constraints ~ {approx_constraints}, approx DOF ~ {approx_dof} / {total_cells}")

    # 히트맵
    fig, ax = plt.subplots(figsize=(min(14, 1.0+0.4*len(months_order)), min(18, 1.0+0.25*len(products_order))))
    im = ax.imshow(level, aspect="auto")
    ax.set_title(f"{title_prefix} granularity (0:none, 1:month, 2:product, 3:both, 4:cell)")
    ax.set_xticks(range(len(months_order)))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in months_order], rotation=90)
    ax.set_yticks(range(len(products_order)))
    ax.set_yticklabels(products_order)
    cbar = plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "anchor_granularity.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_gamma_distribution(cal_df, mode="FULL12", top=None):
    """
    제품별 보정배율(γ) 분포 시각화
    mode: "FULL12" | "PUBLIC6" | "PIECEWISE"
    """
    if mode == "FULL12":
        col = "gamma"
    elif mode == "PUBLIC6":
        col = "gamma_pub"
    elif mode == "PIECEWISE":
        # PIECEWISE는 pub/priv 두 개. 둘 다 보여줌.
        fig, axes = plt.subplots(1, 2, figsize=(14, max(3, 0.25*len(cal_df))))
        for ax, c in zip(axes, ["gamma_pub","gamma_priv"]):
            t = cal_df[["product_name", c]].copy().fillna(1.0).sort_values(c)
            if top is not None:
                t = t.tail(top)
            ax.barh(t["product_name"], t[c])
            ax.set_title(f"{c} distribution")
            ax.set_xlabel("gamma")
            ax.invert_yaxis()
        plt.tight_layout(); plt.show()
        return
    else:
        print(f"[plot_gamma_distribution] unknown mode: {mode}")
        return

    t = cal_df[["product_name", col]].copy().fillna(1.0).sort_values(col)
    if top is not None:
        t = t.tail(top)
    plt.figure(figsize=(10, max(3, 0.25*len(t))))
    plt.barh(t["product_name"], t[col])
    plt.title(f"{col} distribution by product")
    plt.xlabel("gamma")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"gamma_distribution_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_product_timeseries(product_name, trend_df, pred_df, PRED_MONTHS):
    """
    한 제품의 s_sm(검색), c_sm(클릭), quantity(보정 전/후)를 같이 보기
    - 좌축: s_sm, c_sm (0~1 스케일)
    - 우측: quantity, quantity_calib
    """
    tt = trend_df[trend_df["product_name"]==product_name].copy()
    pp = pred_df[pred_df["product_name"]==product_name].copy().sort_values("date")
    if tt.empty or pp.empty:
        print(f"[plot_product_timeseries] no data for: {product_name}")
        return
    # 예측기간만 정렬
    tt = tt[tt["date"].isin(PRED_MONTHS)].sort_values("date")

    fig, ax1 = plt.subplots(figsize=(12,4))
    ax1.plot(tt["date"], tt["s_sm"], marker="o", label="search(smoothed)")
    ax1.plot(tt["date"], tt["c_sm"], marker="o", label="click(smoothed)")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("normalized trend (0~1)")
    ax1.legend(loc="upper left")
    ax1.set_title(f"[{product_name}] Trend & Prediction")

    ax2 = ax1.twinx()
    ax2.plot(pp["date"], pp["quantity"], linestyle="--", marker="x", label="pred (raw)")
    if "quantity_calib" in pp.columns:
        ax2.plot(pp["date"], pp["quantity_calib"], linestyle="-", marker="s", label="pred (calibrated)")
    ax2.set_ylabel("units")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"timeseries_{product_name}.png", dpi=200, bbox_inches="tight")
    plt.close()

products_order = sample["product_name"].tolist()

# =========================================================
# [IPF] 연앵커 기반: "월×제품 동시 제약"
#  - 카테고리 12개월 앵커(연 앵커)를 월비중/제품비중으로 쪼개서
#    -> (월별 총합)과 (제품별 12개월 총합)을 동시에 강제
#  - 초기값은 현 quantity_calib를 사용, IPF로 수렴
# =========================================================

# 0) 카테고리-제품 매핑 (필요한 것만 명시)
CAT2PRODS = {
    "발효유": ["덴마크 하이그릭요거트 400g"],
    "조제커피": ["소화가 잘되는 우유로 만든 카페라떼 250mL",
               "소화가 잘되는 우유로 만든 바닐라라떼 250mL"],
    "조미료": ["동원참치액 순 500g","동원참치액 순 900g",
             "동원참치액 진 500g","동원참치액 진 900g",
             "프리미엄 동원참치액 500g","프리미엄 동원참치액 900g"],
    "식육가공품": ["리챔 오믈레햄 200g","리챔 오믈레햄 340g"],
    # ✅ 요청: 참치캔 = 참기름 4종
    "참치캔": ["동원맛참 고소참기름 90g","동원맛참 고소참기름 135g",
            "동원맛참 매콤참기름 90g","동원맛참 매콤참기름 135g"],
}

# 제품 단가(원/개) — 제품명 정확히 일치해야 합니다.
PRICE_PER_UNIT = {
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

prod2cat = {p:c for c, ps in CAT2PRODS.items() for p in ps}

# 1) 연앵커 로드: 평가 12개월(2024H2+2025H1)을 50:50로 결합
ysum = pd.read_csv("./_save/anchor_step/yearly_anchor_summary.csv").set_index("category")

# 닐슨: 전체시장(세분시장) 매출과 동원F&B 매출에서 2024년 점유율 계산
tot = pd.read_excel("./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2024_세분시장_매출액.xlsx")
dw  = pd.read_excel("./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2024_동원 F&B_매출액.xlsx")

# 필요한 칼럼만 정리 (파일 헤더가 다르면 이름만 맞춰주세요)
tot = tot.rename(columns={"연도":"year","카테고리":"category","매출액(백만원)":"total_mm"})
dw  = dw.rename(columns={"연도":"year","카테고리":"category","매출액(백만원)":"dw_mm","비고":"note"})

# 'etc로 분류됨'은 동원 단독매출로 보기 어려우니 제외(보수적으로)
if "note" in dw.columns:
    dw.loc[dw["note"].astype(str).str.contains("etc", case=False, na=False), "dw_mm"] = pd.NA

# 2024년 기준 카테고리별 점유율 (동원/전체), 결측은 앞뒤 보간 후 중앙값 대체
share_2024 = (tot[tot["year"]==2024][["category","total_mm"]]
              .merge(dw[dw["year"]==2024][["category","dw_mm"]], on="category", how="left"))
share_2024["share"] = share_2024["dw_mm"] / share_2024["total_mm"]
share_2024["share"] = share_2024["share"].fillna(share_2024["share"].median())

# 필요하면 보수적 수동 보정(예: 조제커피는 etc 영향으로 낮게 가정)
manual = {"조제커피": 0.007}   # 필요 없으면 비우세요 {}
for k,v in manual.items():
    share_2024.loc[share_2024["category"]==k, "share"] = v

share_2024 = share_2024.set_index("category")["share"]

# 2025는 정보가 없으니 2024 점유율 유지(혹은 다른 가정 사용)
share_2025 = share_2024.copy()

# === ysum(전체시장 앵커)을 동원 몫으로 축소한 표: ysum_dw ===
ysum_dw = ysum.copy()
ysum_dw["y2024_dw"]      = ysum_dw["y2024"]      * share_2024.reindex(ysum_dw.index).fillna(share_2024.mean())
ysum_dw["y2025_pred_dw"] = ysum_dw["y2025_pred"] * share_2025.reindex(ysum_dw.index).fillna(share_2024.mean())

# === [PATCH C] 카테고리 전환율(click→sales) 보정: DW 앵커 규모에 맞춤 ===
# (1) 제품별 12개월 클릭합
tmp_click_pm = (click.groupby(["product_name","date"])["clicks"].mean()
                     .reset_index())
click12_by_prod = (tmp_click_pm[tmp_click_pm["date"].isin(PRED_MONTHS)]
                   .groupby("product_name")["clicks"].sum())

# (2) 카테고리별 보정배수 계산
cat_rate_mult = {}
for c, prods in CAT2PRODS.items():
    print(f"[AUDIT] category={c} | in ysum={c in ysum.index} | in ysum_dw={c in ysum_dw.index} | n_prods={len(prods)}")
    prods = [p for p in prods if p in click12_by_prod.index]
    if not prods or (c not in ysum_dw.index):
        continue

    # 동원 연앵커(2025예측, 백만원→원)
    A_rev = float(ysum_dw.loc[c, "y2025_pred_dw"]) * 1_000_000.0

    # 현재 전환율×클릭×가격 합
    numer = 0.0
    for p in prods:
        rate_p  = product_params.get(p, {}).get("click_to_sales_rate", CLICK_TO_SALES_RATE_DEF)
        clicks  = float(click12_by_prod.get(p, 0.0))
        price_p = float(PRICE_PER_UNIT.get(p, np.nan))
        if not np.isfinite(price_p) or price_p <= 0:
            # 가격 결측 시 카테고리 중앙값으로 보정
            pv = [PRICE_PER_UNIT.get(pp) for pp in prods if PRICE_PER_UNIT.get(pp) is not None]
            price_p = float(np.nanmedian([x for x in pv if np.isfinite(x) and x>0])) if pv else 1000.0
        numer += rate_p * clicks * price_p

    if numer > 0:
        cat_rate_mult[c] = A_rev / numer
    else:
        cat_rate_mult[c] = 1.0

# (3) 제품별 전환율에 카테고리 보정배수 적용
for p in product_params.keys():
    c = prod2cat.get(p)
    if c in cat_rate_mult:
        product_params[p]["click_to_sales_rate"] *= float(cat_rate_mult[c])

# --- 앵커 모드 설정 ---
ANCHOR_MODE = "absolute"            # "shape" | "growth" | "absolute"
ANCHOR_UNIT_MULT = 1_000_000.0    # 연앵커가 '백만 원'일 때
ANCHOR_GROWTH_STRENGTH = 0.5      # 성장률 반영 강도(0~1)

# 2) 제품분해비(카테고리 내부): 최근 12개월 클릭비중 (그대로 사용)
click12 = (click_pm_raw[click_pm_raw["date"].isin(PRED_MONTHS)]
           .groupby("product_name")["clicks"].sum())
eps = 1e-9
share_prod_in_cat = {}
for c, prods in CAT2PRODS.items():
    prods = [p for p in prods if p in click12.index]
    if not prods: continue
    v = (click12.loc[prods].fillna(0.0) + eps)
    share_prod_in_cat[c] = (v / v.sum()).reindex(prods)

# 3) 월비중(카테고리): 최근 12개월 클릭 합의 월분포 (그대로 사용)
cat_month_share = {}
for c, prods in CAT2PRODS.items():
    cm = (click_pm_raw[(click_pm_raw["product_name"].isin(prods)) &
                       (click_pm_raw["date"].isin(PRED_MONTHS))].groupby("date")["clicks"].sum())
    cm = cm.reindex(PRED_MONTHS, fill_value=0.0) + eps
    cat_month_share[c] = (cm / cm.sum()) if cm.sum() > 0 else pd.Series(1.0/len(PRED_MONTHS), index=PRED_MONTHS)

# === [PATCH A] POS 기반 월비중: marketlink POS 2020~2023에서 월 패턴 추출 ===
PATH_POS_MONTH = DATA_DIR / "pos_data/marketlink_POS_2020_2023_동원 F&B_매출액.xlsx"

def build_pos_month_share(df_pos, category, pred_months, lam_uni=0.2, min_mshare=0.005):
    """
    df_pos: _prepare_pos()로 전처리된 DataFrame 가정
            필요한 컬럼: 'is_dw'(bool), '카테고리_norm', '_월', '_매출', '비고_norm'(선택)
    category: CAT2PRODS의 카테고리 문자열 (예: '발효유')
    pred_months: PRED_MONTHS(datetime index)
    """
    if df_pos is None or df_pos.empty:
        return None

    cat_norm = _norm_txt(category)

    # 1) 동원 F&B + 카테고리 일치만 남기기
    t = df_pos.copy()
    if "is_dw" in t.columns:
        t = t[t["is_dw"] == True]
    if "카테고리_norm" in t.columns:
        t = t[t["카테고리_norm"] == cat_norm]

    # 2) 'etc' 제외 (있을 때만)
    if "비고_norm" in t.columns:
        t = t[~t["비고_norm"].str.contains("etc", case=False, na=False)]

    # 3) 유효 월/매출만
    t = t[(t["_월"].notna()) & (t["_매출"].fillna(0) > 0)]

    if t.empty:
        print(f"[POS WARN] 카테고리 '{category}': 필터 후 행이 없습니다.")
        return None

    # 4) 월 패턴 (월 1~12 합 기준 비중)
    by_m = t.groupby("_월")["_매출"].sum().astype(float)
    by_m = by_m[by_m.index.isin(range(1,13))]  # 안전장치
    if by_m.empty or by_m.sum() <= 0:
        print(f"[POS WARN] 카테고리 '{category}': 월별 매출 집계가 비어있습니다.")
        return None

    share_m = by_m / by_m.sum()  # 1~12 합=1

    # 5) 예측구간 달에 매핑
    s = pd.Series(index=pred_months, dtype=float)
    for d in pred_months:
        m = d.month
        s.loc[d] = float(share_m.get(m, 1/12))

    # 6) 균등 혼합 + 하한 + 재정규화
    s = (1 - lam_uni) * s + lam_uni * (1.0/len(pred_months))
    s = s.clip(lower=min_mshare)
    s = s / s.sum()
    return s



# POS 로드 & 카테고리별 월비중 사전 구축
try:
    posm = pd.read_excel(PATH_POS_MONTH)
except Exception:
    posm = None

import unicodedata

def _norm_txt(x):
    # 유니코드 정규화 + 양쪽 공백 제거 + non-breaking space 제거 + 다중 공백 압축
    s = unicodedata.normalize("NFKC", str(x)).replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _prepare_pos(df):
    df = df.copy()

    # 1) 컬럼명 정규화
    df.columns = [_norm_txt(c) for c in df.columns]

    # 2) 주요 텍스트 컬럼 정규화 복사본
    for col in ["구분", "카테고리", "비고"]:
        if col in df.columns:
            df[col + "_norm"] = df[col].map(_norm_txt)
    # ‘동원 F&B’ 표기 변형 흡수용: 동원F&B / 동원 F & B / 동원 F&B 외
    if "구분_norm" in df.columns:
        df["is_dw"] = df["구분_norm"].str.contains(r"동원\s*F\s*&\s*B", regex=True)
    else:
        df["is_dw"] = False

    # 3) 월 컬럼 정수화 (예: '1', '1월' → 1)
    month_col = "월" if "월" in df.columns else None
    if month_col:
        df["_월"] = df[month_col].astype(str).str.extract(r"(\d+)").astype(float).astype("Int64")
    else:
        df["_월"] = pd.NA

    # 4) 매출액 숫자화
    sales_col = "매출액(백만원)" if "매출액(백만원)" in df.columns else None
    if sales_col:
        df["_매출"] = pd.to_numeric(df[sales_col], errors="coerce")
    else:
        df["_매출"] = pd.NA

    return df

# === 적용 & 진단 출력 ===
if posm is not None:
    posm = _prepare_pos(posm)

    print("\n[POS DEBUG] 컬럼:", list(posm.columns))
    if "구분_norm" in posm.columns:
        print("[POS DEBUG] 구분 unique 상위 10:", posm["구분_norm"].dropna().unique()[:10])
        print("[POS DEBUG] '동원 F&B' 정규식 매칭 수:", int(posm["is_dw"].sum()))
    if "카테고리_norm" in posm.columns:
        print("[POS DEBUG] 카테고리 값/건수 상위:")
        print(posm["카테고리_norm"].value_counts().head(10).to_string())
    if "_월" in posm.columns:
        print("[POS DEBUG] 월 값 샘플:", posm["_월"].dropna().unique()[:12])
    if "_매출" in posm.columns:
        print("[POS DEBUG] 매출 NaN 건수:", int(posm["_매출"].isna().sum()))

cat_month_share_pos = {}
if posm is not None and not posm.empty:
    for c in CAT2PRODS.keys():
        mshare = build_pos_month_share(posm, c, PRED_MONTHS, lam_uni=0.2, min_mshare=0.005)
        if mshare is not None:
            cat_month_share_pos[c] = mshare
            print(f"[POS OK] {c}: month-share sum={mshare.sum():.6f}, head6={mshare.iloc[:6].sum():.3f}, tail6={mshare.iloc[6:].sum():.3f}")
        else:
            print(f"[POS MISS] {c}: fallback to CLICK month-share")



# 기존 "클릭 기반 월비중"은 fallback 용도로만 유지
cat_month_share_click = {}
for c, prods in CAT2PRODS.items():
    cm = (click.groupby(["product_name","date"])["clicks"].mean().reset_index()
                 .query("product_name in @prods and date in @PRED_MONTHS")
                 .groupby("date")["clicks"].sum())
    cm = cm.reindex(PRED_MONTHS, fill_value=0.0) + 1e-9
    cat_month_share_click[c] = cm/cm.sum() if cm.sum() > 0 else pd.Series(1.0/len(PRED_MONTHS), index=PRED_MONTHS)

# 최종 월비중: POS 우선, 없으면 클릭
cat_month_share = {c: cat_month_share_pos.get(c, cat_month_share_click.get(c)) for c in CAT2PRODS.keys()}

P("\n=== [STEP6] MONTH-SHARE SOURCE & IMBALANCE ===")
for c in CAT2PRODS.keys():
    src = "POS" if c in cat_month_share_pos else "CLICK"
    s = (cat_month_share_pos.get(c) if src=="POS" else cat_month_share_click.get(c))
    if s is None:
        P(f" - {c}: NO SHARE AVAILABLE -> will cause fallback/skip later")
        continue
    head6 = float(s.iloc[:6].sum()); tail6 = float(s.iloc[6:].sum())
    P(f" - {c:>6s} | src={src:<5s} | head6={head6:5.3f} | tail6={tail6:5.3f} | min={float(s.min()):.4f}")


# 4) IPF 함수
def ipf(matrix, row_targets, col_targets, tol=1e-6, max_iter=400, tiny=1e-12):
    """
    matrix: (P×M) 초기 값 (양수)
    row_targets: 길이 P (제품별 12개월 합 타깃)
    col_targets: 길이 M (월별 총합 타깃)
    반환: 제약을 동시에 만족하는 조정 행렬 (P×M)
    """
    X = matrix.astype(float).copy()
    X[X<=0] = tiny

    rtar = np.asarray(row_targets, float).copy()
    ctar = np.asarray(col_targets, float).copy()
    rtar[rtar<0] = 0.0
    ctar[ctar<0] = 0.0

    for _ in range(max_iter):
        # row scale
        rs = X.sum(axis=1)
        rs[rs<=0] = tiny
        X *= (rtar/rs)[:,None]

        # col scale
        cs = X.sum(axis=0)
        cs[cs<=0] = tiny
        X *= (ctar/cs)[None,:]

        # 수렴 체크
        r_err = np.max(np.abs(X.sum(axis=1) - rtar) / (rtar + 1e-9))
        c_err = np.max(np.abs(X.sum(axis=0) - ctar) / (ctar + 1e-9))
        if max(r_err, c_err) < tol:
            break
    return X

# 5) 카테고리별로 IPF 실행 — ★ 매출 기준(IPF in revenue) ★
pred_df = pred_df.copy()
start_col = "quantity_calib" if "quantity_calib" in pred_df.columns else "quantity"
if start_col not in pred_df.columns:
    start_col = "quantity"

monthly_total_targets = pd.Series(0.0, index=PRED_MONTHS)  # 시각화용(커버리지)
prod_total_targets = []                                    # 시각화용(커버리지)

P("\n=== [STEP7] IPF INPUT/OUTPUT AUDIT ===")
P(f"- categories = {list(CAT2PRODS.keys())}")

for c, prods in CAT2PRODS.items():
    prods = [p for p in prods if p in pred_df["product_name"].unique()]
    if not prods or (c not in ysum.index): 
        continue

    s_p = share_prod_in_cat.get(c); s_m = cat_month_share.get(c)
    if (s_p is None) or (s_m is None):
        continue

    # (a) 초기 '수량' 행렬
    base_qty = (pred_df[(pred_df["product_name"].isin(prods)) &
                        (pred_df["date"].isin(PRED_MONTHS))]
                .pivot(index="product_name", columns="date", values=start_col)
                .reindex(index=prods, columns=PRED_MONTHS).fillna(eps).astype(float))

    # (b) 제품 가격 벡터
    price_vec = np.array([PRICE_PER_UNIT.get(p, np.nan) for p in prods], float)
    fill_price = np.nanmedian(price_vec[np.isfinite(price_vec) & (price_vec > 0)])
    price_vec = np.where(np.isfinite(price_vec) & (price_vec > 0), price_vec, fill_price)

    # (c) 카테고리 '매출' 타깃 산출  ➜  ★동원 점유율 반영 절대 앵커★
    #  - 기존 growth/shape 대신 동원 몫의 '절대' 연 앵커(백만원→원)를 직접 사용
    #  - ysum_dw에서 카테고리별 동원 매출(백만원)을 이미 계산해 둠
    if c not in ysum_dw.index:
        # 연앵커가 없는 카테고리는 기존 로직으로 fallback (growth/shape)
        y24 = float(ysum.loc[c, "y2024"])
        y25 = float(ysum.loc[c, "y2025_pred"])
        base_rev_total = float((base_qty.values * price_vec[:, None]).sum())
        if ANCHOR_MODE == "absolute":
            A_rev = y25 * ANCHOR_UNIT_MULT
        elif ANCHOR_MODE == "growth":
            g_anchor = y25 / max(y24, 1e-9)
            A_rev = base_rev_total * (g_anchor ** ANCHOR_GROWTH_STRENGTH)
        else:  # "shape"
            A_rev = base_rev_total
    else:
        # ✅ 동원 몫만큼 축소된 '절대' 연앵커(원)
        y25_dw = float(ysum_dw.loc[c, "y2025_pred_dw"])  # 단위: 백만원
        A_rev  = y25_dw * ANCHOR_UNIT_MULT               # 원 단위

    # === AUDIT-2: IPF 입력 타깃 점검 (카테고리별) ===
    print(f"\n[AUDIT-2] category={c}")
    print(f"  - prods={len(prods)}, in ysum={c in ysum.index}, in ysum_dw={c in ysum_dw.index}")
    print(f"  - month_share_sum={cat_month_share[c].sum():.6f}, prod_share_sum={share_prod_in_cat[c].sum():.6f}")

    # (d) 매출 기준 IPF
    base_rev = base_qty.values * price_vec[:, None]
    row_targets_rev = (A_rev * s_p).reindex(prods).values
    col_targets_rev = (A_rev * s_m).reindex(PRED_MONTHS).values
    X_rev = ipf(base_rev, row_targets_rev, col_targets_rev, tol=1e-6, max_iter=400)

    # --- [AUDIT-8] 카테고리별 IPF 결과 월 분해(head6/tail6) 점검 ---
    rev_by_month = X_rev.sum(axis=0)  # 월별 매출 합
    rev_sum = float(rev_by_month.sum()) if np.isfinite(rev_by_month).all() else 0.0
    head6_share = float(rev_by_month[:6].sum() / (rev_sum + 1e-9))
    tail6_share = float(rev_by_month[6:].sum() / (rev_sum + 1e-9))

    print(f"[AUDIT-8] {c} | A_rev={A_rev:,.0f} | head6={head6_share:.3f} | tail6={tail6_share:.3f}")
    if tail6_share < 0.05:
        print(f"  [WARN] {c}: tail6 share {tail6_share:.3f} (<5%). month_share/product_share/targets 확인 필요")

    # 행/열 타깃 일치 여부도 함께 체크
    row_err = np.max(np.abs(X_rev.sum(axis=1) - row_targets_rev) / (np.abs(row_targets_rev) + 1e-9))
    col_err = np.max(np.abs(X_rev.sum(axis=0) - col_targets_rev) / (np.abs(col_targets_rev) + 1e-9))
    print(f"  [AUDIT-8] {c} | row_err={row_err:.2e} | col_err={col_err:.2e}")

    rev_by_month = X_rev.sum(axis=0)
    head6 = float(rev_by_month[:6].sum() / max(rev_by_month.sum(), 1e-9))
    tail6 = float(rev_by_month[6:].sum() / max(rev_by_month.sum(), 1e-9))
    if tail6 < 0.05:
        P(f"  [WARN] {c}: tail6 share {tail6:.3f} (<5%). Check month share / row_targets_rev / col_targets_rev.")    

    print(f"  - A_rev(원)={A_rev:,.0f}")
    print(f"  - row_targets_sum={row_targets_rev.sum():,.0f} | col_targets_sum={col_targets_rev.sum():,.0f} | base_rev_sum={base_rev.sum():,.0f}")

    # === AUDIT-3a: 카테고리별 IPF 결과 월 붕괴 감지 ===
    rev_by_month = X_rev.sum(axis=0)
    head6 = float(rev_by_month[:6].sum() / max(rev_by_month.sum(), 1e-9))
    tail6 = float(rev_by_month[6:].sum() / max(rev_by_month.sum(), 1e-9))
    zero_m = int((rev_by_month[6:] < 1e-8).sum())
    print(f"  - IPF result: head6%={head6:5.3f}, tail6%={tail6:5.3f}, zeros_in_7_12={zero_m}")

    # (e) 다시 '수량'으로 환산
    X_qty = X_rev / price_vec[:, None]
    adj_df = pd.DataFrame(X_qty, index=prods, columns=PRED_MONTHS).stack().reset_index()
    adj_df.columns = ["product_name","date","quantity_ipf"]

    # (f) 결과 반영
    pred_df = pred_df.merge(adj_df, on=["product_name","date"], how="left")
    mask = pred_df["quantity_ipf"].notna()
    pred_df.loc[mask, "quantity_calib"] = pred_df.loc[mask, "quantity_ipf"]
    pred_df.drop(columns=["quantity_ipf"], inplace=True)

    # (g) 시각화용(커버리지용) 집계 — 값 자체는 의미보다 존재 유무가 중요
    monthly_total_targets = monthly_total_targets.add(A_rev * s_m, fill_value=0.0)
    for p in prods:
        prod_total_targets.append({"product_name": p, "total_units_12m": float((A_rev * s_p.loc[p]) / price_vec[prods.index(p)])})

    # 카테고리 루프 안 X_rev 계산 직후
    assert np.isfinite(X_rev).all()
    assert np.isclose(X_rev.sum(), A_rev, rtol=1e-4), f"{c} 매출합 불일치"

check_rows = []
for c, prods in CAT2PRODS.items():
    if c not in ysum_dw.index: 
        continue
    y25_dw = float(ysum_dw.loc[c, "y2025_pred_dw"]) * ANCHOR_UNIT_MULT
    sub = pred_df[pred_df["product_name"].isin(prods) & pred_df["date"].isin(PRED_MONTHS)].copy()
    sub["price"] = sub["product_name"].map(PRICE_PER_UNIT).astype(float)
    rev_pred = float((sub["quantity_calib"] * sub["price"]).sum())
    check_rows.append([c, y25_dw, rev_pred, rev_pred/y25_dw if y25_dw>0 else None])

chk = pd.DataFrame(check_rows, columns=["category","anchor_rev(원)","pred_rev(원)","ratio"])
print("\n[DW Anchor Check]\n", chk.to_string(index=False))

# (선택) 앵커 촘촘함 히트맵 갱신: 월앵커+제품앵커가 동시에 들어가므로 level 3가 되어야 정상
monthly_totals_df = monthly_total_targets.rename("total_units").reset_index().rename(columns={"index":"date"})
product_totals_df = pd.DataFrame(prod_total_targets)

visualize_anchor_granularity(
    pred_df=pred_df,
    PRED_MONTHS=PRED_MONTHS,
    products_order=products_order,
    VAL_COL="quantity_calib",
    monthly_totals=monthly_totals_df,
    product_totals=product_totals_df,
    cell_anchors=None,
    title_prefix="[YearAnchor IPF]"
)

# === AUDIT-3b: 전 제품 전역 스캔 (7~12개월≈0 감지) ===
print("\n[AUDIT-3b] Global scan for late-month collapse (products with tail6 < 5%)")
warn = []
for p, g in pred_df.groupby("product_name"):
    g = g.sort_values("date")
    if not set(PRED_MONTHS).issubset(set(g["date"])): 
        continue
    # 매출 기준으로 보기(가격 반영), 가격 없으면 수량으로 대체
    price = PRICE_PER_UNIT.get(p)
    vals = g["quantity_calib"].values.astype(float)
    if price and np.isfinite(price) and price > 0:
        vals = vals * float(price)
    head6 = vals[:6].sum()
    tail6 = vals[6:].sum()
    total = head6 + tail6 + 1e-9
    if tail6/total < 0.05:  # 뒤 6개월 비중 5% 미만
        warn.append((p, head6/total, tail6/total))
if warn:
    for p, h, t in sorted(warn, key=lambda x: x[2]):
        print(f" - {p}: head6={h:5.3f}, tail6={t:5.3f}")
else:
    print(" - none")



def compare_target_vs_pred_sum(cal_df, pred_df, mode="FULL12"):
    """
    제품별 '앵커 목표 합' vs '보정 후 예측 합' 비교 막대그래프(상위 N)
    - FULL12: target = click_to_sales_rate * raw12
    - PUBLIC6: target = click_to_sales_rate * raw6 (퍼블릭 6개월 합)
    - PIECEWISE: 12개월 합 기준으로 비교
    """
    if mode == "FULL12":
        target = (cal_df[["product_name","click_to_sales_rate","raw12"]]
                  .assign(target=lambda x: x["click_to_sales_rate"]*x["raw12"]))
        agg = (pred_df.groupby("product_name", as_index=False)["quantity_calib"].sum()
               .rename(columns={"quantity_calib":"pred_after"}))
        show = target.merge(agg, on="product_name", how="left")
        show["gap"] = show["pred_after"] - show["target"]

    elif mode == "PUBLIC6":
        target = (cal_df[["product_name","click_to_sales_rate","raw6"]]
                  .assign(target=lambda x: x["click_to_sales_rate"]*x["raw6"]))
        pub6 = pd.date_range("2024-07-01","2024-12-01", freq="MS")
        agg = (pred_df[pred_df["date"].isin(pub6)]
               .groupby("product_name", as_index=False)["quantity_calib"].sum()
               .rename(columns={"quantity_calib":"pred_after"}))
        show = target.merge(agg, on="product_name", how="left")
        show["gap"] = show["pred_after"] - show["target"]

    elif mode == "PIECEWISE":
        target = (cal_df[["product_name","click_to_sales_rate","raw12"]]
                  .assign(target=lambda x: x["click_to_sales_rate"]*x["raw12"]))
        agg = (pred_df.groupby("product_name", as_index=False)["quantity_calib"].sum()
               .rename(columns={"quantity_calib":"pred_after"}))
        show = target.merge(agg, on="product_name", how="left")
        show["gap"] = show["pred_after"] - show["target"]
    else:
        print(f"[compare_target_vs_pred_sum] unknown mode: {mode}")
        return

    show = show.fillna(0.0).sort_values("target", ascending=False)
    topN = min(15, len(show))
    showN = show.head(topN)

    fig, ax = plt.subplots(figsize=(12, max(3, 0.35*topN)))
    idx = np.arange(topN)
    barw = 0.35
    ax.barh(idx - barw/2, showN["target"], height=barw, label="target (anchor)")
    ax.barh(idx + barw/2, showN["pred_after"], height=barw, label="pred (after calib)")
    ax.set_yticks(idx)
    ax.set_yticklabels(showN["product_name"])
    ax.invert_yaxis()
    ax.set_xlabel("units (12m)")
    ax.set_title(f"Target vs Predicted Sum ({mode}) - Top {topN}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"target_vs_pred_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# 7) 제출 포맷으로 변환 (sample_submission과 동일)
# =========================================================

print("\n=== [STEP9] FINAL SANITY BEFORE SUBMISSION ===")
try:
    _tmp = pred_df.copy()
    _tmp["price"] = _tmp["product_name"].map(PRICE_PER_UNIT).astype(float)
    _tmp["rev"] = _tmp["quantity_calib"].astype(float) * _tmp["price"].fillna(1.0)

    rev_by_month = (_tmp.groupby("date")["rev"]
                        .sum()
                        .reindex(PRED_MONTHS)
                        .fillna(0.0))
    head6_rev = float(rev_by_month.iloc[:6].sum())
    tail6_rev = float(rev_by_month.iloc[6:].sum())
    total_rev = head6_rev + tail6_rev

    print(f"- month revenue sum: head6={head6_rev:,.0f} | tail6={tail6_rev:,.0f} | tail6_share={(tail6_rev/(total_rev+1e-9)):.3f}")
    if tail6_rev <= 0 or (tail6_rev/(total_rev+1e-9)) < 0.05:
        print("  [WARN] 뒤 6개월 매출이 비정상적으로 낮습니다. 월비중/제품비중/γ/동원앵커 분배를 재점검하세요.")
except Exception as e:
    print(f"[WARN] STEP9 failed: {e}")


pivot_pred = pred_df.pivot(index="product_name", columns="date", values="quantity_calib").reindex(products_order)
pivot_pred = pivot_pred.reindex(columns=PRED_MONTHS).fillna(0.0)
pivot_pred.columns = [f"months_since_launch_{i+1}" for i in range(len(PRED_MONTHS))]
pivot_pred.reset_index(inplace=True)

out_path = SAVE_DIR / "submission_persona_trend_personas2.csv"
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


# 1) 앵커 촘촘함 히트맵
product_totals = cal[["product_name"]].copy()
product_totals["total_units_12m"] = cal["click_to_sales_rate"] * cal["raw12"]

visualize_anchor_granularity(
    pred_df=pred_df,
    PRED_MONTHS=PRED_MONTHS,
    products_order=products_order,
    VAL_COL="quantity_calib",
    monthly_totals=None,                # 월앵커가 있다면 DF를 넣으세요.
    product_totals=product_totals,      # 제품 12개월 앵커
    cell_anchors=None,
    title_prefix="[Clicks12m anchor]"
)

# 2) 보정배율(γ) 분포 보기
plot_gamma_distribution(cal, mode=CALIB_MODE, top=None)

# 3) 제품별 타임시리즈 : 모든 제품
for p in products_order:
    plot_product_timeseries(p, trend_df=trend_w, pred_df=pred_df, PRED_MONTHS=PRED_MONTHS)

# 4) 제품별 '앵커 목표합 vs 보정후 예측합' 비교
compare_target_vs_pred_sum(cal, pred_df, mode=CALIB_MODE)
