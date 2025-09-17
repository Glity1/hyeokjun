# -*- coding: utf-8 -*-
"""
Steps 5 & 6 — 판매량 환산 + 캘리브레이션/정합 (+ Step 7 제출 포맷 생성)
[개편] '월' 대신 months_since_launch(1~12)만 사용
       2024-07 → MSL=1, ..., 2025-06 → MSL=12
[패치] '2024-1'/'2024.1'/'2024/1' → '2024-10' 강제 교정(엑셀 자릿수 깨짐 대응)

입력
  - ./_save/sku_step/sku_monthly_amounts.csv     # (category, segment, month, sku_id, sku_name, amount)
  - ./_data/dacon/dongwon/pos_data/sku_master.xlsx
  - ./_data/dacon/dongwon/product_info.csv       # (선택) 단가 보정
  - ./_data/dacon/dongwon/personas.json          # (선택)
  - ./_data/dacon/dongwon/sample_submission.csv  # 제출 포맷/순서

출력
  - ./_save/qty_step/sku_monthly_qty.csv               # (long) sku_id, months_since_launch, qty
  - ./_save/qty_step/sku_monthly_qty_detailed.csv      # (점검) 정합 및 금액/단가/정수화 결과
  - ./_save/qty_step/sku_qty_check.csv                 # (점검) 그룹 정합 오차
  - ./_save/qty_step/submission.csv                    # (최종) sample_submission 포맷
"""

from __future__ import annotations
from pathlib import Path
import re, json
import numpy as np
import pandas as pd

# -----------------------------
# 경로
# -----------------------------
IN_SKU_MONTHLY   = Path("./_save/sku_step/sku_monthly_amounts.csv")
SKU_MASTER_FP    = Path("./_data/dacon/dongwon/pos_data/sku_master.xlsx")
PRODUCT_INFO_FP  = Path("./_data/dacon/dongwon/product_info.csv")   # 선택
PERSONAS_JSON    = Path("./_data/dacon/dongwon/personas.json")      # 선택
SAMPLE_SUBMIT_FP = Path("./_data/dacon/dongwon/sample_submission.csv")

OUTDIR = Path("./_save/qty_step"); OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_SUBMIT_RAW   = OUTDIR / "sku_monthly_qty.csv"               # (long, MSL)
OUT_DETAILED     = OUTDIR / "sku_monthly_qty_detailed.csv"
OUT_CHECK        = OUTDIR / "sku_qty_check.csv"
OUT_SUBMISSION   = OUTDIR / "submission.csv"

# -----------------------------
# 상수/옵션
# -----------------------------
LAUNCH_START = (2024, 7)   # 2024-07 → MSL=1
LAUNCH_END   = (2025, 6)   # 2025-06 → MSL=12
MONTHS_12 = list(range(1, 13))  # 1..12

# 단가 정책: 팩 가격(PRICE) 사용(해석이 쉬움). 필요시 size로 나눠 단위가로 바꿀 수 있음.
USE_PACK_PRICE_ONLY = True
# 아주 작은 금액으로 전부 0개가 되는 걸 피하고 싶으면 True (가장 싼 SKU에 최소 1개)
ALLOW_MIN1_WHEN_SMALL = False

# -----------------------------
# 유틸
# -----------------------------
def _gcol(df: pd.DataFrame, keys) -> str | None:
    keys = [k.lower() for k in keys]
    for c in df.columns:
        if any(k in str(c).lower() for k in keys):
            return c
    return None

def _read_table(fp: Path) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(fp)
    if fp.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(fp)
    return pd.read_csv(fp, encoding="utf-8")

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def ym_to_msl(val) -> int | None:
    """
    입력 month(문자/엑셀날짜/숫자형 등)를 MSL(1..12)로 변환.
      - 2024-07 → 1, ..., 2025-06 → 12
      - 범위 밖이면 None
    """
    if pd.isna(val):
        return None
    s = str(val).strip()

    # 1) 날짜 파싱 시도
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        y, m = dt.year, dt.month
    else:
        # 2) 구분자 통일 후 'YYYY-M(M)(-DD)' 추출
        s2 = s.replace("/", "-").replace(".", "-")
        mobj = re.match(r"^\s*(\d{4})-(\d{1,2})(?:-\d{1,2})?\s*$", s2)
        if mobj:
            y, m = int(mobj.group(1)), int(mobj.group(2))
        else:
            # 3) YYYYMM 또는 'YYYY M' 류
            mobj2 = re.match(r"^\s*(\d{4})\s*[-]?\s*(\d{1,2})\s*$", s)
            if mobj2:
                y, m = int(mobj2.group(1)), int(mobj2.group(2))
            else:
                return None

    # MSL 계산: (y-2024)*12 + m - 6
    msl = (y - LAUNCH_START[0]) * 12 + m - (LAUNCH_START[1] - 1)
    if 1 <= msl <= 12:
        return int(msl)
    return None

# -----------------------------
# 1) SKU 월 매출액 로드 (→ MSL 변환)
# -----------------------------
dfm = _read_table(IN_SKU_MONTHLY)
c_col = _gcol(dfm, ["category","카테고리"])
s_col = _gcol(dfm, ["segment","세분","구분"])
m_col = _gcol(dfm, ["month","월","ym","연월","months_since_launch"])
id_col= _gcol(dfm, ["sku_id","id","코드"])
nm_col= _gcol(dfm, ["sku_name","상품명","제품명","name","product"])
v_col = _gcol(dfm, ["amount","매출","revenue","value"])

need = [c_col, s_col, m_col, id_col, nm_col, v_col]
if any(x is None for x in need):
    raise KeyError(f"[sku_monthly_amounts] 필요한 컬럼을 찾지 못했습니다: {list(dfm.columns)}")

dfm = dfm[[c_col,s_col,m_col,id_col,nm_col,v_col]].copy()
dfm.columns = ["category","segment","month_raw","sku_id","sku_name","amount"]
for col in ["category","segment","sku_id","sku_name"]:
    dfm[col] = dfm[col].astype(str).map(_clean)
dfm["amount"] = pd.to_numeric(dfm["amount"], errors="coerce").fillna(0.0)

# === [PATCH] month_raw 교정: '2024-1'/'2024.1'/'2024/1' → '2024-10' ===
_raw = dfm["month_raw"].astype(str).str.strip()
dfm["month_raw"] = _raw.str.replace(r'^\s*2024[\-\.\/]\s*1\s*$', '2024-10', regex=True)

# 월 → MSL (이미 1~12 숫자가 들어온 경우도 그대로 사용)
def to_msl_safe(x):
    try:
        xi = int(x)
        if 1 <= xi <= 12:
            return xi
    except Exception:
        pass
    return ym_to_msl(x)

dfm["msl"] = dfm["month_raw"].map(to_msl_safe)
bad = dfm["msl"].isna().sum()
if bad:
    print(f"[WARN] MSL 변환 실패 행 {bad}개: 범위(2024-07~2025-06) 밖이거나 파싱 불가 → 제외")
dfm = dfm.dropna(subset=["msl"]).copy()
dfm["msl"] = dfm["msl"].astype(int).clip(1,12)

# 디버그(선택): 월별 행 수 확인
print("[INFO] rows per MSL:", dfm["msl"].value_counts().sort_index().to_dict())

# -----------------------------
# 2) 단가 소스 로드 (sku_master + product_info 보정)
# -----------------------------
sku = _read_table(SKU_MASTER_FP)
name_col = _gcol(sku, ["sku_name","제품명","상품명","name","product"])
id2_col  = _gcol(sku, ["sku_id","id","코드"])
cat_col  = _gcol(sku, ["category","카테고리"])
seg_col  = _gcol(sku, ["segment","세분","구분"])
price_col= _gcol(sku, ["unit_price","price","판매가","가격","listprice"])
size_col = _gcol(sku, ["pack_size","용량","규격","size"])
unit_col = _gcol(sku, ["pack_unit","단위","unit"])

need2 = [name_col, cat_col, seg_col, price_col]
if any(x is None for x in need2):
    raise KeyError(f"[sku_master] 필수 컬럼이 부족합니다: {list(sku.columns)}")

ren = {name_col:"sku_name", cat_col:"category", seg_col:"segment", price_col:"price"}
if id2_col: ren[id2_col] = "sku_id"
if size_col:ren[size_col]= "size"
if unit_col:ren[unit_col]= "unit"
sku = sku.rename(columns=ren)
sku["sku_name"] = sku["sku_name"].astype(str).map(_clean)
sku["category"] = sku["category"].astype(str).map(_clean)
sku["segment"]  = sku["segment"].astype(str).map(_clean)
sku["price"]    = pd.to_numeric(sku["price"], errors="coerce")

if "sku_id" not in sku.columns:
    sku["sku_id"] = sku["sku_name"].str.replace(r"[^0-9A-Za-z가-힣]+","_", regex=True).str.strip("_")

# (선택) product_info로 단가 보정
if PRODUCT_INFO_FP.exists():
    pi = _read_table(PRODUCT_INFO_FP)
    pi_id = _gcol(pi, ["sku_id","id","코드"])
    pi_nm = _gcol(pi, ["sku_name","제품명","상품명","name"])
    pi_pr = _gcol(pi, ["unit_price","price","가격","판매가"])
    if pi_pr is not None and (pi_id or pi_nm):
        keep = {}
        if pi_id: keep[pi_id] = "sku_id"
        if pi_nm: keep[pi_nm] = "sku_name"
        keep[pi_pr] = "price_pi"
        pi = pi.rename(columns=keep)[list(keep.values())].copy()
        sku = sku.merge(pi, on="sku_id", how="left") if "sku_id" in keep.values() \
              else sku.merge(pi, on="sku_name", how="left")
        sku["price"] = sku["price_pi"].combine_first(sku["price"])
        if "price_pi" in sku.columns: sku.drop(columns=["price_pi"], inplace=True)

# 단가 선택: 팩 가격
if USE_PACK_PRICE_ONLY:
    unit_price_col = "price"
else:
    if "size" in sku.columns:
        sku["size"] = pd.to_numeric(sku["size"], errors="coerce")
        sku["price_per_unit"] = sku["price"] / sku["size"].replace(0, np.nan)
        unit_price_col = "price_per_unit"
    else:
        unit_price_col = "price"

sku["unit_price"] = pd.to_numeric(sku[unit_price_col], errors="coerce")
med_all = float(sku["unit_price"].median())
def _fill_unit_price(g):
    m = g["unit_price"].median()
    g["unit_price"] = g["unit_price"].fillna(m if np.isfinite(m) and m>0 else med_all)
    return g
sku = sku.groupby(["category","segment"], group_keys=False).apply(_fill_unit_price)

# -----------------------------
# 3) 월 매출액 + 단가 결합 → 판매량(float)
# -----------------------------
base = dfm.merge(
    sku[["sku_id","sku_name","category","segment","unit_price"]],
    on=["sku_id","sku_name","category","segment"], how="left"
)
if base["unit_price"].isna().any():
    miss = base[base["unit_price"].isna()][["sku_id","sku_name"]].drop_duplicates()
    print("[WARN] 단가 미확보 SKU 수:", len(miss))
    tmp = base[base["unit_price"].isna()].drop(columns=["unit_price"]).merge(
        sku[["sku_name","unit_price"]].drop_duplicates(),
        on="sku_name", how="left", suffixes=("","_nm")
    )
    base.loc[tmp.index, "unit_price"] = tmp["unit_price"]
if base["unit_price"].isna().any():
    base["unit_price"] = base["unit_price"].fillna(med_all)

base["unit_price"] = base["unit_price"].clip(lower=1e-9)
base["qty_float"]  = base["amount"] / base["unit_price"]

# -----------------------------
# 4) 그룹 정합(스케일) + 정수화(라운딩, 잔차 최소화)
#     그룹키: (category, segment, msl)
# -----------------------------
def integerize_group(g: pd.DataFrame) -> pd.DataFrame:
    target = float(g["amount"].sum())
    rev_from_float = float(np.sum(g["qty_float"] * g["unit_price"]))
    scale = (target / rev_from_float) if rev_from_float > 0 else 1.0
    g = g.copy()
    g["qty_float_scaled"] = g["qty_float"] * scale

    q_floor = np.floor(g["qty_float_scaled"]).astype(int)
    spent   = np.sum(q_floor * g["unit_price"])
    budget  = target - spent

    if ALLOW_MIN1_WHEN_SMALL and budget > 0 and q_floor.sum() == 0 and len(g) > 0:
        i_min = g["unit_price"].idxmin()
        q_floor.loc[i_min] = 1
        spent  = np.sum(q_floor * g["unit_price"])
        budget = target - spent

    g["qty_int"] = q_floor
    g["frac"]    = g["qty_float_scaled"] - q_floor

    if budget > 0:
        order = g.sort_values(["frac", "unit_price"], ascending=[False, True]).index
        for i in order:
            price = float(g.loc[i, "unit_price"])
            if budget >= price - 1e-9:
                g.loc[i, "qty_int"] += 1
                budget -= price
            if budget <= 1e-9:
                break
    elif budget < 0:
        order = g.sort_values(["frac", "unit_price"], ascending=[True, False]).index
        for i in order:
            if g.loc[i, "qty_int"] <= 0:
                continue
            price = float(g.loc[i, "unit_price"])
            g.loc[i, "qty_int"] -= 1
            budget += price
            if budget >= -1e-9:
                break

    g["qty_int"] = g["qty_int"].clip(lower=0).astype(int)
    g["recon_amount"] = g["qty_int"] * g["unit_price"]
    return g

group_keys = ["category","segment","msl"]
out = base.groupby(group_keys, group_keys=False).apply(integerize_group)

# -----------------------------
# 5) 점검표 & 저장 (MSL 기반)
# -----------------------------
check = out.groupby(group_keys, as_index=False).agg(
    seg_amount=("amount","sum"),
    recon_sum =("recon_amount","sum")
)
check["abs_diff"] = (check["seg_amount"] - check["recon_sum"]).abs()
check["rel_diff"] = check["abs_diff"] / check["seg_amount"].replace(0, np.nan)
check.to_csv(OUT_CHECK, index=False, encoding="utf-8-sig")

# 제출용 long (MSL)
submit_raw = out[["sku_id","msl","qty_int"]].rename(columns={"qty_int":"qty"})
submit_raw.to_csv(OUT_SUBMIT_RAW, index=False, encoding="utf-8-sig")

# 상세 파일(점검용)
detailed = out[["category","segment","msl","sku_id","sku_name","unit_price",
                "amount","qty_float","qty_float_scaled","qty_int","recon_amount"]]
detailed["rev_diff"] = detailed["amount"] - detailed["recon_amount"]
detailed.to_csv(OUT_DETAILED, index=False, encoding="utf-8-sig")

print("[SAVE]", OUT_SUBMIT_RAW.resolve())
print("[SAVE]", OUT_DETAILED.resolve())
print("[SAVE]", OUT_CHECK.resolve())
print("그룹별 정합 최대 절대오차:", f"{check['abs_diff'].max():,.2f}")
print("그룹별 정합 평균 상대오차:", f"{check['rel_diff'].mean():.6f}")

# -----------------------------
# 6) 제출 포맷 생성 (샘플과 동일: months_since_launch_1..12)
# -----------------------------
sample = _read_table(SAMPLE_SUBMIT_FP)
if "product_name" not in sample.columns:
    raise KeyError("sample_submission.csv에 'product_name' 열이 필요합니다.")

# sku_master에서 product_name(=sku_name) 매핑
name_map = sku[["sku_id","sku_name"]].drop_duplicates().rename(columns={"sku_name":"product_name"})
sub = submit_raw.merge(name_map, on="sku_id", how="left")

# 1~12 범위만 사용 → 피벗
sub = sub[(sub["msl"]>=1) & (sub["msl"]<=12)].copy()
wide = (sub.pivot_table(index="product_name", columns="msl", values="qty", aggfunc="sum")
           .reindex(columns=MONTHS_12, fill_value=0)  # 누락열 0 채움
           .fillna(0).round().astype(int))
wide.columns = [f"months_since_launch_{int(c)}" for c in wide.columns]
wide = wide.reset_index()

# 샘플 순서/열로 맞추기
final = sample[["product_name"]].merge(wide, on="product_name", how="left").fillna(0).copy()
for i in MONTHS_12:
    col = f"months_since_launch_{i}"
    if col not in final.columns:
        final[col] = 0
final = final[["product_name"] + [f"months_since_launch_{i}" for i in MONTHS_12]]

final.to_csv(OUT_SUBMISSION, index=False, encoding="utf-8-sig")
print("[SAVE]", OUT_SUBMISSION.resolve())
print(f"[SUBMIT] rows={len(final):,}, cols={len(final.columns)}  (샘플 포맷 OK)")
