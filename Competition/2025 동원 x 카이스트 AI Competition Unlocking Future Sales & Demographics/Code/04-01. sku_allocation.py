# -*- coding: utf-8 -*-
"""
Step 4 — SKU 배분 (입력 형식 자동 인식 + 월 표기 깨짐(2024-10→2024-1) 강제 복구 패치)

입력:
  1) 세분시장 월 매출액: ./_save/monthly_step/segment_monthly_202407_202506.csv
     - 형식A: category, segment, month(or ym), amount
     - 형식B: product_name, month(or ym), amount  ← sku_master로부터 (category, segment) 복구
  2) SKU 마스터:        ./_data/dacon/dongwon/pos_data/sku_master.xlsx  (또는 csv)
     - 최소 columns: category, segment, (sku_id | sku_name), (unit_price | price)
     - 있으면 좋음: pack_size/pack_unit → price_per_pack = price/size 로 cheapness 강화
  3) (선택) 퍼소나 JSON: ./_data/dacon/dongwon/personas.json
     - 형식 예: {"sku_bias":{"TUNA001":1.2,"COF001":0.9}}

출력:
  - ./_save/sku_step/sku_monthly_amounts.csv   # [category, segment, month(YYYY.MM), sku_id, sku_name, amount]
"""

from __future__ import annotations
from pathlib import Path
import json, re
import numpy as np
import pandas as pd

# --------------------------
# 설정
# --------------------------
SEG_MONTHLY_FP = Path("./_save/monthly_step/segment_monthly_202407_202506.csv")
SKU_MASTER_FP  = Path("./_data/dacon/dongwon/pos_data/sku_master.xlsx")
PERSONAS_JSON  = Path("./_data/dacon/dongwon/personas.json")

OUTDIR = Path("./_save/sku_step"); OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / "sku_monthly_amounts.csv"

# cheapness 강도 (0=균등, 1=그대로, >1=강화)
CHEAPNESS_GAMMA = 1.0

# --------------------------
# 유틸
# --------------------------
def _gcol(df: pd.DataFrame, keys) -> str | None:
    keys = [k.lower() for k in keys]
    for c in df.columns:
        if any(k in str(c).lower() for k in keys):
            return c
    return None

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def _read_table(fp: Path) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(fp)
    if fp.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(fp)
    return pd.read_csv(fp, encoding="utf-8")

def normalize_ym(x) -> str | float:
    """
    다양한 월 표기(YYYY.M, YYYY.MM, YYYY-M, YYYY/MM, YYYY-MM,
    YYYY-MM-DD, 실제 날짜/엑셀 일련번호 등)를 안전하게 'YYYY.MM'로 변환.
    실패하면 np.nan 반환.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()

    # 1) 먼저 날짜로 파싱 (엑셀 날짜/문자열 날짜 모두 대응)
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return f"{dt.year}.{dt.month:02d}"

    # 2) 구분자 통일
    s = s.replace("/", "-").replace(".", "-")

    # 3) 패턴: YYYY-M 또는 YYYY-MM (YYYY-MM-DD도 허용)
    m = re.search(r"^\s*(\d{4})-(\d{1,2})(?:-\d{1,2})?\s*$", s)
    if m:
        y = int(m.group(1)); mo = int(m.group(2))
        if 1 <= mo <= 12:
            return f"{y}.{mo:02d}"

    # 4) 패턴: YYYYMM (또는 YYYY M)
    m = re.search(r"^\s*(\d{4})\s*[-]?\s*(\d{1,2})\s*$", s)
    if m:
        y = int(m.group(1)); mo = int(m.group(2))
        if 1 <= mo <= 12:
            return f"{y}.{mo:02d}"

    return np.nan

# 선택: 정렬/조인용 숫자형(YYYYMM) 생성
def to_ym_num(ym: str) -> int | float:
    try:
        y, m = ym.split(".")
        return int(y) * 100 + int(m)
    except Exception:
        return np.nan

# --------------------------
# 1) SKU 마스터 로드 (먼저 읽어, 새 형식(product_name) 매핑에 활용)
# --------------------------
sku = _read_table(SKU_MASTER_FP)

name_col = _gcol(sku, ["sku_name","제품명","상품명","product","name"])
id_col   = _gcol(sku, ["sku_id","id","코드"])
cat_col  = _gcol(sku, ["category","카테고리"])
seg_col  = _gcol(sku, ["segment","세분","구분"])
size_col = _gcol(sku, ["pack_size","용량","규격","size"])
unit_col = _gcol(sku, ["pack_unit","단위","unit"])
price_col= _gcol(sku, ["unit_price","price","가격","판매가","listprice"])

need = [name_col, cat_col, seg_col, price_col]
if any(x is None for x in need):
    raise KeyError(f"[sku_master] 필수 컬럼이 부족합니다: {list(sku.columns)}")

ren = {name_col:"sku_name", cat_col:"category", seg_col:"segment", price_col:"price"}
if id_col:   ren[id_col]   = "sku_id"
if size_col: ren[size_col] = "size"
if unit_col: ren[unit_col] = "unit"
sku = sku.rename(columns=ren)

sku["sku_name"] = sku["sku_name"].astype(str).map(_clean)
sku["category"] = sku["category"].astype(str).map(_clean)
sku["segment"]  = sku["segment"].astype(str).map(_clean)
sku["price"]    = pd.to_numeric(sku["price"], errors="coerce")
if "size" in sku.columns:
    sku["size"] = pd.to_numeric(sku["size"], errors="coerce")

if "sku_id" not in sku.columns:
    sku["sku_id"] = (sku["sku_name"]
                     .str.replace(r"[^0-9A-Za-z가-힣]+","_", regex=True)
                     .str.strip("_").str.lower())

# 팩당 가격(선택)
if {"price","size"}.issubset(sku.columns) and sku["size"].fillna(0).gt(0).any():
    sku["price_per_pack"] = sku["price"] / sku["size"].replace(0, np.nan)
    BASE_PRICE_COL = "price_per_pack"
else:
    BASE_PRICE_COL = "price"

# --------------------------
# 2) 세분시장 월 매출 로드 (형식 자동 인식) + 월 정규화 패치
# --------------------------
segm_raw = _read_table(SEG_MONTHLY_FP)

# 공통 열 후보
c_col = _gcol(segm_raw, ["category","카테고리"])
s_col = _gcol(segm_raw, ["segment","세분","구분"])
m_col = _gcol(segm_raw, ["month","월","ym","연월"])
v_col = _gcol(segm_raw, ["amount","매출","value","revenue"])
p_col = _gcol(segm_raw, ["product_name","sku_name","제품명","상품명"])

if v_col is None or m_col is None:
    raise KeyError(f"[segment_monthly] month/amount 컬럼을 찾지 못했습니다: {list(segm_raw.columns)}")

segm = segm_raw.copy()
segm["month"]  = segm[m_col].apply(normalize_ym)      # ★ 2024-10 → 2024.10로 강제 복구
segm["ym_num"] = segm["month"].apply(to_ym_num)       # (선택) 정렬/진단용
segm["amount"] = pd.to_numeric(segm[v_col], errors="coerce").fillna(0.0)

if c_col and s_col:
    # 형식 A: category, segment 있음
    segm["category"] = segm[c_col].astype(str).map(_clean)
    segm["segment"]  = segm[s_col].astype(str).map(_clean)
    segm = segm[["category","segment","month","ym_num","amount"]]
elif p_col:
    # 형식 B: product_name만 있음 → sku_master에서 (category, segment) 복구
    segm["sku_name"] = segm[p_col].astype(str).map(_clean)
    key = sku[["sku_name","category","segment"]].drop_duplicates()
    segm = segm.merge(key, on="sku_name", how="left")
    miss = segm[segm["category"].isna() | segm["segment"].isna()]["sku_name"].drop_duplicates()
    if len(miss):
        raise ValueError(f"[segment_monthly] product_name을 sku_master에서 찾지 못했습니다(샘플): {list(miss.head(10))}")
    segm = segm[["category","segment","month","ym_num","amount"]]
else:
    raise KeyError("[segment_monthly] 인식 가능한 열 조합이 없습니다. "
                   "형식A: category/segment/ym/amount  또는 형식B: product_name/ym/amount")

# --------------------------
# 3) (선택) personas.json 보정치 로드
# --------------------------
sku_bias = {}
if PERSONAS_JSON.exists():
    try:
        js = json.loads(Path(PERSONAS_JSON).read_text(encoding="utf-8"))
        if isinstance(js, dict) and "sku_bias" in js and isinstance(js["sku_bias"], dict):
            sku_bias = {str(k): float(v) for k, v in js["sku_bias"].items() if pd.notna(v)}
            print(f"[INFO] personas.json sku_bias 로드: {len(sku_bias)}개")
    except Exception as e:
        print("[WARN] personas.json 파싱 실패 → 무시:", e)

# --------------------------
# 4) α 가중치(cheapness * bias) 계산
# --------------------------
def compute_alpha(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["sku_id","sku_name","category","segment", BASE_PRICE_COL]].copy()

    def _block(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        # cheapness
        if g[BASE_PRICE_COL].fillna(0).gt(0).any():
            p = g[BASE_PRICE_COL].fillna(g[BASE_PRICE_COL].median())
            cheap = 1.0 / p
            if CHEAPNESS_GAMMA != 1.0:
                cheap = np.power(cheap, CHEAPNESS_GAMMA)
        else:
            cheap = pd.Series(1.0, index=g.index)

        # bias
        if sku_bias:
            bias = g["sku_id"].map(lambda x: float(sku_bias.get(str(x), 1.0)))
        else:
            bias = pd.Series(1.0, index=g.index)

        g["alpha"] = (cheap * bias).astype(float)
        s = float(g["alpha"].sum())
        g["alpha_norm"] = (g["alpha"] / s) if s > 0 else (1.0 / len(g))
        return g

    return out.groupby(["category","segment"], as_index=False, group_keys=False).apply(_block)

alpha_df = compute_alpha(sku)

# --------------------------
# 5) 세분시장 월 매출액 → SKU 분배
# --------------------------
rows = []
missing_pairs = set()
alpha_key = alpha_df.set_index(["category","segment"])

for (cat, seg), g_month in segm.groupby(["category","segment"]):
    try:
        block = alpha_key.loc[(cat, seg)].reset_index()
    except KeyError:
        missing_pairs.add((cat, seg))
        continue

    alphas = block["alpha_norm"].to_numpy(dtype=float)
    sku_ids = block["sku_id"].astype(str).to_numpy()
    sku_nms = block["sku_name"].astype(str).to_numpy()

    # 월 정렬(선택): ym_num 기준
    g_month = g_month.sort_values(["ym_num", "month"])

    for _, r in g_month.iterrows():
        amt = float(r["amount"])
        if amt <= 0:
            continue
        alloc = (alphas * amt).astype(float)
        for j in range(len(sku_ids)):
            rows.append({
                "category": cat,
                "segment":  seg,
                "month":    r["month"],   # ← 항상 'YYYY.MM'
                "sku_id":   sku_ids[j],
                "sku_name": sku_nms[j],
                "amount":   alloc[j],
            })

sku_monthly = pd.DataFrame(rows)

# --------------------------
# 6) 정합 검증 & 저장
# --------------------------
if not sku_monthly.empty:
    left = segm.groupby(["category","segment","month"], as_index=False)["amount"].sum().rename(columns={"amount":"seg_amount"})
    right = sku_monthly.groupby(["category","segment","month"], as_index=False)["amount"].sum().rename(columns={"amount":"sku_sum"})
    chk = left.merge(right, on=["category","segment","month"], how="left")
    chk["diff"] = (chk["seg_amount"] - chk["sku_sum"]).abs()
    print(f"[CHECK] 분배 정합 오차(최대) = {chk['diff'].max():,.6f}")
else:
    print("[WARN] 결과가 비었습니다. SKU 매핑이 없는 세분만 있었을 수 있습니다.")

sku_monthly.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("[SAVE]", OUT_CSV.resolve())

if missing_pairs:
    print("[NOTE] SKU 매핑 없는 (카테고리,세분) 조합 수:", len(missing_pairs))
    # 자세한 리스트가 필요하면 주석 해제
    # print(sorted(missing_pairs))
