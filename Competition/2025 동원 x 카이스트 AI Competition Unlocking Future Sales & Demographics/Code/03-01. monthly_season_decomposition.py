# -*- coding: utf-8 -*-
"""
Step 3 — 월별 시즌 분해 (POS 2020~2023 → 2024-07~2025-06 월 매출)
입력:
  - 세분 연 매출(₩):   ./_save/segment_step/segment_yearly_amounts.csv
  - POS 월 매출(₩):    ./_data/dacon/dongwon/pos_data/marketlink_POS_2020_2023_월별 매출액.xlsx
출력:
  - ./_save/monthly_step/monthly_weights.csv            # (category,segment,month,weight,source)
  - ./_save/monthly_step/segment_monthly_202407_202506.csv  # (category,segment,ym,amount)
  - ./_save/monthly_step/plots/<카테고리>.png
설명:
  - 2024(연) 총액을 12개월로 분해 후 7~12월만 출력, 2025(연) 총액을 12개월로 분해 후 1~6월만 출력
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -----------------------------
# 폰트(환경에 맞게 필요시 변경)
# -----------------------------
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.family'] = 'Malgun Gothic'   # mac: AppleGothic, linux: NanumGothic 등

# -----------------------------
# 경로/상수
# -----------------------------
SEG_YEARLY_FP = Path("./_save/segment_step/segment_yearly_amounts.csv")
POS_MONTHLY_FP = Path("./_data/dacon/dongwon/pos_data/marketlink_POS_2020_2023_월별 매출액.xlsx")

OUTDIR = Path("./_save/monthly_step"); OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR = OUTDIR / "plots"; PLOTDIR.mkdir(parents=True, exist_ok=True)

# 출력 기간
H2_2024 = [("2024", m) for m in range(7,13)]    # 2024-07..12
H1_2025 = [("2025", m) for m in range(1,7)]     # 2025-01..06
TARGET_MONTHS = H2_2024 + H1_2025               # 총 12개월

# -----------------------------
# 유틸
# -----------------------------
def _guess_col(df: pd.DataFrame, keys):
    keys = [str(k).lower() for k in keys]
    for c in df.columns:
        s = str(c).lower()
        if any(k in s for k in keys):
            return c
    return None

def _clean(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

# -----------------------------
# 1) 세분 연 매출 로드 (Step2 결과)
# -----------------------------
seg_year = pd.read_csv(SEG_YEARLY_FP, encoding="utf-8")
c_col = _guess_col(seg_year, ["category","카테고리"])
s_col = _guess_col(seg_year, ["segment","세분","구분"])
y_col = _guess_col(seg_year, ["year","연도"])
v_col = _guess_col(seg_year, ["amount","매출","value"])

if any(x is None for x in [c_col, s_col, y_col, v_col]):
    raise ValueError(f"[세분 연 매출] 컬럼 감지 실패: {list(seg_year.columns)}")

seg_year = seg_year[[c_col, s_col, y_col, v_col]].copy()
seg_year.columns = ["category","segment","year","amount"]
seg_year["category"] = seg_year["category"].map(_clean)
seg_year["segment"]  = seg_year["segment"].map(_clean)
seg_year["year"]     = pd.to_numeric(seg_year["year"], errors="coerce").astype("Int64")
seg_year["amount"]   = pd.to_numeric(seg_year["amount"], errors="coerce").fillna(0.0)

# 2024/2025만 확보(둘 중 하나가 없으면 경고)
avail_years = seg_year["year"].dropna().unique().tolist()
if 2024 not in avail_years:
    print("[WARN] 세분 연 매출에 2024가 없습니다. 2024-07~12 배분은 0으로 나옵니다.")
if 2025 not in avail_years:
    print("[WARN] 세분 연 매출에 2025가 없습니다. 2025-01~06 배분은 0으로 나옵니다.")

# -----------------------------
# 2) POS 월 매출 로드 → (카테고리, 세분)별 월 패턴
# -----------------------------
pos = pd.read_excel(POS_MONTHLY_FP) if POS_MONTHLY_FP.suffix.lower() in [".xlsx",".xls"] else pd.read_csv(POS_MONTHLY_FP, encoding="utf-8")

cat_col = _guess_col(pos, ["카테고리","category","cat"])
seg_col = _guess_col(pos, ["구분","세분","segment","subcat"])
ycol    = _guess_col(pos, ["연도","year"])
mcol    = _guess_col(pos, ["월","month"])
dcol    = _guess_col(pos, ["일자","date"])   # 있으면 여기서 연월 추출
vcol    = _guess_col(pos, ["매출","수량","amount","value","qty"])

if cat_col is None or seg_col is None or vcol is None or (ycol is None and dcol is None):
    raise ValueError(f"[POS] 컬럼 감지 실패: {list(pos.columns)}")

tmp = pos.copy()
tmp[cat_col] = tmp[cat_col].map(_clean)
tmp[seg_col] = tmp[seg_col].map(_clean)

if dcol:
    dt = pd.to_datetime(tmp[dcol], errors="coerce")
    tmp["year"] = dt.dt.year
    tmp["month"] = dt.dt.month
else:
    tmp["year"] = pd.to_numeric(tmp[ycol], errors="coerce")
    if mcol:
        tmp["month"] = pd.to_numeric(tmp[mcol], errors="coerce")
    else:
        raise ValueError("[POS] 월(month) 정보를 찾을 수 없습니다.")

tmp["amount"] = pd.to_numeric(tmp[vcol], errors="coerce").fillna(0.0)
tmp = tmp.dropna(subset=["year","month"])
tmp["year"] = tmp["year"].astype(int)
tmp["month"] = tmp["month"].astype(int)

# 학습 범위: 2020~2023
tmp = tmp[(tmp["year"] >= 2020) & (tmp["year"] <= 2023)]
if tmp.empty:
    print("[WARN] POS 2020~2023 데이터가 비었습니다. 모든 패턴은 균등 분배로 대체합니다.")

# (category, segment, month) 합계 → 월 비중
pos_seg = (tmp.groupby([cat_col, seg_col, "month"], as_index=False)["amount"]
             .sum().rename(columns={cat_col:"category", seg_col:"segment"}))

# 세분 패턴
seg_piv = pos_seg.pivot_table(index=["category","segment"], columns="month", values="amount", aggfunc="sum").fillna(0.0)
# 카테고리 패턴
pos_cat = (tmp.groupby([cat_col, "month"], as_index=False)["amount"]
             .sum().rename(columns={cat_col:"category"}))
cat_piv = pos_cat.pivot_table(index=["category"], columns="month", values="amount", aggfunc="sum").fillna(0.0)

# 월 비중(1~12 합=1)
def _to_weights(row_like: pd.Series) -> np.ndarray:
    arr = np.array([row_like.get(m, 0.0) for m in range(1,13)], dtype=float)
    s = arr.sum()
    if s <= 0:
        return np.ones(12)/12.0
    return arr / s

# (category, segment) → weight(12), source
weights_rows = []

# 모든 (category,segment) 조합을 세분연매출 기준으로 생성
pairs = seg_year[["category","segment"]].drop_duplicates().values.tolist()
for cat, seg in pairs:
    src = "segment"
    if (cat, seg) in seg_piv.index:
        w = _to_weights(seg_piv.loc[(cat,seg)])
    elif cat in cat_piv.index:
        w = _to_weights(cat_piv.loc[cat]); src = "category_fallback"
    else:
        w = np.ones(12)/12.0; src = "uniform_fallback"

    for m in range(1,13):
        weights_rows.append({
            "category": cat, "segment": seg, "month": m,
            "weight": float(w[m-1]), "source": src
        })

monthly_weights = pd.DataFrame(weights_rows)
monthly_weights.to_csv(OUTDIR/"monthly_weights.csv", index=False, encoding="utf-8-sig")
print("[SAVE]", (OUTDIR/"monthly_weights.csv").resolve())

# -----------------------------
# 3) 2024/2025 연 매출 → 월 분해 (2024-07..2025-06만 출력)
# -----------------------------
# 연→월 분해 함수
def split_year_to_months(annual_amt: float, w12: np.ndarray) -> np.ndarray:
    # annual_amt를 12개월 비중대로 나눔
    return annual_amt * w12

out_rows = []

for (cat, seg), g in seg_year.groupby(["category","segment"]):
    # 월 가중치 확보
    w = monthly_weights[(monthly_weights["category"]==cat) & (monthly_weights["segment"]==seg)] \
                        .sort_values("month")["weight"].to_numpy()
    if len(w) != 12:
        w = np.ones(12)/12.0

    # 2024, 2025 연총액
    y24 = float(g.loc[g["year"]==2024, "amount"].sum()) if (g["year"]==2024).any() else 0.0
    y25 = float(g.loc[g["year"]==2025, "amount"].sum()) if (g["year"]==2025).any() else 0.0

    # 연→월
    m24 = split_year_to_months(y24, w)  # index 0..11 => 1..12월
    m25 = split_year_to_months(y25, w)

    # 2024-07..12
    for m in range(7,13):
        out_rows.append({
            "category": cat, "segment": seg,
            "ym": f"2024-{m:02d}",
            "amount": float(m24[m-1])
        })
    # 2025-01..06
    for m in range(1,7):
        out_rows.append({
            "category": cat, "segment": seg,
            "ym": f"2025-{m:02d}",
            "amount": float(m25[m-1])
        })

# -------------------------------------------------
# A) 대표 product_name 매핑 만들기 (sku_master.xlsx에서)
# -------------------------------------------------
SKU_MASTER_FP = Path("./_data/dacon/dongwon/pos_data/sku_master.xlsx")

def _gcol(df, keys):
    keys = [k.lower() for k in keys]
    for c in df.columns:
        if any(k in str(c).lower() for k in keys):
            return c
    return None

sku = pd.read_excel(SKU_MASTER_FP) if SKU_MASTER_FP.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(SKU_MASTER_FP, encoding="utf-8")
cat_col = _gcol(sku, ["category","카테고리"])
seg_col = _gcol(sku, ["segment","세분","구분"])
nm_col  = _gcol(sku, ["sku_name","제품명","상품명","product_name","name"])
size_col= _gcol(sku, ["pack_size","용량","규격","size"])
rep_col = _gcol(sku, ["대표","representative","is_rep","rep"])

if any(x is None for x in [cat_col, seg_col, nm_col]):
    raise KeyError("[sku_master]에서 (category, segment, product_name) 컬럼을 찾지 못했습니다.")

sku = sku.rename(columns={cat_col:"category", seg_col:"segment", nm_col:"product_name"})
sku["category"] = sku["category"].astype(str).str.strip()
sku["segment"]  = sku["segment"].astype(str).str.strip()
sku["product_name"] = sku["product_name"].astype(str).str.strip()

# 대표 선택 규칙:
# 1) '대표' 표기가 있으면 해당=1인 행을 우선
# 2) 없으면 pack_size가 가장 큰 행
# 3) 그래도 없으면 이름 사전순 첫 행
if rep_col and rep_col in sku.columns:
    sku["_rep"] = (sku[rep_col].fillna(0).astype(str).str.strip().isin(["1","True","true","Y","y"])).astype(int)
else:
    sku["_rep"] = 0

if size_col and size_col in sku.columns:
    sku["_size"] = pd.to_numeric(sku[size_col], errors="coerce").fillna(0)
else:
    sku["_size"] = 0

# 대표 선택
rep_map = (
    sku.sort_values(by=["_rep","_size","product_name"], ascending=[False, False, True])
       .groupby(["category","segment"], as_index=False)
       .first()[["category","segment","product_name"]]
)

# -------------------------------------------------
# B) 기존 monthly_out에 product_name 부여해서 저장 형식 변경
# -------------------------------------------------
monthly_out = pd.DataFrame(out_rows).sort_values(["category","segment","ym"]).reset_index(drop=True)

# (category, segment) → 대표 product_name merge
monthly_out = monthly_out.merge(rep_map, on=["category","segment"], how="left")

# 매칭 안된 건 fallback: "카테고리__세분"
mask_na = monthly_out["product_name"].isna()
monthly_out.loc[mask_na, "product_name"] = monthly_out.loc[mask_na, "category"] + "__" + monthly_out.loc[mask_na, "segment"]

# 최종 저장 형식: product_name, ym(=YYYY-MM), amount(단위: 입력 그대로, 보통 '백만원')
final_out = monthly_out[["product_name","ym","amount"]].copy()
final_out["amount"] *= 1_000_000
final_csv = OUTDIR / "segment_monthly_202407_202506.csv"
final_out.to_csv(final_csv, index=False, encoding="utf-8-sig")
print("[SAVE]", final_csv.resolve())

# -----------------------------
# 4) 간단 시각화 (카테고리별 12개월 스택)
# -----------------------------
for cat, g in monthly_out.groupby("category"):
    plt.figure(figsize=(10,4.8))
    # wide: index=ym, columns=segment
    pw = g.pivot_table(index="ym", columns="segment", values="amount", aggfunc="sum").fillna(0.0)
    # ym 정렬
    pw = pw.loc[sorted(pw.index)]
    idx = np.arange(len(pw))
    bottom = np.zeros(len(pw))
    for seg in pw.columns:
        vals = pw[seg].values
        plt.bar(idx, vals, bottom=bottom, label=seg)
        bottom += vals

    # x tick
    plt.xticks(idx, pw.index, rotation=45)
    # 2025-01 경계선
    if "2025-01" in pw.index:
        k = list(pw.index).index("2025-01")
        plt.axvline(k-0.5, color="gray", linestyle="--", alpha=0.6)

    plt.title(f"[월 매출 분해] {cat} — 2024-07 ~ 2025-06")
    plt.ylabel("월 매출액 (₩)")
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    outpng = PLOTDIR / f"{cat}.png"
    plt.savefig(outpng, dpi=150)
    plt.close()
    print("[PLOT]", outpng.resolve())

print("완료: POS 월 패턴 기반 세분시장 월 매출(2024-07~2025-06) 산출 및 저장")
