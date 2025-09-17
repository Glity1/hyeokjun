# qc_after_anchor.py  (교체본)
import pandas as pd, numpy as np
from pathlib import Path

SAVE_DIR = Path("./_save/anchor_pipeline")

# 1) 어떤 제출본을 QC할지 자동 선택 (조정본 우선)
sub_fp = SAVE_DIR / "submission_anchor_adjusted.csv"
if not sub_fp.exists():
    sub_fp = SAVE_DIR / "submission_anchor.csv"

# 2) 파일 로드 (BOM 안전)
sub_adj = pd.read_csv(sub_fp, encoding="utf-8-sig")
anch    = pd.read_csv(SAVE_DIR / "anchor_summary.csv", encoding="utf-8-sig")

# 3) 화이트리스트 카테고리 매핑 (파이프라인과 동일)
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

# ---- 스키마/타입 체크
val_cols = [c for c in sub_adj.columns if c != "product_name"]
assert sub_adj.columns[0] == "product_name" and len(val_cols) == 12, "샘플 스키마(12개월) 불일치"
assert sub_adj["product_name"].is_unique, "product_name 중복 존재"
assert (sub_adj[val_cols].fillna(0) >= 0).all().all(), "음수 존재"
# 정수 여부는 dtype이 int 계열인지 확인
assert all(np.issubdtype(sub_adj[c].dtype, np.integer) for c in val_cols), "정수가 아닌 칸 존재"

# ---- 카테고리 매핑 및 정규화
sub_adj["cat"] = sub_adj["product_name"].map(CATEGORY_OF)
if sub_adj["cat"].isna().any():
    missing = sub_adj.loc[sub_adj["cat"].isna(), "product_name"].unique().tolist()
    raise ValueError(f"CATEGORY_OF에 없는 제품명: {missing}")

def norm_cat(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(r"\s+", "", regex=True)  # 내부 공백 제거
              .str.strip())

sub_adj["cat_norm"] = norm_cat(sub_adj["cat"])
anch["category_norm"] = norm_cat(anch["category"])

# ---- 반기 합계 계산
h2_cols = [f"months_since_launch_{i}" for i in range(1, 7)]   # 2024-07~12
h1_cols = [f"months_since_launch_{i}" for i in range(7, 13)]  # 2025-01~06

qc = (sub_adj.assign(sum_h2=sub_adj[h2_cols].sum(axis=1),
                     sum_h1=sub_adj[h1_cols].sum(axis=1))
               .groupby("cat_norm")[["sum_h2", "sum_h1"]]
               .sum()
               .reset_index())

# ---- 앵커 매핑(dict 기반; 라운드 정수)
anch_map_h2 = (anch.set_index("category_norm")["anchor_2024H2"]
                  .round().astype("Int64").to_dict())
anch_map_h1 = (anch.set_index("category_norm")["anchor_2025H1"]
                  .round().astype("Int64").to_dict())

qc["anchor_h2_int"] = qc["cat_norm"].map(anch_map_h2)
qc["anchor_h1_int"] = qc["cat_norm"].map(anch_map_h1)

# 앵커 매핑 누락 하드 체크
missing_anchor = qc[qc[["anchor_h2_int", "anchor_h1_int"]].isna().any(axis=1)]["cat_norm"].tolist()
if missing_anchor:
    raise ValueError(f"앵커 매핑 실패 카테고리: {missing_anchor} "
                     f"(anchor_summary.csv 의 category 철자/공백 확인)")

# ---- 차이 계산 & 출력
qc["diff_h2"] = qc["sum_h2"] - qc["anchor_h2_int"].astype(int)
qc["diff_h1"] = qc["sum_h1"] - qc["anchor_h1_int"].astype(int)

print("\n=== 산출물 QC — 반기 합계 vs 앵커(정수) ===")
print(qc[["cat_norm","sum_h2","anchor_h2_int","diff_h2","sum_h1","anchor_h1_int","diff_h1"]]
      .rename(columns={"cat_norm":"category"})
      .to_string(index=False))

# ---- 하드 체크: 모두 0이어야 통과
assert (qc["diff_h2"] == 0).all() and (qc["diff_h1"] == 0).all(), "❌ 반기 합계가 앵커(정수)와 일치하지 않습니다."

# ---- 제출값 정수/비음수 재확인
vals = sub_adj[[c for c in sub_adj.columns if c.startswith("months_since_launch_")]]
assert np.issubdtype(vals.dtypes.unique()[0], np.integer), "정수가 아닌 값 포함"
assert (vals.values >= 0).all(), "음수 값 포함"

# ---- (옵션) CSV로 저장
qc_out = SAVE_DIR / "qc_summary.csv"
qc.rename(columns={"cat_norm":"category"}).to_csv(qc_out, index=False, encoding="utf-8-sig")
print("✅ QC 통과 & 저장:", qc_out)
