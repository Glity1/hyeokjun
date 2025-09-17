import pandas as pd

# 1) 가격 분포 점검
sku = pd.read_excel("./_data/dacon/dongwon/pos_data/sku_master.xlsx")
print(sku.groupby(["category","segment"])["unit_price" if "unit_price" in sku else "price"]
      .describe(percentiles=[.05,.5,.95]).round(2))

# 2) MSL 커버리지 & 0 비율
q = pd.read_csv("./_save/qty_step/sku_monthly_qty.csv")
print("MSL 분포:", q["msl"].value_counts().sort_index())
print("MSL별 0비율:", q.assign(is0=q["qty"]<=0).groupby("msl")["is0"].mean().round(3))

# 3) 시즌 가중치 소스 비중
mw = pd.read_csv("./_save/monthly_step/monthly_weights.csv")
print(mw["source"].value_counts(normalize=True).round(3))

# 4) 세분 커버리지/전략 점검
cov = pd.read_csv("./_save/segment_step/coverage_check.csv")
print(cov.groupby(["category","strategy"])["cov_ratio"].mean().round(3).unstack(fill_value=0))

# 5) 샘플 매핑 누락 확인
sample = pd.read_csv("./_data/dacon/dongwon/sample_submission.csv")
name_map = pd.read_excel("./_data/dacon/dongwon/pos_data/sku_master.xlsx")[["sku_id","sku_name"]] \
             .drop_duplicates().rename(columns={"sku_name":"product_name"})
have = q.merge(name_map, on="sku_id", how="left")["product_name"].dropna().unique()
missing = set(sample["product_name"]) - set(have)
print("샘플 대비 누락 product_name 수:", len(missing))
print(list(sorted(list(missing))[:20]))
