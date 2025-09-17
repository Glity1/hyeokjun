# make_features.py
# 네이버 검색/클릭 트렌드 → 판매량 예측용 피처셋 생성 (학습/예측)

import os
import numpy as np
import pandas as pd

# ========= 0. 설정 =========
SEARCH_PATH = "./_data/dacon/dongwon/naver/search_trend_all.csv"
CLICK_PATH  = "./_data/dacon/dongwon/naver/click_trend_all.csv"

EVAL_START = "2022-01-01"
EVAL_END   = "2025-06-30"

TRAIN_START, TRAIN_END = "2022-01-01", "2024-12-31"
TEST_START,  TEST_END  = "2025-01-01", "2025-06-30"

SAVE_DIR = "./_save/features"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========= 1. 로드 & 기간 필터 =========
search = pd.read_csv(SEARCH_PATH, parse_dates=["date"])
click  = pd.read_csv(CLICK_PATH,  parse_dates=["date"])

search = search[(search["date"] >= EVAL_START) & (search["date"] <= EVAL_END)]
click  = click[(click["date"]  >= EVAL_START) & (click["date"]  <= EVAL_END)]

# ========= 2. 파생치 생성 (제품×성별×연령 단위) =========
def add_features(df, value_col, group_cols=["product_name","gender","age"]):
    df = df.copy()
    df = df.sort_values(group_cols + ["date"])
    # 이동평균
    df[f"{value_col}_ma3"] = df.groupby(group_cols)[value_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df[f"{value_col}_ma6"] = df.groupby(group_cols)[value_col].transform(lambda x: x.rolling(6, min_periods=1).mean())
    # 증감률
    df[f"{value_col}_mom"] = df.groupby(group_cols)[value_col].pct_change().fillna(0)
    df[f"{value_col}_yoy"] = df.groupby(group_cols)[value_col].pct_change(periods=12).fillna(0)
    return df

search_feat = add_features(search, "search_index")
click_feat  = add_features(click,  "clicks")

# ========= 3. 콘솔 요약(방향성 체크) =========
def coverage(df, name, val_col):
    print(f"[{name}] rows={len(df):,}, products={df['product_name'].nunique()}, "
          f"dates=({df['date'].min().date()} ~ {df['date'].max().date()})")
    print(f"  - '{val_col}' NA ratio: {df[val_col].isna().mean():.2%}")

print("\n" + "="*80)
print("데이터 커버리지")
print("="*80)
coverage(search_feat, "SEARCH", "search_index")
coverage(click_feat,  "CLICK",  "clicks")

# 제품×월 평균으로 축약(성별/연령 통합: 평균 / 필요시 sum으로 변경)
search_pm = (search_feat.groupby(["product_name","date"], as_index=False)["search_index"].mean())
click_pm  = (click_feat.groupby(["product_name","date"],  as_index=False)["clicks"].mean())

# ========= 4. 최신 스냅샷 · 상관 · 피크월 요약 =========
print("\n" + "="*80)
print("제품별 최신 스냅샷 (마지막 관측월 기준)")
print("="*80)
last_s = (search_feat.sort_values("date")
          .groupby("product_name").tail(1)[
              ["product_name","date","search_index",
               "search_index_mom","search_index_yoy","search_index_ma3","search_index_ma6"]]
          .rename(columns={"date":"last_date_search"}))
last_c = (click_feat.sort_values("date")
          .groupby("product_name").tail(1)[
              ["product_name","date","clicks",
               "clicks_mom","clicks_yoy","clicks_ma3","clicks_ma6"]]
          .rename(columns={"date":"last_date_click"}))
snap = pd.merge(last_s, last_c, on="product_name", how="outer").sort_values("product_name")
print(snap.to_string(index=False))

print("\n" + "="*80)
print("검색↔클릭 상관 (제품별, 공통월 기준)")
print("="*80)
sc = pd.merge(search_pm, click_pm, on=["product_name","date"], how="inner")
corr_tbl = (sc.groupby("product_name")
              .apply(lambda d: d["search_index"].corr(d["clicks"]))
              .reset_index(name="search_click_corr"))
corr_tbl["search_click_corr"] = corr_tbl["search_click_corr"].fillna(0.0)
print("▶ 상관 높은 TOP 10")
print(corr_tbl.sort_values("search_click_corr", ascending=False).head(10).to_string(index=False))
print("\n▶ 상관 낮은(또는 음) TOP 10")
print(corr_tbl.sort_values("search_click_corr", ascending=True).head(10).to_string(index=False))

print("\n" + "="*80)
print("제품별 피크(최대) 월 요약")
print("="*80)
peak_s = search_pm.loc[search_pm.groupby("product_name")["search_index"].idxmax(),
                       ["product_name","date","search_index"]]
peak_c = click_pm.loc[click_pm.groupby("product_name")["clicks"].idxmax(),
                      ["product_name","date","clicks"]]
peak_s = peak_s.rename(columns={"date":"peak_month_search","search_index":"peak_search_index"})
peak_c = peak_c.rename(columns={"date":"peak_month_click","clicks":"peak_clicks"})
peak = pd.merge(peak_s, peak_c, on="product_name", how="outer").sort_values("product_name")
print(peak.to_string(index=False))

# ========= 5. 학습/예측 구간 분리 & 2025 click 대체 =========
train_search = search_feat[(search_feat["date"]>=TRAIN_START)&(search_feat["date"]<=TRAIN_END)]
train_click  = click_feat [(click_feat ["date"]>=TRAIN_START)&(click_feat ["date"]<=TRAIN_END)]
test_search  = search_feat[(search_feat["date"]>=TEST_START) & (search_feat["date"]<=TEST_END)]

# 2025-01~06 click 부재 → 2024-10~12 clicks_ma3 평균으로 carry-forward (간단 대체)
last_click_ma3 = (click_feat[click_feat["date"]>= "2024-10-01"]
                  .groupby("product_name")["clicks_ma3"].mean().rename("click_cf_ma3"))
test_click_stub = test_search[["product_name","date"]].merge(
    last_click_ma3, on="product_name", how="left"
).rename(columns={"click_cf_ma3":"clicks"})
# 대체치 표식/파생
test_click_stub["clicks_mom"] = 0.0
test_click_stub["clicks_yoy"] = 0.0
test_click_stub["clicks_ma3"] = test_click_stub["clicks"]
test_click_stub["clicks_ma6"] = test_click_stub["clicks"]
test_click_stub["missing_click_flag"] = 1

train_click = train_click.copy()
train_click["missing_click_flag"] = 0

# ========= 6. 제품별 best lag(0~6개월) 탐색 (search → click 선행 정도) =========
def best_lag_for_product(prod_df, max_lag=6):
    d = prod_df.sort_values("date")
    best_lag, best_corr = 0, -1
    for lag in range(0, max_lag+1):
        tmp = d.copy()
        tmp["search_shift"] = tmp["search_index"].shift(lag)
        corr = tmp[["search_shift","clicks"]].corr().iloc[0,1]
        if pd.notna(corr) and corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag, best_corr

# 제품×월 평균에서 래그 탐색
sc_pm = pd.merge(search_pm, click_pm, on=["product_name","date"], how="inner")
lag_rows = []
for prod, g in sc_pm.groupby("product_name"):
    lag, corr = best_lag_for_product(g, max_lag=6)
    lag_rows.append({"product_name": prod, "best_lag": lag, "lag_corr": corr})
lag_table = pd.DataFrame(lag_rows).sort_values(["best_lag","lag_corr"], ascending=[True,False])
print("\n=== [제품별 최적 lag (개월), 그때의 상관)] ===")
print(lag_table.to_string(index=False))
lag_table.to_csv(os.path.join(SAVE_DIR, "best_lag_table.csv"), index=False, encoding="utf-8-sig")

# ========= 7. 시즌성 피처 =========
def add_seasonal(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["sin_m"] = np.sin(2*np.pi*df["month"]/12)
    df["cos_m"] = np.cos(2*np.pi*df["month"]/12)
    return df

# 학습용 통합(제품×월 평균 사용)
train_s = add_seasonal(search_pm[(search_pm["date"]>=TRAIN_START)&(search_pm["date"]<=TRAIN_END)])
train_c = add_seasonal(click_pm [(click_pm ["date"]>=TRAIN_START)&(click_pm ["date"]<=TRAIN_END)])
train = pd.merge(train_s, train_c, on=["product_name","date","month","sin_m","cos_m"], how="inner")

# 예측용(2025H1): search 평균 + stub click
test_s = add_seasonal(search_pm[(search_pm["date"]>=TEST_START)&(search_pm["date"]<=TEST_END)])
test_c = add_seasonal(test_click_stub[["product_name","date","clicks","missing_click_flag"]])
test = pd.merge(test_s, test_c, on=["product_name","date","month","sin_m","cos_m"], how="left")
test["missing_click_flag"] = test["missing_click_flag"].fillna(1).astype(int)

# ========= 8. best lag 적용: search_index_lag{lag} & search_index_bestlag =========
def add_best_lag_value(df, lag_tbl, col="search_index"):
    out = []
    lt = lag_tbl.set_index("product_name")["best_lag"].to_dict()
    for prod, g in df.groupby("product_name"):
        lag = int(lt.get(prod, 0))
        gg = g.sort_values("date").copy()
        lag_col = f"{col}_lag{lag}"
        gg[lag_col] = gg[col].shift(lag)
        gg[f"{col}_bestlag"] = gg[lag_col]
        out.append(gg)
    return pd.concat(out, ignore_index=True)

train = add_best_lag_value(train, lag_table, "search_index")
test  = add_best_lag_value(test,  lag_table, "search_index")

# ========= 9. 결측 처리: 제품별 그룹 ffill 후 0 대체 =========
def group_ffill_zero(df, group_col="product_name"):
    df = df.copy().sort_values([group_col,"date"])
    # search/click 관련 열 자동 선택
    cols_to_fill = [c for c in df.columns if c.startswith("search_index") or c.startswith("clicks")]
    for col in cols_to_fill:
        df[col] = df.groupby(group_col)[col].apply(lambda s: s.ffill()).values
        df[col] = df[col].fillna(0)
    return df

train = group_ffill_zero(train)
test  = group_ffill_zero(test)

# 안정적 정렬
for df in (train, test):
    df.sort_values(["product_name","date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

# ========= 10. 저장 =========
train_path = os.path.join(SAVE_DIR, "train_features.csv")
test_path  = os.path.join(SAVE_DIR, "test_features_2025H1.csv")
train.to_csv(train_path, index=False, encoding="utf-8-sig")
test.to_csv(test_path,  index=False, encoding="utf-8-sig")
print("✅ 저장 완료:", train_path, ",", test_path)
print("✅ 저장 완료:", os.path.join(SAVE_DIR, "best_lag_table.csv"))
