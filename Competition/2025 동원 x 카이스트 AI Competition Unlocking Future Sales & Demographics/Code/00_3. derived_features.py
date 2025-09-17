import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False 


# ======================
# 1. 데이터 로드
# ======================
search = pd.read_csv("./_data/dacon/dongwon/naver/search_trend_all.csv", parse_dates=["date"])
click = pd.read_csv("./_data/dacon/dongwon/naver/click_trend_all.csv", parse_dates=["date"])

# ======================
# 2. 기간 필터링 (2022-01 ~ 2025-06)
# ======================
EVAL_START = "2022-01-01"
EVAL_END   = "2025-06-30"

search = search[(search["date"] >= EVAL_START) & (search["date"] <= EVAL_END)]
click  = click[(click["date"]  >= EVAL_START) & (click["date"]  <= EVAL_END)]

# ======================
# 3. Feature 생성 함수
# ======================
def add_features(df, value_col, group_cols=["product_name"]):
    df = df.copy()
    df = df.sort_values(["product_name", "date"])
    
    # 이동평균 (3개월, 6개월)
    df[f"{value_col}_ma3"] = df.groupby(group_cols)[value_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df[f"{value_col}_ma6"] = df.groupby(group_cols)[value_col].transform(lambda x: x.rolling(6, min_periods=1).mean())
    
    # 증감률 (MoM)
    df[f"{value_col}_mom"] = df.groupby(group_cols)[value_col].pct_change().fillna(0)
    
    # 전년 대비 증감률 (YoY)
    df[f"{value_col}_yoy"] = df.groupby(group_cols)[value_col].pct_change(periods=12).fillna(0)
    
    return df

search_feat = add_features(search, "search_index")
click_feat  = add_features(click, "clicks")

# ======================
# 4. 데이터 병합 (선택)
# ======================
merged = pd.merge(
    search_feat, click_feat,
    on=["date","gender","age","product_name","keyword"],
    how="outer"
)

# ======================
# 5. 시각화 (모든 제품)
# ======================
def plot_features(df, product, value_col):
    df_prod = df[df["product_name"] == product]
    plt.figure(figsize=(12,6))
    plt.plot(df_prod["date"], df_prod[value_col], label=value_col, alpha=0.7)
    if f"{value_col}_ma3" in df_prod.columns:
        plt.plot(df_prod["date"], df_prod[f"{value_col}_ma3"], label=f"{value_col}_ma3", linestyle="--")
    if f"{value_col}_ma6" in df_prod.columns:
        plt.plot(df_prod["date"], df_prod[f"{value_col}_ma6"], label=f"{value_col}_ma6", linestyle="--")
    plt.title(f"{product} - {value_col} Trend")
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.legend()
    plt.grid(True)
    # plt.show()

# # ======================
# # 모든 제품에 대해 실행
# # ======================
# # 검색 트렌드
# for product in search_feat["product_name"].unique():
#     plot_features(search_feat, product, "search_index")

# # 클릭량
# for product in click_feat["product_name"].unique():
#     plot_features(click_feat, product, "clicks")

# import os

# # 저장 폴더 생성
# os.makedirs("./_save/plots", exist_ok=True)

# def plot_features(df, product, value_col):
#     df_prod = df[df["product_name"] == product]
#     plt.figure(figsize=(12,6))
#     plt.plot(df_prod["date"], df_prod[value_col], label=value_col, alpha=0.7)
#     if f"{value_col}_ma3" in df_prod.columns:
#         plt.plot(df_prod["date"], df_prod[f"{value_col}_ma3"], label=f"{value_col}_ma3", linestyle="--")
#     if f"{value_col}_ma6" in df_prod.columns:
#         plt.plot(df_prod["date"], df_prod[f"{value_col}_ma6"], label=f"{value_col}_ma6", linestyle="--")
#     plt.title(f"{product} - {value_col} Trend")
#     plt.xlabel("Date")
#     plt.ylabel(value_col)
#     plt.legend()
#     plt.grid(True)

#     # 파일로 저장
#     filename = f"./_save/plots/{product}_{value_col}.png"
#     plt.savefig(filename, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"✅ 저장 완료: {filename}")

# # 모든 제품 실행
# for product in search_feat["product_name"].unique():
#     plot_features(search_feat, product, "search_index")

# for product in click_feat["product_name"].unique():
#     plot_features(click_feat, product, "clicks")

# ======================
# 6. 콘솔 요약 출력 (방향성 확인용)
# ======================
import numpy as np

print("\n" + "="*80)
print("데이터 커버리지")
print("="*80)
def coverage(df, name, val_col):
    print(f"[{name}] rows={len(df):,}, products={df['product_name'].nunique()}, "
          f"dates=({df['date'].min().date()} ~ {df['date'].max().date()})")
    # 세그먼트 누락 비율(간단 확인): value 결측 비율
    miss = df[val_col].isna().mean()
    print(f"  - '{val_col}' NA ratio: {miss:.2%}")

coverage(search_feat, "SEARCH", "search_index")
coverage(click_feat,  "CLICK",  "clicks")

# 제품×월 단위로 집계(성별/연령 통합: 평균 사용; 필요시 sum으로 교체)
search_prod_month = (search_feat
    .groupby(["product_name","date"], as_index=False)["search_index"].mean())
click_prod_month = (click_feat
    .groupby(["product_name","date"], as_index=False)["clicks"].mean())

print("\n" + "="*80)
print("제품별 최신 스냅샷 (마지막 관측월 기준)")
print(" - Search/Click 모두 마지막 행에서 원시값, MoM, YoY 확인")
print("="*80)
# 최신 행(제품별 마지막 날짜) 뽑기
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

snap = pd.merge(last_s, last_c, on="product_name", how="outer")
# 보기 편하게 정렬
snap = snap.sort_values("product_name")
print(snap.to_string(index=False))

print("\n" + "="*80)
print("검색↔클릭 상관 (제품별, 공통월 기준; 성별/연령 평균 후 산출)")
print("="*80)
sc = pd.merge(search_prod_month, click_prod_month, on=["product_name","date"], how="inner")
corr_tbl = (sc.groupby("product_name")
              .apply(lambda d: d["search_index"].corr(d["clicks"]))
              .reset_index(name="search_click_corr"))
# NaN 방지
corr_tbl["search_click_corr"] = corr_tbl["search_click_corr"].fillna(0.0)
print("▶ 상관 높은 TOP 10")
print(corr_tbl.sort_values("search_click_corr", ascending=False).head(10).to_string(index=False))
print("\n▶ 상관 낮은(또는 음) TOP 10")
print(corr_tbl.sort_values("search_click_corr", ascending=True).head(10).to_string(index=False))

print("\n" + "="*80)
print("제품별 피크(최대) 월 요약")
print(" - search_index 최대 시점 / clicks 최대 시점")
print("="*80)
peak_s_idx = search_prod_month.groupby("product_name")["search_index"].idxmax()
peak_c_idx = click_prod_month.groupby("product_name")["clicks"].idxmax()
peak_s = search_prod_month.loc[peak_s_idx, ["product_name","date","search_index"]]
peak_c = click_prod_month.loc[peak_c_idx, ["product_name","date","clicks"]]
peak_s = peak_s.rename(columns={"date":"peak_month_search","search_index":"peak_search_index"})
peak_c = peak_c.rename(columns={"date":"peak_month_click","clicks":"peak_clicks"})
peak = pd.merge(peak_s, peak_c, on="product_name", how="outer").sort_values("product_name")
print(peak.to_string(index=False))

print("\n" + "="*80)
print("추세 분류(단기 vs 중기, MA3/MA6 비율)")
print(" - MA3 / MA6 >= 1.02: UP, <= 0.98: DOWN, 그 외 FLAT")
print("="*80)
def trend_flag(df, valcol):
    last = (df.sort_values("date").groupby("product_name").tail(1)
              [["product_name", f"{valcol}_ma3", f"{valcol}_ma6"]]
              .copy())
    ratio = last[f"{valcol}_ma3"] / (last[f"{valcol}_ma6"].replace(0, np.nan))
    last[f"{valcol}_trend"] = np.where(ratio >= 1.02, "UP",
                               np.where(ratio <= 0.98, "DOWN", "FLAT"))
    return last[["product_name", f"{valcol}_trend"]]

t_search = trend_flag(search_feat, "search_index")
t_click  = trend_flag(click_feat,  "clicks")
t = pd.merge(t_search, t_click, on="product_name", how="outer")
print(t.sort_values("product_name").to_string(index=False))

print("\n" + "="*80)
print("간단 가이드")
print("="*80)
print("1) 상관이 높은 제품은 검색 증가 → 클릭/구매 전환 가능성도 높으므로, 검색 지표를 주요 선행지표로 사용하세요.")
print("2) 피크 월이 명절·시즌에 몰린 품목은 시즌성 가중(달별 더미/사인·코사인) 반영이 중요합니다.")
print("3) 추세가 UP인 제품은 단기 모멘텀(최근 1~3개월) 계열 feature의 가중을 높여 학습하면 효과적입니다.")
print("4) click은 구매에 가깝고 search는 잠재관심이므로, 모델에서 click feature를 조금 더 높은 가중으로 써보세요.")
print("5) 필요시 위 집계에서 성별/연령을 평균 대신 합(sum)으로 바꾸어 재확인하세요(상품군 성격에 따라 다를 수 있음).")

# 7. 학습/예측 구간 분리
TRAIN_START, TRAIN_END = "2022-01-01", "2024-12-31"
TEST_START,  TEST_END  = "2025-01-01", "2025-06-30"

train_search = search_feat[(search_feat["date"]>=TRAIN_START)&(search_feat["date"]<=TRAIN_END)]
train_click  = click_feat[(click_feat["date"] >=TRAIN_START)&(click_feat["date"] <=TRAIN_END)]

test_search  = search_feat[(search_feat["date"]>=TEST_START)&(search_feat["date"]<=TEST_END)]

# 2025년은 click 부재 → 제품별 MA3로 대체(간단 버전, 나중에 모델로 대체 가능)
# 1) 2024-10~12의 clicks_ma3 평균을 2025-01~06에 carry forward
last_click_ma3 = (click_feat[click_feat["date"]>= "2024-10-01"]
                  .groupby("product_name")["clicks_ma3"].mean().rename("click_cf_ma3"))
# 2) test_search에 머지해서 click 대체치 생성
test_click_stub = test_search[["product_name","date"]].merge(
    last_click_ma3, on="product_name", how="left"
).rename(columns={"click_cf_ma3":"clicks"})
test_click_stub["clicks_mom"] = 0.0  # 대체치라 증감률 0 처리(표식도 함께 둠)
test_click_stub["clicks_yoy"] = 0.0
test_click_stub["clicks_ma3"] = test_click_stub["clicks"]
test_click_stub["clicks_ma6"] = test_click_stub["clicks"]
test_click_stub["missing_click_flag"] = 1

# 학습 구간에는 missing_click_flag=0
train_click = train_click.copy()
train_click["missing_click_flag"] = 0

# 8. 제품별 lead-lag 탐색 (0~6개월 중 최대 상관)
import numpy as np

def best_lag_for_product(prod_df, max_lag=6):
    # prod_df: product_name 고정, 월별 search/click 평균(성별/연령 평균)
    d = prod_df.sort_values("date")
    best_lag, best_corr = 0, -1
    for lag in range(0, max_lag+1):
        tmp = d.copy()
        tmp["search_shift"] = tmp["search_index"].shift(lag)
        corr = tmp[["search_shift","clicks"]].corr().iloc[0,1]
        if pd.notna(corr) and corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag, best_corr

# 제품×월 평균으로 축약
search_pm = (search_feat.groupby(["product_name","date"], as_index=False)["search_index"].mean())
click_pm  = (click_feat.groupby(["product_name","date"], as_index=False)["clicks"].mean())
sc_pm = search_pm.merge(click_pm, on=["product_name","date"], how="inner")

lag_rows = []
for prod, g in sc_pm.groupby("product_name"):
    lag, corr = best_lag_for_product(g, max_lag=6)
    lag_rows.append({"product_name": prod, "best_lag": lag, "lag_corr": corr})
lag_table = pd.DataFrame(lag_rows).sort_values(["best_lag","lag_corr"], ascending=[True,False])
print("\n=== [제품별 최적 lag (개월), 그때의 상관] ===")
print(lag_table.to_string(index=False))

# 래그 테이블을 학습/예측 데이터에 적용할 때는, 제품별로 search_index_shift{lag} 생성
def apply_best_lag(df_search, lag_table):
    out = []
    for prod, g in df_search.groupby("product_name"):
        lag = lag_table.set_index("product_name").loc[prod, "best_lag"] if prod in lag_table["product_name"].values else 0
        gg = g.sort_values("date").copy()
        gg[f"search_index_lag{lag}"] = gg["search_index"].shift(lag)
        out.append(gg)
    return pd.concat(out, ignore_index=True)

train_search_lag = apply_best_lag(search_pm, lag_table)

# 9. 시즌성(월) 피처
def add_seasonal(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    # Fourier(간단): 1년 주기
    df["sin_m"] = np.sin(2*np.pi*df["month"]/12)
    df["cos_m"] = np.cos(2*np.pi*df["month"]/12)
    return df

# 학습용 통합
train_s = add_seasonal(search_pm)
train_c = add_seasonal(click_pm)
train = pd.merge(train_s, train_c, on=["product_name","date","month","sin_m","cos_m"], how="inner")

# 2025 예측용 통합 (search + stub click)
test_s  = add_seasonal(test_search.groupby(["product_name","date"], as_index=False)["search_index"].mean())
test_c  = add_seasonal(test_click_stub[["product_name","date","clicks"]])
test = pd.merge(test_s, test_c, on=["product_name","date","month","sin_m","cos_m"], how="left")
test["missing_click_flag"] = test.get("missing_click_flag", pd.Series(1, index=test.index)).fillna(1)

# 래그 피처(제품별 best_lag) 주입: 학습/예측 동일 규칙
train = train.merge(lag_table, on="product_name", how="left")
test  = test.merge(lag_table,  on="product_name", how="left")

# 제품별로 best_lag만큼 시프트된 search_lag 컬럼 생성
def add_best_lag_value(df, col="search_index"):
    out = []
    for prod, g in df.groupby("product_name"):
        lag = int(g["best_lag"].iloc[0]) if "best_lag" in g.columns and len(g) else 0
        gg = g.sort_values("date").copy()
        gg[f"{col}_lag{lag}"] = gg[col].shift(lag)
        out.append(gg)
    return pd.concat(out, ignore_index=True)

train = add_best_lag_value(train, "search_index")
test  = add_best_lag_value(test,  "search_index")

# 결측 처리(시프트로 생기는 NaN)
for df in (train, test):
    for col in ["search_index","clicks","search_index_lag0","search_index_lag1","search_index_lag2",
                "search_index_lag3","search_index_lag4","search_index_lag5","search_index_lag6"]:
        if col in df.columns:
            df[col] = df[col].fillna(method="ffill").fillna(0)

# 저장
import os
os.makedirs("./_save/features", exist_ok=True)
train.to_csv("./_save/features/train_features.csv", index=False, encoding="utf-8-sig")
test.to_csv("./_save/features/test_features_2025H1.csv", index=False, encoding="utf-8-sig")
print("✅ 저장 완료:",
      "./_save/features/train_features.csv, ./_save/features/test_features_2025H1.csv")
