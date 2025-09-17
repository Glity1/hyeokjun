import pandas as pd
from pathlib import Path

# ===== 경로 설정 =====
DATA_DIR = Path("./_data/dacon/dongwon/pos_data")
files = {
    "monthly": DATA_DIR / "marketlink_POS_master.xlsx",          # 월별
    "quarterly": DATA_DIR / "닐슨코리아_2011_2019_분기별마스터.xlsx", # 분기별
    "halfyear": DATA_DIR / "닐슨코리아_마스터_2020_2024.xlsx",     # 반기별
}

# ===== 데이터 로드 =====
def load_monthly(path):
    df = pd.read_excel(path)
    return df[["연도", "월", "구분", "카테고리", "매출액(백만원)"]]

def load_quarterly(path):
    df = pd.read_excel(path)
    return df[["연도", "분기", "구분", "카테고리", "매출액(백만원)"]]

def load_halfyear(path):
    df = pd.read_excel(path)
    return df[["연도", "반기", "구분", "카테고리", "매출액(백만원)"]]

monthly = load_monthly(files["monthly"])
quarterly = load_quarterly(files["quarterly"])
halfyear = load_halfyear(files["halfyear"])

# ===== 보정계수 산정 함수 =====
def calc_monthly_factors(df):
    """월별 데이터 보정계수 산출"""
    results = []
    for cat, g in df.groupby("카테고리"):
        base = g["매출액(백만원)"].mean()
        # 계절
        summer = g[g["월"].isin([6,7,8])]["매출액(백만원)"].mean() / base
        winter = g[g["월"].isin([12,1,2])]["매출액(백만원)"].mean() / base
        spring = g[g["월"].isin([3,4,5])]["매출액(백만원)"].mean() / base
        autumn = g[g["월"].isin([9,10,11])]["매출액(백만원)"].mean() / base
        # 명절 (연도별 변동 고려 없이 단순 월 기준)
        seollal = g[g["월"].isin([1,2])]["매출액(백만원)"].mean() / base
        chuseok = g[g["월"].isin([9,10])]["매출액(백만원)"].mean() / base
        # 전월 대비 평균 성장률 (MoM)
        g = g.sort_values(["연도","월"])
        mom = (g["매출액(백만원)"].pct_change()+1).mean()
        # 월별 패턴 (12개월)
        month_factors = {f"월{m}": g[g["월"]==m]["매출액(백만원)"].mean() / base for m in range(1,13)}

        row = {
            "카테고리": cat, "기준":"월별",
            "봄":spring, "여름":summer, "가을":autumn, "겨울":winter,
            "설날":seollal, "추석":chuseok,
            "MoM평균": mom
        }
        row.update(month_factors)
        results.append(row)
    return pd.DataFrame(results)

def calc_quarterly_factors(df):
    """분기별 데이터 보정계수 산출"""
    results = []
    for cat, g in df.groupby("카테고리"):
        base = g["매출액(백만원)"].mean()
        q_factors = {f"{q}Q": g[g["분기"]==q]["매출액(백만원)"].mean() / base for q in [1,2,3,4]}
        # 계절
        summer = q_factors.get("3Q", None)
        winter = (q_factors.get("1Q",0)+q_factors.get("4Q",0))/2
        seollal = q_factors.get("1Q", None)
        chuseok = q_factors.get("3Q", None)
        # 전분기 대비 성장률
        g = g.sort_values(["연도","분기"])
        qoq = (g["매출액(백만원)"].pct_change()+1).mean()
        # 연도별 YoY (동일 분기 대비 성장률)
        g["연도lag"] = g["연도"].shift(4)
        yoy = (g["매출액(백만원)"].pct_change(4)+1).mean()

        row = {"카테고리":cat, "기준":"분기별",
               "여름":summer, "겨울":winter, "설날":seollal, "추석":chuseok,
               "QoQ평균":qoq, "YoY평균":yoy}
        row.update(q_factors)
        results.append(row)
    return pd.DataFrame(results)

def calc_halfyear_factors(df):
    """반기별 데이터 보정계수 산출"""
    results = []
    for cat, g in df.groupby("카테고리"):
        base = g["매출액(백만원)"].mean()
        upper = g[g["반기"]=="상반기"]["매출액(백만원)"].mean() / base
        lower = g[g["반기"]=="하반기"]["매출액(백만원)"].mean() / base
        # 상/하반기 집중도
        year_total = g.groupby("연도")["매출액(백만원)"].sum()
        upper_share = (g[g["반기"]=="상반기"].groupby("연도")["매출액(백만원)"].sum() / year_total).mean()
        lower_share = (g[g["반기"]=="하반기"].groupby("연도")["매출액(백만원)"].sum() / year_total).mean()
        # 장기 성장률 (CAGR)
        years = g["연도"].dropna().unique()
        years.sort()
        if len(years) > 1:
            start = g[g["연도"]==years.min()]["매출액(백만원)"].mean()
            end = g[g["연도"]==years.max()]["매출액(백만원)"].mean()
            cagr = (end/start)**(1/(years.max()-years.min())) - 1
        else:
            cagr = None
        results.append({
            "카테고리":cat, "기준":"반기별",
            "상반기":upper, "하반기":lower,
            "상반기비중":upper_share, "하반기비중":lower_share,
            "CAGR":cagr
        })
    return pd.DataFrame(results)

# ===== 보정계수 계산 =====
monthly_cf = calc_monthly_factors(monthly)
quarterly_cf = calc_quarterly_factors(quarterly)
halfyear_cf = calc_halfyear_factors(halfyear)

# ===== 저장 =====
monthly_cf.to_excel(DATA_DIR / "보정계수_월별.xlsx", index=False)
quarterly_cf.to_excel(DATA_DIR / "보정계수_분기별.xlsx", index=False)
halfyear_cf.to_excel(DATA_DIR / "보정계수_반기별.xlsx", index=False)

print("✅ 보정계수 산정 완료")
print("월별:", monthly_cf.shape, "분기별:", quarterly_cf.shape, "반기별:", halfyear_cf.shape)
