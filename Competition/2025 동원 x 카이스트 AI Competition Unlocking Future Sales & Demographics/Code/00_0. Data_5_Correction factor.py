import pandas as pd
from pathlib import Path

# ===== 경로 설정 =====
DATA_DIR = Path("./_data/dacon/dongwon/pos_data")
files = {
    "monthly": DATA_DIR / "marketlink_POS_master.xlsx",          
    "quarterly": DATA_DIR / "닐슨코리아_2011_2019_분기별마스터.xlsx", 
    "halfyear": DATA_DIR / "닐슨코리아_마스터_2020_2024.xlsx",     
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

# ===== 공통 유틸 =====
def add_extra_factors(df, group_cols, value_col="매출액(백만원)"):
    """추가 보정계수 산정 (연도 성장률, 변동성, 이벤트효과, 피크대비비율)"""
    results = []
    for cat, g in df.groupby("카테고리"):
        base = g[value_col].mean()

        # 연도별 성장률 (CAGR)
        yearly = g.groupby("연도")[value_col].mean()
        if len(yearly) > 1:
            cagr = (yearly.iloc[-1] / yearly.iloc[0]) ** (1 / (len(yearly)-1)) - 1
        else:
            cagr = 0

        # 변동성 (표준편차 / 평균)
        volatility = g[value_col].std() / base if base > 0 else 0

        # 이벤트 효과 (예: 1월/9월 기준)
        if "월" in g.columns:
            seollal_effect = g[g["월"].isin([1,2])][value_col].mean() / g[g["월"].isin([3,4])][value_col].mean()
            chuseok_effect = g[g["월"]==9][value_col].mean() / g[g["월"].isin([8,10])][value_col].mean()
        elif "분기" in g.columns:
            seollal_effect = g[g["분기"]==1][value_col].mean() / g[g["분기"]==2][value_col].mean()
            chuseok_effect = g[g["분기"]==3][value_col].mean() / g[g["분기"]==4][value_col].mean()
        elif "반기" in g.columns:
            seollal_effect = g[g["반기"]=="상반기"][value_col].mean() / g[g["반기"]=="하반기"][value_col].mean()
            chuseok_effect = g[g["반기"]=="하반기"][value_col].mean() / g[g["반기"]=="상반기"][value_col].mean()
        else:
            seollal_effect, chuseok_effect = None, None

        # 피크 대비 최저치
        peak_to_trough = g[value_col].max() / g[value_col].min() if g[value_col].min() > 0 else None

        results.append([cat, cagr, volatility, seollal_effect, chuseok_effect, peak_to_trough])

    return pd.DataFrame(results, columns=["카테고리","연도성장률","변동성","설날효과","추석효과","최고/최저비율"])

# ===== 기존 보정계수 + 확장 =====
monthly_cf = add_extra_factors(monthly, ["연도","월"])
quarterly_cf = add_extra_factors(quarterly, ["연도","분기"])
halfyear_cf = add_extra_factors(halfyear, ["연도","반기"])

# ===== 통합 결과 =====
correction_factors = pd.concat([
    monthly_cf.assign(기준="월별"),
    quarterly_cf.assign(기준="분기별"),
    halfyear_cf.assign(기준="반기별")
], ignore_index=True)

# ===== 저장 =====
out_path = DATA_DIR / "보정계수_확장산정결과.xlsx"
correction_factors.to_excel(out_path, index=False)

print("✅ 확장 보정계수 산정 완료:", out_path)
print(correction_factors.head(10))
