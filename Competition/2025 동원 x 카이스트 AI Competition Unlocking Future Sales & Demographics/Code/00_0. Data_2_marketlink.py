import pandas as pd
from pathlib import Path

# =========================
# 0. 경로 정의
# =========================
DATA_DIR = Path("./_data/dacon/dongwon/pos_data")
files = {
    "조미료": DATA_DIR / "마켓링크 조미료_2020~2023_월별정리.xlsx",
    "발효유": DATA_DIR / "마켓링크 발효유_2020~2023_월별정리.xlsx",
    "전통기름": DATA_DIR / "마켓링크  traditional_oil_monthly_sales_2020_2023.xlsx",
    "조제커피": DATA_DIR / "마켓링크 커피_2020_2023_월별.xlsx",
    "어육가공품": DATA_DIR / "마켓링크 _어육가공품_2020_2023.xlsx",
}

# =========================
# 1. 공통 처리 함수
# =========================
def clean_long_format(df, category):
    """조미료, 발효유, 전통기름, 조제커피 처리용"""
    df = df.rename(columns={"연도": "year", "월": "month", "구분": "segment", "매출액(백만원)": "sales"})
    df = df[["year", "month", "segment", "sales"]].copy()
    df["category"] = category
    return df

def clean_wide_format(df, category):
    """어육가공품 처리용: wide → long 변환"""
    df = df.rename(columns={"구분": "segment"})
    df = df.melt(id_vars=["segment"], var_name="ym", value_name="sales")
    df["year"] = df["ym"].str[:4].astype(int)
    df["month"] = df["ym"].str[-2:].astype(int)
    df["category"] = category
    df = df[["year", "month", "segment", "sales", "category"]]
    return df

# =========================
# 2. 파일별 처리
# =========================
df_all = []

# 조미료
df_all.append(clean_long_format(pd.read_excel(files["조미료"]), "조미료"))

# 발효유
df_all.append(clean_long_format(pd.read_excel(files["발효유"]), "발효유"))

# 전통기름
df_all.append(clean_long_format(pd.read_excel(files["전통기름"]), "전통기름"))

# 조제커피
df_all.append(clean_long_format(pd.read_excel(files["조제커피"]), "조제커피"))

# 어육가공품 (wide → long)
df_all.append(clean_wide_format(pd.read_excel(files["어육가공품"]), "어육가공품"))

# =========================
# 3. 통합 마스터 테이블 생성
# =========================
master = pd.concat(df_all, ignore_index=True)

# 데이터 정리
master["sales"] = pd.to_numeric(master["sales"], errors="coerce").fillna(0)
master["segment"] = master["segment"].fillna("전체")

# =========================
# 4. 저장
# =========================
SAVE_PATH = "./_save/marketlink_POS_master.csv"
master.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

print("✅ Master POS Table Saved:", SAVE_PATH)
print(master.head(20))
