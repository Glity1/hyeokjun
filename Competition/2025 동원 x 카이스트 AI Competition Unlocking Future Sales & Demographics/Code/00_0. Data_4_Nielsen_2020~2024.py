import pandas as pd
from pathlib import Path

# ===== 경로 설정 =====
DATA_DIR = Path("./_data/dacon/dongwon/pos_data/개별파일/닐슨코리아 분기별 2020~2024")
SAVE_DIR = Path("./_data/dacon/dongwon/pos_data")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ===== 파일 경로 =====
files = {
    "traditional_oil": DATA_DIR / "닐슨코리아 traditional_oil_sales_2020_2024.xlsx",
    "fermented_milk": DATA_DIR / "닐슨코리아 발효유 분기별 2020~2024 데이터 정리.xlsx",
    "seasoning": DATA_DIR / "닐슨코리아 조미료 분기별 2020~2024 데이터 .xlsx",
    "coffee": DATA_DIR / "닐슨코리아 커피_2020_2024.xlsx",
    "fish_processed": DATA_DIR / "닐슨코리아_어육가공품_매출_2020_2024.xlsx",
}

# ===== 공통 컬럼명 정리 =====
def standardize_columns(df):
    rename_map = {
        "매출액 (단위 : 백만원)": "매출액",
        "매출액(백만원)": "매출액",
    }
    df = df.rename(columns=rename_map)
    # 중복 컬럼 제거
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# ===== 개별 로더 =====
def load_traditional_oil(path):
    df = pd.read_excel(path)
    df = standardize_columns(df)
    print("📌 traditional_oil 컬럼:", df.columns.tolist())
    return df[["연도", "반기", "구분", "카테고리", "매출액"]]

def load_fermented_milk(path):
    df = pd.read_excel(path)
    df = standardize_columns(df)
    print("📌 fermented_milk 컬럼:", df.columns.tolist())
    return df[["연도", "반기", "구분", "카테고리", "매출액"]]

def load_seasoning(path):
    df = pd.read_excel(path)
    df = standardize_columns(df)
    print("📌 seasoning 컬럼:", df.columns.tolist())
    return df[["연도", "반기", "구분", "카테고리", "매출액"]]

def load_coffee(path):
    df = pd.read_excel(path)
    df = standardize_columns(df)
    print("📌 coffee 컬럼:", df.columns.tolist())
    df = df[["연도", "반기", "구분", "매출액"]]
    df["카테고리"] = "커피"
    return df[["연도", "반기", "구분", "카테고리", "매출액"]]

def load_fish_processed(path):
    df = pd.read_excel(path)
    df = standardize_columns(df)
    print("📌 fish_processed 컬럼:", df.columns.tolist())
    # Wide → Long 변환
    df = df.melt(id_vars=["구분"], var_name="기간", value_name="매출액")
    # "2020년 상반기" → "2020", "상반기"
    df["연도"] = df["기간"].str.extract(r"(\d{4})년")
    df["반기"] = df["기간"].str.extract(r"(상반기|하반기)")
    df["카테고리"] = "어육가공품"
    return df[["연도", "반기", "구분", "카테고리", "매출액"]]

# ===== 데이터 통합 =====
dfs = [
    load_traditional_oil(files["traditional_oil"]),
    load_fermented_milk(files["fermented_milk"]),
    load_seasoning(files["seasoning"]),
    load_coffee(files["coffee"]),
    load_fish_processed(files["fish_processed"]),
]

master = pd.concat(dfs, ignore_index=True)

# ===== 저장 =====
out_path = SAVE_DIR / "닐슨코리아_마스터_2020_2024.xlsx"
master.to_excel(out_path, index=False)

print("✅ 마스터 테이블 저장 완료:", out_path)
print(master.head(10))
