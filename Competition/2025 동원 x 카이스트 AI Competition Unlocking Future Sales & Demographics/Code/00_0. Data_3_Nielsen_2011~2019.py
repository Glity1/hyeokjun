import pandas as pd
from pathlib import Path

# =========================
# 1. 경로 설정
# =========================
DATA_DIR = Path("./_data/dacon/dongwon/pos_data/개별파일/닐슨코리아 분기별 2011~2019")
SAVE_DIR = Path("./_data/dacon/dongwon/pos_data")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 파일 경로 정의
# =========================
files = {
    "어육가공품": DATA_DIR / "닐슨코리아_어육가공품_매출_2011_2019.xlsx",
    "전통기름": DATA_DIR / "닐슨코리아 traditional_oil_sales_2011_2019.xlsx",
    "발효유": DATA_DIR / "닐슨코리아 발효유 quarterly_sales_2011_2019.xlsx",
    "조미료": DATA_DIR / "닐슨코리아 조미료_2011_2019.xlsx",
    "커피": DATA_DIR / "닐슨코리아 커피_2011_2019_.xlsx"
}

dfs = []

# =========================
# 3. 어육가공품 (wide → long 변환)
# =========================
df_fish = pd.read_excel(files["어육가공품"])

df_fish_long = df_fish.melt(
    id_vars=[df_fish.columns[0]],  # 보통 "구분"
    var_name="기간", 
    value_name="매출액(백만원)"
)

df_fish_long["연도"] = df_fish_long["기간"].str.extract(r"(\d{4})년")
df_fish_long["분기"] = df_fish_long["기간"].str.extract(r"(\d)분기")
df_fish_long["카테고리"] = "어육가공품"
df_fish_long.rename(columns={df_fish.columns[0]: "구분"}, inplace=True)

dfs.append(df_fish_long[["카테고리", "연도", "분기", "구분", "매출액(백만원)"]])

# =========================
# 4. 나머지 파일들
# =========================
for cat, path in files.items():
    if cat == "어육가공품":
        continue
    df = pd.read_excel(path)
    df["카테고리"] = cat
    dfs.append(df[["카테고리", "연도", "분기", "구분", "매출액(백만원)"]])

# =========================
# 5. 최종 마스터 테이블 생성
# =========================
master_quarterly = pd.concat(dfs, ignore_index=True)

# 연도는 숫자로 변환
master_quarterly["연도"] = master_quarterly["연도"].astype(int)

# 분기: 'Q1', 'Q2', 'Q3', 'Q4' → 1,2,3,4 로 변환
master_quarterly["분기"] = (
    master_quarterly["분기"]
    .astype(str)                       # 문자열로 변환
    .str.replace("Q", "", regex=False) # 'Q' 제거
    .astype(int)                       # 숫자 변환
)

# 최종 저장
save_path = SAVE_DIR / "닐슨코리아_2011_2019_분기별마스터.xlsx"
master_quarterly.to_excel(save_path, index=False)

print(f"✅ 마스터 테이블 저장 완료: {save_path}")
print(master_quarterly.head())

