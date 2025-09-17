import pandas as pd
import re, os
from glob import glob

# ========================
# 1. 제품 매핑 딕셔너리
# ========================
keyword_map = {
    "발효유": ["덴마크 하이그릭요거트 400g"],
    "참치": ["동원맛참 고소참기름 90g", "동원맛참 고소참기름 135g",
                "동원맛참 매콤참기름 90g", "동원맛참 매콤참기름 135g"],
    "조미료": ["동원참치액 순 500g", "동원참치액 순 900g",
           "동원참치액 진 500g", "동원참치액 진 900g",
           "프리미엄 동원참치액 500g", "프리미엄 동원참치액 900g"],
    "식육가공": ["리챔 오믈레햄 200g", "리챔 오믈레햄 340g"],
    "조제커피": ["소화가 잘되는 우유로 만든 카페라떼 250mL",
               "소화가 잘되는 우유로 만든 바닐라라떼 250mL"]
}

# ========================
# 2. 파일 불러오기
# ========================
all_files = glob("./_data/dacon/dongwon/naver/검색어 트랜드/*.xlsx")

dfs = []
for file in all_files:
    fname = os.path.basename(file).replace(".xlsx","")
    print("처리 중:", fname)

    # PC/모바일, 성별, 연령대 추출
    match = re.match(r"(PC|모바일)_(남성|여성)_(\d+대|60대 이상)", fname)
    if not match:
        print("⚠️ 정규식 매칭 실패:", fname)
        continue
    device, gender, age = match.groups()

    # 데이터 읽기 (상단 메타정보 스킵)
    df = pd.read_excel(file, skiprows=6)

    # '날짜' 컬럼 정리 (첫 번째만 사용, 중복 제거)
    date_col = df.columns[0]
    keep_cols = [date_col] + [c for c in df.columns if "날짜" not in str(c)]
    df = df[keep_cols].rename(columns={date_col:"date"})
    df["date"] = pd.to_datetime(df["date"])

    # wide → long 변환
    df_long = df.melt(id_vars=["date"], var_name="keyword", value_name="search_index")

    # keyword 정규화
    df_long["keyword"] = df_long["keyword"].str.strip()

    # 메타 추가
    df_long["device"], df_long["gender"], df_long["age"] = device, gender, age

    # keyword → product_name 매핑
    df_mapped = []
    for _, row in df_long.iterrows():
        if row["keyword"] in keyword_map:
            for prod in keyword_map[row["keyword"]]:
                df_mapped.append({
                    "date": row["date"],
                    "device": row["device"],
                    "gender": row["gender"],
                    "age": row["age"],
                    "keyword": row["keyword"],
                    "search_index": row["search_index"],
                    "product_name": prod
                })
    if df_mapped:
        dfs.append(pd.DataFrame(df_mapped))

# ========================
# 3. 통합 및 정리
# ========================
search_all_mapped = pd.concat(dfs, ignore_index=True)

# PC/모바일 통합 (평균)
search_all_mapped = (
    search_all_mapped
    .groupby(["date","gender","age","keyword","product_name"], as_index=False)["search_index"]
    .mean()
)

# ========================
# 4. 저장
# ========================
os.makedirs("./_save", exist_ok=True)
search_all_mapped.to_csv("./_save/search_trend_all.csv", index=False, encoding="utf-8-sig")

print("✅ 저장 완료: ./_save/search_trend_all.csv")
