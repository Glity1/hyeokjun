import pandas as pd
import re, os
from glob import glob

keyword_map = {
    "요거트": ["덴마크 하이그릭요거트 400g"],
    "참치,연어": ["동원맛참 고소참기름 90g", "동원맛참 고소참기름 135g",
                "동원맛참 매콤참기름 90g", "동원맛참 매콤참기름 135g"],
    "액젓": ["동원참치액 순 500g", "동원참치액 순 900g",
           "동원참치액 진 500g", "동원참치액 진 900g",
           "프리미엄 동원참치액 500g", "프리미엄 동원참치액 900g"],
    "햄": ["리챔 오믈레햄 200g", "리챔 오믈레햄 340g"],
    "커피음료": ["소화가 잘되는 우유로 만든 카페라떼 250mL",
               "소화가 잘되는 우유로 만든 바닐라라떼 250mL"]
}

all_files = glob("./_data/dacon/dongwon/naver/클릭량 추이/**/*.csv", recursive=True)

dfs = []
for file in all_files:
    fname = os.path.basename(file).replace(".csv","")
    print("처리 중:", fname)

    # 파일명에서 성별, 연령대, 키워드 추출
    match = re.match(r"(남성|여성)\s+(10대|20대|30대|40대|50대|60대 이상)\s+(.+)\s+클릭량 추이", fname)
    if not match:
        print("⚠️ 정규식 매칭 실패:", fname)
        continue
    gender, age, keyword = match.groups()
    keyword = keyword.strip()

    # CSV 읽기 (UTF-8 + 메타 줄 skip)
    df = pd.read_csv(file, encoding="utf-8-sig", skiprows=7)  # ✅ skiprows 수정
    df = df.iloc[:, :2]    # 앞의 두 컬럼만 사용
    df.columns = ["date", "clicks"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 매핑 적용
    rows = []
    if keyword in keyword_map:
        for prod in keyword_map[keyword]:
            for _, row in df.iterrows():
                if pd.notna(row["date"]):
                    rows.append({
                        "date": row["date"],
                        "gender": gender,
                        "age": age,
                        "keyword": keyword,
                        "clicks": row["clicks"],
                        "product_name": prod
                    })
    if rows:
        dfs.append(pd.DataFrame(rows))

click_all_mapped = pd.concat(dfs, ignore_index=True)

os.makedirs("./_save", exist_ok=True)
click_all_mapped.to_csv("./_save/click_trend_all.csv", index=False, encoding="utf-8-sig")

print("✅ 저장 완료: ./_save/click_trend_all.csv")
