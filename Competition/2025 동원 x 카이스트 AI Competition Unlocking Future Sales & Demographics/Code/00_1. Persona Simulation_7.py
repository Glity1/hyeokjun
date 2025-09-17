import json
import pandas as pd

# -----------------------------
# 1. JSON 로드
# -----------------------------
with open("./_data/dacon/dongwon/personas.json", "r", encoding="utf-8") as f:
    personas = json.load(f)

# -----------------------------
# 2. 페르소나 기반 월별 수요량 집계
# -----------------------------
rows = []
for product, plist in personas.items():
    # 각 persona 마다 월별 구매 의향 스코어 × 확률 가중
    monthly = None
    for p in plist:
        weights = p.get("purchase_probability", 50) / 100.0
        mvals = pd.Series(p.get("monthly_by_launch", [0]*12))
        if monthly is None:
            monthly = mvals * weights
        else:
            monthly += mvals * weights

    # 제품 단위로 최종 합산
    row = {"product_name": product}
    for i in range(12):
        row[f"months_since_launch_{i+1}"] = round(monthly[i], 2)
    rows.append(row)

df_sim = pd.DataFrame(rows)

# -----------------------------
# 3. sample_submission.csv 로드
# -----------------------------
sample = pd.read_csv("./_data/dacon/dongwon/sample_submission.csv")

# -----------------------------
# 4. sample 스키마에 맞추기
# -----------------------------
# sample은 (product_name + 2024-07 ~ 2025-06) 형태라고 가정
date_cols = sample.columns[1:]  # 기간 컬럼

# df_sim의 months_since_launch → sample 날짜 매핑
rename_map = {f"months_since_launch_{i+1}": date_cols[i] for i in range(12)}
df_sim = df_sim.rename(columns=rename_map)

# sample과 merge하여 제출 포맷 완성
submit = sample[["product_name"]].merge(df_sim, on="product_name", how="left")

# -----------------------------
# 5. 값 정리 & 저장
# -----------------------------
for col in date_cols:
    submit[col] = submit[col].fillna(0).round().astype(int)

out_path = "./_save/submission_persona.csv"
submit.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"✅ 제출 파일 저장 완료: {out_path}")
print(submit.head())
