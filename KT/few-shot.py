# make_few_shot_from_csv.py
# 사용법:
#   python make_few_shot_from_csv.py --csv samples.csv --axis 유형 --k 2 --out fewshot.txt --seed 42
# 축(axis): 유형/극성/시제/확실성 중 하나. 출력은 해당 행의 정답(output)을 그대로 사용.
import argparse, re, random, pandas as pd
from collections import defaultdict

AXIS_INDEX = {"유형":0, "극성":1, "시제":2, "확실성":3}

# 라벨별 트리거(가중치용; 매칭 많이 되는 문장 우선)
TRIGGERS = {
    "유형": {
        "대화형": [r"\?", r"(하세요|해 주세요|줄래|할까|문의|신청|참여|습니까|가능할까요)"],
        "예측형": [r"(예정|계획|전망|될\s*것|향후|내달|내년|곧|겠[다]|[으이]ㄹ\s*예정)"],
        "추론형": [r"(가능성|듯|같다|추정|의혹|논의|검토|보인다|전해졌다|관측|평가)"],
        "사실형": [r"(발표했|확인됐|공시|공개됐|집계|현황|수치|우승|체결|출시|개최|발생|완료)"]
    },
    "극성": {
        "긍정": [r"(증가|상승|개선|성공|달성|승인|체결|출시|우승|흑자|회복|반등|전환)"],
        "부정": [r"(감소|하락|악화|실패|취소|지연|연기|중단|폐지|위반|징계|적자|사고|감염|확진\s*급증)"],
        "미정": []
    },
    "시제": {
        "과거": [r"(했[다]|였다|확인됐|발표했|체결했|우승했|완료했|지난|어제|전날|방금)"],
        "현재": [r"(이다|있다|중|하고\s*있|진행\s*중|보유|운영|개최\s*중)"],
        "미래": [r"(하겠|예정|전망|계획|추진|될\s*것|출시한다|개최한다|착수한다|도입한다|확대한다)"]
    },
    "확실성": {
        "확실": [r"(확정|공식|수치|%|억원|명|건|발표했|밝혔다|공시|집계|단정)"],
        "불확실": [r"(가능성|전망|예상|추정|가정|논의|검토|의혹|전해졌[다]|보인다|\?)"]
    }
}

def extract_axis_label(output_str: str, axis: str) -> str:
    parts = [p.strip() for p in str(output_str).split(",")]
    idx = AXIS_INDEX[axis]
    return parts[idx] if idx < len(parts) else ""

def score_by_triggers(text: str, axis: str, label: str) -> int:
    pats = TRIGGERS.get(axis, {}).get(label, [])
    return sum(len(re.findall(p, text)) for p in pats)

def build_fewshot(df: pd.DataFrame, axis: str, k: int, seed: int):
    random.seed(seed)
    # 해당 축 라벨 컬럼 만들기
    df = df.copy()
    df["__axis__"] = df["output"].apply(lambda s: extract_axis_label(s, axis))
    # 라벨별 후보 수집
    buckets = defaultdict(list)
    for i, row in df.iterrows():
        text = str(row["user_prompt"]).strip().replace("\n"," ").replace("\r"," ")
        label_full = str(row["output"]).strip()
        label_axis = row["__axis__"]
        if not text or not label_axis: 
            continue
        score = score_by_triggers(text, axis, label_axis)
        length_bonus = min(len(text), 140) / 140.0  # 너무 짧거나 너무 길면 페널티 완화
        buckets[label_axis].append((score + 0.25*length_bonus, text, label_full))
    # 라벨별 정렬 후 상위 k개(동률 시 랜덤섞기)
    few = []
    for lab, items in buckets.items():
        random.shuffle(items)
        items.sort(key=lambda x: x[0], reverse=True)
        few.extend([(lab, t, y) for _, t, y in items[:k]])
    # 라벨 그룹 순서: 라벨명 오름차순
    few.sort(key=lambda x: x[0])
    # 포맷팅
    lines = ["[Few-shot]"]
    for idx, (_, text, label_full) in enumerate(few, 1):
        lines.append(f"입력: {idx}) {text}")
        lines.append(f"출력: {idx}.{label_full}")
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--axis", default="유형", choices=list(AXIS_INDEX.keys()))
    p.add_argument("--k", type=int, default=2, help="라벨별 추출 개수")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="fewshot.txt")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if "user_prompt" not in df.columns or "output" not in df.columns:
        raise ValueError("CSV에 'user_prompt'와 'output' 컬럼이 필요합니다.")

    fewshot = build_fewshot(df, axis=args.axis, k=args.k, seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(fewshot + "\n")
    print(f"Few-shot 저장: {args.out}\n")
    print(fewshot)

if __name__ == "__main__":
    main()
