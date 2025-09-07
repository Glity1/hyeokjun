# analyze_labels.py
# ---------------------------------------------------------------------
# 사용법(예시):
#   python analyze_labels.py --csv samples.csv --outdir out --topn 20
#   (축 하나만)  python analyze_labels.py --csv samples.csv --axis 유형 --topn 30
#   (불용어 끄기) python analyze_labels.py --csv samples.csv --no_stopwords
#
# 기능:
# 1) output 컬럼(예: "사실형,긍정,현재,확실")을 4축(유형/극성/시제/확실성)으로 분해
# 2) 축별-라벨별 상위 단어 Top-N 집계 → CSV 저장(top_words_*.csv)
# 3) '유형' 축에 대해 데이터 기반 판별 트리거(로그오즈) 산출 →
#    type_triggers.json / type_triggers_prompt.txt 저장
# ---------------------------------------------------------------------

import argparse
import os
import re
import json
from collections import Counter, defaultdict
from math import log
import pandas as pd

# 4축 인덱스 매핑 (output: "유형,극성,시제,확실성")
AXIS_INDEX = {"유형": 0, "극성": 1, "시제": 2, "확실성": 3}
ALL_AXES = ["유형", "극성", "시제", "확실성"]

# 불용어(조사/접속/기능어 중심). 필요 시 --no_stopwords 로 비활성화 가능
STOPWORDS = {
    "은","는","이","가","을","를","에","에서","으로","로","에게","께","와","과","및","또","또는","그리고","하지만","그러나",
    "것","수","등","중","또한","관련","대한","대해","위해","통해","부터","까지","보다","처럼","으로써","으로서","으로부터",
    "그","그녀","그들","이런","그런","저런","우리","여러","각","모든","어떤","어느","각각","해당","이는",
    "지난","이번","현재","오늘","어제","내일","최근","올해","내년","내달",
    "있다","없다","이다","아니다","합니다","한다","했다","하며","하고","되어","된다","됐다","된","되는","되어","였다",
    "이라며","이라면서","라고","했다며","관계자","측","측은","관계자는",
    "명","건","개","차","호","년","월","일","분","시","개월","시간","있는","것으로","같은", "것이다", "것이", "국내", "위한", "등을", "게임", 
    # 영어 기능어
    "the","and","or","of","to","in","for","on","with","at","by","from","as","is","are","was","were","be","been",
}

def extract_axis_label(output_str: str, axis: str) -> str:
    """output(예: '사실형,긍정,현재,확실')에서 지정 축 라벨 추출"""
    if not isinstance(output_str, str):
        return ""
    parts = [p.strip() for p in output_str.split(",")]
    idx = AXIS_INDEX[axis]
    return parts[idx] if idx < len(parts) else ""

def tokenize_ko(text: str):
    """한글/영문/숫자 시퀀스를 토큰화(영문은 소문자), 숫자/1글자 토큰은 이후 필터링"""
    if not isinstance(text, str):
        return []
    toks = re.findall(r"[가-힣A-Za-z0-9]+", text)
    toks = [t.lower() for t in toks]
    return toks

def top_words_by_label(df: pd.DataFrame, text_col: str, label_col: str, axis: str,
                       topn: int, min_count: int, use_stopwords: bool):
    """특정 축에 대해 라벨별 상위 단어 Top-N 계산"""
    stop = set() if not use_stopwords else set(STOPWORDS)
    df = df.copy()
    df["__AXIS__"] = df[label_col].apply(lambda s: extract_axis_label(s, axis))

    per_label_counts: dict[str, Counter] = defaultdict(Counter)
    for text, lab in zip(df[text_col].astype(str), df["__AXIS__"]):
        toks = tokenize_ko(text)
        for tok in toks:
            if tok.isdigit():      # 숫자만 = 제거
                continue
            if len(tok) < 2:       # 1글자 제거
                continue
            if tok in stop:        # 불용어 제거
                continue
            per_label_counts[lab][tok] += 1

    rows = []
    for lab, counter in per_label_counts.items():
        items = [(w, c) for w, c in counter.items() if c >= min_count]
        items.sort(key=lambda x: (-x[1], x[0]))
        top = items[:topn]
        for rank, (w, c) in enumerate(top, 1):
            rows.append({"축": axis, "라벨": lab, "순위": rank, "단어": w, "빈도": c})

    out_df = pd.DataFrame(rows).sort_values(["라벨", "순위"])
    # 콘솔 요약(Top-1)
    print(f"\n=== [{axis}] 라벨별 최다 단어(Top-1) ===")
    for lab in sorted(per_label_counts.keys()):
        items = [(w, c) for w, c in per_label_counts[lab].items() if c >= min_count]
        if items:
            items.sort(key=lambda x: (-x[1], x[0]))
            w, c = items[0]
            print(f"[{lab}] {w} ({c})")
        else:
            print(f"[{lab}] (없음)")
    return out_df, per_label_counts

# -------- '유형' 축: 데이터 기반 트리거(로그오즈) --------

def build_type_triggers_from_counts(per_label_counts: dict[str, Counter],
                                    alpha0: float = 0.1,
                                    min_count: int = 10,
                                    max_terms: int = 30):
    """
    '유형' 축 전용: 라벨별 단어 카운트에서 로그오즈(정보량) 점수로 판별 트리거 Top-N 추출
    (Monroe et al., 2008 Dirichlet prior 사용)
    """
    classes = sorted(per_label_counts.keys())
    total_counts = Counter()
    for c in classes:
        total_counts.update(per_label_counts[c])

    totals_per_class = {c: sum(per_label_counts[c].values()) for c in classes}
    total_all = sum(totals_per_class.values())
    corpus_sum = sum(total_counts.values())

    def log_odds(word, target):
        y_w = per_label_counts[target][word]
        y_rest = total_counts[word] - y_w
        n_y = totals_per_class[target]
        n_rest = total_all - n_y
        p_w = total_counts[word] / corpus_sum if corpus_sum else 0.0
        a_w = alpha0 * p_w
        A = alpha0  # sum of alphas = alpha0
        num = (y_w + a_w) / max(1e-12, (n_y + A - (y_w + a_w)))
        den = (y_rest + a_w) / max(1e-12, (n_rest + A - (y_rest + a_w)))
        return log(num) - log(den)

    triggers = {}
    scored_view = defaultdict(list)
    for c in classes:
        scored = []
        for w, cnt in per_label_counts[c].items():
            if cnt < min_count:
                continue
            # 영어 1~2자 토큰 제거(잡음)
            if re.fullmatch(r"[A-Za-z]+", w) and len(w) <= 2:
                continue
            s = log_odds(w, c)
            scored.append((w, cnt, s))
        scored.sort(key=lambda x: (x[2], x[1]), reverse=True)
        top = scored[:max_terms]
        triggers[c] = [w for (w, cnt, s) in top]
        for w, cnt, s in top:
            scored_view[c].append({"단어": w, "빈도": cnt, "점수": round(s, 4)})

    return triggers, scored_view

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./_data/dacon/KT/samples.csv", help="입력 CSV 경로(기본: samples.csv)")
    ap.add_argument("--text_col", default="user_prompt", help="문장 텍스트 컬럼명")
    ap.add_argument("--label_col", default="output", help="라벨(4축) 컬럼명")
    ap.add_argument("--axis", default="all", choices=["all"] + ALL_AXES, help="대상 축(all/유형/극성/시제/확실성)")
    ap.add_argument("--topn", type=int, default=20, help="라벨별 상위 N 단어")
    ap.add_argument("--min_count", type=int, default=3, help="최소 등장 빈도(미만 제거)")
    ap.add_argument("--no_stopwords", action="store_true", help="불용어 제거 비활성화")
    ap.add_argument("--outdir", default="out", help="결과 저장 폴더")
    # 유형 트리거 옵션
    ap.add_argument("--type_min_count", type=int, default=10, help="유형 트리거 최소 빈도")
    ap.add_argument("--type_max_terms", type=int, default=30, help="유형 트리거 최대 개수")
    ap.add_argument("--type_alpha0", type=float, default=0.1, help="유형 트리거 Dirichlet prior 크기")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    target_axes = ALL_AXES if args.axis == "all" else [args.axis]

    # 축별 Top 단어 산출 및 저장
    saved = []
    per_axis_counts = {}  # '유형' 트리거 계산용으로 counts 보관
    for axis in target_axes:
        out_df, counts = top_words_by_label(
            df, args.text_col, args.label_col,
            axis, args.topn, args.min_count,
            use_stopwords=(not args.no_stopwords)
        )
        csv_path = os.path.join(args.outdir, f"top_words_{axis}.csv")
        out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        saved.append(csv_path)
        per_axis_counts[axis] = counts

    # '유형' 축 트리거(로그오즈) 파일 저장
    if "유형" in target_axes:
        triggers, scored = build_type_triggers_from_counts(
            per_axis_counts["유형"],
            alpha0=args.type_alpha0,
            min_count=args.type_min_count,
            max_terms=args.type_max_terms
        )
        trig_json = os.path.join(args.outdir, "type_triggers.json")
        with open(trig_json, "w", encoding="utf-8") as f:
            json.dump(triggers, f, ensure_ascii=False, indent=2)

        # 프롬프트용 스니펫
        lines = ["[유형 트리거(데이터 기반)]"]
        for c in sorted(triggers.keys()):
            lines.append(f"- {c}: " + " | ".join(triggers[c]))
        trig_txt = os.path.join(args.outdir, "type_triggers_prompt.txt")
        with open(trig_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        saved.extend([trig_json, trig_txt])

    # 저장 요약
    print("\n저장 파일:")
    for p in saved:
        print(" -", p)

if __name__ == "__main__":
    main()
