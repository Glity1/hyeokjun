# -*- coding: utf-8 -*-
"""
트리거 사전 → 프롬프트 텍스트 생성 + 간단 룰베이스 평가

필요 파일:
- samples.csv (columns: user_prompt, output="유형,극성,시제,확실성")
- strong_triggers_by_axis.json
- good_cross_axis.json
- bad_confusing_tokens.json

산출물:
- prompt_candidate.txt (프롬프트에 붙여쓸 트리거 테이블)
- eval_report.txt (각 축 정확도 및 혼동행렬 요약)
"""

import json
import re
import pandas as pd
from collections import defaultdict, Counter

# ===== 경로 =====
SAMPLES = "./_data/dacon/KT/samples.csv"
STRONG_JSON = "strong_triggers_by_axis.json"
CROSS_JSON  = "good_cross_axis.json"
BAD_JSON    = "bad_confusing_tokens.json"

PROMPT_TXT  = "prompt_candidate.txt"
EVAL_TXT    = "eval_report.txt"

# ===== 라벨 우선순위(동률시) =====
TYPE_PRIORITY    = ["대화형", "예측형", "추론형", "사실형"]
POLARITY_PRIOR   = ["부정", "긍정", "미정"]          # 부정 우선
TENSE_PRIOR      = ["과거", "미래", "현재"]
CERTAINTY_PRIOR  = ["불확실", "확실"]               # 불확실 우선 (신호 있으면)

# ===== 토큰 매칭 =====
def find_hits(text, tokens):
    """토큰 리스트 중 본문에 등장하는 개수를 센다 (간단 substring 기반)"""
    cnt = 0
    for t in tokens:
        if t and t in text:
            cnt += 1
    return cnt

def load_data():
    df = pd.read_csv(SAMPLES, encoding="utf-8-sig")
    df[["유형","극성","시제","확실성"]] = df["output"].str.split(",", expand=True)
    with open(STRONG_JSON, "r", encoding="utf-8") as f:
        strong = json.load(f)
    with open(CROSS_JSON, "r", encoding="utf-8") as f:
        cross = json.load(f)
    with open(BAD_JSON, "r", encoding="utf-8") as f:
        bad = json.load(f)
    return df, strong, cross, bad

def build_prompt_text(strong, cross, bad, top_k_per_label=60):
    """프롬프트에 붙여쓸 트리거 테이블 생성(슬림/정돈)"""
    def join_top(lst, k):
        return ", ".join(lst[:k]) if lst else "(없음)"

    lines = []
    lines.append("## 트리거 표 (데이터 기반 최소 세트)")
    for axis in ["유형","극성","시제","확실성"]:
        lines.append(f"\n[{axis}]")
        for label, toks in strong[axis].items():
            lines.append(f"- {label}: {join_top(toks, top_k_per_label)}")
    # 축 간 긍정 신호
    lines.append("\n[축 간 다중 신호 토큰 (좋은 단어)]")
    lines.append("아래 토큰은 여러 축에서 동시에 강한 트리거로 잡혀, 해당 축들에 동시 가중치를 부여:")
    demo = []
    for i, (tok, lst) in enumerate(cross.items()):
        if i >= 40: break  # 프롬프트 과밀 방지: 예시 40개만
        axes = ", ".join(f"{d['axis']}:{d['label']}" for d in lst)
        demo.append(f"{tok}({axes})")
    lines.append("예: " + (", ".join(demo) if demo else "(없음)"))

    # 같은 축 내 혼동 토큰(나쁜 단어) 안내
    lines.append("\n[같은 축 내 혼동 토큰 (나쁜 단어)]")
    lines.append("다음 토큰은 같은 축 안에서 라벨 간 마진이 낮음 → 분류 신뢰도 저하 시 감점/패널티 권장.")
    for axis, rows in bad.items():
        ex = ", ".join(r["token"] for r in rows[:30]) if rows else "(없음)"
        lines.append(f"- {axis}: {ex}")
    return "\n".join(lines)

def choose_with_priority(scores: dict, prior: list, default_label: str):
    """동률시 우선순위로 선택. 모두 0이면 기본 라벨."""
    if all(v == 0 for v in scores.values()):
        return default_label
    maxv = max(scores.values())
    cands = [k for k, v in scores.items() if v == maxv]
    for p in prior:
        if p in cands:
            return p
    return cands[0]

def predict_row(text, strong, cross, bad):
    """간단 룰베이스 예측기: 강 트리거 가점 + 교차축 가점 + 혼동 토큰 감점"""
    # 기본값
    pred = {"유형": None, "극성": None, "시제": None, "확실성": None}

    # 스코어 테이블
    type_scores   = {k:0.0 for k in strong["유형"].keys()}
    polar_scores  = {k:0.0 for k in strong["극성"].keys()}
    tense_scores  = {k:0.0 for k in strong["시제"].keys()}
    cert_scores   = {k:0.0 for k in strong["확실성"].keys()}

    # 1) 강 트리거 점수 (가중치 1.0)
    for lab, toks in strong["유형"].items():
        type_scores[lab] += find_hits(text, toks) * 1.0
    for lab, toks in strong["극성"].items():
        polar_scores[lab] += find_hits(text, toks) * 1.0
    for lab, toks in strong["시제"].items():
        tense_scores[lab] += find_hits(text, toks) * 1.0
    for lab, toks in strong["확실성"].items():
        cert_scores[lab] += find_hits(text, toks) * 1.0

    # 2) 축 간 겹침(좋은 단어) 보너스 (가중치 0.5)
    for tok, axlab in cross.items():
        if tok in text:
            for d in axlab:
                ax, lab = d["axis"], d["label"]
                if ax == "유형":
                    type_scores[lab]  += 0.5
                elif ax == "극성":
                    polar_scores[lab] += 0.5
                elif ax == "시제":
                    tense_scores[lab] += 0.5
                elif ax == "확실성":
                    cert_scores[lab]  += 0.5

    # 3) 같은 축 내 혼동(나쁜 단어) 패널티 (가중치 -0.25)
    for axis, rows in bad.items():
        for r in rows:
            tok = r["token"]
            if tok and tok in text:
                if axis == "유형":
                    type_scores[r["best_label"]] -= 0.25
                elif axis == "극성":
                    polar_scores[r["best_label"]] -= 0.25
                elif axis == "시제":
                    tense_scores[r["best_label"]] -= 0.25
                elif axis == "확실성":
                    cert_scores[r["best_label"]]  -= 0.25

    # 4) 최종 선택 (동률 우선순위 적용 + 기본값)
    pred["유형"]   = choose_with_priority(type_scores,  TYPE_PRIORITY,   "사실형")
    pred["극성"]   = choose_with_priority(polar_scores, POLARITY_PRIOR,  "미정")
    pred["시제"]   = choose_with_priority(tense_scores, TENSE_PRIOR,     "현재")
    pred["확실성"] = choose_with_priority(cert_scores,  CERTAINTY_PRIOR, "확실")

    return pred

def evaluate(df, strong, cross, bad, sample_size=None):
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    golds = df[["유형","극성","시제","확실성"]].copy()
    preds = []
    for text in df["user_prompt"].astype(str).tolist():
        preds.append(predict_row(text, strong, cross, bad))
    pred_df = pd.DataFrame(preds)

    report_lines = []
    accs = {}
    for axis in ["유형","극성","시제","확실성"]:
        acc = (pred_df[axis] == golds[axis]).mean()
        accs[axis] = acc
        cm = pd.crosstab(golds[axis], pred_df[axis], rownames=["gold"], colnames=["pred"], dropna=False)
        report_lines.append(f"\n[{axis}] acc={acc:.4f}")
        report_lines.append(cm.to_string())

    macro = sum(accs.values()) / 4.0
    report_lines.insert(0, f"축별 정확도: {accs}")
    report_lines.insert(1, f"Macro-avg acc: {macro:.4f}")

    with open(EVAL_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))

def main():
    df, strong, cross, bad = load_data()
    # 프롬프트 텍스트 생성
    prompt_txt = build_prompt_text(strong, cross, bad, top_k_per_label=60)
    with open(PROMPT_TXT, "w", encoding="utf-8") as f:
        f.write(prompt_txt)
    print(f"[OK] 프롬프트 트리거 표 저장 → {PROMPT_TXT}")

    # 간단 평가
    evaluate(df, strong, cross, bad, sample_size=None)
    print(f"[OK] 평가 리포트 저장 → {EVAL_TXT}")

if __name__ == "__main__":
    main()
