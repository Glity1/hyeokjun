import argparse, os, re, json
from collections import Counter, defaultdict
from itertools import islice
import pandas as pd

AXES = ["유형", "극성", "시제", "확실성"]
AXIS_INDEX = {"유형":0, "극성":1, "시제":2, "확실성":3}

# ------------------------- 토크나이저 -------------------------

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
def tokenize_ko(text: str):
    if not isinstance(text, str):
        return []
    toks = TOKEN_RE.findall(text)
    return [t.lower() for t in toks if not t.isdigit() and len(t) >= 2]

def bigrams(tokens):
    for i in range(len(tokens)-1):
        yield tokens[i] + " " + tokens[i+1]

# ------------------------- 패턴 사전 -------------------------

# 따옴표(인용부) 범위 탐지용
QUOTE_RANGES_RE = re.compile(r"[\"“”‘’\'「『][^\"“”‘’\'」』]*[\"“”‘’\'」』]")

REQ_PAT = re.compile(r"(하세요|해\s?주세요|줄래|할까|문의|신청|참여|가능할까요|습니까)")
QUESTION_MARK = "?"

REPORTING_VERBS = re.compile(r"(발표했|밝혔|전했|주장했|요청했|촉구했)")

PAST_MARKERS = re.compile(r"(했[다]|였다|확인됐|발표했|체결했|우승했|완료했|공개됐|지난|어제|전날|방금)")
FUTURE_MARKERS = re.compile(r"(하겠|예정|전망|계획|추진|될\s*것|출시한다|개최한다|착수한다|도입한다|확대한다|시행한다)")

ASPECT_DONE = re.compile(r"(았|었)[다\s.,)]")
ASPECT_PROGRESS = re.compile(r"(고\s*있|진행\s*중|…중|중입니다|중이다)")
ASPECT_WILL = re.compile(r"(겠|예정|계획|추진|될\s*것)")

CERTAINTY_NUMERIC = re.compile(r"(\d|\d+%|%|억원|만원|원|명|건|지수|포인트|톤|대|km|kg)")
CERTAINTY_WORDS = re.compile(r"(확정|공식|의결|공포|판결|공시|집계|발표했|밝혔다|확인됐다|통계|자료)")
UNCERTAINTY_WORDS = re.compile(r"(가능성|전망|예상|추정|가정|논의|검토|의혹|전해졌다|보인다|관계자에\s*따르면|가이던스|컨센서스)")

NEGATION = re.compile(r"(않|못|없|지\s*않다)")
NEG_EVENT = re.compile(r"(증가|감소|상승|하락|개선|악화)")
PHRASE_DEC_NOT = re.compile(r"(감소\s*하지\s*않았다)")
PHRASE_INC_NOT = re.compile(r"(증가\s*하지\s*않았다)")

OVERRIDE_POS = re.compile(r"(전환|회복|반등|흑자\s*전환|상승\s*전환)")
OVERRIDE_CONFIRMED = re.compile(r"(했다|됐다)")
OVERRIDE_BLOCK = re.compile(r"(전망|가능|못했다|꺾였다)")

HEAD_IMPLEMENT = re.compile(r"(시행|개최|출시|도입|개시|발효|상장)")
HEAD_ANNOUNCE = re.compile(r"(발표|밝혔)")

END_Q = re.compile(r"(\?|습니까|ㄹ까요)")
END_DECL = re.compile(r"(다[.?!\s]*$|했다[.?!\s]*$|한다[.?!\s]*$)")

# ------------------------- 유틸 함수 -------------------------

def extract_axes(output_str: str):
    parts = [p.strip() for p in str(output_str).split(",")]
    res = {}
    for a in AXES:
        idx = AXIS_INDEX[a]
        res[a] = parts[idx] if idx < len(parts) else ""
    return res

def strip_quoted(text: str):
    """인용부 제거 텍스트 반환"""
    if not isinstance(text, str):
        return ""
    return QUOTE_RANGES_RE.sub("", text)

def has_question_inside_quotes(text: str):
    matches = list(QUOTE_RANGES_RE.finditer(text or ""))
    for m in matches:
        if QUESTION_MARK in m.group(0):
            return True
    return False

def top_ngram_by_label(df, axis, n=1, topn=20, min_count=3):
    rows = []
    for lab, g in df.groupby(axis):
        counter = Counter()
        for txt in g["user_prompt"].astype(str):
            toks = tokenize_ko(txt)
            grams = toks if n == 1 else list(bigrams(toks))
            counter.update(grams)
        items = [(w,c) for w,c in counter.items() if c >= min_count]
        items.sort(key=lambda x:(-x[1], x[0]))
        for rank, (w,c) in enumerate(items[:topn], 1):
            rows.append({"축":axis,"라벨":lab,"n":n,"순위":rank,"ngram":w,"빈도":c})
    return pd.DataFrame(rows).sort_values(["라벨","n","순위"])

def crosstab_pairs(df, a1, a2):
    ct = pd.crosstab(df[a1], df[a2])
    ct.index.name = a1
    ct.columns.name = a2
    return ct.reset_index()

def head_tail(iterable, n):
    it = iter(iterable)
    return list(islice(it, n))

def read_csv_auto(path: str, encoding_opt: str = "auto") -> pd.DataFrame:
    """utf-8-sig → cp949 → utf-8 순으로 시도"""
    if not os.path.exists(path):
        # 현재 폴더의 samples.csv로 폴백
        alt = os.path.join(os.getcwd(), "samples.csv")
        if os.path.exists(alt):
            path = alt
        else:
            raise SystemExit(f"[ERROR] CSV 파일을 찾을 수 없습니다: {path}")
    if encoding_opt != "auto":
        return pd.read_csv(path, encoding=encoding_opt)
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 마지막 시도(에러 메시지 확인용)
    return pd.read_csv(path)

# ------------------------- 메인 분석 -------------------------

def main():
    ap = argparse.ArgumentParser()
    # 기본값을 samples.csv(절대 경로)로 설정 — 인자 없이도 동작
    ap.add_argument(
        "--csv",
        default=r"C:\python\Study25_집\_data\dacon\KT\samples.csv",
        help="samples.csv 경로(미입력 시 기본값 사용)."
    )
    ap.add_argument("--out", default="out")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--fewshot_k", type=int, default=3, help="라벨별 대표/경계 샘플 수(유형 축)")
    ap.add_argument("--encoding", default="auto", help="csv 인코딩(기본: auto 탐지)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = read_csv_auto(args.csv, args.encoding)
    if "user_prompt" not in df.columns or "output" not in df.columns:
        raise SystemExit("[ERROR] CSV에 'user_prompt'와 'output' 컬럼이 필요합니다.")

    # 라벨 분해
    labs = df["output"].apply(extract_axes).apply(pd.Series)
    for a in AXES:
        df[a] = labs[a]

    # (1) 축별 라벨 분포
    for a in AXES:
        s = df[a].value_counts(dropna=False).rename("count").reset_index().rename(columns={"index":"label"})
        s["ratio"] = (s["count"] / s["count"].sum()).round(6)
        s.to_csv(os.path.join(args.out, f"dist_axis_{a}.csv"), index=False, encoding="utf-8-sig")

    # (2) 라벨별 상위 단어/구(1–2gram)
    for a in AXES:
        top1 = top_ngram_by_label(df, a, n=1, topn=args.topn, min_count=args.min_count)
        top1.to_csv(os.path.join(args.out, f"top1gram_{a}.csv"), index=False, encoding="utf-8-sig")
        top2 = top_ngram_by_label(df, a, n=2, topn=args.topn, min_count=args.min_count)
        top2.to_csv(os.path.join(args.out, f"top2gram_{a}.csv"), index=False, encoding="utf-8-sig")

    # (3) 질문/라우터 신호
    rows = []
    for i, row in df.iterrows():
        txt = str(row["user_prompt"])
        outside = strip_quoted(txt)
        has_q_out = QUESTION_MARK in outside
        has_q_in = has_question_inside_quotes(txt)
        has_req = bool(REQ_PAT.search(outside))
        rows.append({
            "no": i+1,
            "유형": row["유형"],
            "has_q_outside": int(has_q_out),
            "has_q_inside": int(has_q_in),
            "has_2nd_req": int(has_req),
        })
    qdf = pd.DataFrame(rows)
    qsum = qdf.groupby("유형")[["has_q_outside","has_q_inside","has_2nd_req"]].mean().reset_index()
    qsum.to_csv(os.path.join(args.out, "question_router_stats.csv"), index=False, encoding="utf-8-sig")

    # (4) 보도서술 동사 분포 (유형 기준)
    rows = []
    for i,row in df.iterrows():
        txt = str(row["user_prompt"])
        has_rep = int(bool(REPORTING_VERBS.search(txt)))
        rows.append({"no":i+1, "유형":row["유형"], "reporting":has_rep})
    rdc = pd.DataFrame(rows).groupby("유형")["reporting"].mean().reset_index()
    rdc.to_csv(os.path.join(args.out, "reporting_verbs_by_유형.csv"), index=False, encoding="utf-8-sig")

    # (5) 시제 앵커 충돌
    rows = []
    for i,row in df.iterrows():
        txt = str(row["user_prompt"])
        has_past = int(bool(PAST_MARKERS.search(txt)))
        has_fut  = int(bool(FUTURE_MARKERS.search(txt)))
        rows.append({"no":i+1,"유형":row["유형"],"시제":row["시제"],"has_past":has_past,"has_future":has_fut})
    tdf = pd.DataFrame(rows)
    tdf["conflict"] = (tdf["has_past"] & tdf["has_future"]).astype(int)
    tdf.groupby(["유형","시제"])[["has_past","has_future","conflict"]].mean().reset_index()\
        .to_csv(os.path.join(args.out,"tense_conflicts.csv"), index=False, encoding="utf-8-sig")

    # (6) 완료/진행/의지 표지 분포
    rows = []
    for i,row in df.iterrows():
        txt = str(row["user_prompt"])
        rows.append({
            "no":i+1, "유형":row["유형"], "시제":row["시제"],
            "done": int(bool(ASPECT_DONE.search(txt))),
            "prog": int(bool(ASPECT_PROGRESS.search(txt))),
            "will": int(bool(ASPECT_WILL.search(txt))),
        })
    adf = pd.DataFrame(rows)
    adf.groupby(["유형","시제"])[["done","prog","will"]].mean().reset_index()\
        .to_csv(os.path.join(args.out,"aspect_markers_by_axis.csv"), index=False, encoding="utf-8-sig")

    # (7) 확실/불확실 신호 및 충돌
    rows = []
    for i,row in df.iterrows():
        txt = str(row["user_prompt"])
        outside = strip_quoted(txt)
        cert = int(bool(CERTAINTY_NUMERIC.search(txt) or CERTAINTY_WORDS.search(txt)))
        uncert = int(bool(UNCERTAINTY_WORDS.search(outside) or (QUESTION_MARK in outside)))
        rows.append({"no":i+1, "확실성":row["확실성"], "cert":cert, "uncert":uncert, "both": int(cert and uncert)})
    cdf = pd.DataFrame(rows)
    cdf.groupby("확실성")[["cert","uncert","both"]].mean().reset_index()\
        .to_csv(os.path.join(args.out,"certainty_uncertainty_stats.csv"), index=False, encoding="utf-8-sig")

    # (8) 부정 스코프/이중부정
    rows, ex = [], []
    for i,row in df.iterrows():
        txt = str(row["user_prompt"])
        has_neg = int(bool(NEGATION.search(txt)))
        has_event = int(bool(NEG_EVENT.search(txt)))
        dec_not = int(bool(PHRASE_DEC_NOT.search(txt)))
        inc_not = int(bool(PHRASE_INC_NOT.search(txt)))
        if dec_not or inc_not:
            ex.append(f"[{i+1}] {txt}")
        rows.append({"no":i+1,"극성":row["극성"],"has_neg":has_neg,"has_event":has_event,"dec_not":dec_not,"inc_not":inc_not})
    nd = pd.DataFrame(rows)
    nd.groupby("극성")[["has_neg","has_event","dec_not","inc_not"]].mean().reset_index()\
        .to_csv(os.path.join(args.out,"negation_scope_stats.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out,"negation_examples.txt"),"w",encoding="utf-8") as f:
        for line in head_tail(ex, 50):
            f.write(line+"\n")

    # (9) 전환 오버라이드 신호
    rows, ex = [], []
    for i,row in df.iterrows():
        txt = str(row["user_prompt"])
        has_tr = int(bool(OVERRIDE_POS.search(txt)))
        confirmed = int(bool(OVERRIDE_CONFIRMED.search(txt)))
        blocked = int(bool(OVERRIDE_BLOCK.search(txt)))
        if has_tr and (confirmed or blocked):
            ex.append(f"[{i+1}] {txt}")
        rows.append({"no":i+1,"극성":row["극성"],"has_turn":has_tr,"confirmed":confirmed,"blocked":blocked})
    od = pd.DataFrame(rows)
    od.groupby("극성")[["has_turn","confirmed","blocked"]].mean().reset_index()\
        .to_csv(os.path.join(args.out,"override_signals.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out,"override_examples.txt"),"w",encoding="utf-8") as f:
        for line in head_tail(ex, 50):
            f.write(line+"\n")

    # (10) 헤드 이벤트 키워드 분포
    rows = []
    for i,row in df.iterrows():
        txt = str(row["user_prompt"])
        head_impl = int(bool(HEAD_IMPLEMENT.search(txt)))
        head_ann  = int(bool(HEAD_ANNOUNCE.search(txt)))
        rows.append({"no":i+1, "유형":row["유형"], "시제":row["시제"], "head_impl":head_impl, "head_announce":head_ann})
    hd = pd.DataFrame(rows)
    hd.groupby(["유형","시제"])[["head_impl","head_announce"]].mean().reset_index()\
        .to_csv(os.path.join(args.out,"head_keywords_map.csv"), index=False, encoding="utf-8-sig")

    # (11) 숫자/단위/기호 신호
    rows = []
    for i,row in df.iterrows():
        txt = str(row["user_prompt"])
        has_num = int(bool(CERTAINTY_NUMERIC.search(txt)))
        rows.append({"no":i+1, "유형":row["유형"], "확실성":row["확실성"], "has_numeric":has_num})
    pd.DataFrame(rows).groupby(["유형","확실성"])["has_numeric"].mean().reset_index()\
        .to_csv(os.path.join(args.out,"numeric_signals.csv"), index=False, encoding="utf-8-sig")

    # (12) 종결 어미/문장 길이
    rows = []
    for i,row in df.iterrows():
        txt = str(row["user_prompt"]).strip()
        end_q = int(bool(END_Q.search(txt))) or int(txt.endswith("?"))
        end_decl = int(bool(END_DECL.search(txt)))
        rows.append({"no":i+1,"유형":row["유형"],"length_char":len(txt),"length_tok":len(tokenize_ko(txt)),"end_q":end_q,"end_decl":end_decl})
    ed = pd.DataFrame(rows)
    ed.groupby("유형")[["length_char","length_tok","end_q","end_decl"]].mean().reset_index()\
        .to_csv(os.path.join(args.out,"endings_lengths.csv"), index=False, encoding="utf-8-sig")

    # (13) Few-shot 제안(유형 축): 대표/경계 샘플
    def score_for_type(text, label):
        s = 0
        if label == "대화형":
            outside = strip_quoted(text)
            s += 3*int(QUESTION_MARK in outside) + 2*int(bool(REQ_PAT.search(outside)))
        if label == "예측형":
            s += 2*int(bool(FUTURE_MARKERS.search(text)))
        if label == "추론형":
            s += 2*int(bool(UNCERTAINTY_WORDS.search(strip_quoted(text))))
        if label == "사실형":
            s += 2*int(bool(CERTAINTY_WORDS.search(text))) + 1*int(bool(CERTAINTY_NUMERIC.search(text)))
        return s

    rep_lines = ["[Few-shot 대표(유형, 각 라벨 상위)]"]
    border_lines = ["[Few-shot 경계(혼합 신호)]"]
    df["__score__"] = [score_for_type(t, lab) for t, lab in zip(df["user_prompt"], df["유형"])]

    # 대표 샘플
    for lab, g in df.groupby("유형"):
        gg = g.sort_values("__score__", ascending=False).head(args.fewshot_k)
        for _, r in gg.iterrows():
            rep_lines.append(f"입력: {r['user_prompt']}\n출력: {r['output']}")

    # 경계 샘플(혼합: 과거+미래, 확실+불확실 동시)
    mix_mask = []
    for i,row in df.iterrows():
        txt = str(row["user_prompt"])
        outside = strip_quoted(txt)
        both_tense = bool(PAST_MARKERS.search(txt)) and bool(FUTURE_MARKERS.search(txt))
        cert = bool(CERTAINTY_NUMERIC.search(txt) or CERTAINTY_WORDS.search(txt))
        uncert = bool(UNCERTAINTY_WORDS.search(outside) or (QUESTION_MARK in outside))
        mix_mask.append(int(both_tense or (cert and uncert)))
    df["__mix__"] = mix_mask
    gg = df[df["__mix__"]==1].head(args.fewshot_k*len(df["유형"].unique()))
    for _, r in gg.iterrows():
        border_lines.append(f"입력: {r['user_prompt']}\n출력: {r['output']}")

    with open(os.path.join(args.out,"fewshot_suggestions_유형.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(rep_lines) + "\n\n" + "\n".join(border_lines) + "\n")

    # (14) 축 간 교차표
    pairs = [("유형","시제"),("유형","확실성"),("유형","극성"),("시제","확실성")]
    for a,b in pairs:
        ctab = crosstab_pairs(df, a, b)
        ctab.to_csv(os.path.join(args.out,f"crosstab_{a}_{b}.csv"), index=False, encoding="utf-8-sig")

    # SUMMARY.md
    summary_lines = ["# SUMMARY",
                     "본 리포트는 samples.csv 기준 14개 분석 지표를 산출했습니다.",
                     "주요 파일 목록은 아래를 참고하세요.\n"]
    summary_files = [
        "dist_axis_유형.csv","dist_axis_극성.csv","dist_axis_시제.csv","dist_axis_확실성.csv",
        "top1gram_유형.csv","top2gram_유형.csv","question_router_stats.csv",
        "reporting_verbs_by_유형.csv","tense_conflicts.csv","aspect_markers_by_axis.csv",
        "certainty_uncertainty_stats.csv","negation_scope_stats.csv","negation_examples.txt",
        "override_signals.csv","override_examples.txt","head_keywords_map.csv",
        "numeric_signals.csv","endings_lengths.csv","fewshot_suggestions_유형.txt",
        "crosstab_유형_시제.csv","crosstab_유형_확실성.csv","crosstab_유형_극성.csv","crosstab_시제_확실성.csv"
    ]
    for fn in summary_files:
        summary_lines.append(f"- {fn}")
    with open(os.path.join(args.out,"SUMMARY.md"),"w",encoding="utf-8") as f:
        f.write("\n".join(summary_lines)+"\n")

    print("[DONE] 결과 저장 폴더:", os.path.abspath(args.out))
    for fn in summary_files:
        p = os.path.join(args.out, fn)
        if os.path.exists(p):
            print(" -", fn)

if __name__ == "__main__":
    main()
