# -*- coding: utf-8 -*-
"""
samples.csv 규칙기반 예측 vs 정답 비교기 (보강판 v3)
- 패치 반영:
  (1) '예정/계획'의 계획공지(사실) vs 전망(예측) 분기 강화(ORG/PLAN_CONF/DATE_MARK)
  (2) 과거 종결 어미 정규식 강화(한글에서도 문장 끝 매칭: (?!\w))
  (3) 확실성: '가능성'만 불확실 유지, '가능하다/할 수 있다'는 확실로 이동(+법/규정 맥락 가산 유지)
  (4) 추론형 시그널 소폭 확대
- 혼동행렬, classification_report, 축별/전체 오분류 CSV 저장
"""

import re
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

CSV_PATH = "./_data/dacon/KT/samples.csv"

# ---------- 공용 유틸 ----------
def read_csv_any(path):
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def cp(xs): return [re.compile(x) for x in xs]
def anym(ps, t): return any(p.search(t) for p in ps)

# ---------- 데이터 로드 ----------
df = read_csv_any(CSV_PATH)

TEXT_COL_CANDIDATES = ["user_prompt", "문장", "text", "input", "sentence"]
text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
if text_col is None:
    print(f"[에러] 텍스트 컬럼을 찾을 수 없습니다. 후보: {TEXT_COL_CANDIDATES}")
    sys.exit(1)

df = df.copy()
df.rename(columns={text_col: "문장"}, inplace=True)

if "output" not in df.columns:
    print("[에러] 'output' 컬럼이 없습니다.")
    sys.exit(1)

split_cols = df["output"].str.split(",", expand=True)
if split_cols.shape[1] != 4:
    print("[에러] 'output'이 '유형,극성,시제,확실성' 4개로 분리되지 않습니다.")
    sys.exit(1)

df[["유형_true","극성_true","시제_true","확실성_true"]] = split_cols

# ---------- 패턴 사전 ----------
# 대화형
DIALOG = cp([
    r"\?", r"(?:하|해)\s*주세요", r"(?:하|하십)세요(?!\w)",
    r"줄래(?!\w)", r"ㄹ까요(?!\w)", r"겠습니까\?",
    r"문의(?:하|하십)?세요(?!\w)", r"답변(?:하|하십)?세요(?!\w)",
    r"참여(?:하|하십)?세요(?!\w)", r"신청(?:하|하십)?세요(?!\w)",
    r"어떻게\b", r"있을까\b", r"어떤\b"
])

# 예측형 키워드(미래 지향)
PRED = cp([
    r"예상", r"전망", r"계획", r"예정", r"추진", r"될\s*것", r"할\s*것이다",
    r"향후", r"내달", r"내년", r"곧", r"다가올", r"전망이다", r"예정이다", r"계획이다"
])

# 계획 공지/확정 신호 + 날짜 신호 + 주체(기관/조직)
PLAN_CONF = cp([r"발표", r"밝혔", r"확정", r"공고", r"승인", r"보도자료", r"공지"])
DATE_MARK = cp([r"오는", r"내달", r"내년", r"\d+\s*월", r"\d+\s*일", r"부터", r"까지"])
ORG = cp([
    r"정부", r"부처", r"위원회", r"(?:시|군|구)(?!\w)", r"공사", r"공단",
    r"법원", r"국회", r"회사", r"대학", r"협회", r"청(?!\w)", r"부(?!\w)"
])

# 추론형(의견/분석/전언/가능성)
INFER = cp([
    r"가능성", r"우려", r"추정", r"듯", r"같다", r"의심", r"해석", r"평가", r"관측",
    r"논의", r"검토", r"의혹", r"가정", r"추정치", r"전해졌다", r"보인다",
    r"평가했다(?!\w)", r"분석했다(?!\w)", r"전망했다(?!\w)", r"관측했다(?!\w)",
    r"진단했다(?!\w)", r"우려했다(?!\w)",
    # 약한 추론 신호 추가
    r"보일\s*것으로\s*보인다", r"예단할\s*수\s*없", r"안갯속"
])

# 사실형(강한 보고/확인 신호)
FACT_STRONG = cp([
    r"했[다습니다](?!\w)", r"하였다(?!\w)", r"밝혔[다습니다](?!\w)", r"발표했[다습니다](?!\w)",
    r"말했[다습니다](?!\w)", r"요구했[다습니다](?!\w)", r"요청했[다습니다](?!\w)",
    r"촉구했[다습니다](?!\w)", r"주장했[다습니다](?!\w)",
    r"확정됐[다습니다](?!\w)", r"체결했[다습니다](?!\w)", r"우승했[다습니다](?!\w)",
    r"완료했[다습니다](?!\w)", r"공개됐[다습니다](?!\w)", r"확인됐[다습니다](?!\w)"
])

# 극성
POS = cp([
    r"증가", r"상승", r"개선", r"성공", r"승인", r"수상", r"회복", r"확대",
    r"달성", r"안정", r"우승", r"합의", r"협약", r"체결", r"출시", r"개통",
    r"개최", r"축하", r"환영", r"선정", r"흑자", r"반등", r"완치", r"복구", r"성과"
])
NEG = cp([
    r"감소", r"하락", r"악화", r"실패", r"취소", r"징계", r"피해", r"적자", r"중단",
    r"연기", r"지연", r"리콜", r"고장", r"사고", r"폭락", r"부진", r"비리",
    r"법위반", r"패배", r"감염", r"확진\s*급증", r"논란"
])
NEG_STRONG = cp([
    r"없다(?!\w)", r"없었다(?!\w)", r"않다(?!\w)", r"않았다(?!\w)",
    r"못했다(?!\w)", r"못한다(?!\w)", r"아니다(?!\w)"
])
POS_EXCEPT = cp([r"문제없다", r"이상\s*없다", r"무혐의"])

# 시제 (시간어 > 서술표지)
PAST_TIME  = cp([r"지난", r"어제", r"전날", r"방금", r"지난해"])
PAST_VERB  = cp([
    r"(?:했|였|었|았)다(?!\w)", r"(?:했|였|었|았)습니다(?!\w)",
    r"(?:였|이었)다(?!\w)",
    r"말했(?:다|습니다)(?!\w)", r"밝혔(?:다|습니다)(?!\w)",
    r"완료했(?:다|습니다)(?!\w)", r"체결했(?:다|습니다)(?!\w)",
    r"확인됐(?:다|습니다)(?!\w)", r"공개됐(?:다|습니다)(?!\w)",
    r"우승했(?:다|습니다)(?!\w)"
])
FUTURE_TIME= cp([r"향후", r"곧", r"내달", r"내년", r"오는"])
FUTURE_VERB= cp([
    r"할\s*것이다", r"예정", r"전망", r"계획", r"추진",
    r"출시한다(?!\w)", r"개최한다(?!\w)", r"착수한다(?!\w)",
    r"도입한다(?!\w)", r"확대한다(?!\w)",
    r"앞두고\s*있다", r"예정돼\s*있다", r"나설\s*예정"
])
PRESENT_HINT= cp([r"오늘", r"최근"])
PRESENT_VERB= cp([r"이다(?!\w)", r"있다(?!\w)", r"중이다(?!\w)", r"운영", r"진행", r"보유", r"유지"])

# 확실성
# 불확실: '가능성'은 유지, '가능하다/할 수 있다' 제거
UNCERTAIN = cp([
    r"가능성", r"예상", r"전망", r"듯", r"같다", r"논의", r"검토",
    r"의혹", r"가정", r"미정", r"추정", r"추정치",
    r"관계자에\s*따르면", r"전해졌다", r"보인다"
])
# 확실: '가능하', '할 수 있다'를 포함(데이터 특성 반영)
CERTAIN = cp([
    r"확인", r"발표", r"체결", r"승인", r"판결", r"공고", r"확정",
    r"밝혔다", r"말했다", r"분명히", r"확실히",
    r"가능하", r"할\s*수\s*있다"
])
LAW_CTX = cp([r"법(?!\w)", r"법원", r"대법원", r"판결", r"규정", r"기준", r"요건",
              r"신청", r"이의신청", r"심판청구", r"과세", r"세법", r"조세"])

# 부속절(계획을 위한 과거 사실 기술)
SUBORD = cp([r"위해", r"목적", r"방안", r"대비해", r"차원에서", r"도록"])

# ---------- 규칙 분류 함수 ----------
def classify_sentence(text):
    t = str(text).strip()
    s = re.sub(r"\s+", " ", t)

    # ---- 유형 ----
    if anym(DIALOG, s):
        y = "대화형"
    else:
        pred_like = anym(PRED, s)
        has_past = anym(PAST_VERB, s) or anym(PAST_TIME, s)
        if pred_like:
            if anym(PLAN_CONF, s) or anym(DATE_MARK, s) or anym(ORG, s):
                # 공식화/날짜/기관 주체 → 계획 공지 = 사실형
                y = "사실형"
            elif anym([re.compile(r"예상"), re.compile(r"전망"), re.compile(r"보인다")], s):
                y = "예측형"
            elif has_past and anym(SUBORD, s):
                # 과거 사실 + '위해/방안 …' 맥락 → 사실형
                y = "사실형"
            elif not has_past:
                y = "예측형"
            else:
                y = "사실형"
        elif anym(INFER, s):
            y = "추론형"
        elif anym(FACT_STRONG, s):
            y = "사실형"
        else:
            y = "사실형"

    # ---- 포괄 결속(유형→다른 축 기본값) ----
    tense_hint = None
    certainty_hint = None
    polarity_hint = None
    if y == "대화형":
        tense_hint = "현재"
        polarity_hint = "미정"
    elif y == "예측형":
        tense_hint = "미래"
        certainty_hint = "불확실"
    elif y == "추론형":
        certainty_hint = "불확실"
    elif y == "사실형":
        certainty_hint = "확실"

    # ---- 시제 ---- (시간어 > 서술표지 > 의미 규칙)
    if anym(PAST_TIME, s) or anym(PAST_VERB, s):
        tense = "과거"
    elif anym(FUTURE_TIME, s) or anym(FUTURE_VERB, s):
        tense = "미래"
    else:
        # '수 있다/가능하다' 만으로는 미래로 끌지 않음(데이터 특성상 설명/능력 용례 多)
        tense = "현재"
    if tense_hint:
        tense = tense_hint

    # ---- 확실성 ----
    if y == "예측형" and anym([re.compile(r"확정"), re.compile(r"공고"), re.compile(r"승인"), re.compile(r"공식\s*일정")], s):
        cert = "확실"
    elif anym(CERTAIN, s):
        cert = "확실"
    elif anym(LAW_CTX, s) and anym([re.compile(r"할\s*수\s*있다")], s):
        # 규범/권한 문맥 + '할 수 있다'는 확실로
        cert = "확실"
    elif anym(UNCERTAIN, s):
        cert = "불확실"
    else:
        cert = "확실"
    if certainty_hint:
        cert = certainty_hint

    # ---- 극성 ----
    if polarity_hint:
        pol = polarity_hint
        if anym(NEG_STRONG + NEG, s):
            pol = "부정"
        elif anym(POS_EXCEPT, s) or anym(POS, s):
            pol = "긍정"
    else:
        if anym(NEG_STRONG + NEG, s):
            pol = "부정"
        elif anym(POS_EXCEPT, s) or anym(POS, s):
            pol = "긍정"
        else:
            pol = "긍정"  # 데이터 분포 편향 상의 전략(원 코드 유지)

    return y, pol, tense, cert

# ---------- 예측/저장/평가 ----------
preds = df["문장"].apply(classify_sentence)
df["유형_pred"] = preds.apply(lambda x: x[0])
df["극성_pred"] = preds.apply(lambda x: x[1])
df["시제_pred"] = preds.apply(lambda x: x[2])
df["확실성_pred"] = preds.apply(lambda x: x[3])

out = df[["문장","output","유형_true","극성_true","시제_true","확실성_true",
          "유형_pred","극성_pred","시제_pred","확실성_pred"]]
out.to_csv("predictions_rule_based_v3.csv", index=False, encoding="utf-8-sig")
print("예측 결과 저장: predictions_rule_based_v3.csv")

axes = ["유형","극성","시제","확실성"]
for col in axes:
    y_true = df[f"{col}_true"]
    y_pred = df[f"{col}_pred"]
    labels = sorted(y_true.unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                             columns=[f"pred_{l}" for l in labels])
    print(f"\n=== [{col}] Confusion Matrix ===")
    print(cm_df)
    print(f"\n=== [{col}] Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"[{col}] Accuracy = {accuracy_score(y_true, y_pred):.4f}")

exact = ((df["유형_true"]==df["유형_pred"]) &
         (df["극성_true"]==df["극성_pred"]) &
         (df["시제_true"]==df["시제_pred"]) &
         (df["확실성_true"]==df["확실성_pred"])).mean()
print(f"\n[전체] 4축 완전 일치 정확도 (Exact Match) = {exact:.4f}")

def dump_errors(col, n=10):
    t, p = f"{col}_true", f"{col}_pred"
    err = df[df[t] != df[p]].copy()
    if err.empty:
        print(f"\n[{col}] 오분류 없음 🎉"); return
    print(f"\n[{col}] 오분류 샘플 상위 {n}개:")
    for _, r in err.head(n).iterrows():
        print(f"- [{col}] true={r[t]} / pred={r[p]} :: {r['문장']}")
    cols = ["문장", t, p, "유형_true","극성_true","시제_true","확실성_true",
            "유형_pred","극성_pred","시제_pred","확실성_pred"]
    err[cols].to_csv(f"errors_{col}_v3.csv", index=False, encoding="utf-8-sig")
    print(f"[{col}] 오분류 저장: errors_{col}_v3.csv (총 {len(err)}건)")

for col in axes:
    dump_errors(col, 10)

any_err = df[(df["유형_true"]!=df["유형_pred"]) |
             (df["극성_true"]!=df["극성_pred"]) |
             (df["시제_true"]!=df["시제_pred"]) |
             (df["확실성_true"]!=df["확실성_pred"])].copy()
if not any_err.empty:
    any_err.to_csv("errors_overall_v3.csv", index=False, encoding="utf-8-sig")
    print(f"\n전체 오분류 저장: errors_overall_v3.csv (총 {len(any_err)}건)")
else:
    print("\n전체 오분류 없음 🎉")
