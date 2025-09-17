# -*- coding: utf-8 -*-
"""
samples.csv에서 '무시어(Stopword) 후보' 자동 발굴 스크립트
- 아이디어: 토큰(단어)이 네 축(유형/극성/시제/확실성) 어느 쪽에도 유의미하게 연관되지 않으면
            규칙 판단을 '흩트리는 배경소음'일 가능성이 높음 → 무시어 후보로 제안
- 방법:
  1) 문장 토크나이즈(가능하면 KoNLPy Okt, 없으면 정규식 토큰화)
  2) CountVectorizer(binary=True, min_df)로 문서-토큰 행렬 X 생성(존재/부재)
  3) 축별로 X vs 라벨에 대해 sklearn.feature_selection.chi2 수행
     - p-value(유의확률)와 효과크기(Cramér's V; 2xK에서 V = sqrt(chi2/n)) 계산
     - Benjamini–Hochberg(FDR) 보정으로 다중검정 제어(선택)
  4) 축 전부에서 (유의X 또는 효과 아주 작음)인 토큰을 무시어 후보로 판정
  5) 결과 저장:
     - token_stats.csv : 축별 df(문서빈도), chi2, p, p_fdr, V, 신호/비신호 여부
     - stopwords_candidates.txt : 무시어 후보 토큰 리스트(정렬)
- 의존: pandas, scikit-learn (scipy는 sklearn 내부에서 사용)
"""

import re
import sys
import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

# ===================== 사용자 설정 =====================
CSV_PATH = "./_data/dacon/KT/samples.csv"

# 텍스트 컬럼 자동 탐색 후보
TEXT_COL_CANDIDATES = ["user_prompt", "문장", "text", "input", "sentence"]

# 최소 문서빈도(너무 희귀한 토큰은 통계 불안정 → 제외)
MIN_DF = 10

# 유의수준 및 효과크기 임계값
ALPHA = 0.05           # 유의수준
V_THRESHOLD = 0.05     # Cramér's V(효과크기) 최소치(작으면 실전 영향 미미로 간주)

# 다중검정 보정(FDR; Benjamini–Hochberg) 사용할지 여부
USE_FDR = True

# 추가 전처리: 무시할 상수 토큰(원하는 경우 여기에 수동 추가)
MANUAL_IGNORE = set([
    # 예: "수", "있다", "그리고", "그러나", "및", "등"
])

# =====================================================

def read_csv_any(path):
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

# ---------- 토크나이저 ----------
def build_tokenizer():
    """
    가능하면 KoNLPy Okt 사용, 없으면 정규식 기반 토크나이저로 대체
    """
    try:
        from konlpy.tag import Okt
        okt = Okt()
        def tokenize_okt(text: str):
            # 불용 부호 제거 후 형태소 단위
            t = str(text)
            toks = okt.morphs(t, stem=False)
            toks = [w.strip() for w in toks if w.strip()]
            return toks
        return tokenize_okt, "Okt"
    except Exception:
        token_re = re.compile(r"[가-힣A-Za-z0-9]+")
        def tokenize_regex(text: str):
            t = str(text)
            toks = token_re.findall(t)
            toks = [w.strip() for w in toks if w.strip()]
            return toks
        return tokenize_regex, "Regex"

def benjamini_hochberg(pvals: np.ndarray):
    """
    Benjamini–Hochberg(FDR) 보정
    - 입력: pvals (길이 m)
    - 출력: p_fdr (같은 길이, 단조증가 보정)
    """
    m = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=float)
    ranks[order] = np.arange(1, m+1)

    p_adj = pvals * m / ranks
    # 누적 최소 보정으로 단조 증가 유지
    p_adj_sorted = np.minimum.accumulate(p_adj[order][::-1])[::-1]
    p_fdr = np.empty(m, dtype=float)
    p_fdr[order] = np.minimum(p_adj_sorted, 1.0)
    return p_fdr

def main():
    # ---------- 데이터 로드 ----------
    df = read_csv_any(CSV_PATH)

    text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
    if text_col is None:
        print(f"[에러] 텍스트 컬럼을 찾을 수 없습니다. 후보: {TEXT_COL_CANDIDATES}")
        sys.exit(1)

    if "output" not in df.columns:
        print("[에러] 'output' 컬럼이 없습니다. (예: '유형,극성,시제,확실성')")
        sys.exit(1)

    # output 분해
    split_cols = df["output"].astype(str).str.split(",", expand=True)
    if split_cols.shape[1] != 4:
        print("[에러] 'output'이 '유형,극성,시제,확실성' 4개로 분리되지 않습니다.")
        sys.exit(1)
    df = df.copy()
    df.rename(columns={text_col: "문장"}, inplace=True)
    df[["유형_true","극성_true","시제_true","확실성_true"]] = split_cols

    # ---------- 토큰화 & 벡터화 ----------
    tokenizer, tok_name = build_tokenizer()
    print(f"[정보] 토크나이저: {tok_name}")

    vectorizer = CountVectorizer(
        tokenizer=tokenizer,
        lowercase=False,
        min_df=MIN_DF,
        binary=True  # 존재/부재
    )
    X = vectorizer.fit_transform(df["문장"].astype(str).tolist())
    vocab = np.array(vectorizer.get_feature_names_out())
    n_docs = X.shape[0]
    print(f"[정보] 문서 수: {n_docs:,}, 특징 수(토큰): {X.shape[1]:,}, min_df={MIN_DF}")

    # 문서 빈도(존재 문서 수)
    dfreq = np.asarray(X.sum(axis=0)).ravel()

    # ---------- 축별 χ² / p / FDR / V 계산 ----------
    axes = [("유형", "유형_true"),
            ("극성", "극성_true"),
            ("시제", "시제_true"),
            ("확실성", "확실성_true")]

    # 결과 저장용 딕셔너리
    stats = {
        "token": vocab,
        "df": dfreq
    }

    for axis_name, col in axes:
        y = df[col].astype(str).values
        chi2_vals, p_vals = chi2(X, y)   # sklearn: 특성 vs 다중 클래스 카이제곱
        # 2xK의 Cramér's V: sqrt(chi2 / n)
        with np.errstate(divide="ignore", invalid="ignore"):
            V = np.sqrt(np.maximum(chi2_vals, 0) / max(n_docs, 1))

        if USE_FDR:
            p_fdr = benjamini_hochberg(p_vals)
        else:
            p_fdr = p_vals.copy()

        # 신호/비신호(=무시) 판정: (p_fdr < ALPHA) & (V >= V_THRESHOLD) 이면 '신호'
        signal = (p_fdr < ALPHA) & (V >= V_THRESHOLD)

        stats[f"chi2_{axis_name}"] = chi2_vals
        stats[f"p_{axis_name}"] = p_vals
        stats[f"pFDR_{axis_name}"] = p_fdr
        stats[f"V_{axis_name}"] = V
        stats[f"signal_{axis_name}"] = signal

        print(f"[정보] 축 '{axis_name}': 신호 토큰 {signal.sum():,} / {len(signal):,} "
              f"(기준: p{'_FDR' if USE_FDR else ''} < {ALPHA}, V ≥ {V_THRESHOLD})")

    # ---------- 무시어 후보 선정 ----------
    # 네 축 모두에서 '비신호'인 토큰만 선별
    signal_masks = np.column_stack([stats[f"signal_{a}"] for a, _ in axes])
    non_signal_all = ~signal_masks.any(axis=1)

    # 수동 무시 목록 적용(있다면 강제 무시)
    manual_mask = np.array([t in MANUAL_IGNORE for t in vocab])
    ignore_mask = non_signal_all | manual_mask

    # 결과 DataFrame
    stat_df = pd.DataFrame(stats)
    stat_df["non_signal_axes_cnt"] = (~signal_masks).sum(axis=1)
    stat_df["ignore_candidate"] = ignore_mask

    # 정렬: (무시 후보 우선) → (non_signal 축 개수 desc) → (df desc) → (토큰)
    stat_df.sort_values(
        by=["ignore_candidate", "non_signal_axes_cnt", "df", "token"],
        ascending=[False, False, False, True],
        inplace=True
    )

    # 저장
    stat_df.to_csv("token_stats.csv", index=False, encoding="utf-8-sig")
    stopwords = stat_df.loc[stat_df["ignore_candidate"], "token"].tolist()
    with open("stopwords_candidates.txt", "w", encoding="utf-8") as f:
        for w in stopwords:
            f.write(w + "\n")

    # 미리보기 출력
    print("\n=== 무시어(Stopword) 후보 상위 50개(미리보기) ===")
    preview = stat_df.loc[stat_df["ignore_candidate"], ["token", "df", "non_signal_axes_cnt"]].head(50)
    for _, r in preview.iterrows():
        print(f"- {r['token']} (df={int(r['df'])}, 비신호축={int(r['non_signal_axes_cnt'])}/4)")

    print(f"\n[완료] token_stats.csv 저장")
    print(f"[완료] stopwords_candidates.txt 저장 (총 {len(stopwords):,}개)")

if __name__ == "__main__":
    main()
