# -*- coding: utf-8 -*-
"""
samples.csv → 강/약 트리거 사전 생성 (안정 수정판)

요구사항
- 축(유형/극성/시제/확실성) "사이" 겹침 단어 = 좋은 신호 → good_cross_axis.json
- 같은 축 "내" 라벨 간 애매(겹침) 단어 = 나쁜 신호 → bad_confusing_tokens.json / .csv
- 같은 축 내 strong trigger는 margin 기반 exclusive → strong_triggers_by_axis.json
- 감사표(trigger_audit.csv) 저장

변경점
- 공통 Vectorizer 1회 fit → 축별로 행만 바꿔 log-odds 계산(축 간 vocabulary 정합 유지)
- audit_rows(): argsort로 열별 1등/2등 라벨·점수 계산(인덱싱 오류 근본 해결)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# 설정
# =========================
CSV_PATH = "./_data/dacon/KT/samples.csv"

# Vectorizer
NGRAM_RANGE = (1, 2)       # uni+bi-gram
MIN_DF = 2                 # 2회 미만 토큰 제거
TOKEN_PATTERN = r'(?u)\b\w\w+\b'  # 2글자 이상

# Log-odds / 임계값
ALPHA = 0.01
TOPK_PER_LABEL = 120
MARGIN_GOOD = 0.60          # strong trigger로 인정할 최소 마진
MARGIN_BAD  = 0.25          # 혼동 토큰으로 간주할 최대 마진

# 혼동/노이즈 어휘(선택)
STOPWORDS = {
    "때문이다", "필요하다", "중요하다",
    "있습니다", "없습니다", "갑니다", "검니다",
}

# 출력 파일
OUT_STRONG_JSON = "strong_triggers_by_axis.json"   # 축별·라벨별 strong trigger (exclusive)
OUT_CROSS_JSON  = "good_cross_axis.json"           # 여러 축에서 strong으로 겹친 좋은 단어
OUT_BAD_JSON    = "bad_confusing_tokens.json"      # 같은 축 내 라벨간 애매한 단어 목록
OUT_BAD_CSV     = "bad_confusing_tokens.csv"
OUT_AUDIT_CSV   = "trigger_audit.csv"

# =========================
# 유틸
# =========================
def read_csv_safely(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def split_output_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "output" not in df.columns:
        raise ValueError("samples.csv에 'output' 컬럼이 없습니다.")
    parts = df["output"].astype(str).str.split(",", expand=True)
    if parts.shape[1] != 4:
        raise ValueError("output 컬럼은 '유형,극성,시제,확실성' 4개 값이어야 합니다.")
    df[["유형", "극성", "시제", "확실성"]] = parts
    return df

def build_vectorizer(min_df=2, ngram_range=(1,2), token_pattern=TOKEN_PATTERN) -> CountVectorizer:
    return CountVectorizer(min_df=min_df, ngram_range=ngram_range, token_pattern=token_pattern)

# =========================
# 공통 vocab/X 생성 (1회 fit)
# =========================
def fit_corpus(texts: pd.Series, vec: CountVectorizer):
    X = vec.fit_transform(texts)
    words = np.array(vec.get_feature_names_out())
    V = words.shape[0]
    # 전체 빈도/문서수
    freq_total = np.asarray(X.sum(axis=0)).ravel()
    X_bin = X.copy(); X_bin.data[:] = 1
    doc_freq = np.asarray(X_bin.sum(axis=0)).ravel()
    return X, words, freq_total, doc_freq

# =========================
# 축별 log-odds (공통 vocab/X 사용)
# =========================
def log_odds_by_axis(X, labels: pd.Series, alpha=ALPHA) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    labels = labels.astype(str)
    unique = sorted(labels.unique().tolist())
    f_total = np.asarray(X.sum(axis=0)).ravel()
    V = f_total.shape[0]

    z_by_label: Dict[str, np.ndarray] = {}
    label_counts: Dict[str, int] = {}

    for lab in unique:
        mask = (labels == lab).values
        X_c  = X[mask]
        X_nc = X[~mask]

        f_c  = np.asarray(X_c.sum(axis=0)).ravel()
        f_nc = np.asarray(X_nc.sum(axis=0)).ravel()

        N_c  = f_c.sum()
        N_nc = f_nc.sum()

        num1 = (f_c + alpha)
        den1 = (N_c + alpha * V) - f_c - alpha
        num2 = (f_nc + alpha)
        den2 = (N_nc + alpha * V) - f_nc - alpha

        eps = 1e-12
        z = np.log((num1 + eps) / (den1 + eps)) - np.log((num2 + eps) / (den2 + eps))
        z_by_label[lab] = z
        label_counts[lab] = int(mask.sum())

    return z_by_label, label_counts

# =========================
# 감사표 (argsort로 1등/2등 계산)
# =========================
def make_audit_df(words: np.ndarray, axis_name: str, z_by_label: Dict[str, np.ndarray], freq_total: np.ndarray) -> pd.DataFrame:
    labels = sorted(list(z_by_label.keys()))
    Z = np.vstack([z_by_label[lab] for lab in labels])  # (L, V)
    L, V = Z.shape

    # 열별 내림차순 정렬 인덱스
    order = np.argsort(Z, axis=0)        # 오름차순
    best_idx   = order[-1, :]            # 1등 라벨 인덱스
    second_idx = order[-2, :] if L >= 2 else order[-1, :]

    best_val   = Z[best_idx,   np.arange(V)]
    second_val = Z[second_idx, np.arange(V)]
    margin     = best_val - second_val

    df = pd.DataFrame({
        "token": words,
        "axis": axis_name,
        "best_label":  [labels[i] for i in best_idx],
        "best_z":      best_val,
        "second_label":[labels[i] for i in second_idx],
        "second_z":    second_val,
        "margin":      margin,
        "freq":        freq_total
    })
    # 필요시 라벨별 z도 추가
    for lab in labels:
        df[f"z::{lab}"] = z_by_label[lab]
    return df

# =========================
# 같은 축 내 strong trigger(독점성 margin 기반 exclusive)
# =========================
def strong_triggers_same_axis(audit_df_axis: pd.DataFrame, topk=TOPK_PER_LABEL, margin_min=MARGIN_GOOD) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for lab in sorted(audit_df_axis["best_label"].unique()):
        sub = audit_df_axis[(audit_df_axis["best_label"] == lab) & (audit_df_axis["margin"] >= margin_min)]
        sub = sub.sort_values(["best_z", "margin", "freq"], ascending=[False, False, False]).head(topk)
        out[lab] = sub["token"].tolist()
    return out

# =========================
# 메인
# =========================
def main():
    # 1) 데이터 로드/파싱
    df = read_csv_safely(CSV_PATH)
    if "user_prompt" not in df.columns:
        raise ValueError("samples.csv에 'user_prompt' 컬럼이 없습니다.")
    df = split_output_cols(df)

    # 2) STOPWORDS 제거(간단 치환)
    texts = df["user_prompt"].astype(str)
    if STOPWORDS:
        def _drop_sw(s: str) -> str:
            out = s
            for sw in STOPWORDS:
                out = out.replace(sw, " ")
            return out
        texts = texts.apply(_drop_sw)

    # 3) 공통 Vectorizer 1회 fit → X, words
    vec = build_vectorizer(min_df=MIN_DF, ngram_range=NGRAM_RANGE, token_pattern=TOKEN_PATTERN)
    X, words, freq_total, doc_freq = fit_corpus(texts, vec)

    axis_cols = ["유형", "극성", "시제", "확실성"]
    audit_all = []
    strong_maps: Dict[str, Dict[str, List[str]]] = {}

    # 4) 축별 log-odds → 감사표 → strong trigger
    for axis in axis_cols:
        z_by_label, label_counts = log_odds_by_axis(X, df[axis])
        audit_axis = make_audit_df(words, axis, z_by_label, freq_total)
        audit_all.append(audit_axis)

        strong_map = strong_triggers_same_axis(audit_axis, topk=TOPK_PER_LABEL, margin_min=MARGIN_GOOD)
        strong_maps[axis] = strong_map

    # 5) 감사표 저장(축 합본)
    audit_df = pd.concat(audit_all, ignore_index=True)
    audit_df.sort_values(["axis", "margin", "best_z"], ascending=[True, False, False]) \
            .to_csv(OUT_AUDIT_CSV, index=False, encoding="utf-8-sig")

    # 6) (좋은) 축 간 겹치는 단어: strong_maps 기준으로 여러 축에서 strong에 든 토큰
    token_to_axes_labels: Dict[str, List[Dict[str, str]]] = {}
    for axis, lab_map in strong_maps.items():
        for lab, toks in lab_map.items():
            for t in toks:
                token_to_axes_labels.setdefault(t, []).append({"axis": axis, "label": lab})
    good_cross_axis = {t: v for t, v in token_to_axes_labels.items() if len(v) > 1}

    # 7) (나쁜) 같은 축 내 라벨 간 애매한 토큰: 감사표에서 margin < MARGIN_BAD
    bad_confusing = {}
    bad_rows = []
    for axis in axis_cols:
        sub = audit_df[audit_df["axis"] == axis]
        bad = sub[sub["margin"] < MARGIN_BAD].copy()
        bad = bad.sort_values(["margin", "freq"], ascending=[True, False])
        rows = []
        for _, r in bad.iterrows():
            item = {
                "token": r["token"],
                "best_label": r["best_label"],
                "second_label": r["second_label"],
                "margin": float(r["margin"]),
                "freq": int(r.get("freq", 0)),
            }
            rows.append(item)
            bad_rows.append({"axis": axis, **item})
        bad_confusing[axis] = rows

    pd.DataFrame(bad_rows).to_csv(OUT_BAD_CSV, index=False, encoding="utf-8-sig")

    # 8) 저장
    with open(OUT_STRONG_JSON, "w", encoding="utf-8") as f:
        json.dump(strong_maps, f, ensure_ascii=False, indent=2)
    with open(OUT_CROSS_JSON, "w", encoding="utf-8") as f:
        json.dump(good_cross_axis, f, ensure_ascii=False, indent=2)
    with open(OUT_BAD_JSON, "w", encoding="utf-8") as f:
        json.dump(bad_confusing, f, ensure_ascii=False, indent=2)

    # 9) 요약
    print("=== 저장 완료 ===")
    print(f"- 강한 트리거(축 내 exclusive): {OUT_STRONG_JSON}")
    print(f"- 다중 축 강 트리거(긍정 신호): {OUT_CROSS_JSON} (tokens={len(good_cross_axis)})")
    print(f"- 혼동 토큰(축 내 마진<{MARGIN_BAD}): {OUT_BAD_JSON}, {OUT_BAD_CSV}")
    print(f"- 감사표: {OUT_AUDIT_CSV}")
    for ax in axis_cols:
        sizes = {lab: len(strong_maps[ax][lab]) for lab in strong_maps[ax]}
        print(f"  {ax}: {sizes}")

if __name__ == "__main__":
    main()
