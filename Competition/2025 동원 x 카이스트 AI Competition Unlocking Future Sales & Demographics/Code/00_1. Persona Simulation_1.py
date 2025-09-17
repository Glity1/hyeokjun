# -*- coding: utf-8 -*-
"""
단일 제품(덴마크 하이그릭요거트 400g) 페르소나 기반 구매 시뮬레이션 → 제출 파일 생성
- sample_submission.csv의 컬럼/순서를 그대로 유지하여 저장
- 샘플에 동일 제품 행이 있으면 해당 행만 갱신, 없으면 같은 형식으로 새 행 추가
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

# =======================
# 0) 경로 & 기본 설정
# =======================
SAMPLE_SUB_PATH = Path("./_data/dacon/dongwon/sample_submission.csv")   # ← 위치 조정
OUTPUT_PATH     = Path("./submission_from_sim_yogurt.csv")

PRODUCT_KEY     = "덴마크 하이그릭요거트 400g"
N_SIM           = 5000
USE_POISSON     = True       # 월별 변동성 샘플링(포아송)
PRESERVE_TOTAL  = False      # 연간 총량 보존 모드
RANDOM_SEED     = 42
np.random.seed(RANDOM_SEED)

# =======================
# 1) 페르소나 JSON (사용자 제공)
# =======================
persona_json = {
  "덴마크 하이그릭요거트 400g": [
    {
      "연령": {"value": "28세", "weight": 0.8},
      "성별": {"value": "여성", "weight": 0.7},
      "소득 구간": {"value": "월 250~350만원", "weight": 0.7},
      "거주 지역": {"value": "서울 강남구", "weight": 0.9},
      "직업": {"value": "마케팅 대리", "weight": 0.7},
      "가족 구성": {"value": "싱글", "weight": 0.9},
      "라이프스타일": {"value": "헬스·요가 중심 자기관리형", "weight": 0.9},
      "건강 관심도": {"value": "매우 높음", "weight": 0.95},
      "브랜드 충성도": {"value": "중간", "weight": 0.6},
      "구매 채널": {"value": "온라인 마켓컬리·쿠팡", "weight": 0.9},
      "프로모션 민감도": {"value": "높음", "weight": 0.7},
      "계절 이벤트 반응": {"value": "연말 건강·선물 프로모션 반응↑", "weight": 0.8},
      "트렌드 반응도": {"value": "SNS·신제품 트렌드 민감", "weight": 0.9},
      "purchase_probability": 77,
      "monthly_by_launch": [3, 3, 4, 3, 4, 6, 5, 3, 3, 3, 4, 6]
    },
    {
      "연령": {"value": "42세", "weight": 0.8},
      "성별": {"value": "남성", "weight": 0.6},
      "소득 구간": {"value": "월 500~700만원", "weight": 0.9},
      "거주 지역": {"value": "경기 분당구", "weight": 0.8},
      "직업": {"value": "IT 기업 팀장", "weight": 0.8},
      "가족 구성": {"value": "아내+자녀2명", "weight": 0.9},
      "라이프스타일": {"value": "가족 건강 중심, 주말 캠핑", "weight": 0.7},
      "건강 관심도": {"value": "높음", "weight": 0.8},
      "브랜드 충성도": {"value": "높음", "weight": 0.85},
      "구매 채널": {"value": "대형마트·온라인 정기배송", "weight": 0.8},
      "프로모션 민감도": {"value": "중간", "weight": 0.5},
      "계절 이벤트 반응": {"value": "설날·추석 선물세트 선호", "weight": 0.9},
      "트렌드 반응도": {"value": "낮음", "weight": 0.4},
      "purchase_probability": 82,
      "monthly_by_launch": [4, 5, 7, 4, 5, 6, 8, 5, 4, 5, 5, 7]
    },
    {
      "연령": {"value": "22세", "weight": 0.9},
      "성별": {"value": "여성", "weight": 0.7},
      "소득 구간": {"value": "월 100~150만원", "weight": 0.6},
      "거주 지역": {"value": "부산 해운대구", "weight": 0.6},
      "직업": {"value": "대학생", "weight": 0.9},
      "가족 구성": {"value": "자취", "weight": 0.8},
      "라이프스타일": {"value": "SNS·인플루언서 트렌드 추종", "weight": 0.9},
      "건강 관심도": {"value": "보통", "weight": 0.5},
      "브랜드 충성도": {"value": "낮음", "weight": 0.3},
      "구매 채널": {"value": "편의점·배달 플랫폼", "weight": 0.9},
      "프로모션 민감도": {"value": "매우 높음", "weight": 0.9},
      "계절 이벤트 반응": {"value": "연말 한정판 구매↑", "weight": 0.8},
      "트렌드 반응도": {"value": "매우 높음", "weight": 0.95},
      "purchase_probability": 61,
      "monthly_by_launch": [2, 2, 3, 2, 3, 4, 3, 3, 2, 2, 3, 5]
    }
  ]
}

# =======================
# 2) 유틸
# =======================
def detect_columns_keep_order(df: pd.DataFrame):
    """샘플 제출 파일에서 ID 컬럼과 월 컬럼(12개)을 원래 순서 그대로 추출"""
    cols = list(df.columns)
    id_col = cols[0]
    # 우선 months_since_launch_1..12 패턴으로 탐색
    pat = re.compile(r"^months_since_launch_(\d{1,2})$")
    month_cols = [c for c in cols if pat.match(c)]
    if len(month_cols) == 12:
        # 숫자 순으로 정렬하되, 원본 순서가 이미 정렬되어 있으면 그대로 둬도 무방
        month_cols = sorted(month_cols, key=lambda c: int(pat.match(c).group(1)))
    else:
        # fallback: ID 제외 나머지를 모두 월 컬럼으로 보고 순서 유지
        month_cols = [c for c in cols if c != id_col]
        if len(month_cols) != 12:
            raise RuntimeError("샘플에서 월 컬럼(12개) 탐지 실패: columns=" + ",".join(cols))
    return id_col, month_cols

def expected_total_12m(personas: list) -> float:
    s = 0.0
    for p in personas:
        prob = float(p.get("purchase_probability", 0)) / 100.0
        base = np.array(p["monthly_by_launch"], dtype=float)
        s += prob * base.sum()
    return float(s)

def simulate_once(personas: list, use_poisson: bool = True) -> np.ndarray:
    monthly = np.zeros(12, dtype=float)
    for p in personas:
        prob = float(p.get("purchase_probability", 0)) / 100.0
        if np.random.rand() >= prob:
            continue
        base = np.array(p["monthly_by_launch"], dtype=float)
        sampled = np.random.poisson(lam=np.clip(base, 0, None)) if use_poisson else base
        monthly += sampled
    return monthly

def run_simulation(personas: list, n_sim: int, use_poisson: bool, preserve_total: bool) -> np.ndarray:
    res = np.zeros((n_sim, 12), dtype=float)
    target_T = expected_total_12m(personas) if preserve_total else None
    for i in range(n_sim):
        m = simulate_once(personas, use_poisson=use_poisson)
        if preserve_total:
            s = m.sum()
            if s > 0 and target_T and target_T > 0:
                m = m * (target_T / s)
        res[i, :] = m
    return res

def fill_submission_like_sample(sample_df: pd.DataFrame,
                                id_col: str,
                                month_cols: list,
                                product_key: str,
                                monthly_values: np.ndarray) -> pd.DataFrame:
    """샘플 형식/순서를 그대로 유지하여 한 제품의 월 값을 채움(행이 없으면 추가)"""
    out = sample_df.copy()
    vals = np.maximum(np.round(monthly_values).astype(int), 1)  # 최소 1, 정수
    mask = out[id_col] == product_key
    if mask.any():
        out.loc[mask, month_cols] = vals
    else:
        row = {id_col: product_key}
        row.update({c: v for c, v in zip(month_cols, vals)})
        # 샘플 컬럼 순서를 보존하여 추가
        append_df = pd.DataFrame([row], columns=list(out.columns))
        out = pd.concat([out, append_df], axis=0, ignore_index=True)
    # NaN 방지
    for c in month_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(1).astype(int)
    return out

# =======================
# 3) 메인
# =======================
def main():
    # 샘플 제출 파일 로드 & 형식 파악
    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(f"샘플 제출 파일을 찾을 수 없습니다: {SAMPLE_SUB_PATH}")
    sample_df = pd.read_csv(SAMPLE_SUB_PATH)
    id_col, month_cols = detect_columns_keep_order(sample_df)

    # 시뮬레이션 실행
    personas = persona_json.get(PRODUCT_KEY, [])
    if not personas:
        raise RuntimeError(f"페르소나가 없습니다: {PRODUCT_KEY}")
    sim_results = run_simulation(personas, N_SIM, USE_POISSON, PRESERVE_TOTAL)
    monthly_mean = sim_results.mean(axis=0)  # 기대 월별 판매량

    # 샘플 형식 그대로 채워 넣기
    submission = fill_submission_like_sample(sample_df, id_col, month_cols, PRODUCT_KEY, monthly_mean)

    # 저장(샘플과 동일한 헤더/순서, UTF-8-SIG)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {OUTPUT_PATH}  |  rows={len(submission)}  id_col='{id_col}'  months={month_cols[0]}..{month_cols[-1]}")

if __name__ == "__main__":
    main()
