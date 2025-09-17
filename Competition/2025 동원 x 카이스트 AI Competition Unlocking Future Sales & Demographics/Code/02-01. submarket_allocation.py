# -*- coding: utf-8 -*-
"""
Step 2 — 세분시장 배분(정합 모드, 2012 누락 패치, 동의어 정규화, 2025 캐리) + 시각화

입력:
  1) 연도별 총액(앵커):  ./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2024 연도별 매출액.xlsx
     - Step1 결과(yearly_anchor_forecast.csv)가 있으면 그 파일이 우선(2011~2025 포함)
  2) 세분시장:         ./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2024_세분시장_매출액.xlsx

출력(정합 모드 기준):
  - ./_save/segment_step/segment_shares.csv              # (category, year, segment, share), 각 연도 share 합=1
  - ./_save/segment_step/segment_yearly_amounts.csv      # (category, year, segment, amount), 각 연도 세분 합=앵커 총액
  - ./_save/segment_step/coverage_check.csv              # 세분합/총액 커버리지, 백필 시작연도/마지막 다세분 연도 등
  - ./_save/segment_step/plots_shares/<카테고리>.png      # 100% 스택 구간그래프(2025 음영)

옵션(보고/비교용):
  - 엄격 모드 결과 추가 저장: segment_shares_strict.csv, segment_yearly_amounts_strict.csv
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -----------------------------
# 설정
# -----------------------------
plt.rcParams["axes.unicode_minus"] = False
rcParams["font.family"] = "Malgun Gothic"   # (mac이면 AppleGothic, linux면 NanumGothic 등으로 교체 가능)

# 입력 경로
YEARLY_TOTAL_FP = Path("./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2024 연도별 매출액.xlsx")
SEGMENT_FP      = Path("./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2024_세분시장_매출액.xlsx")

# (선택) Step1 결과(2011~2025 예측 포함)가 있으면 그걸 앵커로 우선 사용
ANCHOR_FROM_STEP1 = Path("./_save/anchor_step/yearly_anchor_forecast.csv")

# 출력 경로
OUTDIR  = Path("./_save/segment_step"); OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR = OUTDIR / "plots_shares"; PLOTDIR.mkdir(parents=True, exist_ok=True)

# 모드 토글
MODE = "reconcile"   # "reconcile"(정합/기본) | "strict"(엄격, 미분류 노출) — 아래 SAVE_STRICT를 True로 하면 병행 저장도 가능
SAVE_STRICT = True   # 엄격 모드 결과도 참고용으로 추가 저장할지 여부

# 스무딩/백필/캐리 설정
EWMA_ALPHA = 0.35    # 세분 share EWMA 스무딩 강도(0=off, 0.2~0.5 권장)
NONZERO_EPS = 1e-12  # 비중/금액 판단 임계
APPLY_LAST_MULTI_SEGMENT_CARRY = True    # True면: 마지막 '다세분(>=2)' 연도의 분포를 미래연도(예: 2025)에 carry
MULTI_SEG_MIN_SEGMENTS = 2               # '다세분' 판단 기준(>=2개 세분이 eps 초과)
CARRY_TARGET_YEARS = [2025]              # 캐리 적용 대상 연도 리스트
CARRY_BLEND = 1.0                        # 1.0=완전 캐리(덮어쓰기), 0.5=기존과 절반 혼합 등

# 시각화
HIGHLIGHT_2025 = True

# -----------------------------
# 유틸: 컬럼 탐지/정규화/동의어
# -----------------------------
def _guess_col(df: pd.DataFrame, keys):
    keys = [k.lower() for k in keys]
    for c in df.columns:
        s = str(c).lower()
        if any(k in s for k in keys):
            return c
    return None

def _clean_txt(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_segment(cat: str, seg: str) -> str:
    """세분 동의어 매핑 — 필요시 확장"""
    c = _clean_txt(cat)
    s = _clean_txt(seg)

    s_low = s.lower().replace(" ", "").replace("-", "").replace("/", "_")

    # 참치캔: '라이트스탠다드/레귤러' → '일반' 통일
    if c == "참치캔":
        if ("라이트" in s) or ("스탠다드" in s) or ("라이트스탠다드" in s) or ("light" in s_low) or ("standard" in s_low) or ("regular" in s_low) or ("레귤러" in s):
            return "일반"
        # 자주 쓰는 세분 우선 반환
        for k in ["일반","고추","야채","김치","불고기","기타","기타가미"]:
            if k in s:
                return k
        return s

    # 조제커피: 띄어쓰기 통일(공백→언더스코어)
    if c == "조제커피":
        return s.replace(" ", "_")

    # 발효유/식육가공품/조미료: 기본 원문 유지(필요시 보강)
    return s

# -----------------------------
# 로더
# -----------------------------
def load_yearly_total(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp) if fp.suffix.lower() in [".xlsx",".xls"] else pd.read_csv(fp, encoding="utf-8")
    ycol = _guess_col(df, ["연도","year"])
    ccol = _guess_col(df, ["카테고리","category","cat"])
    vcol = _guess_col(df, ["매출","amount","value","revenue"])
    gcol = _guess_col(df, ["구분","type","segment"])
    if any(x is None for x in [ycol, ccol, vcol]):
        raise ValueError(f"[총액] 컬럼 감지 실패: {list(df.columns)}")

    if gcol and gcol in df.columns:
        tmp = df.copy()
        tmp[gcol] = tmp[gcol].astype(str).str.strip()
        mask_total = tmp[gcol].astype(str).str.contains("총매출")
        out = tmp.loc[mask_total, [ycol, ccol, vcol]].copy()
    else:
        out = df[[ycol, ccol, vcol]].copy()

    out.columns = ["year","category","amount"]
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["category"] = out["category"].astype(str).map(_clean_txt)
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.dropna(subset=["year","category","amount"]).sort_values(["category","year"])
    return out

def load_segment(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp) if fp.suffix.lower() in [".xlsx",".xls"] else pd.read_csv(fp, encoding="utf-8")
    ycol = _guess_col(df, ["연도","year"])
    ccol = _guess_col(df, ["카테고리","category","cat"])
    scol = _guess_col(df, ["구분","세분","segment","subcat"])
    vcol = _guess_col(df, ["매출","amount","value","revenue"])
    if any(x is None for x in [ycol, ccol, scol, vcol]):
        raise ValueError(f"[세분] 컬럼 감지 실패: {list(df.columns)}")

    out = df[[ycol, ccol, scol, vcol]].copy()
    out.columns = ["year","category","segment","amount"]
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["category"] = out["category"].astype(str).map(_clean_txt)
    out["segment"] = out.apply(lambda r: _norm_segment(r["category"], str(r["segment"])), axis=1)
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.dropna(subset=["year","category","segment","amount"]).sort_values(["category","year","segment"])
    return out

# -----------------------------
# 메인 로직
# -----------------------------
# 1) 앵커(총액) 확보
if ANCHOR_FROM_STEP1.exists():
    anc = pd.read_csv(ANCHOR_FROM_STEP1, encoding="utf-8")
    ycol = _guess_col(anc, ["year"]); ccol = _guess_col(anc, ["category"]); vcol = _guess_col(anc, ["amount"])
    anc = anc[[ccol, ycol, vcol]].copy(); anc.columns = ["category","year","amount"]
    anc["year"] = pd.to_numeric(anc["year"], errors="coerce").astype("Int64")
    anc["category"] = anc["category"].astype(str).map(_clean_txt)
    anc["amount"] = pd.to_numeric(anc["amount"], errors="coerce")
    anc = anc.dropna(subset=["category","year","amount"])
else:
    anc = load_yearly_total(YEARLY_TOTAL_FP)

seg = load_segment(SEGMENT_FP)

# 2) 피벗 집계
anc_piv = anc.groupby(["category","year"], as_index=False)["amount"].sum()
seg_piv = seg.groupby(["category","year","segment"], as_index=False)["amount"].sum()

# 3) 세분 비중(정합용) 계산 + 스무딩 + 백필 + 2025 캐리
share_rows = []
cov_rows   = []

for cat, g_total in anc_piv.groupby("category"):
    years = sorted(g_total["year"].astype(int).unique().tolist())
    g_seg = seg_piv[seg_piv["category"]==cat].copy()

    # wide pivot: index=year, columns=segment
    pw = g_seg.pivot_table(index="year", columns="segment", values="amount", aggfunc="sum").sort_index()
    pw = pw.reindex(years).fillna(0.0)  # 빈 해=0

    # 연도별 세분합
    row_sum = pw.sum(axis=1)

    # 첫 '양(+) 세분합' 연도 탐색(2012 누락 방지 핵심)
    pos_years = row_sum[row_sum > 0].index.tolist()
    if len(pos_years) == 0:
        # 해당 카테고리에 세분 데이터가 전무 → 강제 단일 세분으로 분배
        pw = pd.DataFrame({"Unsegmented": np.ones(len(years))}, index=years)
        row_sum = pw.sum(axis=1)
        first_pos = years[0]
        used_strategy = "no_segment_data_fallback"
    else:
        first_pos = int(min(pos_years))
        used_strategy = "first_positive_backfill"

    # 기본 비중(0 나누기 회피)
    share = pw.div(row_sum.replace(0.0, np.nan), axis=0)

    # EWMA 스무딩
    if EWMA_ALPHA and 0 < EWMA_ALPHA < 1:
        share = share.ewm(alpha=EWMA_ALPHA, adjust=False).mean()

    # 백필: first_pos 이전 연도 = first_pos 분포로 채움
    if len(pos_years) > 0:
        first_dist = share.loc[first_pos].fillna(0.0).values
        share.loc[:first_pos, :] = first_dist

    # 결측 보간 후 행 합=1 재정규화
    share = share.ffill().bfill()
    share = share.div(share.sum(axis=1), axis=0).fillna(0.0)

    # ---- 2025 캐리(마지막 다세분 연도 분포를 미래로)
    if APPLY_LAST_MULTI_SEGMENT_CARRY:
        # 다세분 연도 탐색(가장 최근)
        seg_counts = (share > NONZERO_EPS).sum(axis=1)
        multi_years = seg_counts[seg_counts >= MULTI_SEG_MIN_SEGMENTS].index.tolist()
        last_multi_year = int(max(multi_years)) if len(multi_years) else None

        if last_multi_year is not None:
            carry_vec = share.loc[last_multi_year].values  # 기준 분포
            for ty in CARRY_TARGET_YEARS:
                if ty in share.index:
                    # 캐리 대상 해의 기존 분포와 혼합
                    if CARRY_BLEND >= 1.0:
                        new_vec = carry_vec
                    elif CARRY_BLEND <= 0.0:
                        new_vec = share.loc[ty].values
                    else:
                        new_vec = CARRY_BLEND * carry_vec + (1.0 - CARRY_BLEND) * share.loc[ty].values
                    # 재정규화 후 적용
                    s = float(np.nansum(new_vec))
                    if s > 0:
                        share.loc[ty, :] = new_vec / s
        else:
            last_multi_year = np.nan
    else:
        last_multi_year = np.nan

    # 커버리지/메타 저장
    g_cov = anc_piv[anc_piv["category"]==cat].merge(
        row_sum.rename("seg_sum"), left_on="year", right_index=True, how="left"
    )
    g_cov["seg_sum"] = g_cov["seg_sum"].fillna(0.0)
    g_cov["cov_ratio"] = g_cov.apply(
        lambda r: (r["seg_sum"]/r["amount"]) if (pd.notna(r["amount"]) and r["amount"]>0) else np.nan, axis=1
    )
    g_cov["first_positive_year"] = first_pos
    g_cov["last_multi_segment_year"] = last_multi_year
    g_cov["multi_seg_threshold"] = MULTI_SEG_MIN_SEGMENTS
    g_cov["strategy"] = used_strategy + ("+carry" if APPLY_LAST_MULTI_SEGMENT_CARRY else "")
    cov_rows.append(g_cov.assign(category=cat))

    # long rows 저장
    for y in years:
        for seg_name in share.columns:
            share_rows.append({
                "category": cat,
                "year": int(y),
                "segment": seg_name,
                "share": float(share.loc[y, seg_name])
            })

segment_shares = pd.DataFrame(share_rows).sort_values(["category","year","segment"]).reset_index(drop=True)
coverage_check = pd.concat(cov_rows, ignore_index=True)[
    ["category","year","amount","seg_sum","cov_ratio","first_positive_year","last_multi_segment_year","multi_seg_threshold","strategy"]
].rename(columns={"amount":"cat_total"}).sort_values(["category","year"]).reset_index(drop=True)

# 4-A) 정합 모드 분배: share * 앵커 총액
base = anc_piv.rename(columns={"amount":"anchor_total"})
seg_amounts = (
    segment_shares.merge(base, on=["category","year"], how="left")
                  .assign(amount=lambda d: d["share"] * d["anchor_total"])
                  [["category","year","segment","amount"]]
                  .sort_values(["category","year","segment"])
)

# 4-B) (옵션) 엄격 모드(보고/비교용): 원 세분합/총액 그대로 → 미분류 포함
if SAVE_STRICT:
    strict_rows = []
    for (c, y), g in seg_piv.groupby(["category","year"]):
        cat_total = float(anc_piv[(anc_piv["category"]==c) & (anc_piv["year"]==y)]["amount"].sum())
        seg_sum   = float(g["amount"].sum())
        if not (cat_total > 0):  # 총액이 0/결측이면 스킵
            continue
        cur_sum_share = 0.0
        if seg_sum > 0:
            for _, r in g.iterrows():
                share_strict = float(r["amount"] / seg_sum) * (seg_sum / cat_total)  # = r.amount / cat_total
                strict_rows.append({"category": c, "year": int(y), "segment": r["segment"], "share": share_strict})
                cur_sum_share += share_strict
        # 미분류 잔여
        residual = max(0.0, 1.0 - cur_sum_share)
        if residual > 1e-12:
            strict_rows.append({"category": c, "year": int(y), "segment": "미분류", "share": residual})
    segment_shares_strict = pd.DataFrame(strict_rows)
    segment_shares_strict = (segment_shares_strict
                             .groupby(["category","year","segment"], as_index=False)["share"].sum())
    # 금액 환산
    seg_amounts_strict = (segment_shares_strict.merge(base, on=["category","year"], how="left")
                          .assign(amount=lambda d: d["share"] * d["anchor_total"])
                          [["category","year","segment","amount"]]
                          .sort_values(["category","year","segment"]))
else:
    segment_shares_strict = None
    seg_amounts_strict = None

# 5) 저장
SEG_SHARE_FP = OUTDIR / "segment_shares.csv"
SEG_AMT_FP   = OUTDIR / "segment_yearly_amounts.csv"
COV_FP       = OUTDIR / "coverage_check.csv"
segment_shares.to_csv(SEG_SHARE_FP, index=False, encoding="utf-8-sig")
seg_amounts.to_csv(SEG_AMT_FP, index=False, encoding="utf-8-sig")
coverage_check.to_csv(COV_FP, index=False, encoding="utf-8-sig")
print("[SAVE]", SEG_SHARE_FP.resolve())
print("[SAVE]", SEG_AMT_FP.resolve())
print("[SAVE]", COV_FP.resolve())

if SAVE_STRICT:
    SEG_SHARE_STRICT_FP = OUTDIR / "segment_shares_strict.csv"
    SEG_AMT_STRICT_FP   = OUTDIR / "segment_yearly_amounts_strict.csv"
    segment_shares_strict.to_csv(SEG_SHARE_STRICT_FP, index=False, encoding="utf-8-sig")
    seg_amounts_strict.to_csv(SEG_AMT_STRICT_FP, index=False, encoding="utf-8-sig")
    print("[SAVE]", SEG_SHARE_STRICT_FP.resolve())
    print("[SAVE]", SEG_AMT_STRICT_FP.resolve())

# 6) 시각화(정합 모드 100% 스택) — 2025 음영
for cat in sorted(segment_shares["category"].unique()):
    g = segment_shares[segment_shares["category"]==cat].copy()
    pw = g.pivot_table(index="year", columns="segment", values="share", aggfunc="mean").sort_index()
    years = pw.index.values

    plt.figure(figsize=(10, 4.8))
    bottom = np.zeros(len(pw))
    for seg_name in pw.columns:
        vals = pw[seg_name].values
        plt.bar(years, vals, bottom=bottom, label=seg_name)
        bottom += vals

    if HIGHLIGHT_2025 and (years.min() <= 2025 <= years.max()):
        plt.axvspan(2024.5, 2025.5, color="tab:gray", alpha=0.08)

    plt.ylim(0, 1.02)
    plt.title(f"[세분 비중] {cat} — 2011–{int(years.max())} (정합)")
    plt.xlabel("연도"); plt.ylabel("세분 비중(합=1)")
    plt.xticks(range(int(years.min()), int(years.max())+1, 1))
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    outpng = PLOTDIR / f"{cat}.png"
    plt.savefig(outpng, dpi=150)
    plt.close()
    print("[PLOT]", outpng.resolve())

print("완료: 세분 배분(정합) + (옵션)엄격 결과 저장 + 시각화 생성")
