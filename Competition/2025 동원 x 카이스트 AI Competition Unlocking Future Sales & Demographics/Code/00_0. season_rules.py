# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# =============================
# 저장 경로 (사용자가 직접 지정)
# =============================
SAVE_DIR = Path("./_save/season_analysis")  # TODO: 저장할 경로 지정
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# 한글 폰트 (윈도우 기준)
# =============================
try:
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\malgun.ttf")
    rcParams["font.family"] = "Malgun Gothic"
    rcParams["axes.unicode_minus"] = False
except:
    pass

# =============================
# 유틸
# =============================
def standardize(df):
    """연도/월 → ym 컬럼 생성"""
    if "연도" in df.columns and "월" in df.columns:
        df["ym"] = pd.to_datetime(df["연도"].astype(str) + "-" + df["월"].astype(str).str.zfill(2))
    elif "year" in df.columns and "month" in df.columns:
        df["ym"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2))
    else:
        if "ym" in df.columns:
            df["ym"] = pd.to_datetime(df["ym"])
    return df

def monthly_pattern(df, cat_col, val_col, category, start, end):
    """특정 카테고리의 윈도우별 월비중 패턴"""
    sub = df[(df["ym"] >= start) & (df["ym"] <= end) & (df[cat_col] == category)].copy()
    if sub.empty:
        return None
    sub = sub.groupby("ym")[val_col].sum()
    sub = sub / sub.sum()
    return sub

def plot_pattern(series_dict, title, fname):
    """여러 소스(DW/시장/세분)를 한 그래프에 비교"""
    plt.figure(figsize=(8,4))
    for label, s in series_dict.items():
        if s is not None:
            plt.plot(s.index, s.values, marker="o", label=label)
    plt.xticks(rotation=45)
    plt.title(title); plt.ylabel("월비중"); plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()
    print("[SAVE]", fname)

# =============================
# 메인 로직
# =============================
def compare_pos_patterns(path_dw, path_total, path_seg):
    # 데이터 로드
    df_dw = standardize(pd.read_excel(path_dw))
    df_total = standardize(pd.read_excel(path_total))
    df_seg = standardize(pd.read_excel(path_seg))

    # 윈도우 정의
    WINDOWS = [
        ("2020-07","2021-06"),
        ("2021-07","2022-06"),
        ("2022-07","2023-06"),
    ]

    # 카테고리 목록 (DW 기준)
    categories = df_dw["카테고리"].unique()

    for cat in categories:
        for (s,e) in WINDOWS:
            start, end = pd.to_datetime(s), pd.to_datetime(e)

            pat_dw = monthly_pattern(df_dw,"카테고리","매출액(백만원)",cat,start,end)
            pat_total = monthly_pattern(df_total,"카테고리","매출액(백만원)",cat,start,end)
            pat_seg = monthly_pattern(df_seg,"카테고리","매출액(백만원)",cat,start,end)

            # CSV 저장 (소스별 따로)
            for label, pat in [("DW",pat_dw),("TOTAL",pat_total),("SEG",pat_seg)]:
                if pat is not None:
                    out_csv = SAVE_DIR / f"pattern_{label}_{cat}_{s}_{e}.csv"
                    pat.to_csv(out_csv, encoding="utf-8-sig")
                    print("[SAVE]", out_csv)

            # 비교 그래프 저장
            out_png = SAVE_DIR / f"compare_{cat}_{s}_{e}.png"
            plot_pattern({"DW":pat_dw,"TOTAL":pat_total,"SEG":pat_seg},
                         f"{cat} {s}~{e}", out_png)

# =============================
# 실행 예시
# =============================
if __name__ == "__main__":
    compare_pos_patterns(
        "./_data/dacon/dongwon/pos_data/marketlink_POS_2020_2023_동원 F&B_매출액.xlsx",
        "./_data/dacon/dongwon/pos_data/marketlink_POS_2020_2023_월별 매출액.xlsx",
        "./_data/dacon/dongwon/pos_data/marketlink_POS_2020_2023_세분시장_매출액.xlsx"
    )
