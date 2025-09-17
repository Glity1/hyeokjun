# -*- coding: utf-8 -*-
"""
Nielsen Half-Year Forecast Eval (YoY vs ETS/ARIMA) — 중복 집계 안정화 버전
- 컬럼 고정: ['연도','분기' or '반기','구분','카테고리','매출액(백만원)']
- 2011~2019: 분기 → 반기 집계
- 2020~2024: 반기 그대로 사용
- 훈련: ~2024H1 (2024H2는 검증용으로 제외)
- 출력: 카테고리별 2024H2 예측치(YoY/ETS/ARIMA), YoY 비율(%), 그래프, 평가표 CSV
"""

from __future__ import annotations
import re, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Korean font setup for Matplotlib (Windows/macOS/Linux) ----
import os, platform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

def setup_korean_font():
    # 한글 마이너스 기호 깨짐 방지
    rcParams["axes.unicode_minus"] = False

    # OS별 후보 폰트
    system = platform.system()
    if system == "Windows":
        candidates = ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Noto Sans CJK KR"]
    elif system == "Darwin":  # macOS
        candidates = ["AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
    else:  # Linux/기타
        candidates = ["NanumGothic", "Noto Sans CJK KR", "AppleGothic", "Malgun Gothic"]

    # 설치 폰트 탐색
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in installed:
            chosen = name
            break

    if chosen is None:
        # 수동 경로 등록 옵션 (필요 시 경로 지정)
        # 예시:
        # font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        # font_manager.fontManager.addfont(font_path)
        # chosen = font_manager.FontProperties(fname=font_path).get_name()
        print("⚠️ 한글 폰트를 찾지 못했습니다. 'NanumGothic' 또는 'Noto Sans CJK KR' 설치/경로 등록을 권장합니다.")
    else:
        rcParams["font.family"] = chosen
        print(f"[Matplotlib] Using Korean font: {chosen}")

    # (선택) 폰트 캐시 문제시: 캐시 삭제 후 재생성 안내
    # import shutil
    # cache_dir = matplotlib.get_cachedir()
    # print("Matplotlib cache:", cache_dir)
    # shutil.rmtree(cache_dir, ignore_errors=True)

setup_korean_font()


# ============== 설정(경로) ==============
NIELSEN_Q_FP = Path("./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2019_분기별 매출액.xlsx")
NIELSEN_H_FP = Path("./_data/dacon/dongwon/pos_data/닐슨코리아_2020_2024 반기별 매출액.xlsx")
SAVE_DIR     = Path("./_save/market_all/nielsen_half_forecast"); SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 로더(한국어 헤더 고정) ----------
def read_nielsen_quarter_kor(fp: Path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(fp, sheet_name=sheet_name)
    need = ['연도','분기','카테고리','매출액(백만원)']
    for c in need:
        if c not in df.columns:
            raise ValueError(f"분기 파일 컬럼 누락: {c} / 실제: {list(df.columns)}")
    out = df[need].copy()
    out.columns = ["year","quarter","category","amount"]
    out["quarter"] = out["quarter"].astype(str).str.extract(r"(\d+)")[0].astype(int).clip(1,4)
    out["amount"]  = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    # 카테고리 공백/이상치 정리
    out["category"] = out["category"].astype(str).str.strip()
    return out

def read_nielsen_half_kor(fp: Path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(fp, sheet_name=sheet_name)
    half_col = "반기" if "반기" in df.columns else ("분기" if "분기" in df.columns else None)
    if half_col is None:
        raise ValueError(f"반기 파일에 '반기' 또는 '분기' 컬럼이 없습니다. 실제: {list(df.columns)}")
    need = ['연도', half_col, '카테고리', '매출액(백만원)']
    for c in need:
        if c not in df.columns:
            raise ValueError(f"반기 파일 컬럼 누락: {c} / 실제: {list(df.columns)}")

    out = df[need].copy()
    out.columns = ["year","half_raw","category","amount"]
    s = out["half_raw"].astype(str).str.upper()
    out["half"] = np.where(s.str.contains("2"), "H2", "H1")  # 'H1/H2' 또는 '1/2' 모두 허용
    out["amount"]  = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out["category"] = out["category"].astype(str).str.strip()
    return out[["category","year","half","amount"]]

def quarter_to_half(df_q: pd.DataFrame) -> pd.DataFrame:
    qq = df_q.copy()
    qq["half"] = np.where(qq["quarter"].isin([1,2]), "H1", "H2")
    hh = qq.groupby(["category","year","half"], as_index=False)["amount"].sum()
    return hh

def concat_half_series(h_from_q: pd.DataFrame, h_direct: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([h_from_q, h_direct], ignore_index=True)
    # ★ 중요: 분기/반기 소스 결합 후에도 중복이 있을 수 있으므로 한 번 더 합산
    df = df.groupby(["category","year","half"], as_index=False)["amount"].sum()
    df["_ho"] = df["half"].map({"H1":1,"H2":2})
    df = df.sort_values(["category","year","_ho"]).drop(columns="_ho").reset_index(drop=True)
    return df

# ---------- 평가지표 ----------
def mape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mask = y > 0
    return np.mean(np.abs((yhat[mask]-y[mask]) / y[mask])) * 100 if mask.any() else np.nan

def smape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = (np.abs(y) + np.abs(yhat)).clip(1e-9, None)
    return np.mean(np.abs(y - yhat) / denom) * 100

def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y,float) - np.asarray(yhat,float))))

def wape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.sum(np.abs(y)).clip(1e-9)
    return np.sum(np.abs(y - yhat)) / denom * 100

# ---------- 예측 (집계 안전장치 포함) ----------
def _aggregate_half(df_cat: pd.DataFrame) -> pd.DataFrame:
    """카테고리 단일 서브프레임을 연도·반기별로 합산하여 중복 제거."""
    g = (df_cat.groupby(["year","half"], as_index=False)["amount"].sum()
               .sort_values(["year","half"]))
    return g

def forecast_yoy_for_cat(train_half_df_cat: pd.DataFrame) -> dict:
    g = _aggregate_half(train_half_df_cat)
    piv = g.pivot(index="year", columns="half", values="amount").sort_index()
    a_2023h1 = float(piv.loc[2023, "H1"]) if (2023 in piv.index and "H1" in piv.columns) else np.nan
    a_2023h2 = float(piv.loc[2023, "H2"]) if (2023 in piv.index and "H2" in piv.columns) else np.nan
    a_2024h1 = float(piv.loc[2024, "H1"]) if (2024 in piv.index and "H1" in piv.columns) else np.nan

    if np.isfinite(a_2023h2) and np.isfinite(a_2023h1) and np.isfinite(a_2024h1) and a_2023h1>0:
        r1 = a_2024h1 / a_2023h1
        y_2024h2 = a_2023h2 * r1
    else:
        r1 = np.nan; y_2024h2 = a_2023h2 if np.isfinite(a_2023h2) else np.nan

    if np.isfinite(y_2024h2) and np.isfinite(a_2023h2) and np.isfinite(a_2024h1) and a_2023h2>0:
        r2 = y_2024h2 / a_2023h2
        y_2025h1 = a_2024h1 * r2
    else:
        r2 = np.nan; y_2025h1 = a_2024h1 if np.isfinite(a_2024h1) else np.nan

    return {
        "2024H2": y_2024h2, "2025H1": y_2025h1,
        "r1_2024H2_from_2024H1_over_2023H1": r1,
        "r2_2025H1_from_2024H2_over_2023H2": r2
    }

def forecast_ETS_for_cat(train_half_df_cat: pd.DataFrame) -> dict:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception:
        return {"2024H2": np.nan, "2025H1": np.nan}
    g = _aggregate_half(train_half_df_cat)
    y = g["amount"].astype(float).values
    if len(y) < 6:
        return {"2024H2": np.nan, "2025H1": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        f = fit.forecast(2)
    return {"2024H2": float(f[0]), "2025H1": float(f[1])}

def forecast_ARIMA_for_cat(train_half_df_cat: pd.DataFrame) -> dict:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        return {"2024H2": np.nan, "2025H1": np.nan}
    g = _aggregate_half(train_half_df_cat)
    y = g["amount"].astype(float).values
    if len(y) < 6:
        return {"2024H2": np.nan, "2025H1": np.nan}
    def _try(order):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(y, order=order).fit()
            f = res.forecast(2)
            return float(f.iloc[0]), float(f.iloc[1])
    for order in [(1,1,1),(0,1,1),(1,1,0),(0,1,0)]:
        try:
            a,b = _try(order); return {"2024H2": a, "2025H1": b}
        except Exception:
            continue
    return {"2024H2": np.nan, "2025H1": np.nan}

# ---------- 시각화 ----------
def plot_history_vs_forecast(cat: str, train_half_df_cat: pd.DataFrame,
                             ets_pred: dict, arima_pred: dict, save_dir: Path):
    g = _aggregate_half(train_half_df_cat)
    x_hist = np.arange(len(g))
    labels_hist = [f"{y}-{h}" for y,h in zip(g["year"], g["half"])]
    y_hist = g["amount"].values

    x_f = np.arange(len(g), len(g)+2)
    labels_f = ["2024-H2","2025-H1"]
    y_ets = [ets_pred.get("2024H2", np.nan), ets_pred.get("2025H1", np.nan)]
    y_ari = [arima_pred.get("2024H2", np.nan), arima_pred.get("2025H1", np.nan)]

    plt.figure(figsize=(9,4.5))
    plt.plot(x_hist, y_hist, marker="o", label="History (to 2024H1)")
    plt.plot(x_f, y_ets, marker="o", linestyle="--", label="ETS Forecast")
    plt.plot(x_f, y_ari, marker="o", linestyle=":",  label="ARIMA Forecast")
    plt.axvline(x=len(g)-0.5, color="gray", linestyle="--", linewidth=1)
    plt.xticks(list(x_hist)+list(x_f), labels_hist+labels_f, rotation=45)
    plt.title(f"[{cat}] 반기 시계열: 히스토리 vs ETS/ARIMA (단위: 백만원)")
    plt.xlabel("Half"); plt.ylabel("Amount (백만원)"); plt.legend(); plt.tight_layout()
    fn = save_dir / f"history_vs_forecast_{re.sub(r'[^0-9A-Za-z가-힣]+','_',cat)}.png"
    plt.savefig(fn, dpi=150); plt.close()

def plot_actual_vs_pred_2024H2(cat: str, actual: float, yoy: float, ets: float, arima: float, save_dir: Path):
    labels = ["Actual 2024H2","YoY","ETS","ARIMA"]
    vals   = [actual, yoy, ets, arima]
    x = np.arange(len(labels))
    plt.figure(figsize=(6.8,4.2))
    plt.bar(x, vals)
    plt.xticks(x, labels)
    plt.title(f"[{cat}] 2024H2: 실제 vs 예측 (단위: 백만원)")
    plt.ylabel("Amount (백만원)")
    plt.tight_layout()
    fn = save_dir / f"actual_vs_pred_2024H2_{re.sub(r'[^0-9A-Za-z가-힣]+','_',cat)}.png"
    plt.savefig(fn, dpi=150); plt.close()

# ---------- 메인 ----------
if __name__ == "__main__":
    # 1) 로드 & 반기로 통일 (+중복합산)
    qdf = read_nielsen_quarter_kor(NIELSEN_Q_FP)
    hdf = read_nielsen_half_kor(NIELSEN_H_FP)
    h_from_q  = quarter_to_half(qdf)
    half_all  = concat_half_series(h_from_q, hdf)       # 이미 category/year/half로 합산

    cats = sorted(half_all["category"].unique())

    print("\n================ 예측 & 검증 요약(단위: 백만원) ================")
    eval_rows = []
    for cat in cats:
        g_all   = half_all[half_all["category"]==cat].copy()
        # train: 2024H2 제외
        g_train = g_all[~((g_all["year"]==2024) & (g_all["half"]=="H2"))].copy()

        # 실제 2024H2
        y_true_row = g_all[(g_all["year"]==2024) & (g_all["half"]=="H2")]
        y_true = float(y_true_row["amount"].sum()) if not y_true_row.empty else np.nan

        # 예측
        yoy  = forecast_yoy_for_cat(g_train)
        ets  = forecast_ETS_for_cat(g_train)
        arim = forecast_ARIMA_for_cat(g_train)

        y_yoy   = yoy.get("2024H2", np.nan)
        y_ets   = ets.get("2024H2", np.nan)
        y_arima = arim.get("2024H2", np.nan)

        # 지표
        def _m(yhat): return mape([y_true],[yhat]) if np.isfinite(y_true) and np.isfinite(yhat) else np.nan
        def _s(yhat): return smape([y_true],[yhat]) if np.isfinite(y_true) and np.isfinite(yhat) else np.nan
        def _a(yhat): return mae([y_true],[yhat])  if np.isfinite(y_true) and np.isfinite(yhat) else np.nan

        mape_yoy, mape_ets, mape_arima = _m(y_yoy), _m(y_ets), _m(y_arima)
        smape_yoy, smape_ets, smape_arima = _s(y_yoy), _s(y_ets), _s(y_arima)
        mae_yoy, mae_ets, mae_arima = _a(y_yoy), _a(y_ets), _a(y_arima)

        def _fmt(x):  return "nan" if (x is None or not np.isfinite(x)) else f"{x:,.0f}"
        def _pct(r):  return "nan" if (r is None or not np.isfinite(r)) else f"{(r-1)*100:+.1f}%"

        print(f"\n■ 카테고리: {cat}")
        print(f"  - Actual 2024H2: {_fmt(y_true)}")
        print(f"  - YoY 예측      : 2024H2={_fmt(y_yoy)}, 2025H1={_fmt(yoy.get('2025H1'))}")
        print(f"    · YoY 비율(2024H2): 2024H1 / 2023H1 = {_pct(yoy.get('r1_2024H2_from_2024H1_over_2023H1'))}")
        print(f"    · YoY 비율(2025H1): 2024H2 / 2023H2 = {_pct(yoy.get('r2_2025H1_from_2024H2_over_2023H2'))}")
        print(f"  - ETS  예측      : 2024H2={_fmt(y_ets)},  2025H1={_fmt(ets.get('2025H1'))}")
        print(f"  - ARIMA예측      : 2024H2={_fmt(y_arima)}, 2025H1={_fmt(arim.get('2025H1'))}")
        if np.isfinite(y_true):
            print(f"  - SMAPE(%) YoY/ETS/ARIMA: {smape_yoy:.2f} / {smape_ets:.2f} / {smape_arima:.2f}")
            print(f"  -  MAPE(%) YoY/ETS/ARIMA: {mape_yoy:.2f} / {mape_ets:.2f} / {mape_arima:.2f}")

        eval_rows.append({
            "category": cat,
            "actual_2024H2": y_true,
            "pred_yoy_2024H2": y_yoy, "pred_ets_2024H2": y_ets, "pred_arima_2024H2": y_arima,
            "mape_yoy": mape_yoy, "mape_ets": mape_ets, "mape_arima": mape_arima,
            "smape_yoy": smape_yoy, "smape_ets": smape_ets, "smape_arima": smape_arima,
            "mae_yoy": mae_yoy, "mae_ets": mae_ets, "mae_arima": mae_arima
        })

        # 그래프 저장
        plot_history_vs_forecast(cat, g_train, ets_pred=ets, arima_pred=arim, save_dir=SAVE_DIR)
        if np.isfinite(y_true):
            plot_actual_vs_pred_2024H2(cat, y_true, y_yoy, y_ets, y_arima, save_dir=SAVE_DIR)

    # 3) 요약 저장
    ev = pd.DataFrame(eval_rows)
    ev.to_csv(SAVE_DIR/"eval_2024H2.csv", index=False, encoding="utf-8-sig")
    print("\n저장:", str((SAVE_DIR/'eval_2024H2.csv').resolve()))
    print("그래프 예:", str((SAVE_DIR/'history_vs_forecast_...png').resolve()))
