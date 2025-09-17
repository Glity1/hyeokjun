# -*- coding: utf-8 -*-
"""
연 매출 앵커 — 빠른 개선안 버전
- 카테고리별 롤링-오리진 1스텝 CV로 방법 가중치 학습(sMAPE 역수)
- 방법 후보: ETS, Damped ETS, Theta(가능시), YoY(중위수 성장률), Linear-Log
- 식육가공품: 2019 이후 공백 → 2020~2025 멀티스텝 예측(끊김 없는 점선)
- 간이 예측구간(PI, 80%): 로그수익률 변동성 기반

입력:  ./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2024 연도별 매출액.xlsx
출력:
  - ./_save/anchor_step/yearly_anchor_forecast.csv
  - ./_save/anchor_step/yearly_anchor_summary.csv
  - ./_save/anchor_step/yearly_anchor_forecast_decomposition.csv   # 학습된 가중치 + 방법별 2025
  - ./_save/anchor_step/plots/<카테고리>.png
"""

from __future__ import annotations
import warnings, math
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -----------------------------
# 한글 폰트 설정
# -----------------------------
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.family'] = 'Malgun Gothic'   # mac: AppleGothic / linux: NanumGothic 등으로 변경 가능

# -----------------------------
# 경로
# -----------------------------
INPATH = Path("./_data/dacon/dongwon/pos_data/닐슨코리아_2011_2024 연도별 매출액.xlsx")
OUTDIR = Path("./_save/anchor_step"); OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR = OUTDIR / "plots"; PLOTDIR.mkdir(parents=True, exist_ok=True)

TARGET_YEAR = 2025
CV_MIN_WINDOW = 6   # CV 시작 최소 길이(연)
EPS = 1e-8

# -----------------------------
# 로더
# -----------------------------
def _guess_col(df: pd.DataFrame, keys):
    keys = [k.lower() for k in keys]
    for c in df.columns:
        s = str(c).lower()
        if any(k in s for k in keys):
            return c
    return None

def load_yearly(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path) if path.suffix.lower() in [".xlsx",".xls"] else pd.read_csv(path, encoding="utf-8")
    ycol = _guess_col(df, ["연도","year"])
    ccol = _guess_col(df, ["카테고리","category","cat"])
    vcol = _guess_col(df, ["매출","amount","value","revenue"])
    if any(x is None for x in [ycol, ccol, vcol]):
        raise ValueError(f"컬럼 추론 실패: year={ycol}, category={ccol}, value={vcol}, 실제={list(df.columns)}")
    out = df[[ycol, ccol, vcol]].copy()
    out.columns = ["year","category","amount"]
    out["year"]   = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out["category"] = out["category"].astype(str).str.strip()
    out = out.dropna(subset=["year","category","amount"]).sort_values(["category","year"]).reset_index(drop=True)
    return out

# -----------------------------
# 방법 정의
# -----------------------------
@dataclass
class MethodForecast:
    name: str
    weight: float = 0.0
    valid: bool = False
    yhat_2025: float = np.nan

def _ets_1step(y: np.ndarray) -> float:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated").fit(optimized=True)
    return float(fit.forecast(1)[0])

def _damped_ets_1step(y: np.ndarray) -> float:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated").fit(optimized=True)
    return float(fit.forecast(1)[0])

def _theta_1step(y: np.ndarray) -> float:
    # ThetaModel이 있으면 사용, 없으면 간단 SES 대체
    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ThetaModel(y).fit()
            f = fit.forecast(1)
        return float(f.iloc[0] if hasattr(f, "iloc") else f[0])
    except Exception:
        # SES fallback
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
        return float(fit.forecast(1)[0])

def _yoy_median_1step(y: np.ndarray) -> float:
    # 최근 성장률(최대 3개)의 '중위수' 적용
    if len(y) < 2 or y[-1] <= 0:
        raise ValueError("insufficient")
    ratios = []
    for i in range(len(y)-1, 0, -1):
        if y[i-1] > 0 and np.isfinite(y[i]) and np.isfinite(y[i-1]):
            ratios.append(y[i]/y[i-1])
        if len(ratios) == 3: break
    if not ratios:
        raise ValueError("no ratios")
    g = float(np.median(ratios))
    return float(max(0.0, y[-1]*g))

def _linlog_1step(y: np.ndarray, years: np.ndarray) -> float:
    if np.any(y <= 0) or len(y) < 3:
        raise ValueError("insufficient")
    x = years.astype(float)
    ly = np.log(y.astype(float))
    X = np.c_[np.ones_like(x), x]
    beta = np.linalg.lstsq(X, ly, rcond=None)[0]
    return float(np.exp(beta[0] + beta[1]*(x[-1]+1)))

METHODS = [
    ("ETS", _ets_1step),
    ("DampedETS", _damped_ets_1step),
    ("Theta", _theta_1step),
    ("YoY_med", _yoy_median_1step),
    ("LinearLog", _linlog_1step),
]

# -----------------------------
# 성능지표 & CV
# -----------------------------
def smape(a: float, f: float) -> float:
    den = (abs(a) + abs(f))
    return 0.0 if den == 0 else 2.0*abs(f-a)/den

def learn_weights_cv(years: np.ndarray, y: np.ndarray) -> dict:
    """
    롤링-오리진 1스텝 CV로 각 방법의 sMAPE 평균을 구하고,
    weight = 1/(eps + sMAPE) 로 역오차 가중치 산출
    """
    T = len(y)
    if T < CV_MIN_WINDOW + 1:
        # 데이터가 짧으면 균등 가중(혹은 기본 비율)
        base = {"ETS":0.6, "DampedETS":0.2, "Theta":0.1, "YoY_med":0.08, "LinearLog":0.02}
        s = sum(base.values())
        return {k:v/s for k,v in base.items()}

    errors = {name: [] for name,_ in METHODS}
    for t in range(CV_MIN_WINDOW, T-1):
        y_train = y[:t+1]
        yr_train = years[:t+1]
        a = y[t+1]  # next actual
        for name, fn in METHODS:
            try:
                if name == "LinearLog":
                    f = fn(y_train, yr_train)
                else:
                    f = fn(y_train)
                e = smape(float(a), float(f))
                if np.isfinite(e):
                    errors[name].append(e)
            except Exception:
                # 실패한 방법은 스킵
                pass

    avg = {k: (np.mean(v) if len(v) else np.nan) for k,v in errors.items()}
    # 유효한 방법만 역오차 가중
    valid = {k: v for k,v in avg.items() if np.isfinite(v)}
    if not valid:
        base = {"ETS":0.6, "DampedETS":0.2, "Theta":0.1, "YoY_med":0.08, "LinearLog":0.02}
        s = sum(base.values())
        return {k:v/s for k,v in base.items()}
    inv = {k: 1.0/(EPS+v) for k,v in valid.items()}
    s = sum(inv.values())
    w = {k: inv[k]/s for k in valid.keys()}
    return w

# -----------------------------
# 1-스텝 예측(학습 가중 사용)
# -----------------------------
def ensemble_1step(years: np.ndarray, y: np.ndarray, weights: dict) -> tuple[float, dict[str, float]]:
    preds = {}
    for name, fn in METHODS:
        if name not in weights:  # CV에서 탈락
            continue
        try:
            if name == "LinearLog":
                preds[name] = float(fn(y, years))
            else:
                preds[name] = float(fn(y))
        except Exception:
            continue
    if not preds:
        return (np.nan, {})
    # 가중 재정규화
    ws = {k: weights[k] for k in preds.keys()}
    s = sum(ws.values())
    if s <= 0:
        w_norm = {k: 1.0/len(ws) for k in ws.keys()}
    else:
        w_norm = {k: ws[k]/s for k in ws.keys()}
    yhat = sum(w_norm[k]*preds[k] for k in preds.keys())
    return (max(0.0, float(yhat)), preds)

# -----------------------------
# 식육가공품 멀티스텝(2020~2025), 그 외 1스텝
# -----------------------------
def forecast_path(years: np.ndarray, y: np.ndarray, target_year: int, category: str) -> tuple[list[int], list[float], dict[str, float]]:
    last = int(years[-1])
    # CV로 가중 학습
    w = learn_weights_cv(years, y)

    if category == "식육가공품":
        H = target_year - last
        pred_years, pred_vals = [], []
        cur_y = y.copy()
        cur_yr = years.copy()
        for h in range(1, H+1):
            yhat, _ = ensemble_1step(cur_yr, cur_y, w)
            pred_years.append(last + h)
            pred_vals.append(yhat)
            # 다음 스텝을 위한 이어붙이기
            cur_y = np.r_[cur_y, [yhat]]
            cur_yr = np.r_[cur_yr, [last + h]]
        return pred_years, pred_vals, w
    else:
        # 2024 → 2025 한 스텝
        yhat, _preds = ensemble_1step(years, y, w)
        return [last+1], [yhat], w

# -----------------------------
# 예측구간(80%) — 로그수익률 변동성 기반
# -----------------------------
def vol_pi(path_vals: list[float], base_series: np.ndarray, alpha: float = 0.20) -> list[tuple[float,float]]:
    """
    path_vals: 1..H 예측값
    base_series: 과거 실측 (양수)
    alpha=0.2 -> 약 80% 대칭 구간, z≈1.28
    """
    z = 1.2816
    y = base_series.astype(float)
    y = y[np.isfinite(y) & (y > 0)]
    if len(y) < 3:
        return [(np.nan, np.nan) for _ in path_vals]
    lr = np.log(y[1:] / y[:-1])
    sig = float(np.std(lr, ddof=1)) if len(lr) >= 2 else float(np.std(lr))
    out = []
    for h, v in enumerate(path_vals, start=1):
        if not np.isfinite(v) or v <= 0 or not np.isfinite(sig):
            out.append((np.nan, np.nan)); continue
        factor = math.exp(z * sig * math.sqrt(h))
        lo = v / factor
        hi = v * factor
        out.append((lo, hi))
    return out

# -----------------------------
# 메인
# -----------------------------
df = load_yearly(INPATH)
cats = sorted(df["category"].unique())

rows = []
decomp_rows = []
summary_rows = []

for c in cats:
    g = df[df["category"]==c].sort_values("year")
    years = g["year"].astype(int).to_numpy()
    vals  = g["amount"].astype(float).to_numpy()

    # 경로 예측
    pred_years, pred_vals, learned_w = forecast_path(years, vals, TARGET_YEAR, c)
    # 80% PI
    pis = vol_pi(pred_vals, vals, alpha=0.20)
    # 저장(실측)
    for y0, v0 in zip(years, vals):
        rows.append({"category": c, "year": int(y0), "amount": float(v0), "is_forecast": 0, "lo80": np.nan, "hi80": np.nan})
    # 저장(예측)
    for y1, v1, (lo, hi) in zip(pred_years, pred_vals, pis):
        rows.append({"category": c, "year": int(y1), "amount": float(v1), "is_forecast": 1, "lo80": lo, "hi80": hi})

    # 분해(학습된 가중치 + 방법별 2025 예측)
    # 2025 개별값도 기록(있을 경우)
    method_preds_2025 = {}
    for name, fn in METHODS:
        if name not in learned_w:  # CV로 제외된 방법
            continue
        try:
            if name == "LinearLog":
                phat = float(fn(vals, years))
            else:
                phat = float(fn(vals))
            method_preds_2025[name] = phat
        except Exception:
            continue

    # decomp 저장
    for name in sorted(learned_w.keys()):
        decomp_rows.append({
            "category": c,
            "method": name,
            "weight": float(learned_w[name]),
            "yhat_2025": float(method_preds_2025.get(name, np.nan)),
            "ok": 1 if name in method_preds_2025 else 0
        })

    # 요약행
    obs_last = int(years.max())
    y2025 = [v for (yy,v) in zip(pred_years, pred_vals) if yy==2025]
    y2025_pred = float(y2025[0]) if y2025 else np.nan
    lo = [lo for (yy,(lo,_hi)) in zip(pred_years, pis) if yy==2025]
    hi = [hi for (yy,(_lo,hi)) in zip(pred_years, pis) if yy==2025]
    lo_2025 = float(lo[0]) if lo else np.nan
    hi_2025 = float(hi[0]) if hi else np.nan

    # 최근 3개년도(실측 기준) 요약
    def _val(yq):
        v = g.loc[g["year"]==yq, "amount"]
        return float(v.values[0]) if len(v) else np.nan
    y2022 = _val(min(2022, obs_last))
    y2023 = _val(min(2023, obs_last))
    y2024 = _val(min(2024, obs_last))
    if np.isfinite(y2022) and y2022>0 and np.isfinite(y2024):
        cagr_22_24 = (y2024/y2022)**(1/2) - 1
    else:
        cagr_22_24 = np.nan
    base = y2024 if np.isfinite(y2024) and y2024>0 else _val(obs_last)
    chg_to_25 = (y2025_pred/base - 1) if (np.isfinite(y2025_pred) and np.isfinite(base) and base>0) else np.nan

    summary_rows.append({
        "category": c,
        "y2022": y2022, "y2023": y2023, "y2024": y2024,
        "y2025_pred": y2025_pred,
        "lo80_2025": lo_2025, "hi80_2025": hi_2025,
        "cagr_22_24": cagr_22_24,
        "chg_to_25": chg_to_25
    })

# 저장
res = pd.DataFrame(rows).sort_values(["category","year"]).reset_index(drop=True)
res.to_csv(OUTDIR/"yearly_anchor_forecast.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(decomp_rows).to_csv(OUTDIR/"yearly_anchor_forecast_decomposition.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(summary_rows).sort_values("category").to_csv(OUTDIR/"yearly_anchor_summary.csv", index=False, encoding="utf-8-sig")
print(f"[SAVE] {OUTDIR/'yearly_anchor_forecast.csv'}")
print(f"[SAVE] {OUTDIR/'yearly_anchor_forecast_decomposition.csv'}")
print(f"[SAVE] {OUTDIR/'yearly_anchor_summary.csv'}")

# -----------------------------
# 시각화
# -----------------------------
for c in res["category"].unique():
    g = res[res["category"]==c].sort_values("year")
    x  = g["year"].astype(int).to_numpy()
    yv = g["amount"].astype(float).to_numpy()
    m  = g["is_forecast"].to_numpy().astype(bool)
    lo = g["lo80"].to_numpy(dtype=float)
    hi = g["hi80"].to_numpy(dtype=float)

    plt.figure(figsize=(10.0,4.8))

    # 실측
    if (~m).any():
        plt.plot(x[~m], yv[~m], label=f"{c} (실측)")

    # 예측 — 선 끊김 방지
    if m.any():
        if (~m).any():
            # 마지막 실측점 포함해 연결된 점선
            x_conn = np.r_[x[~m][-1], x[m]]
            y_conn = np.r_[yv[~m][-1], yv[m]]
        else:
            x_conn = x[m]; y_conn = yv[m]
        plt.plot(x_conn, y_conn, linestyle="--", label=f"{c} (예측)")
        # PI 음영 (예측 연도만)
        mask_pred = m
        if (~m).any():
            # 마지막 실측→첫 예측 사이도 채우고 싶으면 아래 주석 해제
            pass
        # fill_between은 정확히 예측 연도 구간만
        plt.fill_between(x[mask_pred], lo[mask_pred], hi[mask_pred], alpha=0.15, linewidth=0)

        # 2025 점 강조 & 주석
        if (x[m] == TARGET_YEAR).any():
            y2025 = yv[m][np.where(x[m]==TARGET_YEAR)[0][0]]
            plt.scatter([TARGET_YEAR], [y2025], s=80, zorder=5)
            plt.annotate(f"{y2025:,.0f}", (TARGET_YEAR, y2025),
                         xytext=(8,8), textcoords="offset points",
                         fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))
        # 예측 영역 음영
        last_obs = int(x[~m].max()) if (~m).any() else int(x.min())-1
        plt.axvspan(last_obs+0.5, TARGET_YEAR+0.5, alpha=0.08)

    # x축 매년
    xmin = int(x.min()); xmax = int(x.max())
    plt.xlim(xmin - 0.2, xmax + 0.8)
    plt.xticks(list(range(xmin, xmax+1, 1)))

    # 부제: 학습 가중치 요약
    # (파일에서 읽어 표시할 수도 있지만 간단히 생략 가능. 필요 시 summary 표시 로직 추가)
    plt.title(f"[연 매출액] {c} — 실측·예측(→{TARGET_YEAR})")
    plt.xlabel("연도"); plt.ylabel("매출액 (원/또는 백만원)")
    plt.grid(True, alpha=0.3); plt.legend(loc="best"); plt.tight_layout()
    outpng = PLOTDIR / f"{c}.png"
    plt.savefig(outpng, dpi=150); plt.close()
    print(f"[PLOT] {outpng}")

print("완료: CV-가중 앙상블 기반 연 매출 앵커/PI/그래프 저장.")

