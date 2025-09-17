# gridsearch_backtest.py
# -*- coding: utf-8 -*-
"""
Rolling backtest grid search for the persona-based forecasting pipeline.

What this does
--------------
- Dynamically imports your forecasting pipeline module (PIPELINE_PATH).
- Runs rolling backtests on multiple 12-month windows.
- Grid-searches key knobs (custom vs POS month-share, weights, seasonal smooth, etc.).
- Scores shape accuracy with sMAPE/RMSE at product & category levels.
- Saves a CSV of all trials and a JSON of the best config.

How to run
---------
1) Set PIPELINE_PATH to your pipeline .py file path.
2) python gridsearch_backtest.py

Outputs
-------
- _save/persona_sim/gridsearch_results.csv
- _save/persona_sim/best_config.json
- (optional) preview CSV for the best run on the most recent backtest window
"""

from __future__ import annotations

import sys
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import importlib.util
import numpy as np
import pandas as pd

# ====== 1) Point this to your forecasting pipeline file ======
# e.g., "persona_forecast_customshare.py"
PIPELINE_PATH = "./동원/00_1. 페르소나 구매 시뮬레이션8_검증 및 고도화2.py"  # ← 반드시 네 파일명/경로로 수정

# ====== 2) Backtest windows (12-month rolling) ======
BACKTEST_WINDOWS = [
    ("2020-07-01", "2021-06-01"),
    ("2021-07-01", "2022-06-01"),
    ("2022-07-01", "2023-06-01"),
]

# ====== 3) Parameter grid (edit as needed) ======
PARAM_GRID = {
    # month-share source: "custom" uses your CUSTOM_MONTH_SHARE or CSV,
    # "pos" uses POS (fallback CLICK) category distribution
    "share_source": ["custom", "pos"],

    # For "custom" only: blending weight (1.0 = full custom, 0.0 = model base shape)
    "CUSTOM_SHARE_WEIGHT": [1.0, 0.8, 0.6, 0.4],

    # POS season smoothing (for "pos" only). Higher = more uniform
    "POS_LAM_UNI": [0.10, 0.20, 0.30],

    # Search/Click seasonal template uniform mix (applies to both)
    "SC_LAM_UNI": [0.00, 0.05, 0.10, 0.15],

    # Season start month for templates (typically 7; can test 1)
    "SC_SEASON_START": [7],  # [7, 1] to explore more

    # Calibration mode mostly affects scale (shape preserved by month-share),
    # but keep here for completeness. FULL12 is usually fine.
    "CALIB_MODE": ["FULL12"],
}


# ==========================
# Utils
# ==========================

def dynamic_import(module_path: str):
    # 절대경로로 정리 (비ASCII/공백 경로도 안전)
    module_path = os.path.abspath(module_path)

    # 모듈 이름은 고정해도 되지만, 충돌이 싫으면 파일명 기반으로 유니크하게 줄 수도 있습니다.
    mod_name = "pfmod"

    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load module from: {module_path}")

    mod = importlib.util.module_from_spec(spec)

    # ✅ 중요: exec_module 전에 반드시 sys.modules에 등록
    sys.modules[spec.name] = mod

    spec.loader.exec_module(mod)
    return mod


def smape(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)))

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def month_range(start_ym: str, end_ym: str) -> pd.DatetimeIndex:
    return pd.date_range(start_ym, end_ym, freq="MS")


# ==========================
# Single run (one config + one window)
# ==========================
def run_single(mod, pred_months: pd.DatetimeIndex, config_overrides: dict):
    """
    Executes the forecasting pipeline steps using the imported module `mod`,
    with CONFIG overridden by `config_overrides` and horizon = `pred_months`.
    Returns: pred_df (with 'quantity_final'), click_df (raw), sample_df.
    """

    # --- Override CONFIG values for this run ---
    mod.CONFIG.PRED_MONTHS = pred_months
    if "SC_LAM_UNI" in config_overrides:
        mod.CONFIG.SC_LAM_UNI = float(config_overrides["SC_LAM_UNI"])
    if "SC_SEASON_START" in config_overrides:
        mod.CONFIG.SC_SEASON_START = int(config_overrides["SC_SEASON_START"])
    if "POS_LAM_UNI" in config_overrides:
        mod.CONFIG.POS_LAM_UNI = float(config_overrides["POS_LAM_UNI"])
    if "CALIB_MODE" in config_overrides:
        mod.CONFIG.CALIB_MODE = str(config_overrides["CALIB_MODE"])

    # Switch month-share source
    share_source = config_overrides.get("share_source", "custom")
    if share_source == "custom":
        mod.CONFIG.USE_CUSTOM_MONTH_SHARE = True
        mod.CONFIG.CUSTOM_SHARE_WEIGHT = float(config_overrides.get("CUSTOM_SHARE_WEIGHT", 1.0))
        # ensure custom map exists (dict or CSV in your pipeline)
        # leave mod.CUSTOM_MONTH_SHARE as-is (user provided)
    else:
        # POS/CLICK branch only: ensure custom is OFF and dict cleared
        mod.CONFIG.USE_CUSTOM_MONTH_SHARE = False
        # Temporarily clear the in-module default dict to avoid short-circuiting
        try:
            if hasattr(mod, "CUSTOM_MONTH_SHARE"):
                mod.CUSTOM_MONTH_SHARE.clear()
        except Exception:
            pass

    # --- Load data ---
    personas = mod.load_personas(mod.CONFIG.PATH_PERSONA)
    search, click, sample = mod.load_trends()
    pos_total, _, pos_seg = mod.load_pos_months()

    # --- Trends → persona weighting ---
    search = mod.normalize_per_product(search, "search_index", "search_norm")
    click_n = mod.normalize_per_product(click,  "clicks",        "click_norm")
    search = mod.to_buckets(search); click_b = mod.to_buckets(click_n)
    seg_w = mod.build_segment_weights(personas)
    sw = mod.weighted_by_persona(search, "search_norm", seg_w)
    cw = mod.weighted_by_persona(click_b, "click_norm", seg_w)

    # --- Seasonal templates projected to pred horizon ---
    if getattr(mod.CONFIG, "USE_SEASONAL_SC", True):
        sw = mod._seasonalize_to_pred(
                sw[["product_name","date","search_norm_w"]], "search_norm_w",
                mod.CONFIG.PRED_MONTHS,
                season_start=mod.CONFIG.SC_SEASON_START,
                lam_uni=mod.CONFIG.SC_LAM_UNI,
                decay=getattr(mod.CONFIG, "SC_DECAY", 1.0)
            ).rename(columns={"search_norm_w":"s"})
        cw = mod._seasonalize_to_pred(
                cw[["product_name","date","click_norm_w"]], "click_norm_w",
                mod.CONFIG.PRED_MONTHS,
                season_start=mod.CONFIG.SC_SEASON_START,
                lam_uni=mod.CONFIG.SC_LAM_UNI,
                decay=getattr(mod.CONFIG, "SC_DECAY", 1.0)
            ).rename(columns={"click_norm_w":"c"})
    else:
        sw = sw[sw["date"].isin(mod.CONFIG.PRED_MONTHS)].rename(columns={"search_norm_w":"s"})
        cw = cw[cw["date"].isin(mod.CONFIG.PRED_MONTHS)].rename(columns={"click_norm_w":"c"})

    trend_w = pd.merge(sw, cw, on=["product_name","date"], how="outer").fillna(0.0)

    # --- Params, predict, calibrate ---
    product_params = mod.build_product_params(personas, click)
    trend_w_sm = mod.add_ema_and_momentum(trend_w, product_params)
    pred_df = mod.predict(personas, trend_w_sm, product_params, mod.CONFIG.PRED_MONTHS)
    pred_df, cal = mod.calibrate(pred_df, click, product_params, mod.CONFIG.CALIB_MODE)

    # --- Month-share application ---
    if mod.CONFIG.USE_CUSTOM_MONTH_SHARE:
        custom_map = mod.build_custom_monthshare_map(mod.CONFIG.PRED_MONTHS)
        pred_df = mod.apply_monthshare_blend(
            pred_df=pred_df,
            month_share_map=custom_map,
            pred_months=mod.CONFIG.PRED_MONTHS,
            mu_by_cat=mod.CONFIG.CUSTOM_SHARE_WEIGHT,   # scalar ok
            out_col="quantity_final"
        )
    else:
        month_share = mod.build_month_share_by_category(click, pos_total, pos_seg)
        pred_df = mod.apply_monthshare_blend(
            pred_df=pred_df,
            month_share_map=month_share,
            pred_months=mod.CONFIG.PRED_MONTHS,
            mu_by_cat=mod.MU_BY_CAT,                    # dict by category
            out_col="quantity_final"
        )

    return pred_df, click, sample


# ==========================
# Scoring (shape accuracy)
# ==========================
def score_shape(mod, pred_df: pd.DataFrame, click_df: pd.DataFrame,
                pred_months: pd.DatetimeIndex) -> Dict[str, float]:
    """
    Compares monthly shape (shares) between predictions and CLICK reference.
    Returns aggregated metrics:
      - prod_smape_share_weighted
      - prod_rmse_share_weighted
      - cat_smape_share_weighted
      - cat_rmse_share_weighted
    Weights are monthly CLICK volumes (sum over months per entity).
    """

    eps = 1e-9
    # CLICK monthly by product
    click_pm = (click_df.groupby(["product_name","date"], as_index=False)["clicks"].mean())

    # PRODUCT-LEVEL
    prod_scores = []
    prod_weights = []
    for prod, g in pred_df.groupby("product_name", group_keys=False):
        # predicted shares
        p = (g.set_index("date")["quantity_final"]
               .reindex(pred_months, fill_value=0.0).to_numpy(dtype=float))
        ptot = float(p.sum())
        if ptot <= 0:  # skip zero predictions
            continue
        psh = p / (ptot + eps)

        # reference: product click shares
        c = (click_pm[(click_pm["product_name"]==prod) &
                      (click_pm["date"].isin(pred_months))]
                .set_index("date")["clicks"]
                .reindex(pred_months, fill_value=0.0).to_numpy(dtype=float))
        ctot = float(c.sum())
        csh = (c / (ctot + eps)) if ctot > 0 else np.full(len(pred_months), 1.0/len(pred_months))

        prod_scores.append((smape(psh, csh), rmse(psh, csh)))
        prod_weights.append(max(ctot, eps))  # weight by click volume

    if prod_scores:
        smapes = np.array([s for s,_ in prod_scores], float)
        rmses  = np.array([r for _,r in prod_scores], float)
        w      = np.array(prod_weights, float)
        prod_smape_w = float(np.sum(smapes * w) / (np.sum(w) + eps))
        prod_rmse_w  = float(np.sum(rmses  * w) / (np.sum(w) + eps))
    else:
        prod_smape_w, prod_rmse_w = 1.0, 1.0  # fallback

    # CATEGORY-LEVEL
    # Predicted category monthly totals
    prod2cat = getattr(mod, "PROD2CAT", {})
    pred_df["_cat"] = pred_df["product_name"].map(prod2cat).fillna("UNK")
    pred_cat = (pred_df.groupby(["_cat","date"])["quantity_final"].sum().reset_index())

    cat_scores = []
    cat_weights = []
    for cat, g in pred_cat.groupby("_cat", group_keys=False):
        p = (g.set_index("date")["quantity_final"]
               .reindex(pred_months, fill_value=0.0).to_numpy(dtype=float))
        ptot = float(p.sum())
        if ptot <= 0:
            continue
        psh = p / (ptot + eps)

        c = (click_pm[click_pm["product_name"].isin([k for k,v in prod2cat.items() if v==cat])]
                .groupby("date")["clicks"].sum()
                .reindex(pred_months, fill_value=0.0).to_numpy(dtype=float))
        ctot = float(c.sum())
        csh = (c / (ctot + eps)) if ctot > 0 else np.full(len(pred_months), 1.0/len(pred_months))

        cat_scores.append((smape(psh, csh), rmse(psh, csh)))
        cat_weights.append(max(ctot, eps))

    if cat_scores:
        smapes = np.array([s for s,_ in cat_scores], float)
        rmses  = np.array([r for _,r in cat_scores], float)
        w      = np.array(cat_weights, float)
        cat_smape_w = float(np.sum(smapes * w) / (np.sum(w) + eps))
        cat_rmse_w  = float(np.sum(rmses  * w) / (np.sum(w) + eps))
    else:
        cat_smape_w, cat_rmse_w = 1.0, 1.0

    return dict(
        prod_smape_share_weighted = prod_smape_w,
        prod_rmse_share_weighted  = prod_rmse_w,
        cat_smape_share_weighted  = cat_smape_w,
        cat_rmse_share_weighted   = cat_rmse_w,
    )


# ==========================
# Grid search loop
# ==========================
def iterate_param_grid(param_grid: dict):
    """
    Yields dicts of parameter combinations. Skips irrelevant params
    (e.g., CUSTOM_SHARE_WEIGHT when share_source='pos').
    """
    from itertools import product
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in product(*values):
        d = {k: v for k, v in zip(keys, combo)}
        # prune irrelevant
        if d.get("share_source") == "custom":
            # POS-only knobs are irrelevant but harmless; we keep them fixed.
            pass
        else:
            # POS mode doesn't use CUSTOM_SHARE_WEIGHT
            d.pop("CUSTOM_SHARE_WEIGHT", None)
        yield d


def main():
    # import pipeline module
    mod = dynamic_import(PIPELINE_PATH)
    SAVE_DIR = Path(getattr(mod.CONFIG, "SAVE_DIR", Path("./_save/persona_sim")))
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    best = None
    best_key_metric = math.inf  # lower is better
    best_detail = None
    best_pred_preview = None
    best_cfg = None

    # run grid
    for params in iterate_param_grid(PARAM_GRID):
        # aggregate metrics across windows
        win_metrics = []
        for (s, e) in BACKTEST_WINDOWS:
            pred_months = month_range(s, e)
            pred_df, click_df, sample_df = run_single(mod, pred_months, params)
            m = score_shape(mod, pred_df, click_df, pred_months)
            m["window"] = f"{s[:7]}_{e[:7]}"
            win_metrics.append(m)

        # summarize across windows (weighted average by category click total is already inside metrics)
        dfm = pd.DataFrame(win_metrics)
        summary = {
            "share_source": params.get("share_source"),
            "CUSTOM_SHARE_WEIGHT": params.get("CUSTOM_SHARE_WEIGHT", None),
            "POS_LAM_UNI": params.get("POS_LAM_UNI", None),
            "SC_LAM_UNI": params.get("SC_LAM_UNI", None),
            "SC_SEASON_START": params.get("SC_SEASON_START", None),
            "CALIB_MODE": params.get("CALIB_MODE", None),
            # averages
            "avg_prod_smape": float(dfm["prod_smape_share_weighted"].mean()),
            "avg_prod_rmse":  float(dfm["prod_rmse_share_weighted"].mean()),
            "avg_cat_smape":  float(dfm["cat_smape_share_weighted"].mean()),
            "avg_cat_rmse":   float(dfm["cat_rmse_share_weighted"].mean()),
        }
        # also attach per-window for trace
        for _, row in dfm.iterrows():
            w = row["window"]
            summary[f"{w}__prod_smape"] = float(row["prod_smape_share_weighted"])
            summary[f"{w}__cat_smape"]  = float(row["cat_smape_share_weighted"])

        results.append(summary)

        # selection metric: category sMAPE (can switch to prod or composite)
        key_metric = summary["avg_cat_smape"]
        if key_metric < best_key_metric:
            best_key_metric = key_metric
            best = summary
            best_cfg = params
            # keep a preview with the latest window
            pred_months = month_range(*BACKTEST_WINDOWS[-1])
            best_pred_preview, _, _ = run_single(mod, pred_months, params)
            best_detail = dfm.copy()

    # save results
    res_df = pd.DataFrame(results).sort_values("avg_cat_smape", ascending=True)
    out_csv = SAVE_DIR / "gridsearch_results.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # save best config + detail
    best_json = {
        "best_params": best_cfg,
        "best_summary": best,
        "detail_by_window": best_detail.to_dict(orient="records"),
    }
    with open(SAVE_DIR / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_json, f, ensure_ascii=False, indent=2)

    # optional preview export for the best run (latest backtest window)
    if best_pred_preview is not None:
        best_pred_preview.to_csv(SAVE_DIR / "best_preview_latest_window.csv",
                                 index=False, encoding="utf-8-sig")

    print("\n[GRID SEARCH DONE]")
    print(f"Saved: {out_csv}")
    print("Top-5 (by avg_cat_smape):")
    print(res_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
