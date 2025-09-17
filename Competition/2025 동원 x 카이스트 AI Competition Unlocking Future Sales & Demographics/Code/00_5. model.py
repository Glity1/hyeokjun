# train_lgbm.py
import os
import gc
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# =========================
# 0) 경로/설정
# =========================
FEATURE_DIR = "./_save/features"
TRAIN_FEAT  = os.path.join(FEATURE_DIR, "train_features.csv")
TEST_FEAT   = os.path.join(FEATURE_DIR, "test_features_2025H1.csv")
LAG_TABLE   = os.path.join(FEATURE_DIR, "best_lag_table.csv")

# 판매량(정답) 파일: 아래 중 하나를 준비해 주세요.
#  - 컬럼 예: product_name,date,target   (date는 'YYYY-MM-01')
TRAIN_TARGET_PATH = "./_data/dacon/dongwon/train_sales.csv"   # <- 여기에 실제 경로 지정

SAMPLE_SUB_PATH   = "./_data/dacon/dongwon/sample_submission.csv"  # 대회 제공
PRODUCT_INFO_PATH = "./_data/dacon/dongwon/product_info.csv"       # (선택) 매핑에 사용

SAVE_DIR = "./_save/models"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_COL = "target"  # 판매량(또는 대회 타깃) 컬럼명
KEY_COLS   = ["product_name", "date"]

SEEDS = [42, 74, 99, 111, 333]
N_SPLITS = 5

# =========================
# 1) 로드 & 병합 (A: 타깃 조인)
# =========================
train_feat = pd.read_csv(TRAIN_FEAT, parse_dates=["date"])
test_feat  = pd.read_csv(TEST_FEAT,  parse_dates=["date"])

# 타깃 로드
if not os.path.exists(TRAIN_TARGET_PATH):
    raise FileNotFoundError(
        f"[필수] 판매량 파일이 없습니다: {TRAIN_TARGET_PATH}\n"
        f"최소 컬럼 예시는 다음과 같습니다: {KEY_COLS + [TARGET_COL]}\n"
        f"예: product_name,date,target  (date=YYYY-MM-01)"
    )
train_tgt = pd.read_csv(TRAIN_TARGET_PATH, parse_dates=["date"])

# 조인 키 정규화
def normalize_dates(df):
    # 일자를 월초로 정규화(예: 2025-06-17 -> 2025-06-01)
    df = df.copy()
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df

train_feat = normalize_dates(train_feat)
test_feat  = normalize_dates(test_feat)
train_tgt  = normalize_dates(train_tgt)

# 조인
train = pd.merge(train_feat, train_tgt[KEY_COLS+[TARGET_COL]], on=KEY_COLS, how="inner")
print(f"[INFO] 학습 데이터 크기: {train.shape}, 테스트 데이터 크기: {test_feat.shape}")

# =========================
# 2) 피처 선택 (B: 검증전략 전에 컬럼 확정)
# =========================
# 후보 피처 자동 선택
num_cols = []
for c in train.columns:
    if c in KEY_COLS + [TARGET_COL]:
        continue
    if train[c].dtype.kind in "biufc":   # 숫자형
        num_cols.append(c)

# 추천 포함: search/click 원시값, *_mom, *_yoy, *_ma3/ma6, search_index_bestlag, month/sin/cos, missing_click_flag
# 필요시 화이트리스트로 좁히기:
# whitelist = ["search_index","clicks","search_index_mom","search_index_yoy",
#              "clicks_mom","clicks_yoy","search_index_ma3","search_index_ma6",
#              "clicks_ma3","clicks_ma6","search_index_bestlag","month","sin_m","cos_m",
#              "missing_click_flag","best_lag"]
# num_cols = [c for c in num_cols if c in whitelist and c in train.columns]

print(f"[INFO] 사용 피처 수: {len(num_cols)}")
print("[INFO] 샘플 피처:", num_cols[:20])

# =========================
# 3) 검증전략 (TimeSeriesSplit) & 모델 학습 (C,D)
# =========================
def smape(y_true, y_pred):
    # SMAPE: 2*|y - yhat| / (|y|+|yhat|), 0/0 방지
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(1e-8, None)
    return np.mean(2.0 * np.abs(y_true - y_pred) / denom)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

# 정렬 후 인덱스 기반 시계열 분할
train = train.sort_values(["date","product_name"]).reset_index(drop=True)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)  # 단순 시간순 분할(샘플 순서가 시간 정렬되어 있어야 함)

oof_pred = np.zeros(len(train))
test_pred_seeds = np.zeros((len(test_feat), len(SEEDS)))
feature_importances = []

for s_idx, seed in enumerate(SEEDS):
    print(f"\n===== SEED {seed} =====")
    fold_preds = np.zeros(len(train))
    test_pred = np.zeros(len(test_feat))

    for fold, (trn_idx, val_idx) in enumerate(tscv.split(train)):
        trn = train.iloc[trn_idx]
        val = train.iloc[val_idx]

        X_trn = trn[num_cols]
        y_trn = trn[TARGET_COL].values
        X_val = val[num_cols]
        y_val = val[TARGET_COL].values

        model = LGBMRegressor(
            objective="rmse",
            n_estimators=5000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=200, verbose=False),
                log_evaluation(200)
            ]
        )

        val_pred = model.predict(X_val)
        fold_preds[val_idx] = val_pred
        test_pred += model.predict(test_feat[num_cols]) / N_SPLITS

        # 피처 중요도 수집
        feature_importances.append(
            pd.DataFrame({"feature": num_cols, "importance": model.feature_importances_, "seed": seed, "fold": fold})
        )

        print(f"[FOLD {fold}] RMSE={rmse(y_val, val_pred):.4f} | SMAPE={smape(y_val, val_pred):.4f}")

    # seed별 OOF/TEST 축적
    oof_pred += fold_preds / len(SEEDS)
    test_pred_seeds[:, s_idx] = test_pred

    print(f"[SEED {seed}] OOF RMSE={rmse(train[TARGET_COL], fold_preds):.4f} | SMAPE={smape(train[TARGET_COL], fold_preds):.4f}")

# 전체 OOF
print("\n===== OVERALL (OOF across seeds) =====")
print(f"OOF RMSE={rmse(train[TARGET_COL], oof_pred):.4f} | SMAPE={smape(train[TARGET_COL], oof_pred):.4f}")

# =========================
# 4) 예측/제출 생성 (E)
# =========================
# TEST 예측: seed 평균
test_pred = test_pred_seeds.mean(axis=1)
test_out = test_feat[KEY_COLS].copy()
test_out["prediction"] = test_pred

# (선택) product_info로 id 매핑 필요 시
# product_info 예시에 product_id, product_name 등이 있으면 조인해 id 생성 가능
if os.path.exists(PRODUCT_INFO_PATH):
    info = pd.read_csv(PRODUCT_INFO_PATH)
    # 가정: info 안에 product_name이 있고, sample_submission의 id가 (product_id + date)에서 파생된다면 하단 로직 조정
    # test_out = test_out.merge(info[["product_name","product_id"]], on="product_name", how="left")

# sample_submission과 매핑
if os.path.exists(SAMPLE_SUB_PATH):
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    # 대회별 포맷이 다르므로, 간단한 예시만 제공합니다.
    # 가정 1) sample_submission에 product_name / date 컬럼이 있는 경우:
    candidates = [c.lower() for c in sub.columns]
    has_pn = "product_name" in sub.columns
    has_date = "date" in sub.columns

    if has_pn and has_date:
        sub["date"] = pd.to_datetime(sub["date"]).dt.to_period("M").dt.to_timestamp()
        sub = sub.merge(test_out, on=["product_name","date"], how="left")
        # 대회 타깃 컬럼명에 맞춰 rename
        if "target" in sub.columns:
            sub["target"] = sub["prediction"]
        elif "quantity" in sub.columns:
            sub["quantity"] = sub["prediction"]
        else:
            sub["prediction"] = sub["prediction"]
    else:
        # 가정 2) id만 있고, 별도 키가 없다면 임시로 product_name/date 순 정렬 후 대입 (권장X)
        print("[WARN] sample_submission에 키 컬럼(product_name/date)이 없습니다. 단순 대입 방식을 사용하세요.")
        sub = sub.copy()
        sub["prediction"] = 0.0
        # 필요시 여기서 별도 매핑 로직 구현
    sub_path = os.path.join(SAVE_DIR, "submission.csv")
    sub.to_csv(sub_path, index=False, encoding="utf-8-sig")
    print("✅ 제출 파일 저장:", sub_path)

# 보조 아웃풋들 저장
oof_df = train[KEY_COLS + [TARGET_COL]].copy()
oof_df["oof_pred"] = oof_pred
oof_path = os.path.join(SAVE_DIR, "oof_predictions.csv")
oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
print("✅ OOF 저장:", oof_path)

# 피처 중요도
if len(feature_importances):
    fi = pd.concat(feature_importances, ignore_index=True)
    fi_agg = (fi.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index())
    fi_path = os.path.join(SAVE_DIR, "feature_importance.csv")
    fi.to_csv(fi_path, index=False, encoding="utf-8-sig")
    fi_agg_path = os.path.join(SAVE_DIR, "feature_importance_agg.csv")
    fi_agg.to_csv(fi_agg_path, index=False, encoding="utf-8-sig")
    print("✅ 중요도 저장:", fi_path, ",", fi_agg_path)

print("🎉 Done.")
