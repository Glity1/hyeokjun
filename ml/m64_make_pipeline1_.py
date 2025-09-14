# 이진 분류
# 06. cancer
# 07. dacon_diabetes
# 08. kaggle_bank

# 다중 분류
# 09. wine
# 11. digits

# 12. kaggle_santander
# 13. kaggle_otto

# # 전처리 하고 난뒤에 모델에다가 연결 하자
# import os
# import pandas as pd
# import numpy as np
# from sklearn.datasets import load_wine, load_digits, load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# #1. 데이터
# def load_csv_dataset(path, target_col, drop_cols=None):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"{path} 파일이 없습니다.")
#     df = pd.read_csv(path)
#     drop_cols = drop_cols or []
#     X = df.drop([target_col] + drop_cols, axis=1)
#     y = df[target_col]
#     return X.values, y.values


# # 2. 데이터셋 목록
# datasets = [
#     (lambda: load_breast_cancer(return_X_y=True), "cancer", [], {}),   # 06
#     (lambda: load_wine(return_X_y=True), "wine", [], {}),              # 09
#     (lambda: load_digits(return_X_y=True), "digits", [], {}),          # 11
#     (load_csv_dataset, "dacon_diabetes", ["./_data/dacon/diabetes/train.csv"], {"target_col": "Outcome"}),
#     (load_csv_dataset, "kaggle_bank", ["./_data/kaggle/bank/train.csv"], {"target_col": "y"}),
#     (load_csv_dataset, "kaggle_santander", ["./_data/kaggle/santander/train.csv"], {"target_col": "TARGET"}),
#     (load_csv_dataset, "kaggle_otto", ["./_data/kaggle/otto/train.csv"], {"target_col": "target", "drop_cols": ["id"]}),
# ]

# for loader, name, args, kwargs in datasets:
#     try:
#         X, y = loader(*args, **kwargs) if args or kwargs else loader()
#     except Exception as e:
#         print(f"[{name}] 로드 실패: {e}")
#         continue

#     # 데이터 분할
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, train_size=0.8, stratify=y, random_state=777
#     )

#     # 모델 파이프라인
#     model = make_pipeline(
#         MinMaxScaler(),
#         SVC()
#     )

#     # 학습 & 평가
#     model.fit(X_train, y_train)
#     acc = accuracy_score(y_test, model.predict(X_test))
#     print(f"[{name}] acc: {acc:.4f}")


# -*- coding: utf-8 -*-
# 여러 데이터셋(내장 + CSV)을 한 번에 전처리 → 모델(SVC) 학습/평가
# PCA 미사용, 수치형: MinMaxScaler / 범주형: OneHotEncoder

import os
import sys
import numpy as np
import pandas as pd

from sklearn import __version__ as skl_version
from packaging.version import Version
from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# =========================
# 0) 유틸
# =========================
def safe_drop(df: pd.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    return df.drop(columns=cols) if cols else df

def to_dataframe(X, feature_names=None):
    if isinstance(X, pd.DataFrame):
        return X
    if feature_names is None:
        # feature_names가 없으면 자동 생성
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names)

def build_preprocessor(X_df: pd.DataFrame) -> ColumnTransformer:
    """수치/범주 자동 분리 후 전처리 구성"""
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # sklearn 1.2 이상에서는 sparse_output, 그 이하는 sparse
    if Version(skl_version) >= Version("1.2"):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
        n_jobs=None,
    )
    return pre

# =========================
# 1) 데이터 로더
# =========================
def load_builtin_cancer():
    data = load_breast_cancer()
    X_df = to_dataframe(data.data, data.feature_names)
    y = pd.Series(data.target, name="target")
    return X_df, y

def load_builtin_wine():
    data = load_wine()
    X_df = to_dataframe(data.data, data.feature_names)
    y = pd.Series(data.target, name="target")
    return X_df, y

def load_builtin_digits():
    data = load_digits()
    # digits는 feature_names가 없으므로 자동 생성
    X_df = to_dataframe(data.data)
    y = pd.Series(data.target, name="target")
    return X_df, y

def load_csv_dataset(path, target_col, drop_cols=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 파일이 없습니다.")
    df = pd.read_csv(path)
    drop_cols = (drop_cols or []) + [
        # 흔한 ID/인덱스 후보들(있을 때만 제거)
        "id", "ID", "Id", "train_id", "test_id", "ID_code",
        "SK_ID_CURR", "Unnamed: 0", "index"
    ]
    df = safe_drop(df, drop_cols)
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' 이(가) 컬럼에 없습니다. 존재 컬럼: {list(df.columns)[:10]} ...")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y

# =========================
# 2) 데이터셋 목록
# =========================
datasets = [
    # 이진 분류
    (load_builtin_cancer, "cancer", [], {}),  # 06

    # 다중 분류
    (load_builtin_wine,   "wine",   [], {}),  # 09
    (load_builtin_digits, "digits", [], {}),  # 11

    # CSV (경로/타깃 확인해서 필요시 수정)
    (load_csv_dataset, "dacon_diabetes",
        ["./_data/dacon/diabetes/train.csv"], {"target_col": "Outcome"}),  # 07
    (load_csv_dataset, "kaggle_bank",
        ["./_data/kaggle/bank/train.csv"], {"target_col": "Exited"}),           # 08
    (load_csv_dataset, "kaggle_santander",
        ["./_data/kaggle/santander/train.csv"], {"target_col": "target"}), # 12
    (load_csv_dataset, "kaggle_otto",
        ["./_data/kaggle/otto/train.csv"], {"target_col": "target"}),      # 13 (id는 safe_drop에서 자동 처리)
]

# =========================
# 3) 실행 루프
# =========================
def main():
    results = []
    for loader, name, args, kwargs in datasets:
        try:
            X, y = loader(*args, **kwargs) if (args or kwargs) else loader()
        except Exception as e:
            print(f"[{name}] 로드 실패: {e}")
            continue

        # 통일된 DataFrame 형태 유지
        X = to_dataframe(X)

        # stratify가 가능하도록 y 확인
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, stratify=y, random_state=777
            )
        except ValueError:
            # 클래스 불균형/단일 클래스 등 특수 상황일 때 stratify 없이 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, random_state=777
            )

        pre = build_preprocessor(X_train)

        model = Pipeline([
            ("pre", pre),     # 수치: MinMax, 문자: OneHot
            ("clf", SVC())    # PCA 없음
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[{name}] acc: {acc:.4f}")
        results.append((name, acc))

    if results:
        print("\n=== Summary (acc desc) ===")
        for n, a in sorted(results, key=lambda x: -x[1]):
            print(f"{n:>16} | acc={a:.4f}")

if __name__ == "__main__":
    # 한글 경로/파일 문제 방지 (선택)
    try:
        import locale
        locale.setlocale(locale.LC_ALL, "")
    except Exception:
        pass

    np.set_printoptions(suppress=True)
    pd.set_option("display.max_columns", 50)
    main()
