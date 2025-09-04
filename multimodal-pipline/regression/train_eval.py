import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

TRAIN_CSV = "./multimodal-pipline/regression/datasets/train.csv"
MODEL_DIR = "./multimodal-pipline/regression/models"
MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression_pipeline.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics_linear_regression.json")

os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    df = pd.read_csv(TRAIN_CSV)
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    numerical_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object"]).columns.tolist()

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, numerical_feats),
        ("cat", cat_transformer, cat_feats),
    ])

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()
    print(f"Cross-validated MAE (5-fold): {cv_mae:.2f}")

    joblib.dump(model, MODEL_PATH)
    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "CV_MAE_5fold": float(cv_mae),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model → {MODEL_PATH}")
    print(f"Saved metrics → {METRICS_PATH}")

if __name__ == "__main__":
    main()
