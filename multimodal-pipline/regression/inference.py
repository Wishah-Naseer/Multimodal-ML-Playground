import os
import pandas as pd
import joblib


TEST_CSV = "./multimodal-pipline/regression/datasets/test.csv"
MODEL_PATH = "./multimodal-pipline/regression/models/linear_regression_pipeline.joblib"
SUBMISSION_PATH = "./multimodal-pipline/regression/datasets/submission.csv"

def main():
    df_test = pd.read_csv(TEST_CSV)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            f"Run train_eval.py first to create it."
        )
    model = joblib.load(MODEL_PATH)

    preds = model.predict(df_test)

    submission = df_test.loc[:, ["Id"]].copy()
    submission["PredictedSalePrice"] = preds
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote predictions → {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
