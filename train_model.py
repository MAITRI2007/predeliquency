import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "merged_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.pkl")


def add_gig_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    weekly_cols = [
        "salary_week_1",
        "salary_week_2",
        "salary_week_3",
        "salary_week_4",
        "salary_week_5"
    ]

    for col in weekly_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

 
    if "customer_segment" in df.columns:
        segment = df["customer_segment"].astype(str).str.strip().str.lower()
        df["is_gig_worker"] = segment.isin(["gig_worker", "gig", "freelancer"]).astype(int)
    else:
        df["is_gig_worker"] = 0

    df["is_gig_worker"] = np.where(
        (df["is_gig_worker"] == 1) | (df[weekly_cols].gt(0).sum(axis=1) > 1),
        1,
        0
    )

    df["income_5w_avg"] = df[weekly_cols].mean(axis=1)

 
    df["current_week_income"] = df["salary_week_5"]
    df["current_week_income"] = np.where(
        df["current_week_income"] > 0,
        df["current_week_income"],
        df["salary_week_4"]
    )

    df["zero_income_weeks_5w"] = (df[weekly_cols] == 0).sum(axis=1)

    weekly_mean = df[weekly_cols].mean(axis=1).replace(0, np.nan)
    weekly_std = df[weekly_cols].std(axis=1)

    df["income_volatility_5w"] = (
        weekly_std / weekly_mean
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    df["gig_income_drop_flag"] = np.where(
        (df["is_gig_worker"] == 1) &
        (df["current_week_income"] <= 0.5 * df["income_5w_avg"]) &
        (df["zero_income_weeks_5w"] >= 2),
        1,
        0
    )

    return df


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Run merge_csv.py first.")

df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError("CSV dataset is empty.")

print("Columns found in dataset:")
print(df.columns.tolist())

TARGET_COL = "will_default_2_4_weeks"

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")


model_df = df.copy()
model_df = add_gig_features(model_df)

model_df = model_df.drop(
    ["customer_id", "year_month", "emi_payment_date"],
    axis=1,
    errors="ignore"
)

if "risk_bucket" in model_df.columns:
    model_df["risk_bucket"] = (
        model_df["risk_bucket"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "low": 0,
            "medium": 1,
            "high": 2
        })
        .fillna(0)
    )

if "income_type" in model_df.columns:
    model_df["income_type"] = (
        model_df["income_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "monthly_salary": 0,
            "weekly_gig": 1,
            "self_employed_irregular": 2
        })
        .fillna(0)
    )

if "customer_segment" in model_df.columns:
    model_df["customer_segment"] = (
        model_df["customer_segment"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "standard": 0,
            "gig_worker": 1,
            "gig": 1,
            "freelancer": 1
        })
        .fillna(0)
    )

y = pd.to_numeric(model_df[TARGET_COL], errors="coerce").fillna(0).astype(int)
X = model_df.drop(TARGET_COL, axis=1)

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = X.select_dtypes(include=["int64", "float64", "int32", "float32"])

if X.empty:
    raise ValueError("No usable numeric feature columns found after preprocessing.")

print("\nTraining feature columns:")
print(X.columns.tolist())

print("\nTraining rows:", len(X))
print("Positive class count:", int(y.sum()))
print("Negative class count:", int((y == 0).sum()))


model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X, y)


with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(COLUMNS_PATH, "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\n✅ Model trained successfully from CSV")
print(f"✅ Model saved to: {MODEL_PATH}")
print(f"✅ Columns saved to: {COLUMNS_PATH}")