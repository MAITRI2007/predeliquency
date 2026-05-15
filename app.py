from flask import Flask, jsonify, render_template
import pandas as pd
import pickle
import os
import numpy as np

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "merged_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.pkl")


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}")

if not os.path.exists(COLUMNS_PATH):
    raise FileNotFoundError(f"columns.pkl not found at {COLUMNS_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(COLUMNS_PATH, "rb") as f:
    columns = pickle.load(f)


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

    # Latest non-zero week as current week proxy
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

    # Original gig flag logic
    df["gig_income_drop_flag"] = np.where(
        (df["is_gig_worker"] == 1) &
        (df["current_week_income"] <= 0.5 * df["income_5w_avg"]) &
        (df["zero_income_weeks_5w"] >= 2),
        1,
        0
    )

    return df


def load_csv_df():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


def get_latest_snapshot(display_df):
    if display_df.empty:
        return pd.DataFrame()

    if "customer_id" not in display_df.columns:
        return display_df.copy()

    snapshot_df = display_df.copy()

    if "year_month" in snapshot_df.columns:
        snapshot_df["year_month"] = snapshot_df["year_month"].astype(str)
        snapshot_df = snapshot_df.sort_values(["customer_id", "year_month"])
        snapshot_df = snapshot_df.groupby("customer_id", as_index=False).tail(1)
    else:
        snapshot_df = snapshot_df.drop_duplicates(subset="customer_id", keep="last")

    return snapshot_df.reset_index(drop=True)


def get_primary_risk_reason(row):
    if row.get("income_stress_flag", 0) == 1:
        return "Income below historical average"
    elif row.get("gig_income_drop_flag", 0) == 1:
        return "Gig income instability"
    elif row.get("credit_utilization_ratio", 0) >= 0.8:
        return "High credit utilization"
    elif row.get("emi_late_days", 0) > 0:
        return "Late EMI payments"
    elif row.get("negative_balance_days", 0) > 0:
        return "Negative balance days"
    elif row.get("overdraft_flag", 0) == 1:
        return "Overdraft usage"
    elif row.get("income_stop_flag", 0) == 1:
        return "Income disruption"
    else:
        return "Elevated model risk"


def prepare_display_df():
    df = load_csv_df()

    if df.empty:
        return pd.DataFrame()


    model_df = df.copy()
    model_df = add_gig_features(model_df)

    model_df = model_df.drop(
        ["customer_id", "year_month", "emi_payment_date"],
        axis=1,
        errors="ignore"
    )

    if "will_default_2_4_weeks" in model_df.columns:
        model_df = model_df.drop("will_default_2_4_weeks", axis=1, errors="ignore")

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

    model_df = model_df.apply(pd.to_numeric, errors="coerce")
    model_df = model_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = model_df.reindex(columns=columns, fill_value=0)
    preds = model.predict_proba(X)[:, 1]


    display_df = df.copy()
    display_df = add_gig_features(display_df)
    display_df["risk_score"] = preds


    high_threshold = display_df["risk_score"].quantile(0.80)
    medium_threshold = display_df["risk_score"].quantile(0.50)

    display_df["status"] = np.where(
        display_df["risk_score"] >= high_threshold,
        "HIGH RISK",
        np.where(
            display_df["risk_score"] >= medium_threshold,
            "MEDIUM RISK",
            "LOW RISK"
        )
    )


    display_df["income_stress_flag"] = np.where(
        (display_df["income_5w_avg"] > 0) &
        (display_df["current_week_income"] <= 0.5 * display_df["income_5w_avg"]) &
        (display_df["zero_income_weeks_5w"] >= 2),
        1,
        0
    )

    
    display_df["status"] = np.where(
        (display_df["income_stress_flag"] == 1) & (display_df["status"] == "LOW RISK"),
        "MEDIUM RISK",
        display_df["status"]
    )

    
    display_df["status"] = np.where(
        (display_df["income_stress_flag"] == 1) & (display_df["status"] == "MEDIUM RISK"),
        "HIGH RISK",
        display_df["status"]
    )

    # Keep original gig logic influential too
    display_df["status"] = np.where(
        (display_df["gig_income_drop_flag"] == 1) & (display_df["status"] == "MEDIUM RISK"),
        "HIGH RISK",
        display_df["status"]
    )

    display_df["primary_risk_reason"] = display_df.apply(get_primary_risk_reason, axis=1)

    display_df = display_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return display_df


def build_reason_chart_data(display_df):
    if display_df.empty or "primary_risk_reason" not in display_df.columns:
        return []

    high_risk_df = display_df[display_df["status"] == "HIGH RISK"]

    if high_risk_df.empty:
        return []

    reason_counts = high_risk_df["primary_risk_reason"].value_counts().reset_index()
    reason_counts.columns = ["reason", "count"]
    return reason_counts.to_dict(orient="records")


def build_customer_timeline(display_df, customer_id=None):
    if display_df.empty:
        return {"labels": [], "scores": []}

    if "customer_id" not in display_df.columns or "year_month" not in display_df.columns:
        return {"labels": [], "scores": []}

    if customer_id is None:
        customer_id = display_df.iloc[0]["customer_id"]

    customer_df = display_df[
        display_df["customer_id"].astype(str) == str(customer_id)
    ].copy()

    if customer_df.empty:
        return {"labels": [], "scores": []}

    customer_df = customer_df.sort_values("year_month")

    return {
        "labels": customer_df["year_month"].astype(str).tolist(),
        "scores": customer_df["risk_score"].astype(float).tolist()
    }


def build_status_summary(display_df):
    if display_df.empty or "status" not in display_df.columns:
        return {
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "total_customers": 0
        }

    snapshot_df = get_latest_snapshot(display_df)

    return {
        "high_risk": int((snapshot_df["status"] == "HIGH RISK").sum()),
        "medium_risk": int((snapshot_df["status"] == "MEDIUM RISK").sum()),
        "low_risk": int((snapshot_df["status"] == "LOW RISK").sum()),
        "total_customers": int(snapshot_df["customer_id"].nunique()) if "customer_id" in snapshot_df.columns else int(len(snapshot_df))
    }


def build_liquidity_trend(display_df):
    if display_df.empty or "year_month" not in display_df.columns:
        return []

    if "liquidity_stress_score" not in display_df.columns:
        return []

    liquidity_df = (
        display_df.groupby("year_month", as_index=False)["liquidity_stress_score"]
        .mean()
        .sort_values("year_month")
    )

    return liquidity_df.to_dict(orient="records")


def build_risk_distribution(display_df):
    if display_df.empty or "status" not in display_df.columns:
        return {"low": 0, "medium": 0, "high": 0}

    return {
        "low": int((display_df["status"] == "LOW RISK").sum()),
        "medium": int((display_df["status"] == "MEDIUM RISK").sum()),
        "high": int((display_df["status"] == "HIGH RISK").sum())
    }


def build_dashboard_kpis(display_df):
    if display_df.empty:
        return {
            "total_customers": 0,
            "high_risk": 0,
            "total_loan_exposure": 0,
            "loss_prevented": 0,
            "avg_liquidity": 0
        }

    snapshot_df = get_latest_snapshot(display_df)

    total_customers = (
        int(snapshot_df["customer_id"].nunique())
        if "customer_id" in snapshot_df.columns else int(len(snapshot_df))
    )

    high_risk = int((snapshot_df["status"] == "HIGH RISK").sum())

    total_loan_exposure = (
        float(pd.to_numeric(snapshot_df["total_loan"], errors="coerce").fillna(0).sum())
        if "total_loan" in snapshot_df.columns else 0.0
    )

    expected_loss = (
        float(
            (
                pd.to_numeric(snapshot_df["risk_score"], errors="coerce").fillna(0)
                * pd.to_numeric(snapshot_df["total_loan"], errors="coerce").fillna(0)
            ).sum()
        )
        if "total_loan" in snapshot_df.columns else 0.0
    )

    intervention_rate = 0.25
    loss_prevented = expected_loss * intervention_rate

    avg_liquidity = (
        float(pd.to_numeric(snapshot_df["liquidity_stress_score"], errors="coerce").fillna(0).mean())
        if "liquidity_stress_score" in snapshot_df.columns else 0.0
    )

    return {
        "total_customers": total_customers,
        "high_risk": high_risk,
        "total_loan_exposure": round(total_loan_exposure, 2),
        "loss_prevented": round(loss_prevented, 2),
        "avg_liquidity": round(avg_liquidity, 2)
    }

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/risk-analytics")
def risk_analytics():
    return render_template("riskanalytics.html")


@app.route("/customer")
def customer_drilldown():
    return render_template("customer.html")


@app.route("/compliance")
def compliance():
    return render_template("compliance.html")


@app.route("/load_data", methods=["GET"])
def load_data():
    try:
        display_df = prepare_display_df()

        if display_df.empty:
            return jsonify([])

        return jsonify(display_df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    try:
        display_df = prepare_display_df()

        if display_df.empty:
            return jsonify({
                "table_data": [],
                "reason_chart_data": [],
                "customer_timeline": {"labels": [], "scores": []},
                "status_summary": {
                    "high_risk": 0,
                    "medium_risk": 0,
                    "low_risk": 0,
                    "total_customers": 0
                },
                "dashboard_kpis": {
                    "total_customers": 0,
                    "high_risk": 0,
                    "total_loan_exposure": 0,
                    "loss_prevented": 0,
                    "avg_liquidity": 0
                },
                "liquidity_trend": [],
                "risk_distribution": {"low": 0, "medium": 0, "high": 0}
            })

        snapshot_df = get_latest_snapshot(display_df)

        customer_id = None
        if "customer_id" in display_df.columns and not display_df.empty:
            customer_id = display_df.iloc[0]["customer_id"]

        return jsonify({
            "table_data": snapshot_df.to_dict(orient="records"),
            "reason_chart_data": build_reason_chart_data(snapshot_df),
            "customer_timeline": build_customer_timeline(display_df, customer_id),
            "status_summary": build_status_summary(display_df),
            "dashboard_kpis": build_dashboard_kpis(display_df),
            "liquidity_trend": build_liquidity_trend(display_df),
            "risk_distribution": build_risk_distribution(snapshot_df)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)