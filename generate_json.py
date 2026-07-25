import json
import pandas as pd

try:
    # 1. Dataset Read karo
    df = pd.read_csv("merged_dataset.csv")

    # Column names safe handling
    total_cust = len(df)
    loan_sum = (
        float(df["loan_amount"].sum())
        if "loan_amount" in df.columns
        else float(df.iloc[:, 1].sum() if len(df.columns) > 1 else 1000000)
    )

    # 2. JSON Structure Format
    dashboard_data = {
        "dashboard_kpis": {
            "total_customers": total_cust,
            "high_risk": int(total_cust * 0.2),  # Adjust as needed
            "total_loan_exposure": loan_sum,
            "loss_prevented": float(loan_sum * 0.15),
            "avg_liquidity": 1.45,
        },
        "risk_distribution": {
            "low": int(total_cust * 0.5),
            "medium": int(total_cust * 0.3),
            "high": int(total_cust * 0.2),
        },
        "status_summary": {
            "total_customers": total_cust,
            "high_risk": int(total_cust * 0.2),
            "medium_risk": int(total_cust * 0.3),
        },
        "reason_chart_data": [
            {"reason": "High EMI to Income Ratio", "count": 45},
            {"reason": "Utility Payment Delay", "count": 30},
            {"reason": "Frequent Overdrafts", "count": 25},
        ],
        "liquidity_trend": [
            {"year_month": "2024-01", "liquidity_stress_score": 1.2},
            {"year_month": "2024-02", "liquidity_stress_score": 1.5},
            {"year_month": "2024-03", "liquidity_stress_score": 1.8},
        ],
        "table_data": df.head(50).fillna("").to_dict(orient="records"),
    }

    # 3. File Save
    with open("dashboard_data.json", "w") as f:
        json.dump(dashboard_data, f, indent=4)

    print("Success! 'dashboard_data.json' file ban gayi hai.")

except Exception as e:
    print(f"Error aaya: {e}")
