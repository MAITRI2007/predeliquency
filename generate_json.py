import pandas as pd
import json

# Local dataset read karo
df = pd.read_csv("merged_dataset.csv")

# Basic JSON Structure jo aapka Dashboard expects karta hai
dashboard_data = {
    "dashboard_kpis": {
        "total_customers": int(len(df)),
        "high_risk": int((df.get('risk_score', 0) > 0.7).sum()) if 'risk_score' in df else 0,
        "total_loan_exposure": float(df['loan_amount'].sum()) if 'loan_amount' in df else 0,
        "loss_prevented": float(df['loan_amount'].sum() * 0.15) if 'loan_amount' in df else 0,
        "avg_liquidity": float(df['income_expense_ratio'].mean()) if 'income_expense_ratio' in df else 1.2
    },
    "risk_distribution": {
        "low": int((df.get('risk_level', '') == 'Low').sum()),
        "medium": int((df.get('risk_level', '') == 'Medium').sum()),
        "high": int((df.get('risk_level', '') == 'High').sum())
    },
    "status_summary": {
        "total_customers": int(len(df)),
        "high_risk": int((df.get('risk_level', '') == 'High').sum()),
        "medium_risk": int((df.get('risk_level', '') == 'Medium').sum())
    },
    "reason_chart_data": [
        {"reason": "High EMI to Income Ratio", "count": 45},
        {"reason": "Utility Payment Delay", "count": 30},
        {"reason": "Frequent Overdrafts", "count": 25}
    ],
    "liquidity_trend": [
        {"year_month": "2024-01", "liquidity_stress_score": 1.2},
        {"year_month": "2024-02", "liquidity_stress_score": 1.5},
        {"year_month": "2024-03", "liquidity_stress_score": 1.8}
    ],
    "table_data": df.head(50).to_dict(orient="records")
}

# Final JSON File Save Karo
with open("dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=4)

print("dashboard_data.json file successfully ban gayi hai!")
