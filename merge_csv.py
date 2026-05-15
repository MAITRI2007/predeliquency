import pandas as pd

MAIN_FILE = "main.csv"
WEEKLY_FILE = "weekly.csv"
OUTPUT_FILE = "merged_dataset.csv"

main_df = pd.read_csv(MAIN_FILE)
weekly_df = pd.read_csv(WEEKLY_FILE)


main_df["customer_id"] = pd.to_numeric(main_df["customer_id"], errors="coerce")
weekly_df["customer_id"] = pd.to_numeric(weekly_df["customer_id"], errors="coerce")

main_df["year_month"] = main_df["year_month"].astype(str).str.strip()
weekly_df["year_month"] = weekly_df["year_month"].astype(str).str.strip()

weekly_df["salary_weekly"] = pd.to_numeric(weekly_df["salary_weekly"], errors="coerce").fillna(0)
weekly_df["iso_week"] = pd.to_numeric(weekly_df["iso_week"], errors="coerce")


weekly_df = weekly_df.sort_values(["customer_id", "year_month", "iso_week"]).copy()
weekly_df["week_rank"] = weekly_df.groupby(["customer_id", "year_month"]).cumcount() + 1


weekly_df = weekly_df[weekly_df["week_rank"] <= 5].copy()


weekly_pivot = weekly_df.pivot_table(
    index=["customer_id", "year_month"],
    columns="week_rank",
    values="salary_weekly",
    aggfunc="sum",
    fill_value=0
).reset_index()

weekly_pivot.columns = [
    "customer_id" if col == "customer_id" else
    "year_month" if col == "year_month" else
    f"salary_week_{int(col)}"
    for col in weekly_pivot.columns
]


for col in ["salary_week_1", "salary_week_2", "salary_week_3", "salary_week_4", "salary_week_5"]:
    if col not in weekly_pivot.columns:
        weekly_pivot[col] = 0


merged_df = pd.merge(
    main_df,
    weekly_pivot,
    on=["customer_id", "year_month"],
    how="left"
)

weekly_cols = ["salary_week_1", "salary_week_2", "salary_week_3", "salary_week_4", "salary_week_5"]
merged_df[weekly_cols] = merged_df[weekly_cols].fillna(0)


if "customer_segment" not in merged_df.columns:
    merged_df["customer_segment"] = "standard"
else:
    merged_df["customer_segment"] = merged_df["customer_segment"].fillna("standard")

merged_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Merged CSV created successfully")
print("Saved as:", OUTPUT_FILE)
print("Shape:", merged_df.shape)
print("Columns:", merged_df.columns.tolist())