import pandas as pd
import psycopg2
import psycopg2.extras
import os

df = pd.read_csv("cleaned_data.csv")
df['Customer ID'] = df['Customer ID'].astype(float).astype(int).astype(str) 
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df["TotalPrice"] = df["Quantity"] * df["Price"]

df_grouped = df.groupby(["Customer ID", "Invoice"]).agg(
    InvoiceDate=("InvoiceDate", "max"),
    Amount=("TotalPrice", "sum")
).reset_index()

print(f"Row Count: {len(df_grouped)}")
print(f"Invoice: {df_grouped["Invoice"].nunique()}")

# for invoice in df_grouped["Invoice"].unique():
#     df_f = df_grouped[df_grouped["Invoice"] == invoice]
#     if len(df_f) == 1: continue
#     print(df_f)

YEARS = 14
DAYS_PER_YEAR = 365.2425
delta_days = round(YEARS * DAYS_PER_YEAR)

df_grouped["InvoiceDate"] = df_grouped['InvoiceDate'] + pd.Timedelta(days=delta_days)

conn = psycopg2.connect(os.getenv("DATABASE_URL"))

c = conn.cursor()
psycopg2.extras.execute_values(
    c,
    'INSERT INTO transaction (customer_id, created_at, amount) VALUES %s',
    df_grouped[["Customer ID", "InvoiceDate", "Amount"]].itertuples(index=False)
)
conn.commit() 