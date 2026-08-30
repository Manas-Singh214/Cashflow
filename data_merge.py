"""
data_merge.py
Reads source files from DATASETS/, cleans and merges them into 4 analysis-ready CSVs in MERGED DATASETS/.

Output files:
  1. merged_bank_financials.csv   - Bank health + ATM/card metrics (by bank)
  2. bank_transactions_clean.csv  - Individual transactions (cleaned)
  3. merged_crime_data.csv        - State/district crime enriched with cyber data
  4. cyber_national_trends.csv    - National-level year-wise cyber crime summary
"""

import os
import re
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

BASE      = r"d:\Projects\Data Analyst Portfolio\Complete DA"
BANK_DIR  = os.path.join(BASE, "DATASETS", "Bank Records")
CRIME_DIR = os.path.join(BASE, "DATASETS", "Crime Records")
OUT_DIR   = os.path.join(BASE, "MERGED DATASETS")
os.makedirs(OUT_DIR, exist_ok=True)


# --- Helpers ---

def normalize_bank_name(name):
    name = str(name).upper().strip()
    name = re.sub(r'\s+', ' ', name).rstrip('.')
    for suffix in [' LTD', ' LIMITED', ' BANK', ' (SBI)', '!']:
        name = name.replace(suffix, '')
    return name.strip()


STATE_MAP = {
    "A & N ISLANDS":                            "ANDAMAN & NICOBAR ISLANDS",
    "A&N ISLANDS":                              "ANDAMAN & NICOBAR ISLANDS",
    "ANDAMAN AND NICOBAR ISLANDS":              "ANDAMAN & NICOBAR ISLANDS",
    "D&N HAVELI":                               "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "DAMAN & DIU":                              "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "DELHI UT":                                 "DELHI",
    "JAMMU AND KASHMIR":                        "JAMMU & KASHMIR",
    "J&K":                                      "JAMMU & KASHMIR",
}

def normalize_state(name):
    key = str(name).upper().strip()
    return STATE_MAP.get(key, key.title())


def print_summary(label, df):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(f"  Shape : {df.shape[0]} rows x {df.shape[1]} cols")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(f"  Nulls : {nulls.to_string() if not nulls.empty else 'None'}")
    print(f"\n  Sample (3 rows):\n{df.head(3).to_string(index=False)}")


# ================================================================
# FILE 1 - merged_bank_financials.csv
# Bank financial data merged with RBI ATM/card infrastructure stats
# ================================================================
print("\n  Building merged_bank_financials.csv ...")

banks = pd.read_csv(os.path.join(BANK_DIR, "Indian banks data v2.0.csv"), encoding="utf-8")
banks.columns = banks.columns.str.strip()
banks["Bank"] = banks["Bank"].astype(str).str.strip()

banks_clean = banks[[
    "Bank", "Year", "Bank type",
    "Deposits", "Advances", "Total assets", "Net Profit",
    "Gross NPA", "Net NPA", "Net NPA as % to Net Advances",
    "Return on Assets [%]", "Capital Adequacy Ratio-Basel III [%]",
    "Credit Deposit Ratio", "Business per Employee", "Profit per Employee"
]].copy()

banks_clean.columns = [
    "Bank_Name", "Year", "Bank_Type",
    "Deposits_Cr", "Advances_Cr", "Total_Assets_Cr", "Net_Profit_Cr",
    "Gross_NPA_Cr", "Net_NPA_Cr", "Net_NPA_Pct",
    "Return_on_Assets_Pct", "Capital_Adequacy_Ratio_Pct",
    "Credit_Deposit_Ratio", "Business_Per_Employee_Cr", "Profit_Per_Employee_Lakh"
]
banks_clean["_join_key"] = banks_clean["Bank_Name"].apply(normalize_bank_name)

rbi = pd.read_csv(os.path.join(BANK_DIR, "RBI_ATM_Card_Statistics_Apr2025_Jan2026.csv"), encoding="utf-8")
rbi.columns = rbi.columns.str.strip()
rbi["ATMs_Total"] = rbi["ATMs_On_Site"].fillna(0) + rbi["ATMs_Off_Site"].fillna(0)

# Average RBI metrics per bank across all months
rbi_agg = rbi.groupby("Bank_Name").agg(
    ATMs_Total               = ("ATMs_Total",               "mean"),
    ATMs_OnSite              = ("ATMs_On_Site",              "mean"),
    ATMs_OffSite             = ("ATMs_Off_Site",             "mean"),
    PoS_Terminals            = ("PoS",                       "mean"),
    Micro_ATMs               = ("Micro_ATMs",                "mean"),
    UPI_QR_Codes             = ("UPI_QR_Codes",              "mean"),
    Bharat_QR_Codes          = ("Bharat_QR_Codes",           "mean"),
    Credit_Cards_Outstanding = ("Credit_Cards_Outstanding",  "mean"),
    Debit_Cards_Outstanding  = ("Debit_Cards_Outstanding",   "mean"),
).reset_index()
rbi_agg.rename(columns={"Bank_Name": "_join_key"}, inplace=True)
rbi_agg["_join_key"] = rbi_agg["_join_key"].apply(normalize_bank_name)
numeric_cols = rbi_agg.columns.difference(["_join_key"])
rbi_agg[numeric_cols] = rbi_agg[numeric_cols].round(0)

merged_banks = (
    pd.merge(banks_clean, rbi_agg, on="_join_key", how="left")
    .drop(columns=["_join_key"])
    .sort_values(["Bank_Name", "Year"])
    .reset_index(drop=True)
)

out1 = os.path.join(OUT_DIR, "merged_bank_financials.csv")
merged_banks.to_csv(out1, index=False, encoding="utf-8-sig")
print_summary("merged_bank_financials.csv", merged_banks)


# ================================================================
# FILE 2 - bank_transactions_clean.csv
# Individual transactions: deduped, typed, and filtered
# ================================================================
print("\n  Building bank_transactions_clean.csv ...")

txn = pd.read_csv(os.path.join(BANK_DIR, "bank_transactions.csv"), encoding="utf-8", on_bad_lines="skip")
txn.columns = txn.columns.str.strip()
txn.drop_duplicates(inplace=True)
txn.dropna(subset=["TransactionID", "CustomerID"], inplace=True)

txn["CustomerDOB"]     = pd.to_datetime(txn["CustomerDOB"],     dayfirst=True, errors="coerce")
txn["TransactionDate"] = pd.to_datetime(txn["TransactionDate"], dayfirst=True, errors="coerce")
txn.rename(columns={"TransactionAmount (INR)": "TransactionAmount_INR"}, inplace=True)
txn["CustAccountBalance"]    = pd.to_numeric(txn["CustAccountBalance"],    errors="coerce")
txn["TransactionAmount_INR"] = pd.to_numeric(txn["TransactionAmount_INR"], errors="coerce")
txn["CustGender"]            = txn["CustGender"].str.strip().str.upper()

# Keep only valid positive-amount transactions
txn = txn[txn["TransactionAmount_INR"].notna() & (txn["TransactionAmount_INR"] > 0)]
txn = txn[[
    "TransactionID", "CustomerID", "CustomerDOB", "CustGender",
    "CustLocation", "CustAccountBalance",
    "TransactionDate", "TransactionTime", "TransactionAmount_INR"
]]

out2 = os.path.join(OUT_DIR, "bank_transactions_clean.csv")
txn.to_csv(out2, index=False, encoding="utf-8-sig")
print_summary("bank_transactions_clean.csv", txn)


# ================================================================
# FILE 3 - merged_crime_data.csv
# Socioeconomic crime data enriched with state-level cyber stats
# ================================================================
print("\n  Building merged_crime_data.csv ...")

crime = pd.read_csv(os.path.join(CRIME_DIR, "india_crime_socioeconomic_data_700.csv"), encoding="utf-8", on_bad_lines="skip")
crime.columns = crime.columns.str.strip()
crime.rename(columns={
    "State/UT":               "State_UT",
    "GDP per Capita (INR)":   "GDP_Per_Capita_INR",
    "Literacy Rate (%)":      "Literacy_Rate_Pct",
    "Poverty Rate (%)":       "Poverty_Rate_Pct",
    "Unemployment Rate (%)":  "Unemployment_Rate_Pct",
    "Violent Crimes":         "Violent_Crimes",
    "Property Crimes":        "Property_Crimes",
    "Cyber Crimes":           "Cyber_Crimes",
    "Total Crimes Reported":  "Total_Crimes_Reported",
    "Crime Rate per 100,000": "Crime_Rate_Per_100K",
}, inplace=True)
crime["_state_key"] = crime["State_UT"].apply(normalize_state).str.upper()

# Cyber incidents by state (2016-2018) from NCRB Excel
xls = pd.read_excel(os.path.join(CRIME_DIR, "datafile.xls"), dtype=str)
xls.columns = xls.columns.astype(str).str.strip()
xls = xls[~xls["State/UT"].astype(str).str.upper().str.contains(r"TOTAL|ALL INDIA", na=True)]
xls["_state_key"] = xls["State/UT"].apply(normalize_state).str.upper()
xls_sel = xls[["_state_key", "2016", "2017", "2018", "Rate of Total Cyber Crimes (2018)++"]].copy()
xls_sel.columns = ["_state_key", "Cyber_Incidents_2016", "Cyber_Incidents_2017", "Cyber_Incidents_2018", "Cyber_Crime_Rate_2018"]
for col in xls_sel.columns[1:]:
    xls_sel[col] = pd.to_numeric(xls_sel[col], errors="coerce")

# Cyber fraud amounts by state from Parliament session data
rs17 = pd.read_csv(os.path.join(CRIME_DIR, "RS_Session_267_AU_1517_A_to_E.iii_.csv"), encoding="utf-8", on_bad_lines="skip")
rs17.columns = rs17.columns.str.strip()
rs17 = rs17[~rs17["State/UT-wise"].astype(str).str.upper().str.match(r"^TOTAL$", na=True)]
rs17["_state_key"] = rs17["State/UT-wise"].apply(normalize_state).str.upper()
rs17_sel = rs17[[
    "_state_key", "Total incidents Reported",
    "Amount Reported (Rs in Lakhs)", "Lien Amount (Rs in Lakhs)", "Refunded Amount (Rs in Lakhs)"
]].copy()
rs17_sel.columns = ["_state_key", "Cyber_Fraud_Incidents", "Cyber_Fraud_Amount_Reported_Lakh", "Cyber_Fraud_Lien_Amount_Lakh", "Cyber_Fraud_Refunded_Amount_Lakh"]
for col in rs17_sel.columns[1:]:
    rs17_sel[col] = pd.to_numeric(rs17_sel[col], errors="coerce")

crime_merged = (
    crime
    .merge(xls_sel, on="_state_key", how="left")
    .merge(rs17_sel, on="_state_key", how="left")
    .drop(columns=["_state_key"])
)
# Drop rows with no crime values at all
crime_merged.dropna(subset=["Violent_Crimes", "Property_Crimes", "Cyber_Crimes", "Total_Crimes_Reported"], how="all", inplace=True)
crime_merged.sort_values(["State_UT", "Year"], inplace=True)
crime_merged.reset_index(drop=True, inplace=True)

out3 = os.path.join(OUT_DIR, "merged_crime_data.csv")
crime_merged.to_csv(out3, index=False, encoding="utf-8-sig")
print_summary("merged_crime_data.csv", crime_merged)


# ================================================================
# FILE 4 - cyber_national_trends.csv
# National year-wise cyber crime and financial fraud summary
# ================================================================
print("\n  Building cyber_national_trends.csv ...")

df_1505 = pd.read_csv(os.path.join(CRIME_DIR, "RS_Session_267_AU_1505_A_to_C.ii_.csv"), on_bad_lines="skip")
df_1505.columns = df_1505.columns.str.strip()
df_1505 = df_1505[["Year-wise", "Number of Incidents", "De-Frauded Amount (In Crore)"]].copy()
df_1505.columns = ["Year", "Cyber_Fraud_Incidents", "Cyber_Fraud_Defrauded_Cr"]
df_1505["Year"] = pd.to_numeric(df_1505["Year"], errors="coerce").astype("Int64")

df_1991 = pd.read_csv(os.path.join(CRIME_DIR, "RS_Session_267_AU_1991_A.csv"), on_bad_lines="skip")
df_1991.columns = df_1991.columns.str.strip()
df_1991 = df_1991[["Year", "Number of Incidents (in lakhs)", "Total Amount Involved (Rs. in Crore)"]].copy()
df_1991.columns = ["Year_raw", "Financial_Fraud_Incidents_Lakh", "Financial_Fraud_Amount_Cr"]
df_1991["Year"] = pd.to_numeric(df_1991["Year_raw"].astype(str).str[:4], errors="coerce").astype("Int64")
df_1991.drop(columns=["Year_raw"], inplace=True)

df_528 = pd.read_csv(os.path.join(CRIME_DIR, "RS_Session_267_AU_528_A_to_B_0.csv"), on_bad_lines="skip")
df_528.columns = ["Year", "Cybersecurity_Incidents_Total"]
df_528["Year"] = pd.to_numeric(df_528["Year"], errors="coerce").astype("Int64")

trends = (
    df_528
    .merge(df_1505, on="Year", how="outer")
    .merge(df_1991, on="Year", how="outer")
    .sort_values("Year")
    .reset_index(drop=True)
)

out4 = os.path.join(OUT_DIR, "cyber_national_trends.csv")
trends.to_csv(out4, index=False, encoding="utf-8-sig")
print_summary("cyber_national_trends.csv", trends)


# ================================================================
# FINAL SUMMARY
# ================================================================
print(f"\n{'='*60}")
print("  All files saved to MERGED DATASETS/")
print(f"{'='*60}")
for f in [out1, out2, out3, out4]:
    print(f"  {os.path.basename(f):45s}  {os.path.getsize(f) / 1_048_576:6.2f} MB")
print(f"{'='*60}")