import pandas as pd
import re

# Load raw dataset
df = pd.read_csv("C:\\Users\\Kirbs\\Desktop\\BSIT_3A\\FINALS_DATASETS\\VSRR_Provisional_Drug_Overdose_Death_Counts.csv")

# Rename columns for clarity
df = df.rename(columns={
    "State Name": "state",
    "Year": "year",
    "Month": "month",
    "Indicator": "drug_type",
    "Data Value": "deaths"
})

# Keep only necessary columns
df = df[["state", "year", "month", "drug_type", "deaths"]]

# Drop rows with missing values
df = df.dropna()

# 🔥 REMOVE US DATA AT SOURCE
# Comprehensive pattern matching for all US variations
us_patterns = [
    r"united\s+states",
    r"us",
    r"u\.s\.",
    r"usa",
    r"america",
    r"\s*u\s*s\s*"  # handles "u s" with spaces
]

# Create a boolean mask for non-US states
us_mask = df["state"].apply(
    lambda x: not any(re.search(pattern, str(x), re.IGNORECASE) for pattern in us_patterns)
)

# Apply the mask to filter out US data
df = df[us_mask].copy()

# Remove summary indicators (NOT actual drug categories)
df = df[~df["drug_type"].str.lower().isin([
    "number of deaths"
])]

# Focus on opioid-related deaths only (recommended)
df = df[df["drug_type"].str.contains("opioid", case=False, na=False)]

# Ensure deaths is numeric
df[" deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
df = df.dropna()

# 🚨 VALIDATION CHECK
remaining_us = df["state"].str.contains(
    r"united\s+states|us|u\.s\.|usa|america", 
    case=False, 
    regex=True
).sum()

if remaining_us > 0:
    print(f"❌ Warning: Found {remaining_us} US records that weren't filtered out!")
    print("These records will be included in the cleaned data")
else:
    print("✅ All US data successfully removed")

# Save cleaned data
df.to_csv("C:\\Users\\Kirbs\\Desktop\\BSIT_3A\\FINALS_DATASETS\\clean_overdose_data.csv", index=False)

print("✅ Cleaned file saved as clean_overdose_data.csv")
print(f"📊 Total records: {len(df)}")
print(f"🗺️ States included: {df['state'].nunique()}")
print(f"📅 Year range: {df['year'].min()} - {df['year'].max()}")