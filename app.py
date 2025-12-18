import streamlit as st
import pandas as pd
import plotly.express as px
import re

# ======================= PAGE CONFIG =======================
st.set_page_config(
    page_title="US Opioid Overdose Death Analysis",
    layout="wide"
)

# ======================= LOAD DATA =======================
@st.cache_data
def load_data():
    df = pd.read_csv("clean_overdose_data.csv")

    # Normalize text
    df["state"] = df["state"].astype(str).str.strip().str.lower()
    df["month"] = df["month"].astype(str).str.strip()
    df["drug_type"] = df["drug_type"].astype(str).str.strip()

    # Remove all variations of US
    us_patterns = [
        r"united\s+states",
        r"us",
        r"u\.s\.",
        r"usa",
        r"america",
        r"\s*u\s*s\s*"
    ]
    us_mask = df["state"].apply(
        lambda x: not any(re.search(pattern, x, re.IGNORECASE) for pattern in us_patterns)
    )
    df = df[us_mask].copy()

    # Restore original case
    df["state"] = df["state"].str.title()

    # Type casting
    df["year"] = df["year"].astype(int)
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")

    # Month order
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    df["month"] = pd.Categorical(
        df["month"],
        categories=month_order,
        ordered=True
    )

    # Drop invalid rows
    df = df.dropna(subset=["deaths"])

    return df

df = load_data()

# ======================= TITLE =======================
st.title("🚨 Drug Overdose Death Analysis")
st.caption("State-Level Trends and Outlier Detection")
st.markdown("---")

# ======================= SIDEBAR FILTERS =======================
st.sidebar.header("🔍 Filters")

# State filter
states = sorted(df["state"].unique())
selected_states = st.sidebar.multiselect(
    "Select State(s)", states, default=states[:3]
)

# Year filter
year_min, year_max = int(df["year"].min()), int(df["year"].max())
selected_years = st.sidebar.slider(
    "Select Year Range", year_min, year_max, (year_min, year_max)
)

# Drug filter
drug_types = sorted(df["drug_type"].unique())
selected_drugs = st.sidebar.multiselect(
    "Select Drug Type(s)", drug_types, default=drug_types
)

# Apply filters
filtered_df = df.loc[
    (df["state"].isin(selected_states)) &
    (df["year"].between(selected_years[0], selected_years[1])) &
    (df["drug_type"].isin(selected_drugs))
].copy()

# ======================= METRICS =======================
col1, col2, col3 = st.columns(3)
col1.metric("Total Deaths", f"{filtered_df['deaths'].sum():,}")
col2.metric("Average Deaths", f"{filtered_df['deaths'].mean():,.1f}")
col3.metric("States Selected", filtered_df["state"].nunique())
st.markdown("---")

# ======================= TREND LINE =======================
st.subheader("📈 Monthly Death Trends by State")

# Aggregate deaths by year, month, and state
trend_df = (
    filtered_df
    .groupby(["year", "month", "state"], as_index=False)["deaths"]
    .sum()
)

# Create a proper datetime column for chronological x-axis
trend_df["year_month_dt"] = pd.to_datetime(trend_df["year"].astype(str) + "-" + trend_df["month"].astype(str), format="%Y-%B")

# Plot the line chart
trend_fig = px.line(
    trend_df,
    x="year_month_dt",
    y="deaths",
    color="state",
    markers=True,
    title="Monthly Overdose Death Trends"
)

trend_fig.update_layout(
    xaxis_title="Year & Month",
    hovermode="x unified"
)

# Optionally format x-axis to show "Jan 2020" style
trend_fig.update_xaxes(
    dtick="M1",
    tickformat="%b %Y"
)

st.plotly_chart(trend_fig, use_container_width=True)

# ======================= DRUG TYPE COMPARISON =======================
st.subheader("💊 Deaths by Drug Type")

# Bar Chart: filtered by selected drugs
drug_fig = px.bar(
    filtered_df,
    x="drug_type",
    y="deaths",
    color="state",
    barmode="group",
    title="Deaths by Drug Type and State (Filtered Drugs)"
)
st.plotly_chart(drug_fig, use_container_width=True)

# ======================= PIE CHART (GLOBAL DRUG DISTRIBUTION) =======================
st.subheader("💊 Distribution of Overdose Deaths by Drug Category")

pie_df = (
    df[df["year"].between(selected_years[0], selected_years[1])]
    .groupby("drug_type", as_index=False)["deaths"]
    .sum()
)

pie_fig = px.pie(
    pie_df,
    names="drug_type",
    values="deaths",
    hole=0.4,
    title="Percentage Contribution of Each Drug Category to Total Overdose Deaths"
)

st.plotly_chart(pie_fig, use_container_width=True)

# ======================= OUTLIER DETECTION =======================
st.subheader("🚨 Outlier Detection")

outlier_results = []
for state in filtered_df["state"].unique():
    state_data = filtered_df[filtered_df["state"] == state]
    Q1 = state_data["deaths"].quantile(0.25)
    Q3 = state_data["deaths"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    state_outliers = state_data[state_data["deaths"] > upper_bound]
    outlier_results.append(state_outliers)

if outlier_results:
    outliers = pd.concat(outlier_results)
else:
    outliers = pd.DataFrame()

if not outliers.empty:
    st.warning(f"⚠️ {len(outliers)} outlier records detected")
    st.dataframe(
        outliers[["state", "year", "month", "drug_type", "deaths"]],
        use_container_width=True
    )
else:
    st.success("No significant outliers detected.")

# ======================= FOOTER =======================
st.markdown("---")
st.markdown(
    "Source: "
    "[CDC VSRR Provisional Drug Overdose Death Counts (Kaggle)]"
    "(https://www.kaggle.com/datasets/craigchilvers/opioids-vssr-provisional-drug-overdose-statistics)"
)
