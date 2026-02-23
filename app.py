import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="☕ Coffee Shop Sales Dashboard",
    layout="wide"
)

st.title("☕ Coffee Shop Sales Dashboard")
st.caption("Interactive Sales Analysis & Business Insights")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("CoffeeShopSales-cleaned.csv")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["month"] = df["transaction_date"].dt.month_name()
    return df

df = load_data()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔎 Filter Data")

location = st.sidebar.selectbox(
    "Store Location",
    ["All"] + sorted(df["store_location"].unique())
)

category = st.sidebar.selectbox(
    "Product Category",
    ["All"] + sorted(df["product_category"].unique())
)

month = st.sidebar.selectbox(
    "Month",
    ["All"] + sorted(df["month"].unique())
)

# ---------------- APPLY FILTERS ----------------
filtered_df = df.copy()

if location != "All":
    filtered_df = filtered_df[filtered_df["store_location"] == location]

if category != "All":
    filtered_df = filtered_df[filtered_df["product_category"] == category]

if month != "All":
    filtered_df = filtered_df[filtered_df["month"] == month]

# ---------------- KPI METRICS ----------------
st.subheader("📌 Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric("💰 Total Revenue", f"₹ {filtered_df['total_amount'].sum():,.0f}")
c2.metric("🧾 Transactions", filtered_df.shape[0])
c3.metric("📦 Quantity Sold", int(filtered_df["transaction_qty"].sum()))
c4.metric("🛒 Avg Bill Value", f"₹ {filtered_df['total_amount'].mean():.0f}")

# ---------------- MONTHLY SALES ----------------
st.subheader("📈 Monthly Sales Summary")

monthly_sales = (
    filtered_df
    .groupby("month")["total_amount"]
    .sum()
    .sort_values(ascending=False)
)

st.dataframe(monthly_sales)

# ---------------- PRODUCT PERFORMANCE ----------------
st.subheader("📦 Product Performance (Revenue)")

product_sales = (
    filtered_df
    .groupby("product_type")["total_amount"]
    .sum()
    .sort_values(ascending=False)
)

st.dataframe(product_sales)

# ---------------- WEEKDAY INSIGHTS ----------------
st.subheader("📅 Weekday Sales Insights")

weekday_avg = (
    filtered_df
    .groupby("weekday")["total_amount"]
    .mean()
    .sort_values(ascending=False)
)

best_day = weekday_avg.idxmax()
worst_day = weekday_avg.idxmin()

w1, w2 = st.columns(2)
w1.success(f"🔥 Best Sales Day: **{best_day}**")
w2.warning(f"❄️ Lowest Sales Day: **{worst_day}**")

st.dataframe(weekday_avg)

# ---------------- CATEGORY vs LOCATION ----------------
st.subheader("📊 Category vs Store Location")

pivot_table = pd.pivot_table(
    filtered_df,
    index="product_category",
    columns="store_location",
    values="total_amount",
    aggfunc="sum",
    fill_value=0
)

st.dataframe(pivot_table)

# ---------------- SMART INSIGHTS ----------------
st.subheader("🧠 Smart Business Insights")

top_product = product_sales.idxmax()
top_category = (
    filtered_df
    .groupby("product_category")["total_amount"]
    .sum()
    .idxmax()
)

st.info(
    f"""
    🔹 **Top-Selling Product:** {top_product}  
    🔹 **Best Performing Category:** {top_category}  
    🔹 **Highest Sales Day:** {best_day}
    """
)

# ---------------- RAW DATA ----------------
with st.expander("📄 View Filtered Raw Data"):
    st.dataframe(filtered_df)
    
