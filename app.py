import streamlit as st
import pandas as pd
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="☕ Coffee Shop Premium Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('https://images.unsplash.com/photo-1509042239860-f550ce710b93');
            background-size: cover;
            background-attachment: fixed;
        }}
        .main {{
            background-color: rgba(255,255,255,0.92);
            padding: 20px;
            border-radius: 15px;
        }}
        h1, h2, h3 {{
            color: #6b3e26;
        }}
        .stMetric {{
            background-color: #fbeee6;
            padding: 10px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# ---------------- TITLE SECTION ----------------
st.markdown("<h1 style='text-align:center;'>☕ Coffee Shop Sales Dashboard</h1>", unsafe_allow_html=True)
st.image("https://cdn-icons-png.flaticon.com/512/924/924514.png", width=100)
st.caption("Interactive Sales Analysis & Smart Business Insights")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("CoffeeShopSales-cleaned.csv")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["month"] = df["transaction_date"].dt.month_name()
    df["weekday"] = df["transaction_date"].dt.day_name()
    return df


df = load_data()

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/590/590685.png", width=80)
st.sidebar.title("Navigation")

page  = st.sidebar.radio(...)
    [
        "🏠 Overview",
        "📈 Sales Analysis",
        "📦 Product Insights",
        "📊 Category vs Location",
        "📄 Raw Data"
        "🤖 Sales Prediction (ML)",
        "📊 Advanced Visuals",
    ]
)

st.sidebar.markdown("---")
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

# ---------------- OVERVIEW PAGE ----------------
if page == "🏠 Overview":
    st.subheader("📌 Key Performance Indicators")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("💰 Total Revenue", f"₹ {filtered_df['total_amount'].sum():,.0f}")
    c2.metric("🧾 Transactions", filtered_df.shape[0])
    c3.metric("📦 Quantity Sold", int(filtered_df["transaction_qty"].sum()))
    c4.metric("🛒 Avg Bill Value", f"₹ {filtered_df['total_amount'].mean():.0f}")

    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046784.png", width=120)

# ---------------- SALES ANALYSIS ----------------
elif page == "📈 Sales Analysis":
    st.subheader("📈 Monthly Sales Summary")

    monthly_sales = (
        filtered_df
        .groupby("month")["total_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(monthly_sales)

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
    w1.success(f"🔥 Best Sales Day: {best_day}")
    w2.warning(f"❄️ Lowest Sales Day: {worst_day}")

    st.line_chart(weekday_avg)

# ---------------- PRODUCT INSIGHTS ----------------
elif page == "📦 Product Insights":
    st.subheader("📦 Product Performance")

    product_sales = (
        filtered_df
        .groupby("product_type")["total_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(product_sales)

    top_product = product_sales.idxmax()

    st.info(f"🏆 Top-Selling Product: {top_product}")
    st.image("https://cdn-icons-png.flaticon.com/512/3075/3075977.png", width=120)

# ---------------- CATEGORY VS LOCATION ----------------
elif page == "📊 Category vs Location":
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
    st.area_chart(pivot_table)

# ---------------- RAW DATA ----------------
elif page == "📄 Raw Data":
    st.subheader("📄 Filtered Raw Data")
    st.dataframe(filtered_df)

# ---------------- ML SALES PREDICTION ----------------
elif page == "🤖 Sales Prediction (ML)":
    st.subheader("🤖 AI-Based Sales Prediction")
    
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import plotly.express as px

    # Prepare monthly data for prediction
    monthly_data = (
        df.groupby(df["transaction_date"].dt.to_period("M"))
        ["total_amount"].sum().reset_index()
    )

    monthly_data["transaction_date"] = monthly_data["transaction_date"].astype(str)
    monthly_data["month_index"] = np.arange(len(monthly_data))

    X = monthly_data[["month_index"]]
    y = monthly_data["total_amount"]

    model = LinearRegression()
    model.fit(X, y)

    # Predict next 3 months
    future_index = np.arange(len(monthly_data), len(monthly_data)+3).reshape(-1,1)
    predictions = model.predict(future_index)

    st.write("### 📈 Monthly Sales with Prediction")

    # Animated Plotly Chart
    fig = px.line(
        monthly_data,
        x="transaction_date",
        y="total_amount",
        markers=True,
        title="Historical Monthly Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(f"📊 Predicted Next Month Sales: ₹ {predictions[0]:,.0f}")

# ---------------- ADVANCED VISUALS ----------------
elif page == "📊 Advanced Visuals":
    st.subheader("📊 Interactive & Animated Visualizations")

    import plotly.express as px

    # Revenue by Category (Interactive)
    category_sales = (
        filtered_df.groupby("product_category")["total_amount"]
        .sum().reset_index()
    )

    fig1 = px.pie(
        category_sales,
        names="product_category",
        values="total_amount",
        title="Revenue Distribution by Category",
        hole=0.4
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Animated Scatter by Month
    animated_data = df.copy()
    animated_data["month"] = animated_data["transaction_date"].dt.month_name()

    fig2 = px.scatter(
        animated_data,
        x="transaction_qty",
        y="total_amount",
        animation_frame="month",
        color="product_category",
        size="total_amount",
        title="Animated Sales Trend by Month"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>☕ Designed with Streamlit | AI Powered Coffee Shop Analytics Dashboard</center>", unsafe_allow_html=True)
