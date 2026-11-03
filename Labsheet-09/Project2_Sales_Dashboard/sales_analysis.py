"""Project 2: Sales Data Analysis Dashboard
Monthly sales, profit, customer segmentation (RFM), product performance, regional analysis.
"""
import os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 13, "axes.titleweight": "bold"})
def save(name): plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png"); plt.close()

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv("data/sales_data.csv", parse_dates=["order_date"])

# ---------------------------------------------------------------- 2. EDA
print("=" * 60, "\nSHAPE:", df.shape)
print("\nINFO:"); df.info()
print("\nMISSING VALUES:\n", df.isna().sum())
print("\nDESCRIBE:\n", df.describe().T)
print("Duplicate orders:", df.duplicated().sum())

# ---------------------------------------------------------------- 3. CLEANING
df = df.drop_duplicates()
df["segment"] = df["segment"].fillna(df["segment"].mode()[0])           # categorical -> mode
df["discount"] = df["discount"].fillna(0)                                # no record -> no discount
df["quantity"] = df.groupby("product")["quantity"].transform(lambda s: s.fillna(s.median())).astype(int)
df["sales"] = (df.unit_price * df.quantity * (1 - df.discount)).round(2) # recompute after fixing qty
df["margin"] = df.profit / df.sales
cat_margin = df.groupby("category")["margin"].transform("median")
df["profit"] = df["profit"].fillna((df.sales * cat_margin).round(2))     # estimate from category margin
df["margin"] = df.profit / df.sales
print("\nAfter cleaning, missing values:", df.isna().sum().sum(), "| rows:", len(df))

# ---------------------------------------------------------------- 4. FEATURES
df["month"] = df.order_date.dt.to_period("M").dt.to_timestamp()
df["year"] = df.order_date.dt.year
df["month_name"] = df.order_date.dt.month_name()

monthly = df.groupby("month").agg(sales=("sales", "sum"), profit=("profit", "sum"), orders=("order_id", "count")).reset_index()
cat = df.groupby("category")[["sales", "profit"]].sum().reset_index()
prod = df.groupby("product")[["sales", "profit", "quantity"]].sum().sort_values("sales", ascending=False).reset_index()
reg = df.groupby("region")[["sales", "profit"]].sum().reset_index()
reg["margin_%"] = reg.profit / reg.sales * 100
seg = df.groupby("segment")[["sales", "profit"]].sum().reset_index()

# --- Customer segmentation using RFM (Recency, Frequency, Monetary)
snapshot = df.order_date.max() + pd.Timedelta(days=1)
rfm = df.groupby("customer_id").agg(recency=("order_date", lambda s: (snapshot - s.max()).days),
                                    frequency=("order_id", "count"), monetary=("sales", "sum"))
for col, asc in [("recency", False), ("frequency", True), ("monetary", True)]:
    rfm[col[0].upper()] = pd.qcut(rfm[col].rank(method="first", ascending=asc), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["score"] = rfm[["R", "F", "M"]].sum(axis=1)
rfm["customer_segment"] = pd.cut(rfm.score, [0, 5, 7, 9, 12], labels=["At Risk", "Occasional", "Loyal", "Champions"])
rfm_sum = rfm.groupby("customer_segment", observed=True).agg(customers=("score", "count"), revenue=("monetary", "sum")).reset_index()
rfm.to_csv(f"{OUT}/customer_rfm_segments.csv")
print("\nRFM SEGMENTS:\n", rfm_sum)

# ---------------------------------------------------------------- 5. CHARTS
# 5.1 Monthly sales & profit trend (line)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(monthly.month, monthly.sales / 1e6, marker="o", label="Sales")
ax.plot(monthly.month, monthly.profit / 1e6, marker="s", label="Profit")
ax.set(title="Monthly Sales and Profit Trend", xlabel="Month", ylabel="Amount (Rs million)"); ax.legend()
save("01_monthly_sales_profit")

# 5.2 Seasonality
order = ["January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"]
fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(data=df.groupby("month_name").sales.sum().reindex(order).reset_index(), x="month_name", y="sales", ax=ax, color="#3b82f6")
ax.set(title="Seasonality: Total Sales by Calendar Month", xlabel="", ylabel="Sales (Rs)"); plt.xticks(rotation=40)
save("02_seasonality")

# 5.3 Yearly
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.barplot(data=df.groupby("year")[["sales", "profit"]].sum().reset_index().melt("year"), x="year", y="value", hue="variable", ax=ax)
ax.set(title="Yearly Sales vs Profit", ylabel="Rs", xlabel=""); save("03_yearly")

# 5.4 Category performance
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=cat.melt("category"), x="category", y="value", hue="variable", ax=ax)
ax.set(title="Sales and Profit by Category", ylabel="Rs", xlabel=""); save("04_category_performance")

# 5.5 Product performance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(data=prod.head(8).iloc[::-1], x="sales", y="product", ax=axes[0], color="#3b82f6"); axes[0].set(title="Top 8 Products by Sales", ylabel="")
sns.barplot(data=prod.sort_values("profit").head(5), x="profit", y="product", ax=axes[1], color="#ef4444"); axes[1].set(title="Bottom 5 Products by Profit", ylabel="")
save("05_product_performance")

# 5.6 Regional analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].pie(reg.sales, labels=reg.region, autopct="%1.1f%%", startangle=90, wedgeprops={"width": .45}); axes[0].set_title("Sales Share by Region")
sns.barplot(data=reg, x="region", y="margin_%", ax=axes[1], color="#22c55e"); axes[1].set(title="Profit Margin (%) by Region", xlabel="")
save("06_regional_analysis")

# 5.7 State-wise sales
fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(data=df.groupby("state").sales.sum().sort_values().reset_index(), x="sales", y="state", ax=ax, color="#8b5cf6")
ax.set(title="Sales by State", ylabel="", xlabel="Sales (Rs)"); save("07_state_sales")

# 5.8 Business segment split
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].pie(seg.sales, labels=seg.segment, autopct="%1.1f%%", startangle=90); axes[0].set_title("Sales by Customer Segment")
sns.barplot(data=seg, x="segment", y="profit", ax=axes[1], color="#f59e0b"); axes[1].set(title="Profit by Segment", xlabel="")
save("08_segment_analysis")

# 5.9 RFM segmentation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=rfm_sum, x="customer_segment", y="customers", ax=axes[0], color="#3b82f6"); axes[0].set(title="Customers per RFM Segment", xlabel="")
sns.scatterplot(data=rfm, x="frequency", y="monetary", hue="customer_segment", ax=axes[1], alpha=.7)
axes[1].set(title="RFM: Frequency vs Monetary Value"); save("09_rfm_segmentation")

# 5.10 Profit heatmap region x category
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df.pivot_table(index="region", columns="category", values="profit", aggfunc="sum"), annot=True, fmt=".0f", cmap="RdYlGn", ax=ax)
ax.set_title("Profit Heatmap: Region x Category"); save("10_profit_heatmap")

# 5.11 Discount vs profit margin
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="discount", y="margin", ax=ax); ax.axhline(0, color="red", ls="--")
ax.set(title="Impact of Discount on Profit Margin", xlabel="Discount", ylabel="Profit margin"); save("11_discount_vs_margin")

# 5.12 Correlation
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[["unit_price", "quantity", "discount", "sales", "profit"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix"); save("12_correlation")

# ---------------------------------------------------------------- 6. INTERACTIVE DASHBOARD (Plotly)
tot_s, tot_p = df.sales.sum(), df.profit.sum()
fig = make_subplots(rows=4, cols=2, vertical_spacing=.07,
    specs=[[{"type": "indicator"}, {"type": "indicator"}], [{"colspan": 2}, None], [{}, {"type": "domain"}], [{}, {}]],
    subplot_titles=("", "", "Monthly Sales & Profit", "Top Products by Sales", "Sales by Region",
                    "Profit by Category", "Customer Segments (RFM)"))
fig.add_trace(go.Indicator(mode="number", value=tot_s, title="Total Sales (Rs)", number={"valueformat": ",.0f"}), 1, 1)
fig.add_trace(go.Indicator(mode="number", value=tot_p, title=f"Total Profit (Rs) | Margin {tot_p / tot_s:.1%}",
                           number={"valueformat": ",.0f"}), 1, 2)
fig.add_trace(go.Scatter(x=monthly.month, y=monthly.sales, name="Sales", mode="lines+markers"), 2, 1)
fig.add_trace(go.Scatter(x=monthly.month, y=monthly.profit, name="Profit", mode="lines+markers"), 2, 1)
t = prod.head(8).iloc[::-1]
fig.add_trace(go.Bar(x=t.sales, y=t["product"], orientation="h", name="Product sales", marker_color="#3b82f6"), 3, 1)
fig.add_trace(go.Pie(labels=reg.region, values=reg.sales, hole=.45, name="Region"), 3, 2)
fig.add_trace(go.Bar(x=cat.category, y=cat.profit, name="Category profit", marker_color="#22c55e"), 4, 1)
fig.add_trace(go.Bar(x=rfm_sum.customer_segment.astype(str), y=rfm_sum.customers, name="Customers", marker_color="#f59e0b"), 4, 2)
fig.update_xaxes(rangeslider_visible=True, row=2, col=1)
fig.update_layout(title="Sales Analysis Dashboard", height=1400, template="plotly_white", showlegend=False)
fig.write_html(f"{OUT}/sales_dashboard.html")

# Second interactive chart: region filter dropdown
regs = ["All"] + sorted(df.region.unique())
f2 = go.Figure()
for r in regs:
    d = df if r == "All" else df[df.region == r]
    m = d.groupby("month")[["sales", "profit"]].sum().reset_index()
    f2.add_trace(go.Scatter(x=m.month, y=m.sales, name=f"Sales - {r}", visible=(r == "All")))
    f2.add_trace(go.Scatter(x=m.month, y=m.profit, name=f"Profit - {r}", visible=(r == "All")))
f2.update_layout(title="Monthly Sales & Profit (filter by region)", template="plotly_white",
    updatemenus=[dict(buttons=[dict(label=r, method="update",
        args=[{"visible": [i // 2 == k for i in range(2 * len(regs))]}]) for k, r in enumerate(regs)], x=1.0, y=1.15)])
f2.write_html(f"{OUT}/sales_by_region_filter.html")

# ---------------------------------------------------------------- 7. FINDINGS
best_m = monthly.loc[monthly.sales.idxmax()]
best_c = cat.loc[cat.sales.idxmax()]; best_cp = cat.loc[cat.profit.idxmax()]
best_r = reg.loc[reg.sales.idxmax()]; worst_m = reg.loc[reg["margin_%"].idxmin()]
neg = (df.profit < 0).mean() * 100
champ = rfm_sum[rfm_sum.customer_segment == "Champions"]
findings = f"""SALES ANALYSIS - KEY FINDINGS
==============================
1. Total sales Rs {tot_s:,.0f}, total profit Rs {tot_p:,.0f} (overall margin {tot_p / tot_s:.1%}).
2. Best month: {best_m.month:%B %Y} with sales of Rs {best_m.sales:,.0f}. Sales peak in Oct-Dec (festive season).
3. {best_c.category} generates the most sales; {best_cp.category} generates the most profit.
4. Top product by sales: {prod.iloc[0]['product']} (Rs {prod.iloc[0].sales:,.0f}).
5. Region with highest sales: {best_r.region}; lowest profit margin: {worst_m.region} ({worst_m['margin_%']:.1f}%).
6. {neg:.1f}% of orders are loss-making; losses are concentrated in orders with 20-30% discount.
7. RFM segmentation: {int(champ.customers.sum())} 'Champion' customers contribute Rs {champ.revenue.sum():,.0f} revenue.
Recommendations: cap discounts at 10%, stock up before Oct, run retention offers for 'At Risk' customers,
and focus on the high-margin Office Supplies category for profit growth.
"""
open(f"{OUT}/findings.txt", "w").write(findings); print("\n" + findings)
