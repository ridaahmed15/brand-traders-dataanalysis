"""
Brand Traders · Sales Analytics
================================
A Streamlit dashboard that joins the 7 relational tables
(orders, order_details, products, categories, customers, employees, shippers)
into one analytical model and lets you slice it with global filters.

Run:   streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Brand Traders · Sales Analytics",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"
REQUIRED_TABLES = [
    "categories", "customers", "employees", "order_details",
    "orders", "products", "shippers",
]

# Plotly's "country names" matching is unreliable for e.g. "UK"/"USA", so map explicitly.
ISO3 = {
    "Argentina": "ARG", "Austria": "AUT", "Belgium": "BEL", "Brazil": "BRA",
    "Canada": "CAN", "Denmark": "DNK", "Finland": "FIN", "France": "FRA",
    "Germany": "DEU", "Ireland": "IRL", "Italy": "ITA", "Mexico": "MEX",
    "Norway": "NOR", "Poland": "POL", "Portugal": "PRT", "Spain": "ESP",
    "Sweden": "SWE", "Switzerland": "CHE", "UK": "GBR", "USA": "USA",
    "Venezuela": "VEN",
}

COLORS = px.colors.qualitative.Set2
PRIMARY = "#3b6fd4"


# ══════════════════════════════════════════════════════════════════════════════
# Formatting / UI helpers
# ══════════════════════════════════════════════════════════════════════════════
def fmt_money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.1f}K"
    return f"${x:,.0f}"


def show(fig: go.Figure, height: int = 380) -> None:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")


def table(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)


MONEY = st.column_config.NumberColumn(format="$%.0f")
MONEY2 = st.column_config.NumberColumn(format="$%.2f")
PCT = st.column_config.NumberColumn(format="%.1f%%")


# ══════════════════════════════════════════════════════════════════════════════
# Data loading + joining
# ══════════════════════════════════════════════════════════════════════════════
def _read_csv(raw: bytes) -> pd.DataFrame:
    """customers.csv / products.csv are not UTF-8 (accented names) → fall back to latin-1."""
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    df.columns = df.columns.str.strip()
    return df


def get_sources() -> dict[str, bytes]:
    """Use uploaded files if provided, otherwise the ./data folder."""
    st.sidebar.header("📁 Data")
    uploads = st.sidebar.file_uploader(
        "Upload the 7 CSV files (optional)",
        type="csv",
        accept_multiple_files=True,
        help="Leave empty to use the CSVs in the ./data folder.",
    )
    if uploads:
        src = {Path(u.name).stem.lower(): u.getvalue() for u in uploads}
    elif DATA_DIR.exists():
        src = {p.stem.lower(): p.read_bytes() for p in DATA_DIR.glob("*.csv")}
    else:
        src = {}

    missing = [t for t in REQUIRED_TABLES if t not in src]
    if missing:
        st.title("📊 Brand Traders · Sales Analytics")
        st.error(
            "Missing table(s): **" + ", ".join(f"{m}.csv" for m in missing) + "**. "
            "Put all 7 CSVs in a `data/` folder next to `app.py`, or upload them in the sidebar."
        )
        st.stop()
    return {k: src[k] for k in REQUIRED_TABLES}


@st.cache_data(show_spinner="Loading and joining tables…")
def build_model(sources: dict[str, bytes]):
    """Return (fact table, raw tables, data-quality report)."""
    t = {k: _read_csv(v) for k, v in sources.items()}

    # ── dimension tables (rename clashing column names up-front) ──────────────
    customers = t["customers"].rename(columns={
        "companyName": "customerName", "city": "customerCity", "country": "customerCountry",
        "contactName": "customerContact", "contactTitle": "customerContactTitle"})
    employees = t["employees"].copy()
    employees["managerName"] = employees["reportsTo"].map(
        employees.set_index("employeeID")["employeeName"])
    employees = employees.rename(columns={
        "title": "employeeTitle", "city": "employeeCity", "country": "employeeCountry"}
    ).drop(columns="reportsTo")
    shippers = t["shippers"].rename(columns={"companyName": "shipperName"})
    products = t["products"].rename(columns={"unitPrice": "listPrice"})
    categories = t["categories"][["categoryID", "categoryName"]]

    orders = t["orders"].copy()
    for c in ["orderDate", "requiredDate", "shippedDate"]:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")

    od = t["order_details"].copy()
    od["grossRevenue"] = od["unitPrice"] * od["quantity"]
    od["lineRevenue"] = od["grossRevenue"] * (1 - od["discount"])      # net of discount
    od["discountAmount"] = od["grossRevenue"] - od["lineRevenue"]

    # ── the join: order_details (grain = order line) is the fact table ────────
    fact = (
        od.merge(orders, on="orderID", how="left", validate="many_to_one")
          .merge(customers, on="customerID", how="left", validate="many_to_one")
          .merge(employees, on="employeeID", how="left", validate="many_to_one")
          .merge(shippers, on="shipperID", how="left", validate="many_to_one")
          .merge(products, on="productID", how="left", validate="many_to_one")
          .merge(categories, on="categoryID", how="left", validate="many_to_one")
    )
    fact["orderMonth"] = fact["orderDate"].dt.to_period("M").dt.to_timestamp()
    fact["orderYear"] = fact["orderDate"].dt.year
    fact["orderQuarter"] = (fact["orderDate"].dt.year.astype(str) + " Q"
                            + fact["orderDate"].dt.quarter.astype(str))

    # ── data-quality report ───────────────────────────────────────────────────
    checks = [
        ("Order lines without a matching order", int((~od["orderID"].isin(orders["orderID"])).sum())),
        ("Order lines without a matching product", int((~od["productID"].isin(products["productID"])).sum())),
        ("Orders without a matching customer", int((~orders["customerID"].isin(customers["customerID"])).sum())),
        ("Orders without a matching employee", int((~orders["employeeID"].isin(employees["employeeID"])).sum())),
        ("Orders without a matching shipper", int((~orders["shipperID"].isin(shippers["shipperID"])).sum())),
        ("Products without a matching category", int((~products["categoryID"].isin(categories["categoryID"])).sum())),
        ("Duplicate (order, product) lines", int(od.duplicated(["orderID", "productID"]).sum())),
        ("Rows gained/lost in join (fact vs order_details)", int(len(fact) - len(od))),
        ("Orders shipped before they were placed", int((orders["shippedDate"] < orders["orderDate"]).sum())),
        ("Orders not yet shipped (shippedDate empty)", int(orders["shippedDate"].isna().sum())),
    ]
    informational = {"Orders not yet shipped (shippedDate empty)"}
    quality = pd.DataFrame(checks, columns=["Check", "Count"])
    quality["Status"] = [
        "ℹ️ Info" if name in informational else ("✅ OK" if n == 0 else "⚠️ Review")
        for name, n in checks
    ]
    return fact, t, quality


def order_level(d: pd.DataFrame) -> pd.DataFrame:
    """Collapse order lines → one row per order (freight lives at order level)."""
    g = d.groupby("orderID", as_index=False).agg(
        orderDate=("orderDate", "first"),
        requiredDate=("requiredDate", "first"),
        shippedDate=("shippedDate", "first"),
        freight=("freight", "first"),
        customerID=("customerID", "first"),
        customerName=("customerName", "first"),
        customerCountry=("customerCountry", "first"),
        employeeName=("employeeName", "first"),
        shipperName=("shipperName", "first"),
        revenue=("lineRevenue", "sum"),
        units=("quantity", "sum"),
        items=("productID", "nunique"),
    )
    g["isShipped"] = g["shippedDate"].notna()
    g["isLate"] = g["isShipped"] & (g["shippedDate"] > g["requiredDate"])
    g["daysToShip"] = (g["shippedDate"] - g["orderDate"]).dt.days
    return g


def compute_kpis(d: pd.DataFrame) -> dict:
    if d.empty:
        return dict(revenue=0.0, orders=0, customers=0, units=0, aov=0.0,
                    disc_rate=0.0, freight=0.0, on_time=np.nan)
    ol = order_level(d)
    gross = d["grossRevenue"].sum()
    rev = d["lineRevenue"].sum()
    shipped = ol[ol["isShipped"]]
    return dict(
        revenue=rev,
        orders=len(ol),
        customers=d["customerID"].nunique(),
        units=int(d["quantity"].sum()),
        aov=rev / len(ol),
        disc_rate=(1 - rev / gross) if gross else 0.0,
        freight=ol["freight"].sum(),
        on_time=(1 - shipped["isLate"].mean()) if len(shipped) else np.nan,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Load data + sidebar filters
# ══════════════════════════════════════════════════════════════════════════════
fact, raw_tables, quality = build_model(get_sources())

st.sidebar.header("🔎 Filters")
data_min, data_max = fact["orderDate"].min().date(), fact["orderDate"].max().date()
date_range = st.sidebar.date_input(
    "Order date range", value=(data_min, data_max), min_value=data_min, max_value=data_max)
if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Pick an end date to apply the date filter.")
    st.stop()


def ms(label: str, col: str) -> list:
    opts = sorted(fact[col].dropna().unique().tolist())
    return st.sidebar.multiselect(label, opts, default=opts)


sel_cat = ms("Product category", "categoryName")
sel_country = ms("Customer country", "customerCountry")
sel_emp = ms("Sales employee", "employeeName")
sel_ship = ms("Shipper", "shipperName")
incl_disc = st.sidebar.checkbox("Include discontinued products", value=True)

base = fact[
    fact["categoryName"].isin(sel_cat)
    & fact["customerCountry"].isin(sel_country)
    & fact["employeeName"].isin(sel_emp)
    & fact["shipperName"].isin(sel_ship)
]
if not incl_disc:
    base = base[base["discontinued"] == 0]

start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
span = end - start + pd.Timedelta(days=1)
prev_start, prev_end = start - span, start - pd.Timedelta(days=1)
cur = base[(base["orderDate"] >= start) & (base["orderDate"] <= end)]
prev = base[(base["orderDate"] >= prev_start) & (base["orderDate"] <= prev_end)]
prev_ok = prev_start.date() >= data_min and not prev.empty   # only compare against a full window

st.sidebar.caption(
    f"Dataset covers **{data_min:%d %b %Y} → {data_max:%d %b %Y}**. "
    "First and last months are partial."
)

st.title("📊 Brand Traders · Sales Analytics")
st.caption(
    "Revenue = unit price × quantity × (1 − discount), excluding freight. "
    "Joined from orders, order_details, products, categories, customers, employees and shippers."
)

if cur.empty:
    st.warning("No data for the current filter selection. Widen the filters in the sidebar.")
    st.stop()

ol = order_level(cur)
k = compute_kpis(cur)
kp = compute_kpis(prev) if prev_ok else None

(tab_over, tab_prod, tab_cust, tab_emp,
 tab_ship, tab_disc, tab_data) = st.tabs([
    "🏠 Overview", "📦 Products", "👥 Customers & Geography", "🧑‍💼 Employees",
    "🚚 Shipping", "🏷️ Discounts", "🗂️ Data & Model",
])


# ══════════════════════════════════════════════════════════════════════════════
# 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_over:
    def metric(col, label, value, key, fmt=None):
        delta = None
        if kp is not None and kp.get(key):
            delta = f"{(k[key] / kp[key] - 1) * 100:+.1f}%"
        col.metric(label, value, delta=delta, border=True,
                   help="Change vs the preceding period of equal length." if delta else None)

    r1 = st.columns(4)
    metric(r1[0], "Net revenue", fmt_money(k["revenue"]), "revenue")
    metric(r1[1], "Orders", f"{k['orders']:,}", "orders")
    metric(r1[2], "Avg order value", f"${k['aov']:,.0f}", "aov")
    metric(r1[3], "Active customers", f"{k['customers']:,}", "customers")
    r2 = st.columns(4)
    metric(r2[0], "Units sold", f"{k['units']:,}", "units")
    r2[1].metric("Avg discount rate", f"{k['disc_rate']:.1%}", border=True)
    metric(r2[2], "Freight cost", fmt_money(k["freight"]), "freight")
    r2[3].metric("On-time shipping",
                 "n/a" if np.isnan(k["on_time"]) else f"{k['on_time']:.1%}", border=True,
                 help="Share of shipped orders with shippedDate ≤ requiredDate.")

    # ── key takeaways ─────────────────────────────────────────────────────────
    cat_rev = cur.groupby("categoryName")["lineRevenue"].sum().sort_values(ascending=False)
    ctry_rev = cur.groupby("customerCountry")["lineRevenue"].sum().sort_values(ascending=False)
    mon_rev = cur.groupby("orderMonth")["lineRevenue"].sum()
    cust_rev = cur.groupby("customerName")["lineRevenue"].sum().sort_values(ascending=False)
    prod_rev = cur.groupby("productName")["lineRevenue"].sum().sort_values(ascending=False)
    emp_rev = cur.groupby("employeeName")["lineRevenue"].sum().sort_values(ascending=False)
    top20 = max(1, round(len(cust_rev) * 0.2))
    tot = k["revenue"]

    with st.container(border=True):
        st.subheader("Key takeaways")
        st.markdown(
            f"- **{cat_rev.index[0]}** is the biggest category: {fmt_money(cat_rev.iloc[0])} "
            f"({cat_rev.iloc[0] / tot:.0%} of revenue).\n"
            f"- **{ctry_rev.index[0]}** leads by country with {fmt_money(ctry_rev.iloc[0])} "
            f"({ctry_rev.iloc[0] / tot:.0%}); the top 3 countries make up "
            f"{ctry_rev.head(3).sum() / tot:.0%}.\n"
            f"- Best month: **{mon_rev.idxmax():%b %Y}** at {fmt_money(mon_rev.max())}.\n"
            f"- Top product: **{prod_rev.index[0]}** ({fmt_money(prod_rev.iloc[0])}). "
            f"Top customer: **{cust_rev.index[0]}** ({fmt_money(cust_rev.iloc[0])}).\n"
            f"- The top 20% of customers ({top20}) generate **{cust_rev.head(top20).sum() / tot:.0%}** "
            f"of revenue.\n"
            f"- Best-selling employee: **{emp_rev.index[0]}** ({fmt_money(emp_rev.iloc[0])}).\n"
            f"- Discounts cost {fmt_money(cur['discountAmount'].sum())} "
            f"({k['disc_rate']:.1%} of gross sales)."
        )

    # ── trend ─────────────────────────────────────────────────────────────────
    monthly = (cur.groupby("orderMonth")
                  .agg(revenue=("lineRevenue", "sum"), orders=("orderID", "nunique"))
                  .reset_index())
    monthly["rolling3"] = monthly["revenue"].rolling(3, min_periods=1).mean()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=monthly["orderMonth"], y=monthly["revenue"], name="Revenue",
                marker_color=PRIMARY, opacity=0.8)
    fig.add_scatter(x=monthly["orderMonth"], y=monthly["rolling3"], name="3-month average",
                    mode="lines", line=dict(color="#e4572e", width=3))
    fig.add_scatter(x=monthly["orderMonth"], y=monthly["orders"], name="Orders",
                    mode="lines+markers", line=dict(color="#2a9d8f", dash="dot"), secondary_y=True)
    fig.update_layout(title="Monthly revenue and order count")
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Orders", secondary_y=True, showgrid=False)
    show(fig, 420)
    st.caption("The first and last months of the dataset are partial months, so they look lower.")

    c1, c2 = st.columns(2)
    with c1:
        d = cat_rev.reset_index()
        d.columns = ["Category", "Revenue"]
        fig = px.pie(d, names="Category", values="Revenue", hole=0.5,
                     color_discrete_sequence=COLORS, title="Revenue share by category")
        fig.update_traces(textinfo="percent+label")
        show(fig)
    with c2:
        d = ctry_rev.head(10).reset_index()
        d.columns = ["Country", "Revenue"]
        fig = px.bar(d, x="Revenue", y="Country", orientation="h",
                     title="Top 10 countries by revenue", color_discrete_sequence=[PRIMARY])
        fig.update_yaxes(categoryorder="total ascending")
        show(fig)

    c1, c2 = st.columns(2)
    with c1:
        q = cur.groupby("orderQuarter")["lineRevenue"].sum().reset_index()
        fig = px.bar(q, x="orderQuarter", y="lineRevenue", title="Revenue by quarter",
                     labels={"orderQuarter": "Quarter", "lineRevenue": "Revenue"},
                     color_discrete_sequence=[PRIMARY])
        show(fig)
    with c2:
        fig = px.histogram(ol, x="revenue", nbins=40, title="Order value distribution",
                           labels={"revenue": "Order value ($)"}, color_discrete_sequence=[PRIMARY])
        fig.update_layout(yaxis_title="Orders")
        show(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 2 · PRODUCTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_prod:
    prod = (cur.groupby(["productID", "productName", "categoryName", "discontinued"], as_index=False)
               .agg(revenue=("lineRevenue", "sum"), units=("quantity", "sum"),
                    orders=("orderID", "nunique"), listPrice=("listPrice", "first"),
                    discountGiven=("discountAmount", "sum")))

    c1, c2, c3 = st.columns([1, 1, 2])
    metric_choice = c1.selectbox("Rank by", ["Revenue", "Units", "Orders"], key="prod_metric")
    top_n = c2.slider("How many", 5, 30, 10, key="prod_n")
    col = {"Revenue": "revenue", "Units": "units", "Orders": "orders"}[metric_choice]

    cA, cB = st.columns(2)
    with cA:
        d = prod.nlargest(top_n, col)
        fig = px.bar(d, x=col, y="productName", orientation="h", color="categoryName",
                     color_discrete_sequence=COLORS, title=f"Top {top_n} products by {metric_choice.lower()}",
                     labels={col: metric_choice, "productName": ""})
        fig.update_yaxes(categoryorder="total ascending")
        show(fig, 460)
    with cB:
        d = prod.nsmallest(top_n, col)
        fig = px.bar(d, x=col, y="productName", orientation="h", color="categoryName",
                     color_discrete_sequence=COLORS, title=f"Bottom {top_n} products by {metric_choice.lower()}",
                     labels={col: metric_choice, "productName": ""})
        fig.update_yaxes(categoryorder="total descending")
        show(fig, 460)

    fig = px.treemap(prod[prod["revenue"] > 0], path=["categoryName", "productName"],
                     values="revenue", color="categoryName", color_discrete_sequence=COLORS,
                     title="Revenue treemap: category → product")
    show(fig, 480)

    cA, cB = st.columns(2)
    with cA:
        cat_m = (cur.groupby(["orderMonth", "categoryName"])["lineRevenue"].sum().reset_index())
        fig = px.area(cat_m, x="orderMonth", y="lineRevenue", color="categoryName",
                      color_discrete_sequence=COLORS, title="Monthly revenue by category",
                      labels={"orderMonth": "", "lineRevenue": "Revenue"})
        show(fig)
    with cB:
        fig = px.scatter(prod, x="listPrice", y="units", size="revenue", color="categoryName",
                         hover_name="productName", color_discrete_sequence=COLORS, log_x=True,
                         title="Price vs volume (bubble = revenue)",
                         labels={"listPrice": "List price ($, log)", "units": "Units sold"})
        show(fig)

    cA, cB = st.columns(2)
    with cA:
        cs = (cur.groupby("categoryName")
                 .agg(revenue=("lineRevenue", "sum"), units=("quantity", "sum"),
                      orders=("orderID", "nunique"), products=("productID", "nunique"),
                      avgDiscount=("discount", "mean"))
                 .reset_index().sort_values("revenue", ascending=False))
        cs["avgDiscount"] *= 100
        st.subheader("Category summary")
        table(cs.rename(columns={"categoryName": "Category"}),
              column_config={"revenue": MONEY, "avgDiscount": PCT})
    with cB:
        disc_share = prod.groupby("discontinued")["revenue"].sum()
        disc_rev = disc_share.get(1, 0.0)
        st.subheader("Discontinued products")
        st.metric("Revenue from discontinued items", fmt_money(disc_rev),
                  delta=f"{disc_rev / k['revenue']:.1%} of total", delta_color="off", border=True)
        table(prod[prod["discontinued"] == 1].sort_values("revenue", ascending=False)
              [["productName", "categoryName", "revenue", "units"]],
              column_config={"revenue": MONEY})

    with st.expander("Full product table"):
        table(prod.sort_values("revenue", ascending=False).drop(columns=["productID"]),
              column_config={"revenue": MONEY, "listPrice": MONEY2, "discountGiven": MONEY})


# ══════════════════════════════════════════════════════════════════════════════
# 3 · CUSTOMERS & GEOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════
with tab_cust:
    ref_date = cur["orderDate"].max()
    cust = (ol.groupby(["customerID", "customerName", "customerCountry"], as_index=False)
              .agg(orders=("orderID", "count"), revenue=("revenue", "sum"),
                   units=("units", "sum"), lastOrder=("orderDate", "max")))
    cust["aov"] = cust["revenue"] / cust["orders"]
    cust["recencyDays"] = (ref_date - cust["lastOrder"]).dt.days
    cust = cust.sort_values("revenue", ascending=False).reset_index(drop=True)

    cA, cB = st.columns(2)
    with cA:
        n = st.slider("Top customers to show", 5, 30, 10, key="cust_n")
        d = cust.head(n)
        fig = px.bar(d, x="revenue", y="customerName", orientation="h", color="customerCountry",
                     color_discrete_sequence=COLORS, title=f"Top {n} customers by revenue",
                     labels={"revenue": "Revenue", "customerName": ""})
        fig.update_yaxes(categoryorder="total ascending")
        show(fig, 460)
    with cB:
        cum = cust["revenue"].cumsum() / cust["revenue"].sum()
        pareto = pd.DataFrame({"rank": np.arange(1, len(cust) + 1) / len(cust) * 100,
                               "revenue": cust["revenue"], "cum": cum * 100})
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=pareto["rank"], y=pareto["revenue"], name="Revenue", marker_color=PRIMARY)
        fig.add_scatter(x=pareto["rank"], y=pareto["cum"], name="Cumulative %",
                        mode="lines", line=dict(color="#e4572e", width=3), secondary_y=True)
        fig.update_layout(title="Customer Pareto (customers ranked by revenue)")
        fig.update_xaxes(title_text="% of customers")
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative % of revenue", range=[0, 105],
                         secondary_y=True, showgrid=False)
        show(fig, 460)
        st.caption("Bars: each customer's revenue. Line: how quickly revenue concentrates in top customers.")

    ctry = (cur.groupby("customerCountry")
               .agg(revenue=("lineRevenue", "sum"), orders=("orderID", "nunique"),
                    customers=("customerID", "nunique"))
               .reset_index())
    ctry["aov"] = ctry["revenue"] / ctry["orders"]
    ctry["iso3"] = ctry["customerCountry"].map(ISO3)

    fig = px.choropleth(ctry.dropna(subset=["iso3"]), locations="iso3", color="revenue",
                        hover_name="customerCountry", hover_data={"iso3": False, "orders": True},
                        color_continuous_scale="Blues", title="Revenue by customer country")
    fig.update_geos(showcountries=True, fitbounds="locations", showframe=False)
    show(fig, 480)

    cA, cB = st.columns(2)
    with cA:
        fig = px.scatter(cust, x="orders", y="revenue", size="aov", color="recencyDays",
                         hover_name="customerName", color_continuous_scale="RdYlGn_r",
                         title="Customer value map (bubble = AOV, colour = days since last order)",
                         labels={"orders": "Orders", "revenue": "Revenue", "recencyDays": "Days since last order"})
        show(fig)
    with cB:
        fig = px.bar(ctry.sort_values("customers", ascending=False), x="customerCountry", y="customers",
                     title="Active customers by country", color_discrete_sequence=[PRIMARY],
                     labels={"customerCountry": "", "customers": "Customers"})
        show(fig)

    st.subheader("Customer table")
    table(cust.rename(columns={"customerName": "Customer", "customerCountry": "Country"})
              .drop(columns=["customerID"]),
          column_config={"revenue": MONEY, "aov": MONEY,
                         "lastOrder": st.column_config.DateColumn(format="YYYY-MM-DD")})
    st.subheader("Country table")
    table(ctry.drop(columns=["iso3"]).sort_values("revenue", ascending=False)
              .rename(columns={"customerCountry": "Country"}),
          column_config={"revenue": MONEY, "aov": MONEY})


# ══════════════════════════════════════════════════════════════════════════════
# 4 · EMPLOYEES
# ══════════════════════════════════════════════════════════════════════════════
with tab_emp:
    emp = (cur.groupby(["employeeName", "employeeTitle", "managerName"], as_index=False, dropna=False)
              .agg(revenue=("lineRevenue", "sum"), orders=("orderID", "nunique"),
                   customers=("customerID", "nunique"), units=("quantity", "sum"),
                   discountGiven=("discountAmount", "sum"), gross=("grossRevenue", "sum")))
    emp["aov"] = emp["revenue"] / emp["orders"]
    emp["avgDiscount"] = emp["discountGiven"] / emp["gross"] * 100
    emp = emp.sort_values("revenue", ascending=False)

    cA, cB = st.columns(2)
    with cA:
        fig = px.bar(emp, x="revenue", y="employeeName", orientation="h", color="employeeTitle",
                     color_discrete_sequence=COLORS, title="Revenue by employee",
                     labels={"revenue": "Revenue", "employeeName": ""})
        fig.update_yaxes(categoryorder="total ascending")
        show(fig)
    with cB:
        fig = px.scatter(emp, x="orders", y="aov", size="revenue", color="employeeName",
                         color_discrete_sequence=COLORS, title="Volume vs order value (bubble = revenue)",
                         labels={"orders": "Orders handled", "aov": "Avg order value ($)"})
        show(fig)

    emp_m = cur.groupby(["orderMonth", "employeeName"])["lineRevenue"].sum().reset_index()
    fig = px.line(emp_m, x="orderMonth", y="lineRevenue", color="employeeName",
                  color_discrete_sequence=COLORS, title="Monthly revenue by employee",
                  labels={"orderMonth": "", "lineRevenue": "Revenue"})
    show(fig)

    piv = cur.pivot_table(index="employeeName", columns="categoryName", values="lineRevenue",
                          aggfunc="sum", fill_value=0)
    fig = px.imshow(piv, text_auto=".2s", aspect="auto", color_continuous_scale="Blues",
                    title="Employee × category revenue")
    show(fig, 420)

    st.subheader("Employee scorecard")
    table(emp.drop(columns=["discountGiven", "gross"]).rename(columns={
              "employeeName": "Employee", "employeeTitle": "Title", "managerName": "Reports to"}),
          column_config={"revenue": MONEY, "aov": MONEY, "avgDiscount": PCT})


# ══════════════════════════════════════════════════════════════════════════════
# 5 · SHIPPING
# ══════════════════════════════════════════════════════════════════════════════
with tab_ship:
    shipped = ol[ol["isShipped"]]
    sh = (ol.groupby("shipperName")
            .agg(orders=("orderID", "count"), revenue=("revenue", "sum"),
                 freight=("freight", "sum"), avgFreight=("freight", "mean"),
                 unshipped=("isShipped", lambda s: int((~s).sum())))
            .join(shipped.groupby("shipperName").agg(
                avgDaysToShip=("daysToShip", "mean"), lateRate=("isLate", "mean")))
            .reset_index())
    sh["lateRate"] = sh["lateRate"] * 100

    m = st.columns(4)
    m[0].metric("Orders shipped", f"{len(shipped):,}", border=True)
    m[1].metric("Not yet shipped", f"{int((~ol['isShipped']).sum()):,}", border=True)
    m[2].metric("Avg days to ship", "n/a" if shipped.empty else f"{shipped['daysToShip'].mean():.1f}", border=True)
    m[3].metric("Late shipments", "n/a" if shipped.empty else f"{shipped['isLate'].mean():.1%}",
                help="shippedDate later than requiredDate", border=True)

    cA, cB = st.columns(2)
    with cA:
        fig = px.pie(sh, names="shipperName", values="orders", hole=0.5,
                     color_discrete_sequence=COLORS, title="Order share by shipper")
        show(fig)
    with cB:
        fig = px.bar(sh, x="shipperName", y="avgFreight", color_discrete_sequence=[PRIMARY],
                     title="Average freight per order", text_auto="$.2f",
                     labels={"shipperName": "", "avgFreight": "Avg freight ($)"})
        show(fig)

    cA, cB = st.columns(2)
    with cA:
        if shipped.empty:
            st.info("No shipped orders in this selection.")
        else:
            fig = px.box(shipped, x="shipperName", y="daysToShip", color="shipperName",
                         color_discrete_sequence=COLORS, title="Days from order to shipment",
                         labels={"shipperName": "", "daysToShip": "Days"})
            fig.update_layout(showlegend=False)
            show(fig)
    with cB:
        fig = px.bar(sh.dropna(subset=["lateRate"]), x="shipperName", y="lateRate",
                     color_discrete_sequence=["#e4572e"], title="Late shipment rate (%)",
                     text_auto=".1f", labels={"shipperName": "", "lateRate": "% shipped late"})
        show(fig)

    fr = (ol.groupby("customerCountry").agg(freight=("freight", "sum"), revenue=("revenue", "sum"))
            .reset_index())
    fr["freightPct"] = fr["freight"] / fr["revenue"] * 100
    fig = px.bar(fr.sort_values("freightPct", ascending=False), x="customerCountry", y="freightPct",
                 color_discrete_sequence=[PRIMARY], title="Freight as % of revenue, by destination country",
                 labels={"customerCountry": "", "freightPct": "Freight % of revenue"})
    show(fig)

    st.subheader("Shipper scorecard")
    table(sh.rename(columns={"shipperName": "Shipper"}),
          column_config={"revenue": MONEY, "freight": MONEY, "avgFreight": MONEY2,
                         "avgDaysToShip": st.column_config.NumberColumn(format="%.1f"),
                         "lateRate": PCT})

    open_orders = ol[~ol["isShipped"]].copy()
    if not open_orders.empty:
        open_orders["daysWaiting"] = (pd.Timestamp(data_max) - open_orders["orderDate"]).dt.days
        with st.expander(f"Orders not yet shipped ({len(open_orders)})"):
            table(open_orders[["orderID", "orderDate", "requiredDate", "customerName", "customerCountry",
                               "shipperName", "revenue", "daysWaiting"]].sort_values("daysWaiting", ascending=False),
                  column_config={"revenue": MONEY})


# ══════════════════════════════════════════════════════════════════════════════
# 6 · DISCOUNTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_disc:
    dd = cur.copy()
    dd["discountPct"] = (dd["discount"] * 100).round().astype(int)
    dd["hasDiscount"] = np.where(dd["discountPct"] > 0, "Discounted", "Full price")

    m = st.columns(4)
    m[0].metric("Discount given", fmt_money(dd["discountAmount"].sum()), border=True)
    m[1].metric("Effective discount rate", f"{k['disc_rate']:.1%}", border=True)
    m[2].metric("Lines with a discount", f"{(dd['discountPct'] > 0).mean():.1%}", border=True)
    dsc = dd[dd["discountPct"] > 0]
    m[3].metric("Avg discount when applied", "n/a" if dsc.empty else f"{dsc['discount'].mean():.1%}", border=True)

    by_lvl = (dd.groupby("discountPct")
                .agg(lines=("orderID", "count"), revenue=("lineRevenue", "sum"),
                     avgQty=("quantity", "mean"), discountGiven=("discountAmount", "sum"))
                .reset_index())
    by_lvl["label"] = by_lvl["discountPct"].astype(str) + "%"

    cA, cB = st.columns(2)
    with cA:
        fig = px.bar(by_lvl, x="label", y="revenue", color_discrete_sequence=[PRIMARY],
                     title="Revenue by discount level",
                     labels={"label": "Discount", "revenue": "Net revenue"})
        fig.update_xaxes(categoryorder="array", categoryarray=by_lvl["label"].tolist())
        show(fig)
    with cB:
        fig = px.bar(by_lvl, x="label", y="avgQty", color_discrete_sequence=["#2a9d8f"],
                     title="Average quantity per line, by discount level", text_auto=".1f",
                     labels={"label": "Discount", "avgQty": "Avg quantity"})
        fig.update_xaxes(categoryorder="array", categoryarray=by_lvl["label"].tolist())
        show(fig)

    cA, cB = st.columns(2)
    with cA:
        dc = dd.groupby("categoryName").agg(given=("discountAmount", "sum"),
                                            gross=("grossRevenue", "sum")).reset_index()
        dc["rate"] = dc["given"] / dc["gross"] * 100
        fig = px.bar(dc.sort_values("rate", ascending=False), x="categoryName", y="rate",
                     color_discrete_sequence=[PRIMARY], text_auto=".1f",
                     title="Effective discount rate by category (%)",
                     labels={"categoryName": "", "rate": "% of gross sales"})
        show(fig)
    with cB:
        cmp = (dd.groupby("hasDiscount").agg(lines=("orderID", "count"),
                                              avgQty=("quantity", "mean"),
                                              avgLine=("lineRevenue", "mean")).reset_index())
        fig = px.bar(cmp, x="hasDiscount", y="avgLine", color="hasDiscount",
                     color_discrete_sequence=COLORS, text_auto="$.0f",
                     title="Average order-line value: discounted vs full price",
                     labels={"hasDiscount": "", "avgLine": "Avg line revenue ($)"})
        fig.update_layout(showlegend=False)
        show(fig)

    dm = dd.groupby("orderMonth").agg(given=("discountAmount", "sum"), gross=("grossRevenue", "sum")).reset_index()
    dm["rate"] = dm["given"] / dm["gross"] * 100
    fig = px.line(dm, x="orderMonth", y="rate", markers=True, title="Effective discount rate over time (%)",
                  color_discrete_sequence=["#e4572e"], labels={"orderMonth": "", "rate": "% of gross sales"})
    show(fig, 320)
    table(by_lvl.drop(columns=["label"]).rename(columns={"discountPct": "Discount %"}),
          column_config={"revenue": MONEY, "discountGiven": MONEY,
                         "avgQty": st.column_config.NumberColumn(format="%.1f")})


# ══════════════════════════════════════════════════════════════════════════════
# 7 · DATA & MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("How the tables are related")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontname="Helvetica"];
      edge [fontname="Helvetica", fontsize=10];
      order_details -> orders     [label="orderID"];
      order_details -> products   [label="productID"];
      products      -> categories [label="categoryID"];
      orders        -> customers  [label="customerID"];
      orders        -> employees  [label="employeeID"];
      orders        -> shippers   [label="shipperID"];
      employees     -> employees  [label="reportsTo"];
    }""")
    st.caption("The analytical fact table has one row per order line (order_details) enriched "
               "with every dimension above. Freight is an order-level field, so order-level "
               "metrics de-duplicate lines first to avoid double counting.")

    cA, cB = st.columns([1, 1.4])
    with cA:
        st.subheader("Source tables")
        table(pd.DataFrame({"Table": list(raw_tables), "Rows": [len(v) for v in raw_tables.values()],
                            "Columns": [v.shape[1] for v in raw_tables.values()]}))
    with cB:
        st.subheader("Data-quality checks")
        table(quality)

    st.subheader("Joined dataset (current filters)")
    show_cols = ["orderID", "orderDate", "customerName", "customerCountry", "employeeName", "shipperName",
                 "categoryName", "productName", "unitPrice", "quantity", "discount",
                 "grossRevenue", "lineRevenue", "freight", "shippedDate"]
    table(cur[show_cols].sort_values(["orderDate", "orderID"]).reset_index(drop=True),
          column_config={"lineRevenue": MONEY2, "grossRevenue": MONEY2})
    st.download_button("⬇️ Download filtered joined data (CSV)",
                       cur.drop(columns=["orderMonth", "orderQuarter"]).to_csv(index=False).encode("utf-8"),
                       file_name="brand_traders_joined.csv", mime="text/csv")

    with st.expander("Show the pandas join used"):
        st.code("""fact = (order_details
    .merge(orders,     on="orderID")
    .merge(customers,  on="customerID")
    .merge(employees,  on="employeeID")
    .merge(shippers,   on="shipperID")
    .merge(products,   on="productID")
    .merge(categories, on="categoryID"))
fact["lineRevenue"] = fact.unitPrice * fact.quantity * (1 - fact.discount)""", language="python")
