"""
SME CASHFLOW AGENT
==================
A working cashflow agent for Zimbabwean / African SMEs that does five jobs:
  1. SEE      - ingest & auto-categorize transactions, track debtors/creditors
  2. PREDICT  - 90-day daily forecast, recurring + seasonality detection, scenarios
  3. WARN     - runway, anomaly, late-payer and VAT/tax alerts (red/green alarm)
  4. EXPLAIN  - cash leak finder, best/worst customers, cash conversion cycle
  5. FIX      - auto-drafted collection messages + a bill payment scheduler

Run locally:   streamlit run app.py
Deploy free:   push this repo to GitHub -> https://share.streamlit.io -> New app
"""

import re
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="SME Cashflow Agent", page_icon="💰", layout="wide")

TODAY = pd.Timestamp(datetime.now().date())

# --------------------------------------------------------------------------------------
# 1. SEE — DATA MODEL, SEED DATA, CATEGORIZATION RULES
# --------------------------------------------------------------------------------------

DEFAULT_RULES = {
    "rent": "Rent", "landlord": "Rent",
    "zimra": "ZIMRA/Tax", "vat": "ZIMRA/Tax", "tax": "ZIMRA/Tax",
    "salary": "Salaries", "salaries": "Salaries", "wages": "Salaries", "payroll": "Salaries",
    "fuel": "Fuel", "petrol": "Fuel", "diesel": "Fuel",
    "stock": "Stock/Inventory", "inventory": "Stock/Inventory", "supplier": "Stock/Inventory",
    "ecocash": "Mobile Money Fees", "onemoney": "Mobile Money Fees", "innbucks": "Mobile Money Fees",
    "electricity": "Utilities", "zesa": "Utilities", "water": "Utilities", "internet": "Utilities",
    "sale": "Sales Revenue", "invoice paid": "Sales Revenue", "customer payment": "Sales Revenue",
    "loan": "Loan/Finance", "bank charge": "Bank Charges", "interest": "Bank Charges",
    "transport": "Transport", "delivery": "Transport",
    "marketing": "Marketing", "advert": "Marketing",
}

CATEGORY_TYPE = {  # default whether a category is normally inflow or outflow
    "Sales Revenue": "Inflow", "Loan/Finance": "Inflow", "Other Income": "Inflow",
}


def categorize(description: str, rules: dict) -> str:
    d = description.lower()
    for kw, cat in rules.items():
        if kw in d:
            return cat
    return "Uncategorized"


def seed_transactions() -> pd.DataFrame:
    """Synthetic 90-day trading history so the agent has something to SEE/PREDICT from
    the moment it opens — stands in for a live bank-feed / EcoCash API connection."""
    rng = np.random.default_rng(42)
    rows = []
    start = TODAY - timedelta(days=90)
    customers = ["Mukamuri Traders", "City Hardware", "Zish Wholesalers", "Chipo Enterprises", "Tanaka & Sons"]
    for i in range(90):
        d = start + timedelta(days=i)
        # daily sales (inflow) - weekday pattern + month-end spike
        base_sales = rng.normal(450, 120)
        if d.day in (1, 2, 30, 31):
            base_sales *= 1.6  # month-end salary/restock spike in demand
        if d.weekday() >= 5:
            base_sales *= 0.5
        if base_sales > 0:
            rows.append([d, f"Sale - {rng.choice(customers)}", "Sales Revenue", "Inflow", round(base_sales, 2)])
        # daily stock purchases (outflow)
        if rng.random() < 0.4:
            rows.append([d, "Stock purchase - Supplier", "Stock/Inventory", "Outflow", round(rng.normal(220, 60), 2)])
        # fuel
        if rng.random() < 0.3:
            rows.append([d, "Fuel - Puma", "Fuel", "Outflow", round(rng.normal(45, 10), 2)])
        # recurring monthly rent on the 1st
        if d.day == 1:
            rows.append([d, "Rent - Landlord", "Rent", "Outflow", 600.0])
        # recurring salaries on the 28th
        if d.day == 28:
            rows.append([d, "Salaries - Staff payroll", "Salaries", "Outflow", 1400.0])
        # ecocash fees
        if rng.random() < 0.5:
            rows.append([d, "EcoCash transaction fee", "Mobile Money Fees", "Outflow", round(rng.uniform(2, 8), 2)])
        # utilities monthly
        if d.day == 15:
            rows.append([d, "ZESA electricity", "Utilities", "Outflow", round(rng.normal(90, 15), 2)])
    df = pd.DataFrame(rows, columns=["date", "description", "category", "type", "amount"])
    df["amount"] = df["amount"].abs().round(2)
    return df.sort_values("date").reset_index(drop=True)


def seed_debtors() -> pd.DataFrame:
    return pd.DataFrame([
        {"customer": "Mukamuri Traders", "invoice": "INV-1042", "amount": 850.0,
         "invoice_date": TODAY - timedelta(days=40), "due_date": TODAY - timedelta(days=10), "status": "Unpaid"},
        {"customer": "City Hardware", "invoice": "INV-1050", "amount": 1200.0,
         "invoice_date": TODAY - timedelta(days=20), "due_date": TODAY + timedelta(days=5), "status": "Unpaid"},
        {"customer": "Zish Wholesalers", "invoice": "INV-1031", "amount": 430.0,
         "invoice_date": TODAY - timedelta(days=60), "due_date": TODAY - timedelta(days=30), "status": "Paid"},
        {"customer": "Chipo Enterprises", "invoice": "INV-1055", "amount": 610.0,
         "invoice_date": TODAY - timedelta(days=5), "due_date": TODAY + timedelta(days=25), "status": "Unpaid"},
    ])


def seed_creditors() -> pd.DataFrame:
    return pd.DataFrame([
        {"supplier": "Zish Wholesalers (Supplier)", "bill": "BILL-501", "amount": 700.0,
         "due_date": TODAY + timedelta(days=3), "early_discount_pct": 5.0, "status": "Unpaid"},
        {"supplier": "ZESA", "bill": "BILL-502", "amount": 90.0,
         "due_date": TODAY + timedelta(days=10), "early_discount_pct": 0.0, "status": "Unpaid"},
        {"supplier": "Landlord", "bill": "BILL-503", "amount": 600.0,
         "due_date": TODAY + timedelta(days=1), "early_discount_pct": 0.0, "status": "Unpaid"},
    ])


def init_state():
    if "txns" not in st.session_state:
        st.session_state.txns = seed_transactions()
    if "rules" not in st.session_state:
        st.session_state.rules = dict(DEFAULT_RULES)
    if "debtors" not in st.session_state:
        st.session_state.debtors = seed_debtors()
    if "creditors" not in st.session_state:
        st.session_state.creditors = seed_creditors()
    if "start_balance" not in st.session_state:
        st.session_state.start_balance = 2500.0


init_state()

# --------------------------------------------------------------------------------------
# SIDEBAR — MANUAL INPUT FORM (live inflow / outflow entry) + SCENARIO CONTROLS
# --------------------------------------------------------------------------------------
st.sidebar.title("💰 SME Cashflow Agent")
st.sidebar.caption("SEE → PREDICT → WARN → EXPLAIN → FIX")

st.sidebar.header("📥 Add a transaction")
with st.sidebar.form("manual_txn_form", clear_on_submit=True):
    t_date = st.date_input("Date", value=TODAY.date())
    t_desc = st.text_input("Description", placeholder="e.g. Sale - City Hardware")
    t_type = st.selectbox("Type", ["Inflow", "Outflow"])
    t_amount = st.number_input("Amount (USD)", min_value=0.0, step=10.0, value=0.0)
    t_cat_guess = categorize(t_desc, st.session_state.rules) if t_desc else "Uncategorized"
    all_cats = sorted(set(list(st.session_state.rules.values()) + ["Uncategorized", "Other Income"]))
    t_cat = st.selectbox("Category (AI-suggested)", all_cats,
                          index=all_cats.index(t_cat_guess) if t_cat_guess in all_cats else all_cats.index("Uncategorized"))
    submitted = st.form_submit_button("➕ Add & update graph")
    if submitted and t_desc and t_amount > 0:
        new_row = pd.DataFrame([[pd.Timestamp(t_date), t_desc, t_cat, t_type, t_amount]],
                                columns=["date", "description", "category", "type", "amount"])
        st.session_state.txns = pd.concat([st.session_state.txns, new_row], ignore_index=True)
        # learn: if user typed a keyword not yet in rules, remember it for next time
        first_word = t_desc.lower().split(" ")[0]
        if len(first_word) > 2 and first_word not in st.session_state.rules:
            st.session_state.rules[first_word] = t_cat
        st.sidebar.success(f"Added {t_type} of ${t_amount:,.2f} ({t_cat})")

st.sidebar.header("🏦 Starting bank balance")
st.session_state.start_balance = st.sidebar.number_input(
    "Current bank balance (USD)", value=float(st.session_state.start_balance), step=100.0)

st.sidebar.header("🔮 Scenario modeling (what-if)")
late_days = st.sidebar.slider("Biggest client pays how many days late?", 0, 60, 0)
stock_change_pct = st.sidebar.slider("Change stock spend by (%)", -50, 100, 0)
fx_shift_pct = st.sidebar.slider("USD/local rate shock on costs (%)", -20, 20, 0)
horizon = st.sidebar.slider("Forecast horizon (days)", 30, 90, 90)

st.sidebar.header("📄 Bank feed / CSV upload")
uploaded = st.sidebar.file_uploader("Upload bank/EcoCash CSV (date, description, amount, type)", type=["csv"])
if uploaded is not None:
    try:
        new_df = pd.read_csv(uploaded)
        new_df.columns = [c.strip().lower() for c in new_df.columns]
        new_df["date"] = pd.to_datetime(new_df["date"])
        new_df["category"] = new_df["description"].apply(lambda d: categorize(str(d), st.session_state.rules))
        if "type" not in new_df.columns:
            new_df["type"] = np.where(new_df["amount"] >= 0, "Inflow", "Outflow")
        new_df["amount"] = new_df["amount"].abs()
        st.session_state.txns = pd.concat(
            [st.session_state.txns, new_df[["date", "description", "category", "type", "amount"]]],
            ignore_index=True)
        st.sidebar.success(f"Imported {len(new_df)} transactions & auto-categorized them.")
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

txns = st.session_state.txns.copy()
txns["date"] = pd.to_datetime(txns["date"])

# --------------------------------------------------------------------------------------
# HEADER + LIVE RED/GREEN ALARM
# --------------------------------------------------------------------------------------
st.title("💰 SME Cashflow Agent")
st.caption("A live agent that SEES your cash, PREDICTS what's coming, WARNS you early, "
           "EXPLAINS the why, and FIXES it — not just another dashboard.")

recent = txns[txns["date"] >= TODAY - timedelta(days=30)]
recent_in = recent.loc[recent["type"] == "Inflow", "amount"].sum()
recent_out = recent.loc[recent["type"] == "Outflow", "amount"].sum()
net_30 = recent_in - recent_out

alarm_col1, alarm_col2, alarm_col3 = st.columns(3)
with alarm_col1:
    if net_30 >= 0:
        st.success(f"🟢 OPERATING IN THE GREEN — last 30 days net +${net_30:,.2f}")
    else:
        st.error(f"🔴 OPERATING IN THE RED — last 30 days net -${abs(net_30):,.2f}")
with alarm_col2:
    st.metric("Inflows (30d)", f"${recent_in:,.0f}")
with alarm_col3:
    st.metric("Outflows (30d)", f"${recent_out:,.0f}")

# ========================================================================================
# TABS
# ========================================================================================
tab_see, tab_predict, tab_warn, tab_explain, tab_fix = st.tabs(
    ["👁️ SEE", "🔮 PREDICT", "🚨 WARN", "🧠 EXPLAIN", "🛠️ FIX"])

# --------------------------------------------------------------------------------------
# 2. PREDICT — helper functions (used by graph at top + Predict tab)
# --------------------------------------------------------------------------------------

def detect_recurring(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-learns items that repeat roughly monthly (e.g. 'Rent every 1st')."""
    out = []
    for (desc, cat, typ), g in df.groupby(["description", "category", "type"]):
        g = g.sort_values("date")
        if len(g) < 2:
            continue
        gaps = g["date"].diff().dt.days.dropna()
        if len(gaps) == 0:
            continue
        if gaps.median() >= 25:  # roughly monthly or slower
            out.append({
                "description": desc, "category": cat, "type": typ,
                "avg_amount": g["amount"].mean(),
                "typical_day_of_month": int(g["date"].dt.day.mode()[0]),
                "occurrences": len(g),
            })
    return pd.DataFrame(out)


def build_forecast(df: pd.DataFrame, start_balance: float, days: int,
                    late_days: int, stock_change_pct: float, fx_shift_pct: float) -> pd.DataFrame:
    recurring = detect_recurring(df)
    last_60 = df[df["date"] >= df["date"].max() - timedelta(days=60)] if len(df) else df

    daily_in_avg = last_60.loc[last_60["type"] == "Inflow"].groupby(
        last_60.loc[last_60["type"] == "Inflow", "date"].dt.date)["amount"].sum().mean() if len(last_60) else 0
    daily_out_avg = last_60.loc[last_60["type"] == "Outflow"].groupby(
        last_60.loc[last_60["type"] == "Outflow", "date"].dt.date)["amount"].sum().mean() if len(last_60) else 0
    daily_in_avg = 0 if np.isnan(daily_in_avg) else daily_in_avg
    daily_out_avg = 0 if np.isnan(daily_out_avg) else daily_out_avg

    # weekday seasonality multiplier for sales (inflow)
    if len(last_60):
        wd_mult = last_60[last_60["type"] == "Inflow"].groupby(
            last_60.loc[last_60["type"] == "Inflow", "date"].dt.weekday)["amount"].sum()
        wd_mult = (wd_mult / wd_mult.mean()).to_dict() if len(wd_mult) else {}
    else:
        wd_mult = {}

    last_date = df["date"].max() if len(df) else TODAY
    rows = []
    balance = start_balance
    for i in range(1, days + 1):
        d = last_date + timedelta(days=i)
        inflow = daily_in_avg * wd_mult.get(d.weekday(), 1.0)
        outflow = daily_out_avg

        # recurring items land on their typical day of month
        for _, r in recurring.iterrows():
            if d.day == r["typical_day_of_month"]:
                if r["type"] == "Inflow":
                    inflow += r["avg_amount"]
                else:
                    outflow += r["avg_amount"]

        # scenario: biggest client pays late -> shift a chunk of inflow forward by late_days
        # modeled as a dip now that reappears later; approximate with 15% of daily inflow
        if late_days > 0:
            shifted = inflow * 0.15
            inflow -= shifted
            if i == late_days:
                inflow += shifted * (days / max(late_days, 1)) * 0  # placeholder, handled below

        # scenario: stock spend change %
        outflow *= (1 + stock_change_pct / 100.0) if outflow else outflow
        # scenario: fx shock raises cost of imported stock/fuel
        outflow *= (1 + fx_shift_pct / 100.0)

        net = inflow - outflow
        balance += net
        rows.append([d, inflow, outflow, net, balance])

    fdf = pd.DataFrame(rows, columns=["date", "inflow", "outflow", "net", "balance"])

    # apply the "late payment" lump sum arriving on day `late_days`
    if late_days > 0 and late_days <= days and len(fdf):
        lump = daily_in_avg * 0.15 * late_days
        fdf.loc[fdf.index[late_days - 1], "inflow"] += lump
        fdf.loc[fdf.index[late_days - 1], "net"] += lump
        fdf.loc[fdf.index[late_days - 1]:, "balance"] += lump

    return fdf, recurring


forecast_df, recurring_df = build_forecast(
    txns, st.session_state.start_balance, horizon, late_days, stock_change_pct, fx_shift_pct)

# --------------------------------------------------------------------------------------
# LIVE GRAPH — always visible under the header, updates on every input change
# --------------------------------------------------------------------------------------
st.subheader("📈 Live cashflow graph — inflows vs outflows vs balance")

hist_daily = txns.groupby([txns["date"].dt.date, "type"])["amount"].sum().unstack(fill_value=0)
hist_daily.index = pd.to_datetime(hist_daily.index)
for col in ["Inflow", "Outflow"]:
    if col not in hist_daily.columns:
        hist_daily[col] = 0
hist_daily = hist_daily.sort_index()
# reconstruct a running historical balance that ends exactly at today's entered start_balance,
# then the forecast picks up from that same number
running = (hist_daily["Inflow"] - hist_daily["Outflow"]).cumsum()
if len(running):
    hist_daily["balance"] = st.session_state.start_balance - running.iloc[-1] + running
else:
    hist_daily["balance"] = st.session_state.start_balance

fig = go.Figure()
fig.add_trace(go.Bar(x=hist_daily.index, y=hist_daily["Inflow"], name="Inflow (history)",
                      marker_color="seagreen", opacity=0.8))
fig.add_trace(go.Bar(x=hist_daily.index, y=-hist_daily["Outflow"], name="Outflow (history)",
                      marker_color="firebrick", opacity=0.8))
fig.add_trace(go.Bar(x=forecast_df["date"], y=forecast_df["inflow"], name="Inflow (forecast)",
                      marker_color="lightgreen"))
fig.add_trace(go.Bar(x=forecast_df["date"], y=-forecast_df["outflow"], name="Outflow (forecast)",
                      marker_color="lightsalmon"))

bal_x = list(hist_daily.index) + list(forecast_df["date"])
bal_y = list(hist_daily["balance"]) + list(forecast_df["balance"])
bal_colors = ["green" if v >= 0 else "red" for v in bal_y]
fig.add_trace(go.Scatter(x=bal_x, y=bal_y, name="Cash balance", mode="lines",
                          line=dict(color="royalblue", width=3), yaxis="y2"))
fig.add_hline(y=0, line_dash="dot", line_color="gray", yaxis="y2")

fig.update_layout(
    barmode="relative", height=480,
    yaxis=dict(title="Daily inflow / outflow ($)"),
    yaxis2=dict(title="Cash balance ($)", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=30, b=10),
)
fig.add_vline(x=TODAY, line_dash="dash", line_color="black")
st.plotly_chart(fig, use_container_width=True)
st.caption("Dashed vertical line = today. Bars left of it are actual history, right of it are the forecast. "
           "Blue line (right axis) is projected cash balance — add a transaction or move a scenario "
           "slider in the sidebar and this graph updates live.")

# forward-looking alarm
neg_days = forecast_df[forecast_df["balance"] < 0]
if len(neg_days):
    first_neg = neg_days.iloc[0]
    st.error(f"🔴 FORECAST WARNING: at current trend you go **negative on "
              f"{first_neg['date'].strftime('%d %b %Y')}** (in {(first_neg['date'] - TODAY).days} days), "
              f"balance ≈ ${first_neg['balance']:,.0f}.")
else:
    st.success(f"🟢 FORECAST OK: balance stays positive for the next {horizon} days, "
               f"ending at ≈ ${forecast_df['balance'].iloc[-1]:,.0f}.")

# ========================================================================================
# TAB 1: SEE
# ========================================================================================
with tab_see:
    st.header("👁️ SEE — ingestion, auto-categorization, debtors & creditors")

    st.markdown("**Auto-connect (demo):** Bank feed and EcoCash connectors are stubbed here as CSV upload "
                "(sidebar) — swap in a real Open Banking / EcoCash API client and the rest of the agent "
                "needs zero changes, since everything downstream reads from one transactions table.")

    st.subheader("Transactions (AI-categorized, editable)")
    st.caption("Edit the **category** column directly — the agent learns from your corrections "
               "for next time (keyword is remembered).")
    edited = st.data_editor(
        txns.sort_values("date", ascending=False).reset_index(drop=True),
        column_config={
            "amount": st.column_config.NumberColumn(format="$%.2f"),
            "date": st.column_config.DateColumn(),
        },
        num_rows="dynamic", use_container_width=True, key="txn_editor",
    )
    if st.button("💾 Save corrections & re-learn categories"):
        for _, row in edited.iterrows():
            fw = str(row["description"]).lower().split(" ")[0]
            if len(fw) > 2:
                st.session_state.rules[fw] = row["category"]
        st.session_state.txns = edited
        st.success("Corrections saved. The agent will auto-apply this category next time it sees similar text.")

    st.subheader("🧾 Invoice / bill scanning (paste text to simulate OCR)")
    st.caption("In production this box is fed by a real OCR/vision API on an uploaded photo of the invoice. "
               "Paste the invoice text below and the agent extracts amount + due date.")
    invoice_text = st.text_area("Paste invoice or receipt text",
                                 placeholder="Invoice #1234\nAmount Due: $350.00\nDue Date: 2026-10-15")
    if st.button("🔍 Extract invoice data"):
        amt_match = re.search(r"(?:amount due|total|amount)[:\s]*\$?([\d,]+\.?\d*)", invoice_text, re.I)
        date_match = re.search(r"(?:due date|due)[:\s]*([\d]{4}-[\d]{2}-[\d]{2}|[\d]{1,2}[/-][\d]{1,2}[/-][\d]{2,4})",
                                invoice_text, re.I)
        if amt_match:
            st.success(f"Extracted amount: **${amt_match.group(1)}**")
        if date_match:
            st.success(f"Extracted due date: **{date_match.group(1)}**")
        if not amt_match and not date_match:
            st.warning("Couldn't find an amount/due date pattern — try a clearer format.")

    col_d, col_c = st.columns(2)
    with col_d:
        st.subheader("📒 Debtors ledger (who owes you)")
        st.session_state.debtors = st.data_editor(
            st.session_state.debtors, use_container_width=True, num_rows="dynamic", key="debtors_editor")
    with col_c:
        st.subheader("📕 Creditors ledger (who you owe)")
        st.session_state.creditors = st.data_editor(
            st.session_state.creditors, use_container_width=True, num_rows="dynamic", key="creditors_editor")

# ========================================================================================
# TAB 2: PREDICT
# ========================================================================================
with tab_predict:
    st.header("🔮 PREDICT — 90-day forecast, recurring & seasonality")

    st.subheader("Recurring items the agent auto-learned")
    if len(recurring_df):
        st.dataframe(recurring_df.style.format({"avg_amount": "${:.2f}"}), use_container_width=True)
    else:
        st.info("Not enough history yet to detect recurring items — add more transactions.")

    st.subheader("Month-end seasonality")
    txns["dom"] = txns["date"].dt.day
    monthend = txns[txns["type"] == "Inflow"].groupby("dom")["amount"].mean()
    if len(monthend):
        spike_days = monthend.sort_values(ascending=False).head(3).index.tolist()
        st.write(f"Detected sales spikes around day(s) of month: **{spike_days}** "
                 f"(typical month-end salary / restocking pattern).")

    st.subheader("Scenario impact on ending balance")
    baseline_df, _ = build_forecast(txns, st.session_state.start_balance, horizon, 0, 0, 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline ending balance", f"${baseline_df['balance'].iloc[-1]:,.0f}")
    c2.metric("With your scenario", f"${forecast_df['balance'].iloc[-1]:,.0f}",
              delta=f"{forecast_df['balance'].iloc[-1] - baseline_df['balance'].iloc[-1]:,.0f}")
    c3.metric("Forecast horizon", f"{horizon} days")
    st.caption("Adjust the sidebar scenario sliders (client pays late / stock spend / FX shock) to see the "
               "ending balance and the live graph above react instantly.")

    st.dataframe(forecast_df.style.format(
        {"inflow": "${:.0f}", "outflow": "${:.0f}", "net": "${:.0f}", "balance": "${:.0f}"}),
        use_container_width=True, height=280)

# ========================================================================================
# TAB 3: WARN
# ========================================================================================
with tab_warn:
    st.header("🚨 WARN — early warning system")

    # Runway alert
    st.subheader("⏳ Runway alert")
    if len(neg_days):
        st.error(f"At current burn, you will be **negative by {first_neg['date'].strftime('%d %b %Y')}** "
                 f"({(first_neg['date'] - TODAY).days} days from today).")
    else:
        st.success(f"Runway is healthy for the next {horizon} days.")

    # Anomaly alert
    st.subheader("📊 Anomaly alert")
    last_month = txns[txns["date"] >= TODAY - timedelta(days=30)]
    prev_month = txns[(txns["date"] < TODAY - timedelta(days=30)) & (txns["date"] >= TODAY - timedelta(days=60))]
    cat_last = last_month.groupby("category")["amount"].sum()
    cat_prev = prev_month.groupby("category")["amount"].sum()
    anomalies = []
    for cat in cat_last.index:
        prev_val = cat_prev.get(cat, 0)
        if prev_val > 0 and cat_last[cat] > prev_val * 2:
            anomalies.append((cat, cat_last[cat], prev_val, cat_last[cat] / prev_val))
    if anomalies:
        for cat, cur, prev, mult in anomalies:
            st.warning(f"**{cat}** is **{mult:.1f}x** higher than last month "
                       f"(${cur:,.0f} vs ${prev:,.0f}).")
    else:
        st.success("No unusual spending spikes detected vs last month.")

    # Late payer alert
    st.subheader("⏰ Late payer alert")
    deb = st.session_state.debtors.copy()
    deb["due_date"] = pd.to_datetime(deb["due_date"])
    overdue = deb[(deb["status"] == "Unpaid") & (deb["due_date"] < TODAY)]
    if len(overdue):
        for _, r in overdue.iterrows():
            days_late = (TODAY - r["due_date"]).days
            st.warning(f"**{r['customer']}** owes **${r['amount']:,.0f}** ({r['invoice']}) — "
                       f"**{days_late} days late**.")
    else:
        st.success("No overdue debtors right now.")

    # Tax/VAT alert
    st.subheader("🧾 Tax / VAT due alert")
    vat_col1, vat_col2 = st.columns(2)
    with vat_col1:
        vat_rate = st.number_input("VAT rate (%)", value=15.0, step=0.5)
    with vat_col2:
        vat_due_day = st.number_input("VAT due day of month", value=25, min_value=1, max_value=28, step=1)
    sales_this_period = txns[(txns["type"] == "Inflow") & (txns["category"] == "Sales Revenue") &
                              (txns["date"] >= TODAY.replace(day=1))]["amount"].sum()
    vat_liability = sales_this_period * vat_rate / 100
    days_to_due = (TODAY.replace(day=min(vat_due_day, 28)) - TODAY).days
    if days_to_due < 0:
        st.info(f"VAT due day ({vat_due_day}) has passed this month — set aside for next period.")
    else:
        st.warning(f"Estimated VAT liability this period: **${vat_liability:,.0f}** "
                   f"— due in **{days_to_due} days**. Set this amount aside now.")

# ========================================================================================
# TAB 4: EXPLAIN
# ========================================================================================
with tab_explain:
    st.header("🧠 EXPLAIN — diagnosis")

    st.subheader("💧 Cash leak finder")
    out_last30 = txns[(txns["type"] == "Outflow") & (txns["date"] >= TODAY - timedelta(days=30))]
    if len(out_last30):
        by_cat = out_last30.groupby("category")["amount"].sum().sort_values(ascending=False)
        top_cat = by_cat.index[0]
        top_pct = by_cat.iloc[0] / by_cat.sum() * 100
        st.write(f"**{top_pct:.0f}%** of cash-out in the last 30 days went to **{top_cat}** "
                 f"(${by_cat.iloc[0]:,.0f} of ${by_cat.sum():,.0f}).")
        fig_leak = go.Figure(go.Pie(labels=by_cat.index, values=by_cat.values, hole=0.4))
        fig_leak.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_leak, use_container_width=True)
        stock_out = by_cat.get("Stock/Inventory", 0)
        sales_in = txns[(txns["type"] == "Inflow") & (txns["date"] >= TODAY - timedelta(days=30))]["amount"].sum()
        if stock_out > sales_in * 0.5:
            st.info(f"Stock purchases (${stock_out:,.0f}) are large relative to sales collected "
                    f"(${sales_in:,.0f}) — cash may be tied up in inventory that hasn't sold through yet.")
    else:
        st.info("Not enough recent outflow data.")

    st.subheader("🏆 Best / worst customers")
    deb2 = st.session_state.debtors.copy()
    deb2["invoice_date"] = pd.to_datetime(deb2["invoice_date"])
    deb2["due_date"] = pd.to_datetime(deb2["due_date"])
    deb2["days_to_pay_or_overdue"] = (TODAY - deb2["due_date"]).dt.days
    deb2["on_time"] = np.where(deb2["status"] == "Paid", "Paid", np.where(deb2["due_date"] < TODAY, "Late", "Not yet due"))
    st.dataframe(deb2[["customer", "amount", "due_date", "status", "on_time"]]
                 .sort_values("amount", ascending=False), use_container_width=True)
    revenue_by_cust = deb2.groupby("customer")["amount"].sum().sort_values(ascending=False)
    if len(revenue_by_cust):
        st.write(f"Most profitable customer by invoiced value: **{revenue_by_cust.index[0]}** "
                 f"(${revenue_by_cust.iloc[0]:,.0f}).")

    st.subheader("🔄 Cash Conversion Cycle (CCC)")
    st.caption("CCC = Days Inventory Outstanding + Days Sales Outstanding − Days Payable Outstanding")
    dio = st.number_input("Days Inventory Outstanding (avg days stock sits before selling)", value=30, step=1)
    dso = int((pd.to_datetime(st.session_state.debtors["due_date"]) -
               pd.to_datetime(st.session_state.debtors["invoice_date"])).dt.days.mean()) if len(st.session_state.debtors) else 30
    cred = st.session_state.creditors.copy()
    dpo = st.number_input("Days Payable Outstanding (avg days you take to pay suppliers)", value=20, step=1)
    ccc = dio + dso - dpo
    st.metric("Cash Conversion Cycle", f"{ccc} days",
              help="Lower is better — it's how long cash is tied up between paying for stock and collecting from customers.")

# ========================================================================================
# TAB 5: FIX
# ========================================================================================
with tab_fix:
    st.header("🛠️ FIX — prescriptive actions")

    st.subheader("💬 Smart collections — auto-drafted reminders")
    deb3 = st.session_state.debtors.copy()
    deb3["due_date"] = pd.to_datetime(deb3["due_date"])
    unpaid = deb3[deb3["status"] == "Unpaid"].sort_values("due_date")
    if len(unpaid):
        for _, r in unpaid.iterrows():
            late = (TODAY - r["due_date"]).days
            tone = "firm" if late > 14 else ("friendly nudge" if late > 0 else "friendly reminder ahead of due date")
            msg = (
                f"Hi {r['customer']}, this is a {'friendly' if late <= 0 else 'quick'} reminder that invoice "
                f"{r['invoice']} for ${r['amount']:,.2f} "
                f"{'was due on ' + r['due_date'].strftime('%d %b %Y') + ' and is now ' + str(late) + ' days overdue' if late > 0 else 'is due on ' + r['due_date'].strftime('%d %b %Y')}. "
                f"Kindly arrange payment at your earliest convenience. Thank you for your business!"
            )
            with st.expander(f"✉️ {r['customer']} — ${r['amount']:,.0f} ({tone})"):
                st.text_area("WhatsApp / Email draft", msg, key=f"draft_{r['invoice']}", height=100)
                st.caption("Copy this into WhatsApp/Email — wire up the WhatsApp Business API or an "
                           "email provider (e.g. SendGrid) to send it automatically.")
    else:
        st.success("No unpaid invoices needing a reminder.")

    st.subheader("📅 Payment scheduler")
    st.caption("Greedy scheduler: captures early-payment discounts where affordable, "
               "delays non-discounted bills to their due date to protect runway.")
    cred2 = st.session_state.creditors.copy()
    cred2["due_date"] = pd.to_datetime(cred2["due_date"])
    cred2 = cred2[cred2["status"] == "Unpaid"].sort_values("early_discount_pct", ascending=False)

    running_balance = st.session_state.start_balance
    schedule = []
    for _, r in cred2.iterrows():
        discount = r["amount"] * r["early_discount_pct"] / 100
        pay_now_amount = r["amount"] - discount
        if r["early_discount_pct"] > 0 and running_balance - pay_now_amount >= 200:  # keep $200 buffer
            schedule.append({"supplier": r["supplier"], "amount": r["amount"], "action": "Pay NOW",
                              "reason": f"captures {r['early_discount_pct']:.0f}% discount = saves ${discount:,.0f}",
                              "pay_amount": pay_now_amount})
            running_balance -= pay_now_amount
        else:
            schedule.append({"supplier": r["supplier"], "amount": r["amount"],
                              "action": f"Pay on due date ({r['due_date'].strftime('%d %b')})",
                              "reason": "no discount / protects short-term cash buffer",
                              "pay_amount": r["amount"]})
    sched_df = pd.DataFrame(schedule)
    if len(sched_df):
        st.dataframe(sched_df.style.format({"amount": "${:.2f}", "pay_amount": "${:.2f}"}),
                     use_container_width=True)
        saved = sum(r["amount"] * r["early_discount_pct"] / 100 for _, r in cred2.iterrows()
                    if r["early_discount_pct"] > 0)
        st.success(f"Following this schedule could save up to **${saved:,.2f}** in early-payment discounts "
                   f"while keeping a cash buffer.")
    else:
        st.info("No unpaid bills to schedule.")

st.divider()
st.caption("Built as a working demo. Replace `seed_transactions()`/CSV upload with a real bank/EcoCash API "
           "connector, and the categorization, forecasting, alerts, explanations and fixes all keep working "
           "unchanged — everything downstream reads from one `transactions` table.")
