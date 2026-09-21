# SME Cashflow Agent 💰

A working cashflow agent for SMEs (built with the Zimbabwean market in mind — bank
feed + EcoCash style inflows) that does five jobs, not just charts:

| Job | What it does |
|---|---|
| 👁️ **SEE** | Ingests transactions, AI-auto-categorizes them (learns from your corrections), simulates invoice/receipt scanning, tracks a debtors and creditors ledger |
| 🔮 **PREDICT** | 90-day daily cash forecast, auto-detects recurring items (rent, salaries) and month-end seasonality, and lets you run "what-if" scenarios (late-paying client, stock spend change, FX shock) |
| 🚨 **WARN** | Runway alert, spending-anomaly alert, late-payer alert, VAT/tax set-aside reminder — plus a live **red/green** operating alarm |
| 🧠 **EXPLAIN** | Cash leak finder, best/worst customers, Cash Conversion Cycle |
| 🛠️ **FIX** | Auto-drafts WhatsApp/email collection reminders for late payers, and a bill payment scheduler that captures early-payment discounts without breaking your cash buffer |

The header of the app always shows a **live graph** of daily inflows (green) vs
outflows (red) plus the projected cash balance line, and flags whether the
business is operating in the red or green. A sidebar form lets you type in new
inflow/outflow transactions and move scenario sliders — the graph updates
immediately.

## Run it locally

```bash
git clone https://github.com/<your-username>/sme-cashflow-agent.git
cd sme-cashflow-agent
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploy a live public link (free, ~2 minutes)

1. Create a new GitHub repo and push these three files (`app.py`,
   `requirements.txt`, `README.md`) to it.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, set the main file path to
   `app.py`, and click **Deploy**.
4. Streamlit Cloud installs `requirements.txt` and gives you a public URL like
   `https://<your-app-name>.streamlit.app` — that's your live demo link.

## Wiring in real data sources (next step)

Everything in the app reads from one `transactions` table in
`st.session_state`. To go from demo to production:

- Replace `seed_transactions()` / the CSV uploader with a real Open Banking
  API client or an EcoCash merchant API client that appends rows to that same
  table — nothing downstream (categorization, forecasting, alerts,
  explanations, fixes) needs to change.
- Replace the "paste invoice text" box with a real OCR/vision API call
  (e.g. Google Cloud Vision, AWS Textract, or an LLM vision call) that fills
  the same amount/due-date extraction.
- Wire the "Smart collections" drafts to the WhatsApp Business API or an
  email provider (e.g. SendGrid) to actually send them, instead of just
  drafting them for copy/paste.

## Tech stack

- [Streamlit](https://streamlit.io) — UI
- [Plotly](https://plotly.com/python/) — the live inflow/outflow/balance graph
- pandas / numpy — categorization, recurring-item detection, forecasting logic
