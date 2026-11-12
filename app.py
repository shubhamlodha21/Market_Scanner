"""Breakout (swing) scanner for US and Indian stocks - Streamlit app.

Run with:  streamlit run app.py
"""
import os

import pandas as pd
import streamlit as st

import scanner as sc

try:
    import config
except ImportError:  # config.py is optional
    config = None


def setting(name: str) -> str:
    return (getattr(config, name, "") if config else "") or os.environ.get(name, "")


FINNHUB_KEY = setting("FINNHUB_API_KEY")

st.set_page_config(page_title="Breakout Scanner", page_icon="📊", layout="wide")
st.title("📊 Breakout Scanner")
st.caption("Daily swing 'stock to watch' ideas from price, volume, fundamentals and news. "
           "A watchlist tool, not investment advice.")

market_name = st.radio("Market", list(sc.MARKETS), horizontal=True)
market = sc.MARKETS[market_name]

# ------------------------------------------- sidebar (everything collapsed)
with st.sidebar:
    with st.expander("⚙️ Filter settings", expanded=False):
        pivot_lookback = st.slider("Pivot lookback (days)", 20, 100, 50, 5,
                                   help="Pivot = highest high of the previous N trading days.")
        breakout_window = st.slider("Breakout within last N days", 1, 5, 3)
        min_rvol = st.slider("Min. relative volume on breakout day", 1.0, 5.0, 1.5, 0.1,
                             help="Breakout-day volume divided by the prior 20-day average.")
        min_move = st.slider("Min. 20-day move (%)", 0.0, 30.0, 10.0, 0.5)
        max_ext = st.slider("Max. distance above pivot (%)", 1.0, 20.0, 8.0, 0.5,
                            help="Skips stocks that have already run too far from the breakout level.")
        require_mas = st.checkbox("Price above 20/50/200-day MAs", True)
        market_filter = st.checkbox("Only when the index is above its 200-day MA", True,
                                    help="SPY for the US, Nifty 50 for India.")
        skip_repeats = st.checkbox("Skip stocks alerted in the last 10 days", True)

    with st.expander("📝 Output settings", expanded=False):
        top_n = st.slider("Number of picks", 1, 10, 5)
        earn_days = st.slider("Earnings warning (days ahead)", 1, 14, 7)
        news_days = st.slider("News from last N days", 1, 7, 3)

    with st.expander("📨 Telegram", expanded=False):
        tg_token = st.text_input("Bot token", type="password", value=setting("TELEGRAM_BOT_TOKEN"))
        tg_chat = st.text_input("Chat ID", value=setting("TELEGRAM_CHAT_ID"))

    with st.expander("📂 Custom ticker list", expanded=False):
        upload = st.file_uploader("Ticker CSV (optional)", type="csv",
                                  help="One column named Ticker. Leave empty to scan the full index.")


# ------------------------------------------------------------ cached loaders
@st.cache_data(ttl=86400, show_spinner=False)
def load_universe(market_code: str) -> pd.DataFrame:
    return sc.get_universe(market_code)


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers: tuple) -> dict:
    return sc.download_prices(list(tickers))


@st.cache_data(ttl=3600, show_spinner=False)
def load_company(ticker: str, currency: str) -> dict:
    return sc.get_company_data(ticker, currency)


@st.cache_data(ttl=900, show_spinner=False)
def load_news(ticker: str, key: str, days: int, market_code: str, company: str) -> list:
    return sc.get_news(ticker, key, days, market_code=market_code, company=company)


def read_custom_universe(file, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    col = next((c for c in df.columns if c.strip().lower() in ("ticker", "symbol")), df.columns[0])
    names = df[col].dropna().astype(str).str.strip().str.upper()
    names = names[names != ""].drop_duplicates()
    if suffix:  # Indian tickers: add .NS when it is missing
        tickers = names.where(names.str.endswith(suffix), names + suffix)
        names = tickers.str.removesuffix(suffix)
    else:
        tickers = names.str.replace(".", "-", regex=False)
    return pd.DataFrame({"Ticker": tickers.values, "Company": names.values, "Sector": ""})


# ----------------------------------------------------------------- run scan
if st.button("🔍 Run scan", type="primary"):
    if upload is not None:
        universe = read_custom_universe(upload, market["suffix"])
    else:
        try:
            universe = load_universe(market["code"])
        except Exception as exc:
            st.error(f"Could not load the {market_name} stock list ({exc}). "
                     "Upload a ticker CSV under 'Custom ticker list' in the sidebar instead.")
            st.stop()

    bench = market["benchmark"]
    tickers = tuple(dict.fromkeys(list(universe["Ticker"]) + [bench]))
    with st.spinner(f"Downloading daily prices for {len(tickers) - 1} stocks (first run takes 1–2 minutes)…"):
        prices = load_prices(tickers)
    if not prices:
        st.error("No price data came back from Yahoo Finance. Check the internet connection and try again.")
        st.stop()

    prices = {t: sc.drop_partial_bar(df, market["tz"], market["close"]) for t, df in prices.items()}
    regime = sc.market_regime(prices.get(bench))
    wanted = set(universe["Ticker"])
    stocks = {t: df for t, df in prices.items() if t in wanted and len(df)}
    if not stocks:
        st.error("None of the tickers returned price data.")
        st.stop()
    data_date = max(df.index[-1] for df in stocks.values())

    params = sc.ScanParams(pivot_lookback, breakout_window, min_rvol, min_move, max_ext, require_mas)
    results = sc.scan(stocks, params)
    if not results.empty:
        results = results.merge(universe, on="Ticker", how="left")
        if skip_repeats:
            results = results[~results["Ticker"].isin(sc.recently_alerted(sc.load_log(), data_date))]
        results = results.reset_index(drop=True)

    blocked = bool(market_filter and regime and not regime["ok"])
    picks = []
    if not blocked and not results.empty:
        top = results.head(top_n)
        with st.spinner("Fetching fundamentals and news for the top picks…"):
            for _, row in top.iterrows():
                listed_name = row["Company"] if isinstance(row.get("Company"), str) else row["Ticker"]
                company = load_company(row["Ticker"], market["currency"])
                if company["name"] == row["Ticker"]:
                    company = {**company, "name": listed_name}
                news = load_news(row["Ticker"], FINNHUB_KEY, news_days, market["code"], company["name"])
                picks.append({
                    "ticker": row["Ticker"], "symbol": row["Ticker"].removesuffix(market["suffix"] or "\0"),
                    "name": company["name"],
                    "message": sc.format_message(row, company, news, pivot_lookback, earn_days, market),
                    "links": [n for n in news if n["url"]],
                    "pivot": row["Pivot"],
                })
        sc.append_log(top, data_date, market["code"])

    st.session_state["result"] = {
        "market_name": market_name, "market": market,
        "regime": regime, "blocked": blocked, "results": results, "picks": picks,
        "data_date": data_date, "scanned": len(stocks),
        "charts": {p["ticker"]: stocks[p["ticker"]]["Close"].tail(180) for p in picks},
        "last_close": {t: float(df["Close"].iloc[-1]) for t, df in stocks.items()},
    }

# ------------------------------------------------------------------ results
res = st.session_state.get("result")
if res is None:
    st.info("Choose a market and press **Run scan**. Run it after that market's close for final volume numbers.")
    st.stop()

rm = res["market"]
cur = rm["currency"]
regime = res["regime"]
st.markdown(f"**Showing:** {res['market_name']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Data date", res["data_date"].strftime(sc.DATE_FMT))
c2.metric("Stocks scanned", res["scanned"])
c3.metric("Matches", len(res["results"]))
if regime:
    c4.metric(f"{rm['bench_name']} vs 200-day MA", f"{regime['close']:,.2f}",
              f"{(regime['close'] / regime['sma200'] - 1) * 100:+.1f}%")

tab_picks, tab_all, tab_log = st.tabs(["Picks", "All matches", "Pick log"])

with tab_picks:
    if res["blocked"]:
        st.warning(f"{rm['bench_name']} is below its 200-day moving average, so no picks are shown. "
                   "Untick the index filter under 'Filter settings' to see them anyway.")
    elif not res["picks"]:
        st.info("No stock passed all filters today. Loosen the filters under 'Filter settings' and run again.")
    else:
        all_text = "\n\n──────────\n\n".join(p["message"] for p in res["picks"])
        b1, b2 = st.columns(2)
        b1.download_button("⬇️ Download all picks (.txt)", all_text,
                           file_name=f"stocks_to_watch_{rm['code']}_{res['data_date'].strftime(sc.DATE_FMT)}.txt")
        if b2.button("📨 Send picks to Telegram", disabled=not (tg_token and tg_chat)):
            failures = []
            for p in res["picks"]:
                ok, err = sc.send_telegram(tg_token, tg_chat, p["message"])
                if not ok:
                    failures.append(f"{p['symbol']}: {err}")
            if failures:
                st.error("Telegram failed for " + "; ".join(failures))
            else:
                st.success(f"Sent {len(res['picks'])} picks to Telegram.")

        for p in res["picks"]:
            st.subheader(f"{p['symbol']} — {p['name']}")
            st.code(p["message"], language=None, wrap_lines=True)
            for n in p["links"]:
                st.markdown(f"[{n['headline']}]({n['url']})")
            with st.expander("Price chart (last 180 days)"):
                chart = res["charts"][p["ticker"]].to_frame("Close")
                chart["Pivot"] = p["pivot"]
                st.line_chart(chart)

money = st.column_config.NumberColumn(format=f"{cur}%.2f")

with tab_all:
    if res["results"].empty:
        st.info("No matches.")
    else:
        cols = ["Ticker", "Company", "Sector", "Score", "Price", "Pivot", "Above Pivot %",
                "Breakout Date", "Breakout RVOL", "RVOL", "Move 20D %", "SMA20", "SMA50", "SMA200"]
        table = res["results"][[c for c in cols if c in res["results"].columns]]
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={
                "Breakout Date": st.column_config.DateColumn(format="DD-MM-YYYY"),
                "Score": st.column_config.NumberColumn(format="%.0f"),
                "Price": money, "Pivot": money, "SMA20": money, "SMA50": money, "SMA200": money,
                "Above Pivot %": st.column_config.NumberColumn(format="%.1f%%"),
                "Breakout RVOL": st.column_config.NumberColumn(format="%.2fx"),
                "RVOL": st.column_config.NumberColumn(format="%.2fx"),
                "Move 20D %": st.column_config.NumberColumn(format="%+.1f%%"),
            })
        st.download_button("⬇️ Download matches (.csv)",
                           table.to_csv(index=False, date_format=sc.DATE_FMT),
                           file_name=f"matches_{rm['code']}_{res['data_date'].strftime(sc.DATE_FMT)}.csv")

with tab_log:
    log = sc.load_log()
    log = log[log["Market"] == rm["code"]]
    if log.empty:
        st.info("No picks logged for this market yet. Every pick is saved here so you can check how the filters perform.")
    else:
        log = log.drop(columns="Market")
        log["Current Price"] = log["Ticker"].map(res["last_close"])
        log["Return %"] = (log["Current Price"] / log["Alert Price"] - 1) * 100
        log = log.sort_values("Alert Date", ascending=False)
        done = log["Return %"].dropna()
        if len(done):
            m1, m2, m3 = st.columns(3)
            m1.metric("Picks logged", len(log))
            m2.metric("Average return since alert", f"{done.mean():+.1f}%")
            m3.metric("Winners", f"{(done > 0).mean() * 100:.0f}%")
        st.dataframe(
            log, hide_index=True, width="stretch",
            column_config={
                "Alert Date": st.column_config.DateColumn(format="DD-MM-YYYY"),
                "Alert Price": money, "Pivot": money, "Current Price": money,
                "Breakout RVOL": st.column_config.NumberColumn(format="%.2fx"),
                "Move 20D %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Return %": st.column_config.NumberColumn(format="%+.1f%%"),
            })
