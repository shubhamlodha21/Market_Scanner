"""Core logic for the US + India breakout (swing) scanner (no Streamlit code in this file)."""
from __future__ import annotations

import datetime as dt
import io
import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd
import xml.etree.ElementTree as ET

import requests
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; sp500-scanner/1.0)"}
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picks_log.csv")
DATE_FMT = "%d-%m-%Y"  # DD-MM-YYYY everywhere
MIN_BARS = 221  # enough history for a 200-day average

NSE_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
]
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,application/xml,*/*",
    "Referer": "https://www.nseindia.com/",
}

# One entry per market shown in the app's selector
MARKETS = {
    "US — S&P 500": {
        "code": "US", "suffix": "", "benchmark": "SPY", "bench_name": "SPY", "currency": "$",
        "flag": "🇺🇸", "header": "US STOCK TO WATCH", "tz": "America/New_York", "close": (16, 15),
    },
    "India — Nifty 500": {
        "code": "IN", "suffix": ".NS", "benchmark": "^NSEI", "bench_name": "Nifty 50", "currency": "₹",
        "flag": "🇮🇳", "header": "INDIAN STOCK TO WATCH", "tz": "Asia/Kolkata", "close": (15, 45),
    },
}


@dataclass
class ScanParams:
    pivot_lookback: int = 50       # pivot = highest high of the prior N days
    breakout_window: int = 3       # breakout must have happened in the last N bars
    min_rel_volume: float = 1.5    # breakout-day volume / prior 20-day average
    min_move_20d: float = 10.0     # % change over 20 trading days
    max_extension: float = 8.0     # max % above the pivot (avoid late entries)
    require_above_mas: bool = True  # price above 20/50/200-day averages


# ----------------------------------------------------------------- universe
def get_sp500() -> pd.DataFrame:
    """Current S&P 500 constituents from Wikipedia."""
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]
    out = table[["Symbol", "Security", "GICS Sector"]].copy()
    out.columns = ["Ticker", "Company", "Sector"]
    # Yahoo uses BRK-B, not BRK.B
    out["Ticker"] = out["Ticker"].str.strip().str.replace(".", "-", regex=False)
    return out


def get_nifty500() -> pd.DataFrame:
    """Current Nifty 500 constituents from the NSE index file (Yahoo tickers end in .NS)."""
    last_error = None
    for url in NSE_URLS:
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
            resp.raise_for_status()
            table = pd.read_csv(io.StringIO(resp.text))
            table.columns = [c.strip() for c in table.columns]
            out = table[["Symbol", "Company Name", "Industry"]].copy()
            out.columns = ["Ticker", "Company", "Sector"]
            out["Ticker"] = out["Ticker"].astype(str).str.strip() + ".NS"
            return out
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"NSE index file not reachable: {last_error}")


def get_universe(market_code: str) -> pd.DataFrame:
    return get_nifty500() if market_code == "IN" else get_sp500()


# ------------------------------------------------------------------- prices
def download_prices(tickers: list[str], period: str = "2y", batch: int = 100) -> dict[str, pd.DataFrame]:
    """Daily split/dividend-adjusted OHLCV per ticker."""
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        raw = yf.download(chunk, period=period, interval="1d", auto_adjust=True,
                          group_by="ticker", threads=True, progress=False)
        if raw is None or raw.empty:
            continue
        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(subset=["Close"])
            except KeyError:
                continue
            if len(df):
                out[t] = df
    return out


def drop_partial_bar(df: pd.DataFrame, tz: str = "America/New_York", close: tuple = (16, 15)) -> pd.DataFrame:
    """Remove today's bar while that market's session is still running (volume is incomplete)."""
    now = dt.datetime.now(ZoneInfo(tz))
    if len(df) and df.index[-1].date() == now.date() and now.time() < dt.time(*close):
        return df.iloc[:-1]
    return df


# ------------------------------------------------------------------ filters
def analyze(df: pd.DataFrame, p: ScanParams) -> dict | None:
    """Return the setup details if the stock passes every filter, else None."""
    if df is None or len(df) < MIN_BARS:
        return None
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    v = df["Volume"].astype(float)

    pivot = h.rolling(p.pivot_lookback).max().shift(1)   # excludes the current day
    rvol = v / v.rolling(20).mean().shift(1)
    above = (c > pivot).astype(bool)
    fresh = above & ~above.shift(1, fill_value=False).astype(bool)  # first close above pivot

    recent = fresh.iloc[-p.breakout_window:]
    if not recent.any():
        return None
    bday = recent[recent].index[-1]
    level = float(pivot.loc[bday])
    price = float(c.iloc[-1])
    if price <= level:  # fell back under the pivot
        return None

    bo_rvol = float(rvol.loc[bday])
    move20 = (price / float(c.iloc[-21]) - 1) * 100
    ext = (price / level - 1) * 100
    sma = {n: float(c.rolling(n).mean().iloc[-1]) for n in (20, 50, 200)}
    above_ma = {n: price > sma[n] for n in sma}

    if pd.isna(bo_rvol) or bo_rvol < p.min_rel_volume:
        return None
    if move20 < p.min_move_20d or ext > p.max_extension:
        return None
    if p.require_above_mas and not all(above_ma.values()):
        return None

    return {
        "Price": price, "Pivot": level, "Breakout Date": pd.Timestamp(bday),
        "Breakout RVOL": bo_rvol, "RVOL": float(rvol.iloc[-1]),
        "Move 20D %": move20, "Above Pivot %": ext,
        "SMA20": sma[20], "SMA50": sma[50], "SMA200": sma[200],
        "Above 20": above_ma[20], "Above 50": above_ma[50], "Above 200": above_ma[200],
    }


def scan(prices: dict[str, pd.DataFrame], p: ScanParams) -> pd.DataFrame:
    rows = []
    for t, df in prices.items():
        r = analyze(df, p)
        if r:
            rows.append({"Ticker": t, **r})
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    # Rank: half volume strength, half 20-day momentum (0-100)
    res["Score"] = (res["Breakout RVOL"].rank(pct=True) + res["Move 20D %"].rank(pct=True)) * 50
    return res.sort_values("Score", ascending=False).reset_index(drop=True)


def market_regime(spy: pd.DataFrame | None) -> dict | None:
    if spy is None or len(spy) < 200:
        return None
    c = spy["Close"].astype(float)
    close, s50, s200 = float(c.iloc[-1]), float(c.rolling(50).mean().iloc[-1]), float(c.rolling(200).mean().iloc[-1])
    return {"close": close, "sma50": s50, "sma200": s200, "ok": close > s200}


# ------------------------------------------------- fundamentals and earnings
def _money(x: float, currency: str = "$") -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if currency == "₹":
        return f"{sign}₹{x / 1e7:,.0f} Cr"
    if x >= 1e9:
        return f"{sign}${x / 1e9:.1f}B"
    return f"{sign}${x / 1e6:.0f}M"


def fundamentals_line(info: dict, currency: str = "$") -> str:
    parts = []
    if info.get("revenueGrowth") is not None:
        parts.append(f"revenue growth {info['revenueGrowth'] * 100:+.1f}% YoY")
    if info.get("earningsGrowth") is not None:
        parts.append(f"earnings growth {info['earningsGrowth'] * 100:+.1f}% YoY")
    if info.get("freeCashflow") is not None:
        parts.append(f"free cash flow {_money(info['freeCashflow'], currency)}")
    elif info.get("operatingCashflow") is not None:
        parts.append(f"operating cash flow {_money(info['operatingCashflow'], currency)}")
    if info.get("profitMargins") is not None:
        parts.append(f"profit margin {info['profitMargins'] * 100:.1f}%")
    if not parts:
        return "Data not available."
    text = ", ".join(parts) + "."
    return text[0].upper() + text[1:]


def get_company_data(ticker: str, currency: str = "$") -> dict:
    """Name, country, one-line fundamentals and next earnings date (Yahoo Finance)."""
    out = {"name": ticker, "country": "", "fundamentals": "Data not available.", "earnings_date": None}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception:
        return out
    out["name"] = info.get("longName") or info.get("shortName") or ticker
    out["country"] = info.get("country") or ""
    out["fundamentals"] = fundamentals_line(info, currency)
    try:
        cal = tk.calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        today = dt.date.today()
        upcoming = sorted(d for d in (pd.Timestamp(x).date() for x in (dates or [])) if d >= today)
        out["earnings_date"] = upcoming[0] if upcoming else None
    except Exception:
        pass
    return out


# --------------------------------------------------------------------- news
def finnhub_news(ticker: str, api_key: str, days: int, limit: int) -> list[dict]:
    today = dt.date.today()
    resp = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={"symbol": ticker.replace("-", "."), "from": str(today - dt.timedelta(days=days)),
                "to": str(today), "token": api_key},
        timeout=20,
    )
    resp.raise_for_status()
    items = sorted(resp.json(), key=lambda x: x.get("datetime", 0), reverse=True)
    return [{"headline": it.get("headline", "").strip(), "source": it.get("source", ""),
             "date": dt.datetime.fromtimestamp(it["datetime"], dt.timezone.utc).date(),
             "url": it.get("url", "")}
            for it in items[:limit] if it.get("headline") and it.get("datetime")]


def yahoo_news(ticker: str, days: int, limit: int) -> list[dict]:
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    out = []
    for it in yf.Ticker(ticker).news or []:
        c = it.get("content") or it  # newer yfinance nests everything under "content"
        title = c.get("title")
        pub = c.get("pubDate") or c.get("providerPublishTime")
        if not title or pub is None:
            continue
        when = pd.to_datetime(pub, unit="s", utc=True) if isinstance(pub, (int, float)) else pd.to_datetime(pub, utc=True)
        if when < cutoff:
            continue
        out.append({"headline": title.strip(),
                    "source": (c.get("provider") or {}).get("displayName") or c.get("publisher") or "",
                    "date": when.date(),
                    "url": (c.get("canonicalUrl") or {}).get("url") or c.get("link") or ""})
    return sorted(out, key=lambda x: x["date"], reverse=True)[:limit]


def google_news(company: str, days: int, limit: int) -> list[dict]:
    """Google News RSS (India edition) - used for Indian stocks, where Finnhub has no free coverage."""
    resp = requests.get("https://news.google.com/rss/search",
                        params={"q": f'"{company}" stock when:{days}d', "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
                        headers={"User-Agent": BROWSER_HEADERS["User-Agent"]}, timeout=20)
    resp.raise_for_status()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    out = []
    for item in ET.fromstring(resp.content).iter("item"):
        title, pub = item.findtext("title"), item.findtext("pubDate")
        if not title or not pub:
            continue
        when = pd.to_datetime(pub, utc=True)
        if when < cutoff:
            continue
        source = item.findtext("source") or ""
        if source and title.endswith(f" - {source}"):  # Google appends the publisher to the title
            title = title[: -len(source) - 3]
        out.append({"headline": title.strip(), "source": source, "date": when.date(),
                    "url": item.findtext("link") or ""})
    return sorted(out, key=lambda x: x["date"], reverse=True)[:limit]


def get_news(ticker: str, finnhub_key: str = "", days: int = 3, limit: int = 2,
             market_code: str = "US", company: str = "") -> list[dict]:
    if market_code == "IN":
        try:
            items = google_news(company or ticker.replace(".NS", ""), days, limit)
            if items:
                return items
        except Exception:
            pass
    elif finnhub_key:
        try:
            items = finnhub_news(ticker, finnhub_key, days, limit)
            if items:
                return items
        except Exception:
            pass
    try:
        return yahoo_news(ticker, days, limit)
    except Exception:
        return []


# ------------------------------------------------------------------ message
def format_message(row, company: dict, news: list[dict], pivot_lookback: int, earnings_warn_days: int = 7,
                   market: dict | None = None) -> str:
    market = market or MARKETS["US — S&P 500"]
    cur = market["currency"]
    symbol = row["Ticker"].removesuffix(market["suffix"]) if market["suffix"] else row["Ticker"]
    above = [f"{n}-day" for n in (20, 50, 200) if row[f"Above {n}"]]
    below = [f"{n}-day" for n in (20, 50, 200) if not row[f"Above {n}"]]
    if not below:
        ma_text = "trading above 20-day, 50-day & 200-day moving averages"
    else:
        ma_text = (f"above {', '.join(above)} MA; " if above else "") + f"below {', '.join(below)} MA"

    if news:
        news_text = "".join(
            f"\n   – {n['headline']} ({n['source'] + ', ' if n['source'] else ''}{n['date'].strftime(DATE_FMT)})"
            for n in news)
    else:
        news_text = " No major news in the last few days."

    watch = (f"Sustained momentum above the {cur}{row['Pivot']:,.2f} breakout zone, "
             f"supported by volume confirmation.")
    ed = company.get("earnings_date")
    if ed is not None:
        days_to = (ed - dt.date.today()).days
        if 0 <= days_to <= earnings_warn_days:
            watch += f" ⚠️ Earnings on {ed.strftime(DATE_FMT)} (in {days_to} day{'s' if days_to != 1 else ''})."

    return (
        f"📊 {market['header']}:\n\n"
        f"{symbol} — {company.get('name', symbol)} {market['flag']}\n\n"
        f"• Fundamentals: {company.get('fundamentals', 'Data not available.')}\n\n"
        f"• Technical Setup: Breakout above {cur}{row['Pivot']:,.2f} pivot ({pivot_lookback}-day high) on "
        f"{row['Breakout Date'].strftime(DATE_FMT)} with {row['Breakout RVOL']:.2f}x volume.\n\n"
        f"• Key Metrics: Current Price: {cur}{row['Price']:,.2f} | Relative Volume: {row['RVOL']:.2f}x | "
        f"{row['Above Pivot %']:.1f}% above pivot.\n\n"
        f"• Momentum: 20-day move {row['Move 20D %']:+.1f}%; {ma_text}.\n\n"
        f"• News:{news_text}\n\n"
        f"• Watch For: {watch}"
    )


def send_telegram(token: str, chat_id: str, text: str) -> tuple[bool, str]:
    try:
        resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                             json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
                             timeout=20)
        data = resp.json()
        return bool(data.get("ok")), data.get("description", "")
    except Exception as exc:  # network error, bad token format, etc.
        return False, str(exc)


# ----------------------------------------------------------------- pick log
LOG_COLS = ["Alert Date", "Market", "Ticker", "Company", "Alert Price", "Pivot", "Breakout RVOL", "Move 20D %"]


def load_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=LOG_COLS)
    log = pd.read_csv(LOG_PATH)
    log["Alert Date"] = pd.to_datetime(log["Alert Date"], format=DATE_FMT)
    if "Market" not in log.columns:  # logs written before India was added
        log.insert(1, "Market", "US")
    return log


def append_log(picks: pd.DataFrame, alert_date: pd.Timestamp, market_code: str = "US") -> None:
    if picks.empty:
        return
    new = pd.DataFrame({
        "Alert Date": pd.Timestamp(alert_date).normalize(),
        "Market": market_code,
        "Ticker": picks["Ticker"].values,
        "Company": picks["Company"].values,
        "Alert Price": picks["Price"].round(2).values,
        "Pivot": picks["Pivot"].round(2).values,
        "Breakout RVOL": picks["Breakout RVOL"].round(2).values,
        "Move 20D %": picks["Move 20D %"].round(1).values,
    })
    log = load_log()
    log = new if log.empty else pd.concat([log, new], ignore_index=True)
    log = log.drop_duplicates(["Alert Date", "Ticker"], keep="first")
    log.to_csv(LOG_PATH, index=False, date_format=DATE_FMT)


def recently_alerted(log: pd.DataFrame, alert_date: pd.Timestamp, days: int = 10) -> set[str]:
    """Tickers alerted on an earlier day within the last N days (same-day re-runs are kept)."""
    if log.empty:
        return set()
    day = pd.Timestamp(alert_date).normalize()
    mask = (log["Alert Date"] < day) & (log["Alert Date"] >= day - pd.Timedelta(days=days))
    return set(log.loc[mask, "Ticker"])
