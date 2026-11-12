# Market_Scanner

Streamlit app that scans the S&P 500 (US) or Nifty 500 (India) for fresh breakouts and writes a daily "stock to watch" note for each pick: fundamentals, technical setup, key metrics, momentum, news and what to watch for.

## Run

```
pip install -r requirements.txt
cp config.example.py config.py   # then add your Finnhub key (optional)
streamlit run app.py
```

The keys can also be set as environment variables: `FINNHUB_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## How it works

- **Prices:** daily adjusted OHLCV from Yahoo Finance (`yfinance`).
- **Filters:** first close above the 50-day high within the last 3 days, relative volume of at least 1.5x on the breakout day, 20-day move of at least 10%, price above the 20/50/200-day averages, and no more than 8% above the pivot. All adjustable in the sidebar.
- **Market filter:** picks show only when SPY (US) or the Nifty 50 (India) is above its 200-day average.
- **News:** Finnhub or Yahoo for US stocks, Google News for Indian stocks.
- **Pick log:** every pick is saved to `picks_log.csv` with its return since the alert.

Run it after the market close (US: after 4:15 pm ET, India: after 3:45 pm IST) so the day's volume is final.

This is a watchlist tool, not investment advice.
