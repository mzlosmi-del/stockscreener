# Stock Screener API

FastAPI backend for the daily stock picking tool. Uses Yahoo Finance (free, no API key) with `pandas-ta` for precise technical indicator calculation.

## What it calculates (on real OHLCV data)

| Indicator | Params | Used for |
|---|---|---|
| RSI | 14 periods | Overbought/oversold |
| MACD | 12/26/9 | Momentum direction |
| EMA | 20 periods | Trend filter |
| Bollinger Bands | 20 periods, 2σ | Volatility positioning |
| ATR | 14 periods | Stop loss & target sizing |
| Volume ratio | vs 20-day avg | Conviction filter |

**Data**: 15-min delayed intraday via Yahoo Finance. No API key needed.

---

## Deploy to Railway (recommended, free)

1. Push this folder to a GitHub repo
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your repo — Railway auto-detects Python and uses `railway.toml`
4. Once deployed, copy your public URL (e.g. `https://stock-screener-xxx.railway.app`)
5. Paste that URL into the frontend widget where it says `YOUR_API_URL`

Railway free tier: 500 hours/month — enough for daily use.

---

## Deploy to Render (alternative, also free)

1. Push this folder to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → Connect repo
3. Set:
   - **Runtime**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Choose **Free** instance type → Deploy
5. Copy the `.onrender.com` URL into the frontend

Note: Render free tier spins down after 15 min of inactivity (first request takes ~30s to wake up).

---

## Run locally for testing

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open: http://localhost:8000/docs

---

## API Endpoints

### `GET /screen`
Screen the default 25-stock watchlist.

```
GET https://your-api.railway.app/screen
```

Response:
```json
{
  "stocks": [
    {
      "ticker": "NVDA",
      "price": 875.24,
      "change": 2.14,
      "rsi": 58.3,
      "macd_val": 1.24,
      "macd_dir": "rising",
      "score": 8,
      "signal": "strong-buy",
      "stop": 845.10,
      "target": 912.50,
      "risk_reward": 1.5,
      "buy_points": ["Entry near $875.24", "..."],
      "sell_points": ["Stop loss: $845.10 (2× ATR)", "..."],
      "history": [820.1, 831.4, ...],
      "updated_at": "2026-03-15T14:30:00Z",
      "data_note": "15-min delayed via Yahoo Finance"
    }
  ],
  "count": 25,
  "generated_at": "2026-03-15T14:30:00Z"
}
```

### `GET /screen?tickers=AAPL,TSLA,NVDA`
Screen specific tickers (max 50).

### `GET /stock/NVDA`
Single stock full analysis.

---

## Signal Score Logic

Score is 0–10, built from 5 indicators (each contributes up to 2 points):

```
RSI < 30       → +2    RSI 30-45  → +1    RSI > 72 → -1
MACD+ rising   → +2    MACD+ flat → +1    MACD- falling → -1
Price > EMA+3% → +2    Price > EMA → +1
Volume > 2×avg → +2    Volume > 1.5× → +1
BB pos < 15%   → +2    BB pos < 35% → +1  BB pos > 85% → -1

Base offset: +4 (neutral stocks land near 4-5)

Score ≥ 8 → Strong Buy
Score ≥ 6 → Buy
Score ≥ 4 → Hold
Score ≥ 2 → Sell
Score  < 2 → Strong Sell
```

---

## Disclaimer

This tool is for informational and educational purposes only. It is not financial advice. Always do your own research before making investment decisions.
