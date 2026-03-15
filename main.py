"""
Stock Screener API — FastAPI + yfinance + pandas-ta
Calculates RSI, MACD, Bollinger Bands, EMA, ATR, Volume from real OHLCV data.
Data is 15-min delayed intraday via Yahoo Finance (free, no API key needed).
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from datetime import datetime, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback

app = FastAPI(title="Stock Screener API", version="1.0.0")

# Allow all origins so your frontend (Claude artifact or any host) can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=6)

# Default US large-cap watchlist — extend as needed
DEFAULT_TICKERS = [
    "NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
    "JPM", "JNJ", "XOM", "UNH", "BAC", "HD", "PG", "CVX",
    "MRK", "ABBV", "KO", "PFE", "BA", "CAT", "GE", "AMD", "CRM", "NOW",
]

SECTOR_MAP = {
    "NVDA": "Technology", "AAPL": "Technology", "MSFT": "Technology",
    "META": "Technology", "GOOGL": "Technology", "AMZN": "Technology",
    "AMD": "Technology", "CRM": "Technology", "NOW": "Technology",
    "TSLA": "Consumer", "HD": "Consumer", "PG": "Consumer", "KO": "Consumer",
    "JPM": "Financials", "BAC": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "MRK": "Healthcare",
    "ABBV": "Healthcare", "PFE": "Healthcare",
    "XOM": "Energy", "CVX": "Energy",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
}


def compute_signal_score(rsi, macd_val, macd_dir, pct_above_ema, vol_mult, bb_pos, price_change):
    """
    Composite signal score 0-10.
    Each of 5 indicators contributes 0-2 points.
    Score >= 8 = Strong Buy, >= 6 = Buy, >= 4 = Hold, >= 2 = Sell, < 2 = Strong Sell
    """
    score = 0

    # RSI: oversold = bullish, overbought = bearish
    if rsi < 30:
        score += 2
    elif rsi < 45:
        score += 1
    elif rsi > 72:
        score -= 1

    # MACD histogram direction and sign
    if macd_val > 0 and macd_dir == "rising":
        score += 2
    elif macd_val > 0:
        score += 1
    elif macd_val < 0 and macd_dir == "falling":
        score -= 1

    # Price vs EMA20
    if pct_above_ema > 3:
        score += 2
    elif pct_above_ema > 0:
        score += 1

    # Volume surge with positive price (confirms move)
    if vol_mult > 2.0 and price_change > 0:
        score += 2
    elif vol_mult > 1.5:
        score += 1

    # Bollinger Band position (0 = lower band, 1 = upper band)
    if bb_pos < 0.15:
        score += 2
    elif bb_pos < 0.35:
        score += 1
    elif bb_pos > 0.85:
        score -= 1

    score = max(0, min(10, score + 4))  # baseline of 4 (neutral)
    return score


def signal_label(score):
    if score >= 8:
        return "strong-buy"
    elif score >= 6:
        return "buy"
    elif score >= 4:
        return "hold"
    elif score >= 2:
        return "sell"
    return "strong-sell"


def fetch_and_analyze(ticker: str) -> dict:
    """
    Fetch 60 days of daily OHLCV + today's intraday 15-min data for a single ticker.
    Calculates all indicators on real data.
    """
    try:
        tk = yf.Ticker(ticker)

        # 60 days of daily data — enough for EMA20, RSI(14), MACD(26), BB(20), ATR(14)
        hist = tk.history(period="60d", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 30:
            return None

        # Today's intraday 1-day 15m bars for current price
        intra = tk.history(period="1d", interval="15m", auto_adjust=True)
        if not intra.empty:
            current_price = float(intra["Close"].iloc[-1])
            current_vol_today = float(intra["Volume"].sum())
        else:
            current_price = float(hist["Close"].iloc[-1])
            current_vol_today = float(hist["Volume"].iloc[-1])

        close = hist["Close"].astype(float)
        high = hist["High"].astype(float)
        low = hist["Low"].astype(float)
        volume = hist["Volume"].astype(float)

        # ── RSI(14) ──────────────────────────────────────────────────
        rsi_series = ta.rsi(close, length=14)
        rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.isna().all() else 50.0

        # ── MACD(12,26,9) ─────────────────────────────────────────────
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            hist_col = [c for c in macd_df.columns if "MACDh" in c]
            macd_col = [c for c in macd_df.columns if c.startswith("MACD_")]
            macd_val = float(macd_df[hist_col[0]].iloc[-1]) if hist_col else 0.0
            prev_macd_val = float(macd_df[hist_col[0]].iloc[-2]) if hist_col and len(macd_df) > 1 else macd_val
            macd_dir = "rising" if macd_val > prev_macd_val else "falling"
        else:
            macd_val, macd_dir = 0.0, "flat"

        # ── EMA(20) ───────────────────────────────────────────────────
        ema20_series = ta.ema(close, length=20)
        ema20 = float(ema20_series.iloc[-1]) if ema20_series is not None and not ema20_series.isna().all() else current_price
        pct_above_ema = (current_price - ema20) / ema20 * 100

        # ── Bollinger Bands(20, 2) ────────────────────────────────────
        bb_df = ta.bbands(close, length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            lower_col = [c for c in bb_df.columns if "BBL" in c]
            upper_col = [c for c in bb_df.columns if "BBU" in c]
            bb_lower = float(bb_df[lower_col[0]].iloc[-1]) if lower_col else current_price * 0.95
            bb_upper = float(bb_df[upper_col[0]].iloc[-1]) if upper_col else current_price * 1.05
            bb_range = bb_upper - bb_lower
            bb_pos = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
        else:
            bb_lower, bb_upper, bb_pos = current_price * 0.95, current_price * 1.05, 0.5

        # ── ATR(14) ───────────────────────────────────────────────────
        atr_series = ta.atr(high, low, close, length=14)
        atr = float(atr_series.iloc[-1]) if atr_series is not None and not atr_series.isna().all() else current_price * 0.015

        # ── Volume vs 20-day average ──────────────────────────────────
        avg_vol_20 = float(volume.iloc[-21:-1].mean()) if len(volume) > 21 else float(volume.mean())
        # Scale today's intraday volume to full-day equivalent
        vol_mult = (current_vol_today / avg_vol_20) if avg_vol_20 > 0 else 1.0

        # ── 1-day price change ────────────────────────────────────────
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
        price_change = (current_price - prev_close) / prev_close * 100

        # ── 20-day price history for sparkline ───────────────────────
        history_20 = [round(float(p), 2) for p in close.iloc[-20:].tolist()]

        # ── Signal score ──────────────────────────────────────────────
        score = compute_signal_score(rsi, macd_val, macd_dir, pct_above_ema, vol_mult, bb_pos, price_change)
        signal = signal_label(score)

        # ── Buy / sell narratives ─────────────────────────────────────
        buy_points = []
        sell_points = []

        if signal in ("strong-buy", "buy"):
            buy_points.append(f"Entry near current price ${current_price:.2f}")
            if rsi < 45:
                buy_points.append(f"RSI at {rsi:.1f} — oversold conditions support entry")
            if macd_val > 0 and macd_dir == "rising":
                buy_points.append("MACD histogram positive and rising — bullish momentum confirmed")
            if pct_above_ema > 0:
                buy_points.append(f"Price {pct_above_ema:.1f}% above EMA20 (${ema20:.2f}) — uptrend intact")
            if vol_mult > 1.5:
                buy_points.append(f"Volume {vol_mult:.1f}× 20-day average — strong participation")
            if bb_pos < 0.3:
                buy_points.append("Price near lower Bollinger Band — mean-reversion setup")
        else:
            buy_points.append(f"Wait — RSI at {rsi:.1f}, look for drop below 40 before entering")
            buy_points.append(f"Watch for price reclaim above EMA20 (${ema20:.2f})")
            if macd_val < 0:
                buy_points.append("Wait for MACD histogram to cross back into positive territory")

        stop = current_price - atr * 2
        target = current_price + atr * 3
        rr = round(atr * 3 / (atr * 2), 1)

        sell_points.append(f"Stop loss: ${stop:.2f} (2× ATR below entry) — exit immediately if breached")
        sell_points.append(f"Primary target: ${target:.2f} (3× ATR above entry) — take full profit here")
        if rsi > 68:
            sell_points.append(f"RSI at {rsi:.1f} — reduce size or wait for pullback before entry")
        if macd_val > 0 and macd_dir == "falling":
            sell_points.append("MACD flattening — momentum weakening, tighten stop to breakeven")
        if bb_pos > 0.8:
            sell_points.append("Price near upper Bollinger Band — consider partial profit taking (50%)")
        sell_points.append(f"Also exit on daily close below EMA20 (${ema20:.2f}) or gap-down open >2%")

        return {
            "ticker": ticker,
            "name": tk.info.get("shortName", ticker) if hasattr(tk, "info") else ticker,
            "sector": SECTOR_MAP.get(ticker, tk.info.get("sector", "Unknown") if hasattr(tk, "info") else "Unknown"),
            "price": round(current_price, 2),
            "change": round(price_change, 2),
            "rsi": round(rsi, 1),
            "macd_val": round(macd_val, 3),
            "macd_dir": macd_dir,
            "ema20": round(ema20, 2),
            "pct_above_ema": round(pct_above_ema, 2),
            "bb_lower": round(bb_lower, 2),
            "bb_upper": round(bb_upper, 2),
            "bb_pos": round(bb_pos, 3),
            "atr": round(atr, 2),
            "vol_mult": round(vol_mult, 2),
            "score": score,
            "signal": signal,
            "stop": round(stop, 2),
            "target": round(target, 2),
            "risk_reward": rr,
            "history": history_20,
            "buy_points": buy_points,
            "sell_points": sell_points,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "data_note": "15-min delayed via Yahoo Finance",
        }

    except Exception:
        print(f"Error fetching {ticker}: {traceback.format_exc()}")
        return None


@app.get("/")
def root():
    return {"status": "ok", "message": "Stock Screener API — see /docs for endpoints"}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/screen")
async def screen(
    tickers: str = Query(default=None, description="Comma-separated tickers, e.g. AAPL,MSFT. Defaults to built-in watchlist."),
):
    """
    Screen one or more stocks. Returns full indicator data + buy/sell signals.
    Example: GET /screen?tickers=AAPL,NVDA,TSLA
    """
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        ticker_list = DEFAULT_TICKERS

    if len(ticker_list) > 50:
        raise HTTPException(status_code=400, detail="Max 50 tickers per request")

    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(executor, fetch_and_analyze, t) for t in ticker_list]
    results = await asyncio.gather(*tasks)

    stocks = [r for r in results if r is not None]
    stocks.sort(key=lambda s: s["score"], reverse=True)

    return {
        "stocks": stocks,
        "count": len(stocks),
        "screened": len(ticker_list),
        "failed": len(ticker_list) - len(stocks),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_note": "15-min delayed intraday via Yahoo Finance (free, no API key required)",
    }


@app.get("/stock/{ticker}")
async def single_stock(ticker: str):
    """
    Fetch full analysis for a single ticker.
    Example: GET /stock/NVDA
    """
    result = await asyncio.get_event_loop().run_in_executor(
        executor, fetch_and_analyze, ticker.upper()
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {ticker}")
    return result
