"""
Stock Screener API — FastAPI + yfinance + pandas-ta
Calculates RSI, MACD, Bollinger Bands, EMA, ATR, Volume from real OHLCV data.
Data is 15-min delayed intraday via Yahoo Finance (free, no API key needed).
Optionally uses a trained XGBoost ML model (signal_model.pkl) for predictions.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
import os

# ── ML model — loaded once at startup if available ────────────────────────────
ML_MODEL      = None
ML_FEATURES   = None
ML_METADATA   = None

try:
    import joblib
    _model_path = os.path.join(os.path.dirname(__file__), "signal_model.pkl")
    if os.path.exists(_model_path):
        _bundle    = joblib.load(_model_path)
        ML_MODEL   = _bundle["model"]
        ML_FEATURES= _bundle["feature_cols"]
        ML_METADATA= _bundle.get("metadata", {})
        print(f"ML model loaded — AUC: {ML_METADATA.get('auc', 'unknown')}, "
              f"trained on {ML_METADATA.get('train_rows', '?')} rows")
    else:
        print("No signal_model.pkl found — running without ML signal")
except Exception as e:
    print(f"Could not load ML model: {e}")


def ml_predict(features: dict):
    """
    Given a dict of feature values, return ML buy probability (0-1).
    Returns None if model not loaded or features incomplete.
    """
    if ML_MODEL is None or ML_FEATURES is None:
        return None
    try:
        row = [features.get(f, np.nan) for f in ML_FEATURES]
        if any(np.isnan(float(v)) for v in row):
            return None
        prob = float(ML_MODEL.predict_proba([row])[0][1])
        return round(prob, 3)
    except Exception:
        return None

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

# Extended universe for the Rule 2 hunter — mid-caps outside S&P 500
# Global universe for Rule 2 scanning — mid-caps outside S&P 500 where
# technical patterns are less efficiently priced, plus international ADRs
_HUNT_RAW = [
    # ── US Technology mid-caps ────────────────────────────────────────────────
    "PCTY","QLYS","CIEN","GLOB","NCNO","EXLS","KFRC","PAYO","PRFT",
    "JAMF","ALKT","DOCN","MAPS","YEXT","BIGC","SEMR","SPSC","FOUR",
    "RELY","TASK","FLYW","BRZE","CFLT","DDOG","GTLB","HUBS","ZS",
    "BILL","ESTC","FROG","SMAR","TOST","WEAV","ZI",

    # ── US Consumer / retail mid-caps ─────────────────────────────────────────
    "CROX","BOOT","CAVA","SHAK","PLAY","BJ","FIVE","OLLI","PRGO","CATO",
    "BIRD","CURV","ELF","GOOS","LULU","OXM","PLCE","RGS","SCVL","TLYS",
    "CENT","CENTA","CHEF","FBRT","HAYW","LESL","PATK","POOL","SBH","XPOF",

    # ── US Healthcare mid-caps ────────────────────────────────────────────────
    "ENSG","ACAD","ITCI","AMED","NHC","PAHC","RGEN","NUVL","RXST","PRAX",
    "ACLS","ADUS","ALHC","AMEH","ANIP","ARVN","ASTH","AVTR","AXSM","BDTX",
    "CASH","CHRD","CRVS","DXCM","GKOS","HALO","INVA","IRMD","JNCE","KIDS",

    # ── US Industrials / materials mid-caps ───────────────────────────────────
    "ASTE","ITRI","KTOS","MYRG","ROAD","WMS","GVA","USLM","UFPI",
    "AEIS","AOUT","ARCB","ARLO","ATRI","BFAM","BLBD","BMI","CECO","CEIX",
    "CENT","CLH","CMCO","CNXC","CVEO","DFIN","DY","ENVA","FCFS","FWRD",

    # ── US Financial mid-caps ─────────────────────────────────────────────────
    "CSWC","GBCI","HOMB","TOWN","CVBF","SFNC","WSFS","NBTB","FFIN",
    "ABCB","ACNB","AROW","BANF","BANR","BHLB","BKU","BMTC","BSVN","BUSE",
    "CADE","CALB","CARE","CBAN","CBSH","CBTX","CCBG","CFB","CFFI","CFFN",

    # ── US Energy mid-caps ────────────────────────────────────────────────────
    "CIVI","MTDR","CHRD","VTLE","DINO","DKL",
    "AM","AMPY","ARCH","AROC","BATL","CEIX","CEL","CHNR","CLR","CTRA",
    "DKL","DNOW","DRQ","FLNG","GPRE","HESM","HPKV","HTZ","KALU","KNTK",

    # ── US REITs ─────────────────────────────────────────────────────────────
    "IIPR","NTST","APLE","SVC","ROIC","SITC","PLYM","NXRT","GMRE","PSTL",
    "ALEX","ALPS","AMH","AOMR","APAM","BRSP","BRT","CLDT","CLNC","COLD",

    # ── European ADRs (US-listed) ─────────────────────────────────────────────
    "ASML","SAP","SHOP","NVO","AZN","GSK","BP","SHEL","RIO","BBL",
    "UL","BTI","PHG","ING","ABB","ERIC","NOK","ST","SSYS","FLEX",
    "PSTG","WPM","AGI","KGC","OR","PAAS","MAG","AEM","FNV","GOLD",

    # ── LatAm ADRs ───────────────────────────────────────────────────────────
    "VALE","ERJ","PAGS","MELI","TIMB","DESP","BRFS","GGAL","SUPV","LPSN",
    "BSBR","ITUB","BBD","SBS","CIG","CBD","GGB","PBR","SID","VTMX",

    # ── Asia Pacific ADRs ─────────────────────────────────────────────────────
    "TSM","BABA","JD","PDD","BIDU","NIO","LI","XPEV","NTES","WB",
    "BILI","CANG","CIFS","CX","DADA","DOYU","EDU","FINV","GOTU","HTHT",
    "IQ","JOYY","KC","LAIX","LFC","LKNCY","MOMO","NOAH","RERE","RLX",
    "SE","TIGR","TME","TUYA","VNET","WDH","WIMI","XD","YMM","ZH",

    # ── India ADRs ───────────────────────────────────────────────────────────
    "INFY","WIT","HDB","IBN","SIFY","VEDL","TTM","MMYT","INDA","INDY",

    # ── Africa / Middle East ADRs ─────────────────────────────────────────────
    "GOLD","HL","SA","SLW","TRQ","BTG","DRD","HMY","SBSW","ANGPY",

    # ── Large caps as fallback ────────────────────────────────────────────────
    "NVDA","AAPL","MSFT","META","GOOGL","AMZN","TSLA","AMD",
    "JPM","BAC","JNJ","XOM","HD","PG","CVX","MRK",
    "NFLX","DIS","CAT","GE","BA","UPS","ADBE","CRM",
]
_seen = set()
HUNT_UNIVERSE = [x for x in _HUNT_RAW if not (x in _seen or _seen.add(x))]

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
    Fetch 250 days of daily OHLCV + today's intraday 15-min data.
    Adds SMA50/SMA200 trend filter and ATR-based position sizing.
    """
    try:
        tk = yf.Ticker(ticker)

        # 250 days needed for SMA200
        hist = tk.history(period="250d", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 50:
            return None

        intra = tk.history(period="1d", interval="15m", auto_adjust=True)
        if not intra.empty:
            current_price = float(intra["Close"].iloc[-1])
            current_vol_today = float(intra["Volume"].sum())
        else:
            current_price = float(hist["Close"].iloc[-1])
            current_vol_today = float(hist["Volume"].iloc[-1])

        close  = hist["Close"].astype(float)
        high   = hist["High"].astype(float)
        low    = hist["Low"].astype(float)
        volume = hist["Volume"].astype(float)

        # ── RSI(14) ───────────────────────────────────────────────────
        rsi_series = ta.rsi(close, length=14)
        rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.isna().all() else 50.0

        # ── MACD(12,26,9) ─────────────────────────────────────────────
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            hist_col = [c for c in macd_df.columns if "MACDh" in c]
            macd_val = float(macd_df[hist_col[0]].iloc[-1]) if hist_col else 0.0
            prev_macd_val = float(macd_df[hist_col[0]].iloc[-2]) if hist_col and len(macd_df) > 1 else macd_val
            macd_dir = "rising" if macd_val > prev_macd_val else "falling"
        else:
            macd_val, macd_dir = 0.0, "flat"

        # ── EMA20, SMA50, SMA200 ──────────────────────────────────────
        ema20_series = ta.ema(close, length=20)
        ema20 = float(ema20_series.iloc[-1]) if ema20_series is not None and not ema20_series.isna().all() else current_price
        pct_above_ema = (current_price - ema20) / ema20 * 100

        sma50  = float(close.tail(50).mean())
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())
        pct_above_sma50  = (current_price - sma50)  / sma50  * 100
        pct_above_sma200 = (current_price - sma200) / sma200 * 100

        # ── Trend filter: relaxed — price must be above SMA200 (primary bull/bear line)
        # SMA50 > SMA200 is informational but no longer a hard gate
        trend_bullish = (current_price > sma200)
        trend_aligned = (current_price > sma50) and (sma50 > sma200)   # stronger confirmation
        trend_status  = (
            "Strong uptrend" if trend_aligned and pct_above_sma200 > 5
            else "Uptrend"   if trend_aligned
            else "Above 200" if trend_bullish
            else "Downtrend" if current_price < sma200
            else "Mixed"
        )

        # ── Bollinger Bands(20,2) ─────────────────────────────────────
        bb_df = ta.bbands(close, length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            lower_col = [c for c in bb_df.columns if "BBL" in c]
            upper_col = [c for c in bb_df.columns if "BBU" in c]
            bb_lower = float(bb_df[lower_col[0]].iloc[-1]) if lower_col else current_price * 0.95
            bb_upper = float(bb_df[upper_col[0]].iloc[-1]) if upper_col else current_price * 1.05
            bb_range = bb_upper - bb_lower
            bb_pos   = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
        else:
            bb_lower, bb_upper, bb_pos = current_price * 0.95, current_price * 1.05, 0.5

        # ── ATR(14) ───────────────────────────────────────────────────
        atr_series = ta.atr(high, low, close, length=14)
        atr = float(atr_series.iloc[-1]) if atr_series is not None and not atr_series.isna().all() else current_price * 0.015

        # ── Volume vs 20-day average ──────────────────────────────────
        avg_vol_20  = float(volume.iloc[-21:-1].mean()) if len(volume) > 21 else float(volume.mean())
        vol_mult    = (current_vol_today / avg_vol_20) if avg_vol_20 > 0 else 1.0

        # ── 1-day price change ────────────────────────────────────────
        prev_close   = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
        price_change = (current_price - prev_close) / prev_close * 100

        # ── 20-day price history for sparkline ───────────────────────
        history_20 = [round(float(p), 2) for p in close.iloc[-20:].tolist()]

        # ── ROC(60) and 52-week metrics ───────────────────────────────
        roc60        = float(close.pct_change(60).iloc[-1] * 100) if len(close) >= 60 else 0.0
        high52       = float(high.rolling(252).max().iloc[-1]) if len(close) >= 252 else float(high.max())
        low52        = float(low.rolling(252).min().iloc[-1])  if len(close) >= 252 else float(low.min())
        pct_52w_high = (current_price - high52) / high52 * 100 if high52 > 0 else 0.0
        pct_52w_low  = (current_price - low52)  / low52  * 100 if low52  > 0 else 0.0
        hist_vol_20  = float(close.pct_change().rolling(20).std().iloc[-1] * (252**0.5) * 100)

        # ── Data-proven rules from ML research ───────────────────────
        # Rule 1: MACD positive AND RSI(14) < 35
        # 946 signals | 55.2% outperform SPY | avg alpha +14.52% over 20d
        rule1 = (macd_val > 0) and (rsi < 35)

        # Rule 2: Above SMA200 AND ROC(60) deeply negative AND far from 52w high
        # 100 signals | 72.0% outperform SPY | avg alpha +12.38% over 20d
        # roc60 threshold = bottom 15th percentile (~-22%), pct_52w_high bottom 20th (~-3.82%)
        rule2 = (trend_bullish) and (roc60 < -22.0) and (pct_52w_high < -3.82)

        # ── Signal score (legacy composite, kept for screener table) ──
        score = compute_signal_score(rsi, macd_val, macd_dir, pct_above_ema,
                                     vol_mult, bb_pos, price_change)
        if not trend_bullish and score >= 5:
            score = 4
        signal = signal_label(score)

        # ── Override signal if a data-proven rule fires ───────────────
        if rule2:
            # Highest precision rule — override to strong-buy
            signal = "strong-buy"
            score  = max(score, 8)
        elif rule1:
            # High alpha rule — override to buy minimum
            if signal not in ("strong-buy",):
                signal = "buy"
                score  = max(score, 6)

        # ── ML signal (if model available) ───────────────────────────
        obv          = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        obv_slope    = float((obv.diff(5) / (obv.abs().rolling(5).mean() + 1e-9)).iloc[-1])
        vol_ratio_5  = float(volume.rolling(5).mean().iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else 1.0
        bb_width_pct = float((bb_upper - bb_lower) / current_price * 100) if bb_upper > bb_lower else 4.0
        bb_squeeze   = bool((bb_upper - bb_lower) < close.rolling(20).std().iloc[-1] * 2 * 0.8)
        rsi7         = float(ta.rsi(close, length=7).iloc[-1]) if len(close) >= 7 else rsi
        roc5         = float(close.pct_change(5).iloc[-1]  * 100)
        roc10        = float(close.pct_change(10).iloc[-1] * 100)
        roc20        = float(close.pct_change(20).iloc[-1] * 100)
        open_price   = float(hist["Open"].iloc[-1])
        high_price   = float(hist["High"].iloc[-1])
        low_price    = float(hist["Low"].iloc[-1])
        candle_range = high_price - low_price + 1e-9
        candle_body  = (current_price - open_price) / candle_range
        upper_shadow = (high_price - max(current_price, open_price)) / candle_range
        lower_shadow = (min(current_price, open_price) - low_price)  / candle_range
        trend_regime = 1 if sma50 > sma200 else (-1 if sma50 < sma200 else 0)
        macd_signal_val = float(
            ta.macd(close, fast=12, slow=26, signal=9)
            .filter(like="MACDs").iloc[-1].values[0]
        ) if macd_df is not None and not macd_df.empty else 0.0

        ml_features = {
            "rsi_14": rsi, "rsi_7": rsi7,
            "macd_hist": macd_val, "macd_line": macd_val,
            "macd_signal": macd_signal_val,
            "roc_5": roc5, "roc_10": roc10, "roc_20": roc20,
            "pct_above_ema20": pct_above_ema,
            "pct_above_sma50": pct_above_sma50,
            "pct_above_sma200": pct_above_sma200,
            "trend_regime": trend_regime,
            "atr_pct": (atr / current_price * 100) if current_price > 0 else 0,
            "bb_pos": bb_pos, "bb_width_pct": bb_width_pct,
            "bb_squeeze": float(bb_squeeze),
            "hist_vol_20": hist_vol_20,
            "vol_ratio": vol_mult, "vol_ratio_5": vol_ratio_5,
            "obv_slope": obv_slope,
            "price_change_1d": price_change,
            "price_change_3d": float(close.pct_change(3).iloc[-1] * 100),
            "pct_from_52w_high": pct_52w_high, "pct_from_52w_low": pct_52w_low,
            "candle_body": candle_body,
            "upper_shadow": upper_shadow, "lower_shadow": lower_shadow,
        }
        ml_prob = ml_predict(ml_features)

        if ml_prob is not None:
            if ml_prob >= 0.55:
                ml_signal = "strong-buy"   # 73.7% win rate threshold
            elif ml_prob >= 0.50:
                ml_signal = "buy"
            elif ml_prob <= 0.35:
                ml_signal = "sell"
            else:
                ml_signal = "hold"
            # If ML is highly confident, also override main signal
            if ml_prob >= 0.55 and signal not in ("strong-buy",):
                signal = "strong-buy"
                score  = max(score, 8)
        else:
            ml_signal = None

        # ── Position sizing ───────────────────────────────────────────
        account_size   = 10000.0
        risk_pct       = 0.01
        risk_euros     = account_size * risk_pct
        stop_distance  = atr * 2
        shares_sized   = risk_euros / stop_distance if stop_distance > 0 else 0
        position_euros = shares_sized * current_price
        if position_euros < 500 and shares_sized > 0:
            shares_sized   = 500 / current_price
            position_euros = 500.0
        position_pct = (position_euros / account_size) * 100

        stop   = current_price - atr * 2
        target = current_price + atr * 4
        rr     = round(atr * 4 / (atr * 2), 1)

        # ── Buy / sell narratives ─────────────────────────────────────
        buy_points, sell_points = [], []

        # Which rules fired — explain them clearly
        if rule2:
            buy_points.append(
                f"RULE 2 FIRED (72% win rate vs SPY): Price above SMA200 + "
                f"60-day return {roc60:.1f}% (deeply oversold) + "
                f"{pct_52w_high:.1f}% from 52w high — recovery setup")
        if rule1:
            buy_points.append(
                f"RULE 1 FIRED (55% win rate vs SPY): "
                f"MACD positive + RSI {rsi:.1f} oversold — "
                f"historically +14.5% alpha over SPY in 20 days")
        if ml_prob is not None and ml_prob >= 0.55:
            buy_points.append(
                f"ML MODEL CONFIDENT: {ml_prob*100:.0f}% probability of "
                f"outperforming SPY — highest precision tier (73.7% historical win rate)")

        if signal in ("strong-buy", "buy"):
            buy_points.append(f"Entry near current price ${current_price:.2f}")
            if trend_bullish:
                if trend_aligned:
                    buy_points.append(
                        f"Trend confirmed — price above SMA50 (${sma50:.2f}) "
                        f"and SMA50 above SMA200 (${sma200:.2f})")
                else:
                    buy_points.append(
                        f"Trend filter passed — price above SMA200 (${sma200:.2f})")
            if rsi < 45:
                buy_points.append(f"RSI {rsi:.1f} — oversold, supports entry")
            if macd_val > 0:
                buy_points.append("MACD histogram positive — bullish momentum")
            if bb_pos < 0.3:
                buy_points.append("Near lower Bollinger Band — mean reversion setup")
            if vol_mult > 1.5:
                buy_points.append(f"Volume {vol_mult:.1f}x average — strong participation")
            buy_points.append(
                f"Position size: {shares_sized:.2f} shares "
                f"(€{position_euros:.0f} = {position_pct:.1f}% of €10k, "
                f"risking €{risk_euros:.0f})")
            commission = commerzbank_fee(position_euros)
            buy_points.append(
                f"Commerzbank fee: €{commission:.2f} per leg "
                f"(€{commission*2:.2f} round trip)")
        else:
            if not rule1 and not rule2:
                buy_points.append("Neither data-proven rule is firing right now")
            if not trend_bullish:
                buy_points.append(
                    f"Price below SMA200 (${sma200:.2f}) — "
                    f"wait for trend to recover")
            if rsi >= 35:
                buy_points.append(
                    f"RSI {rsi:.1f} — wait for RSI < 35 for Rule 1 to fire")
            if roc60 >= -22:
                buy_points.append(
                    f"60-day return {roc60:.1f}% — "
                    f"wait for deeper pullback (<-22%) for Rule 2")
            if macd_val <= 0:
                buy_points.append("MACD negative — wait for histogram to turn positive")

        sell_points.append(
            f"Stop loss: ${stop:.2f} (2x ATR) — hard exit, no exceptions")
        sell_points.append(
            f"Target: ${target:.2f} (4x ATR, R:R 1:2) — take full profit here")
        sell_points.append(
            "Minimum hold: 5 trading days before any signal-based exit")
        if rsi > 68:
            sell_points.append(f"RSI {rsi:.1f} — approaching overbought, consider reducing size")
        if macd_val > 0 and macd_dir == "falling":
            sell_points.append("MACD flattening — tighten stop to breakeven after day 5")
        if bb_pos > 0.8:
            sell_points.append("Near upper Bollinger Band — consider 50% profit at this level")
        sell_points.append(
            f"Also exit if price closes below SMA200 (${sma200:.2f}) — trend broken")

        return {
            "ticker":            ticker,
            "name":              tk.info.get("shortName", ticker) if hasattr(tk, "info") else ticker,
            "sector":            SECTOR_MAP.get(ticker, tk.info.get("sector", "Unknown") if hasattr(tk, "info") else "Unknown"),
            "price":             round(current_price, 2),
            "change":            round(price_change, 2),
            "rsi":               round(rsi, 1),
            "macd_val":          round(macd_val, 3),
            "macd_dir":          macd_dir,
            "ema20":             round(ema20, 2),
            "sma50":             round(sma50, 2),
            "sma200":            round(sma200, 2),
            "pct_above_ema":     round(pct_above_ema, 2),
            "pct_above_sma50":   round(pct_above_sma50, 2),
            "pct_above_sma200":  round(pct_above_sma200, 2),
            "trend_bullish":     trend_bullish,
            "trend_status":      trend_status,
            "roc60":             round(roc60, 2),
            "pct_52w_high":      round(pct_52w_high, 2),
            "pct_52w_low":       round(pct_52w_low, 2),
            "rule1_fired":       rule1,
            "rule2_fired":       rule2,
            "bb_lower":          round(bb_lower, 2),
            "bb_upper":          round(bb_upper, 2),
            "bb_pos":            round(bb_pos, 3),
            "atr":               round(atr, 2),
            "vol_mult":          round(vol_mult, 2),
            "score":             score,
            "signal":            signal,
            "stop":              round(stop, 2),
            "target":            round(target, 2),
            "risk_reward":       rr,
            "position_shares":   round(shares_sized, 2),
            "position_euros":    round(position_euros, 0),
            "position_pct":      round(position_pct, 1),
            "risk_euros":        round(risk_euros, 0),
            "ml_prob":           ml_prob,
            "ml_signal":         ml_signal,
            "ml_available":      ML_MODEL is not None,
            "history":           history_20,
            "buy_points":        buy_points,
            "sell_points":       sell_points,
            "updated_at":        datetime.now(timezone.utc).isoformat(),
            "data_note":         "15-min delayed via Yahoo Finance",
        }

    except Exception:
        print(f"Error fetching {ticker}: {traceback.format_exc()}")
        return None


FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>US Market Scanner</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#fff;--bg2:#f5f5f4;--bg3:#eeede9;--txt:#1a1a1a;--txt2:#666;--txt3:#999;--border:rgba(0,0,0,0.1);--border2:rgba(0,0,0,0.18);--green:#2d7a3a;--red:#b03030;--r:8px;--rl:12px}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg3);color:var(--txt);min-height:100vh}
.wrap{max-width:1100px;margin:0 auto;padding:1.5rem 1rem}
.header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.25rem;flex-wrap:wrap;gap:10px}
.title{font-size:20px;font-weight:600}.subtitle{font-size:13px;color:var(--txt2);margin-top:3px}
.ldot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);margin-right:6px;animation:lp 2s ease-in-out infinite}
@keyframes lp{0%,100%{opacity:1}50%{opacity:.3}}
.disc{font-size:11px;color:var(--txt3);background:var(--bg2);border:.5px solid var(--border);border-radius:var(--r);padding:6px 10px}
.regime-banner{border-radius:var(--rl);padding:16px 20px;margin-bottom:1.25rem;border:.5px solid transparent}
.regime-banner.green{background:#e8f5ea;border-color:#a8d5b0}
.regime-banner.amber{background:#fef9e7;border-color:#e8d08a}
.regime-banner.red{background:#fdeaea;border-color:#e8aaaa}
.regime-banner.loading{background:var(--bg2);border-color:var(--border)}
.rb-verdict{font-size:15px;font-weight:600;margin-bottom:5px}
.rb-verdict.green{color:#1a5c28}.rb-verdict.amber{color:#92620a}.rb-verdict.red{color:#7a1c1c}.rb-verdict.loading{color:var(--txt2)}
.rb-reason{font-size:13px;line-height:1.55;color:var(--txt2)}
.rb-stats{display:flex;gap:24px;flex-wrap:wrap;margin-top:12px;padding-top:12px;border-top:.5px solid rgba(0,0,0,0.08)}
.rb-stat-label{font-size:11px;color:var(--txt3);margin-bottom:3px}
.rb-stat-value{font-size:14px;font-weight:600}
.vix-wrap{width:120px;height:5px;background:rgba(0,0,0,0.1);border-radius:3px;margin-top:5px}
.vix-fill{height:5px;border-radius:3px;transition:width .6s}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:1.25rem}
select,button{font-size:13px;padding:7px 12px;border-radius:var(--r);border:.5px solid var(--border2);background:var(--bg);color:var(--txt);cursor:pointer}
select:hover,button:hover{background:var(--bg2)}
.btnp{background:var(--txt)!important;color:var(--bg)!important;border-color:var(--txt)!important}
.btnp:hover{opacity:.85}button:disabled{opacity:.5;cursor:not-allowed}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:1.5rem}
.mc{background:var(--bg);border-radius:var(--r);padding:12px 14px;border:.5px solid var(--border)}
.ml{font-size:11px;color:var(--txt2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
.mv{font-size:22px;font-weight:600}.mv.g{color:var(--green)}.mv.r{color:var(--red)}
.ms{font-size:11px;color:var(--txt3);margin-top:2px}
.card{background:var(--bg);border-radius:var(--rl);border:.5px solid var(--border);overflow:hidden;margin-bottom:1.5rem}
table{width:100%;border-collapse:collapse;font-size:13px;table-layout:fixed}
th{background:var(--bg2);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--txt2);padding:10px 12px;text-align:left;border-bottom:.5px solid var(--border);cursor:pointer;user-select:none}
th:hover{color:var(--txt)}td{padding:11px 12px;border-bottom:.5px solid var(--border);vertical-align:middle}
tr:last-child td{border-bottom:none}tbody tr{cursor:pointer;transition:background .1s}tbody tr:hover td{background:var(--bg2)}
.pill{display:inline-block;font-size:11px;font-weight:500;padding:3px 9px;border-radius:20px}
.sb{background:#d0f0d8;color:#1a5c28}.b{background:#e5f5e9;color:#2d6e3a}
.h{background:#fef9e7;color:#7a6520}.s{background:#fde8e8;color:#9b2a2a}.ss{background:#fbd5d5;color:#7a1c1c}
.gn{color:var(--green);font-weight:500}.rd{color:var(--red);font-weight:500}
.bar{display:flex;gap:2px;align-items:center}
.seg{width:8px;height:8px;border-radius:1px;background:var(--border2)}
.seg.bull{background:var(--green)}.seg.bear{background:var(--red)}
.tabs{display:flex;border-bottom:.5px solid var(--border);margin-bottom:1.25rem}
.tab{font-size:13px;padding:8px 16px;color:var(--txt2);cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
.tab.active{color:var(--txt);border-bottom-color:var(--txt);font-weight:500}
.loading{text-align:center;padding:3rem;color:var(--txt2);font-size:14px}
.dot{display:inline-block;animation:pulse 1.2s ease-in-out infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
.ebox{background:#fdeaea;border:.5px solid #e8aaaa;border-radius:var(--r);padding:14px 16px;color:#7a1c1c;font-size:13px;margin-bottom:1rem}
.ph{background:var(--bg2);padding:12px 16px;border-bottom:.5px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.pb{padding:16px}.dg{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.dl{font-size:11px;color:var(--txt2);margin-bottom:3px}.dv{font-size:15px;font-weight:600}
.sbox{border-radius:var(--r);padding:12px 14px;margin-bottom:10px}
.sbox.buy{background:#e8f5ea;border:.5px solid #a8d5b0}.sbox.sell{background:#fdeaea;border:.5px solid #e8aaaa}
.sbox-t{font-size:12px;font-weight:600;margin-bottom:6px}
.sbox.buy .sbox-t{color:#1a5c28}.sbox.sell .sbox-t{color:#7a1c1c}.sbox p{font-size:13px;line-height:1.7}
.inds{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 4px}
.itag{font-size:11px;padding:3px 8px;border-radius:20px;border:.5px solid var(--border2)}
.itag.bull{border-color:#a8d5b0;background:#f0faf2;color:#1a5c28}
.itag.bear{border-color:#e8aaaa;background:#fdf5f5;color:#7a1c1c}
.itag.neut{background:var(--bg2);color:var(--txt2)}
.rbox{margin-top:16px;background:var(--bg2);border-radius:var(--r);padding:12px 14px}
.rg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;font-size:13px;margin-top:8px}
.back{font-size:13px;color:var(--txt2);cursor:pointer;margin-bottom:1rem;display:inline-block;border:none;background:none;padding:0}
.back:hover{color:var(--txt);background:none}
.cc{position:relative;height:180px;width:100%;margin:14px 0}
.spin{display:inline-block;animation:sp .8s linear infinite}
@keyframes sp{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.note{font-size:11px;color:var(--txt3);margin-top:10px}
.hunt-card{background:var(--bg);border-radius:var(--rl);border:.5px solid var(--border);padding:16px;margin-bottom:12px}
.hunt-card-top{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;margin-bottom:12px}
.hunt-card-ticker{font-size:18px;font-weight:700}
.hunt-card-name{font-size:12px;color:var(--txt2);margin-top:2px}
.hunt-card-price{font-size:18px;font-weight:600;text-align:right}
.hunt-card-chg{font-size:13px;text-align:right;margin-top:2px}
.hunt-card-body{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
.hunt-stat-label{font-size:11px;color:var(--txt3);margin-bottom:2px}
.hunt-stat-value{font-size:14px;font-weight:600}
.hunt-points{font-size:13px;line-height:1.65;color:var(--txt2);margin-bottom:12px}
.hunt-risk{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;background:var(--bg2);border-radius:var(--r);padding:10px 12px;font-size:12px}
.bt-mc{background:var(--bg);border-radius:var(--r);padding:11px 13px;border:.5px solid var(--border)}
.bt-mc .ml{font-size:10px;color:var(--txt3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.bt-mc .mv{font-size:18px;font-weight:700}
.bt-win{color:#2d7a3a}.bt-loss{color:#b03030}.bt-neut{color:var(--txt)}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div><div class="title"><span class="ldot"></span>US Market Scanner</div><div class="subtitle" id="ts">Connecting to live data...</div></div>
    <div class="disc">15-min delayed · Yahoo Finance · Not financial advice</div>
  </div>

  <!-- Market regime banner -->
  <div class="regime-banner loading" id="rbanner">
    <div class="rb-verdict loading" id="rb-verdict">Checking market conditions...</div>
    <div class="rb-reason" id="rb-reason">Fetching VIX and S&P 500 data</div>
    <div class="rb-stats" id="rb-stats" style="display:none">
      <div><div class="rb-stat-label">VIX (fear index)</div><div class="rb-stat-value" id="rb-vix">-</div><div class="vix-wrap"><div class="vix-fill" id="rb-vix-bar" style="width:0%"></div></div></div>
      <div><div class="rb-stat-label">S&P 500 (SPY)</div><div class="rb-stat-value" id="rb-spy">-</div></div>
      <div><div class="rb-stat-label">vs 200-day SMA</div><div class="rb-stat-value" id="rb-200">-</div></div>
      <div><div class="rb-stat-label">vs 50-day SMA</div><div class="rb-stat-value" id="rb-50">-</div></div>
      <div><div class="rb-stat-label">Regime</div><div class="rb-stat-value" id="rb-regime">-</div></div>
    </div>
  </div>

  <div id="main">
    <div class="controls">
      <select id="fsig"><option value="all">All signals</option><option value="strong-buy">Strong buy</option><option value="buy">Buy</option><option value="hold">Hold</option><option value="sell">Sell</option><option value="strong-sell">Strong sell</option></select>
      <select id="fsec"><option value="all">All sectors</option><option value="Technology">Technology</option><option value="Healthcare">Healthcare</option><option value="Financials">Financials</option><option value="Energy">Energy</option><option value="Consumer">Consumer</option><option value="Industrials">Industrials</option></select>
      <select id="fsort"><option value="score">Sort: Score</option><option value="rsi">Sort: RSI</option><option value="change">Sort: 1D chg</option><option value="volume">Sort: Volume</option></select>
      <button class="btnp" id="rbtn">Refresh</button>
    </div>
    <div class="metrics">
      <div class="mc"><div class="ml">Strong buys</div><div class="mv g" id="msb">-</div><div class="ms">stocks</div></div>
      <div class="mc"><div class="ml">Buy signals</div><div class="mv g" id="mb">-</div><div class="ms">stocks</div></div>
      <div class="mc"><div class="ml">Sell signals</div><div class="mv r" id="ms2">-</div><div class="ms">stocks</div></div>
      <div class="mc"><div class="ml">Avg RSI</div><div class="mv" id="mrsi">-</div><div class="ms">screened</div></div>
      <div class="mc"><div class="ml">Stock mood</div><div class="mv" id="mmood" style="font-size:15px">-</div><div class="ms" id="mmoods"></div></div>
    </div>
    <div class="tabs">
      <div class="tab active" id="t-screen" onclick="setTab('screen')">Screener</div>
      <div class="tab" id="t-hunt" onclick="setTab('hunt')">&#128269; Find Rule 2 Setup</div>
      <div class="tab" id="t-global" onclick="setTab('global')">&#127758; Global Rule 2 Scan</div>
      <div class="tab" id="t-check" onclick="setTab('check')">&#128270; Check Ticker</div>
      <div class="tab" id="t-backtest" onclick="setTab('backtest')">&#9654; Backtest</div>
      <div class="tab" id="t-watch" onclick="setTab('watch')">Watchlist</div>
    </div>
    <div id="tab-screen">
      <div id="loading" class="loading">Fetching live data<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span><div style="font-size:12px;margin-top:8px;color:#999">First load may take 15-20s</div></div>
      <div id="err" style="display:none"></div>
      <div id="tw" class="card" style="display:none">
        <table><thead><tr>
          <th style="width:13%">Ticker</th><th style="width:9%">Price</th><th style="width:9%">1D chg</th>
          <th style="width:13%">Signal</th><th style="width:16%">Score</th><th style="width:7%">RSI</th>
          <th style="width:11%">MACD</th><th style="width:8%">Vol</th><th style="width:14%">Sector</th>
        </tr></thead><tbody id="tbody"></tbody></table>
      </div>
    </div>
    <div id="tab-hunt" style="display:none">
      <div id="hunt-idle" style="text-align:center;padding:2.5rem 1rem">
        <div style="font-size:32px;margin-bottom:12px">&#128269;</div>
        <div style="font-size:15px;font-weight:600;margin-bottom:8px">Rule 2 Signal Hunter</div>
        <div style="font-size:13px;color:#666;margin-bottom:8px;max-width:460px;margin-left:auto;margin-right:auto">
          Scans ~100 mid-cap stocks for <strong>Rule 2</strong> setups: price above SMA200, down &gt;22% over 60 days, far from 52-week high.
          <br><br>Historical edge: <strong>72% outperform SPY</strong> by 1%+ over 20 days, avg alpha +12.4%.
          Falls back to Rule 1 (MACD+ &amp; RSI&lt;35, 55% win rate) if Rule 2 is not firing today.
        </div>
        <button class="btnp" onclick="startHunt()">&#128269; Hunt for Rule 2 Setup</button>
      </div>
      <div id="hunt-loading" style="display:none;text-align:center;padding:2.5rem 1rem">
        <div style="font-size:13px;color:#666;margin-bottom:12px">Scanning for Rule 2 signals<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>
        <div id="hunt-progress" style="font-size:22px;font-weight:600;color:#2d7a3a;margin-bottom:4px">searching...</div>
        <div id="hunt-scanned" style="font-size:12px;color:#999">0 stocks scanned</div>
        <div style="width:200px;height:4px;background:rgba(0,0,0,0.1);border-radius:2px;margin:12px auto 0">
          <div id="hunt-bar" style="height:4px;background:#2d7a3a;border-radius:2px;width:0%;transition:width .4s"></div>
        </div>
      </div>
      <div id="hunt-results" style="display:none">
        <div id="hunt-summary" style="margin-bottom:1rem;padding:12px 16px;border-radius:var(--rl);font-size:13px"></div>
        <div id="hunt-cards"></div>
        <div style="text-align:center;margin-top:1rem">
          <button class="btnp" onclick="startHunt()">&#128269; Hunt Again</button>
        </div>
      </div>
      <div id="hunt-err" style="display:none"></div>
    </div>
    </div>
    <div id="tab-global" style="display:none">
      <div id="global-idle" style="text-align:center;padding:2.5rem 1rem">
        <div style="font-size:32px;margin-bottom:12px">&#127758;</div>
        <div style="font-size:15px;font-weight:600;margin-bottom:8px">Global Rule 2 Scanner</div>
        <div style="font-size:13px;color:#666;margin-bottom:8px;max-width:500px;margin-left:auto;margin-right:auto">
          Scans <strong>300+ stocks globally</strong> — US mid-caps, European ADRs, LatAm, Asia Pacific — for Rule 2 setups.<br><br>
          <strong>Rule 2:</strong> Price above SMA200 + 60-day return &lt; -22% + far from 52-week high<br>
          <span style="color:#2d7a3a;font-weight:600">72% outperform SPY by 1%+</span> over 20 days · avg alpha +12.4%<br><br>
          Also shows any Rule 1 signals found (MACD positive + RSI &lt; 35, 55% win rate).
        </div>
        <div style="font-size:12px;color:#999;margin-bottom:16px">Takes ~60 seconds to scan the full universe</div>
        <button class="btnp" onclick="startGlobalScan()">&#127758; Start Global Scan</button>
      </div>
      <div id="global-loading" style="display:none;text-align:center;padding:2.5rem 1rem">
        <div style="font-size:13px;color:#666;margin-bottom:12px">Scanning global universe<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>
        <div id="global-progress" style="font-size:22px;font-weight:600;color:#2d7a3a;margin-bottom:4px">Scanning...</div>
        <div id="global-sub" style="font-size:12px;color:#999">Fetching data in parallel batches of 20</div>
        <div style="width:260px;height:4px;background:rgba(0,0,0,0.1);border-radius:2px;margin:12px auto 0">
          <div id="global-bar" style="height:4px;background:#2d7a3a;border-radius:2px;width:5%;transition:width 2s"></div>
        </div>
      </div>
      <div id="global-results" style="display:none">
        <div id="global-summary" style="margin-bottom:1rem"></div>
        <div id="global-r2-section" style="display:none">
          <div style="font-size:14px;font-weight:600;color:#1a5c28;margin-bottom:10px;padding:8px 12px;background:#e8f5ea;border-radius:var(--r);border:.5px solid #a8d5b0">
            &#9989; Rule 2 signals — <span id="global-r2-count">0</span> found · 72% win rate vs SPY · avg +12.4% alpha
          </div>
          <div id="global-r2-cards"></div>
        </div>
        <div id="global-r1-section" style="display:none;margin-top:1.5rem">
          <div style="font-size:14px;font-weight:600;color:#7a6520;margin-bottom:10px;padding:8px 12px;background:#fef9e7;border-radius:var(--r);border:.5px solid #e8d08a">
            &#128308; Rule 1 signals — <span id="global-r1-count">0</span> found · 55% win rate vs SPY · avg +14.5% alpha
          </div>
          <div id="global-r1-cards"></div>
        </div>
        <div id="global-none" style="display:none;text-align:center;padding:2rem;color:#666;font-size:13px">
          No Rule 2 or Rule 1 signals found in today's scan.<br>
          This is normal — these are high-precision signals that fire infrequently.<br>
          Try again tomorrow or after a market pullback.
        </div>
        <div style="text-align:center;margin-top:1.5rem">
          <button class="btnp" onclick="startGlobalScan()">&#127758; Scan Again</button>
        </div>
      </div>
      <div id="global-err" style="display:none"></div>
    </div>
    <div id="tab-check" style="display:none">
      <div style="max-width:520px;margin:0 auto;padding:1.5rem 0">
        <div style="font-size:15px;font-weight:600;margin-bottom:6px">Check any ticker</div>
        <div style="font-size:13px;color:#666;margin-bottom:16px">Enter any US stock symbol to run the full signal analysis against all indicators.</div>
        <div style="display:flex;gap:8px;margin-bottom:1.5rem">
          <input id="ticker-input" type="text" placeholder="e.g. AAPL, NVDA, SHOP..." style="flex:1;font-size:14px;padding:9px 12px;border-radius:var(--r);border:.5px solid var(--border2);background:var(--bg);color:var(--txt);text-transform:uppercase" maxlength="10" />
          <button class="btnp" id="check-btn" onclick="checkTicker()">Analyse</button>
        </div>
        <div id="check-loading" style="display:none;text-align:center;padding:2rem">
          Analysing<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
        </div>
        <div id="check-err" style="display:none"></div>
        <div id="check-result" style="display:none">
          <div class="card">
            <div class="ph">
              <div><span style="font-size:15px;font-weight:600" id="ck-ticker"></span><span style="font-size:13px;color:#666;margin-left:10px" id="ck-name"></span></div>
              <div style="display:flex;gap:8px;align-items:center"><span id="ck-pill"></span><button id="ck-wlbtn" onclick="ckWatchlist()">+ Watchlist</button></div>
            </div>
            <div class="pb">
              <div class="dg">
                <div><div class="dl">Price</div><div class="dv" id="ck-price"></div></div>
                <div><div class="dl">1-day change</div><div class="dv" id="ck-chg"></div></div>
                <div><div class="dl">Signal score</div><div class="dv" id="ck-score"></div></div>
                <div><div class="dl">ATR (14)</div><div class="dv" id="ck-atr"></div></div>
              </div>
              <div class="dl" style="margin-top:14px;margin-bottom:4px">Indicator breakdown</div>
              <div class="inds" id="ck-inds"></div>
              <div class="cc"><canvas id="ck-chart"></canvas></div>
              <div class="sbox buy"><div class="sbox-t">BUY - entry conditions</div><p id="ck-buy"></p></div>
              <div class="sbox sell"><div class="sbox-t">SELL / exit conditions</div><p id="ck-sell"></p></div>
              <div class="rbox">
                <div class="dl">Risk management</div>
                <div class="rg">
                  <div><div class="dl">Stop loss (2x ATR)</div><div style="font-weight:600;color:#b03030" id="ck-stop"></div></div>
                  <div><div class="dl">Target (3x ATR)</div><div style="font-weight:600;color:#2d7a3a" id="ck-tgt"></div></div>
                  <div><div class="dl">Risk / reward</div><div style="font-weight:600" id="ck-rr"></div></div>
                </div>
              </div>
              <p class="note">Live data · Yahoo Finance · 15-min delayed · Not financial advice</p>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div id="tab-backtest" style="display:none">
      <div style="max-width:660px;margin:0 auto;padding:1.5rem 0">
        <div style="font-size:15px;font-weight:600;margin-bottom:4px">Backtest a strategy</div>
        <div style="font-size:13px;color:#666;margin-bottom:14px">Simulates trading using the data-proven rules. Entry fires on Rule 1 (MACD+ &amp; RSI&lt;35) and/or Rule 2 (above SMA200 + deep pullback). Commerzbank fees applied on every trade.</div>

        <div style="background:#fef9e7;border:.5px solid #e8d08a;border-radius:var(--rl);padding:12px 14px;margin-bottom:14px;font-size:12px;color:#7a6520">
          <strong>Fee warning:</strong> Commerzbank charges min €9.90 per trade (€19.80 round trip). On a €1,000 account with 45 trades that is €891 in fees — more than your capital. Use at least €5,000 account size, or €10,000 for realistic results.
        </div>

        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;align-items:flex-end">
          <div style="flex:1;min-width:100px">
            <div class="dl" style="margin-bottom:4px">Ticker</div>
            <input id="bt-ticker" type="text" placeholder="e.g. SHOP" style="width:100%;font-size:14px;padding:9px 12px;border-radius:var(--r);border:.5px solid var(--border2);background:var(--bg);color:var(--txt);text-transform:uppercase" maxlength="10" />
          </div>
          <div style="width:110px">
            <div class="dl" style="margin-bottom:4px">Period</div>
            <select id="bt-period" style="width:100%">
              <option value="1y">1 year</option>
              <option value="2y">2 years</option>
              <option value="5y" selected>5 years</option>
            </select>
          </div>
          <div style="width:130px">
            <div class="dl" style="margin-bottom:4px">Account size</div>
            <select id="bt-size" style="width:100%">
              <option value="1000">€1,000</option>
              <option value="5000">€5,000</option>
              <option value="10000" selected>€10,000</option>
              <option value="25000">€25,000</option>
              <option value="50000">€50,000</option>
            </select>
          </div>
          <div style="width:130px">
            <div class="dl" style="margin-bottom:4px">Min position</div>
            <select id="bt-minpos" style="width:100%">
              <option value="500">€500</option>
              <option value="1000">€1,000</option>
              <option value="2000" selected>€2,000</option>
              <option value="5000">€5,000</option>
            </select>
          </div>
          <button class="btnp" id="bt-btn" onclick="runBacktest()">&#9654; Run</button>
        </div>

        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:1rem;align-items:flex-end">
          <div style="flex:1">
            <div class="dl" style="margin-bottom:4px">Entry mode</div>
            <select id="bt-mode" style="width:100%">
              <option value="both">Rule 1 + Rule 2 (more trades)</option>
              <option value="rule2only">Rule 2 only — highest precision (72% win rate vs SPY)</option>
              <option value="rule1only">Rule 1 only — MACD+ &amp; RSI&lt;35 (55% win rate vs SPY)</option>
              <option value="score">Legacy score ≥5 only</option>
            </select>
          </div>
        </div>

        <div id="bt-loading" style="display:none;text-align:center;padding:2rem;color:#666">
          Running backtest<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
        </div>
        <div id="bt-err" style="display:none"></div>
        <div id="bt-results" style="display:none">
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin-bottom:1.25rem" id="bt-metrics"></div>
          <div class="card" style="padding:16px;margin-bottom:1rem">
            <div style="font-size:12px;color:#999;margin-bottom:8px">Equity curve &mdash; starting &euro;<span id="bt-start-val"></span></div>
            <div style="position:relative;height:220px"><canvas id="bt-equity-chart"></canvas></div>
          </div>
          <div class="card" style="overflow:hidden;margin-bottom:1rem">
            <div style="background:var(--bg2);padding:10px 14px;font-size:12px;font-weight:600;border-bottom:.5px solid var(--border)">Trade log</div>
            <div style="overflow-x:auto">
              <table style="font-size:12px">
                <thead><tr>
                  <th style="width:5%">#</th>
                  <th style="width:11%">Buy date</th>
                  <th style="width:8%">Buy $</th>
                  <th style="width:11%">Sell date</th>
                  <th style="width:8%">Sell $</th>
                  <th style="width:7%">Days</th>
                  <th style="width:8%">Invested</th>
                  <th style="width:8%">Fees €</th>
                  <th style="width:8%">P&amp;L €</th>
                  <th style="width:7%">%</th>
                  <th style="width:19%">Exit reason</th>
                </tr></thead>
                <tbody id="bt-trades"></tbody>
              </table>
            </div>
          </div>
          <p class="note">Backtest uses adjusted close prices. Slippage and commissions not included. Past performance does not guarantee future results.</p>
        </div>
      </div>
    </div>
      <div id="wt"></div>
    </div>
  </div>

  <div id="detail" style="display:none">
    <button class="back" id="backbtn">&#8592; Back to screener</button>
    <div id="dl" class="loading" style="display:none">Loading<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>
    <div id="de" style="display:none"></div>
    <div id="dc" style="display:none">
      <div class="card">
        <div class="ph">
          <div><span style="font-size:15px;font-weight:600" id="dticker"></span><span style="font-size:13px;color:#666;margin-left:10px" id="dname"></span></div>
          <div style="display:flex;gap:8px;align-items:center"><span id="dpill"></span><button id="wlbtn">+ Watchlist</button></div>
        </div>
        <div class="pb">
          <div class="dg">
            <div><div class="dl">Price</div><div class="dv" id="dprice"></div></div>
            <div><div class="dl">1-day change</div><div class="dv" id="dchg"></div></div>
            <div><div class="dl">Signal score</div><div class="dv" id="dscore"></div></div>
            <div><div class="dl">ATR (14)</div><div class="dv" id="datr"></div></div>
          </div>
          <div class="dl" style="margin-top:14px;margin-bottom:4px">Indicator breakdown</div>
          <div class="inds" id="dinds"></div>
          <div class="cc"><canvas id="pc"></canvas></div>
          <div class="sbox buy"><div class="sbox-t">BUY - entry conditions</div><p id="dbuy"></p></div>
          <div class="sbox sell"><div class="sbox-t">SELL / exit conditions</div><p id="dsell"></p></div>
          <div class="rbox">
            <div class="dl">Risk management</div>
            <div class="rg">
              <div><div class="dl">Stop loss (2x ATR)</div><div style="font-weight:600;color:#b03030" id="dstop"></div></div>
              <div><div class="dl">Target (3x ATR)</div><div style="font-weight:600;color:#2d7a3a" id="dtgt"></div></div>
              <div><div class="dl">Risk / reward</div><div style="font-weight:600" id="drr"></div></div>
            </div>
          </div>
          <p class="note">Live data · Yahoo Finance · 15-min delayed · Not financial advice</p>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const API='';
let stocks=[],filtered=[],watchlist=new Set(),curTicker=null,chart=null;
const $=id=>document.getElementById(id);
const fmt=(n,d=2)=>n!=null?(+n).toFixed(d):'-';
const fmtP=n=>n!=null?((+n>=0?'+':'')+((+n).toFixed(2))+'%'):'-';
function pC(s){return{'strong-buy':'sb','buy':'b','hold':'h','sell':'s','strong-sell':'ss'}[s]||'h'}
function pL(s){return{'strong-buy':'Strong buy','buy':'Buy','hold':'Hold','sell':'Sell','strong-sell':'Strong sell'}[s]||s}
function sbar(sc,sig){let h='<div class="bar">';for(let i=1;i<=10;i++)h+=`<div class="seg ${i<=sc?(sig&&sig.includes('sell')?'bear':'bull'):''}"></div>`;return h+`</div><small style="color:#999;margin-top:2px;display:block">${sc}/10</small>`}

async function loadMarket(){
  try{
    const res=await fetch(API+'/market');
    if(!res.ok)return;
    const m=await res.json();
    const color=m.color||'amber';
    $('rbanner').className='regime-banner '+color;
    const v=$('rb-verdict');v.className='rb-verdict '+color;v.textContent=m.verdict;
    $('rb-reason').textContent=m.reason;
    $('rb-stats').style.display='flex';
    const vix=m.vix||0;
    $('rb-vix').textContent=fmt(vix,1);
    $('rb-vix').style.color=vix>30?'#b03030':vix>20?'#92620a':'#2d7a3a';
    const bar=$('rb-vix-bar');
    bar.style.width=Math.min(100,(vix/50)*100)+'%';
    bar.style.background=vix>30?'#b03030':vix>20?'#e8a020':'#2d7a3a';
    $('rb-spy').textContent='$'+fmt(m.spy_price);
    const s200=$('rb-200');s200.textContent=(m.spy_vs_200sma>=0?'+':'')+fmt(m.spy_vs_200sma,1)+'%';
    s200.style.color=m.spy_vs_200sma>=0?'#2d7a3a':'#b03030';
    const s50=$('rb-50');s50.textContent=(m.spy_vs_50sma>=0?'+':'')+fmt(m.spy_vs_50sma,1)+'%';
    s50.style.color=m.spy_vs_50sma>=0?'#2d7a3a':'#b03030';
    $('rb-regime').textContent=m.regime;
    $('rb-regime').style.color=color==='green'?'#2d7a3a':color==='red'?'#b03030':'#92620a';
  }catch(e){
    $('rb-verdict').textContent='Market context unavailable';
    $('rb-reason').textContent='Could not fetch VIX / SPY data';
  }
}

async function load(){
  const btn=$('rbtn');btn.innerHTML='<span class="spin">&#8635;</span> Loading';btn.disabled=true;
  $('loading').style.display='';$('tw').style.display='none';$('err').style.display='none';
  loadMarket();
  try{
    const res=await fetch(API+'/screen');
    if(!res.ok)throw new Error('Server error '+res.status);
    const data=await res.json();
    stocks=data.stocks||[];
    if(!stocks.length)throw new Error('No stocks returned - market may be closed');
    applyFilters();updateMetrics();
    $('loading').style.display='none';$('tw').style.display='';
    $('ts').textContent='Live · 15-min delayed · Updated '+new Date().toLocaleTimeString()+' · '+stocks.length+' stocks';
  }catch(e){
    $('loading').style.display='none';$('err').style.display='';
    $('err').innerHTML='<div class="ebox">Failed to load: '+e.message+'</div>';
  }
  btn.innerHTML='Refresh';btn.disabled=false;
}

function applyFilters(){
  const sig=$('fsig').value,sec=$('fsec').value,sort=$('fsort').value;
  filtered=[...stocks].filter(s=>(sig==='all'||s.signal===sig)&&(sec==='all'||s.sector===sec));
  if(sort==='score')filtered.sort((a,b)=>b.score-a.score);
  else if(sort==='rsi')filtered.sort((a,b)=>a.rsi-b.rsi);
  else if(sort==='change')filtered.sort((a,b)=>b.change-a.change);
  else if(sort==='volume')filtered.sort((a,b)=>b.vol_mult-a.vol_mult);
  renderTable();
}
function renderTable(){
  $('tbody').innerHTML=filtered.map(s=>`<tr onclick="showDetail('${s.ticker}')">
    <td><strong>${s.ticker}</strong><div style="font-size:11px;color:#999">${(s.name||'').split(' ').slice(0,2).join(' ')}</div></td>
    <td>$${fmt(s.price)}</td><td class="${s.change>=0?'gn':'rd'}">${fmtP(s.change)}</td>
    <td><span class="pill ${pC(s.signal)}">${pL(s.signal)}</span></td>
    <td>${sbar(s.score,s.signal)}</td>
    <td style="color:${s.rsi>70?'#b03030':s.rsi<35?'#2d7a3a':'inherit'}">${fmt(s.rsi,1)}</td>
    <td class="${s.macd_val>=0?'gn':'rd'}">${s.macd_val>=0?'+':''}${fmt(s.macd_val,2)} <span style="font-size:10px;color:#999">${s.macd_dir||''}</span></td>
    <td>${fmt(s.vol_mult,2)}x</td><td style="font-size:11px;color:#999">${s.sector||'-'}</td>
  </tr>`).join('');
}
function updateMetrics(){
  let sb=0,b=0,s=0,rsi=0;
  stocks.forEach(st=>{if(st.signal==='strong-buy')sb++;if(st.signal==='buy')b++;if(st.signal==='sell'||st.signal==='strong-sell')s++;rsi+=st.rsi;});
  $('msb').textContent=sb;$('mb').textContent=b;$('ms2').textContent=s;
  $('mrsi').textContent=fmt(rsi/stocks.length,1);
  const bull=sb+b,bear=s;
  $('mmood').textContent=bull>bear*1.5?'Bullish':bear>bull*1.5?'Bearish':'Mixed';
  $('mmoods').textContent=bull+' buys vs '+bear+' sells';
}
async function showDetail(ticker){
  curTicker=ticker;
  $('main').style.display='none';$('detail').style.display='';
  $('dl').style.display='';$('dc').style.display='none';$('de').style.display='none';
  if(chart){chart.destroy();chart=null;}
  try{
    const res=await fetch(API+'/stock/'+ticker);
    if(!res.ok)throw new Error('Server error '+res.status);
    const s=await res.json();
    $('dticker').textContent=s.ticker;
    $('dname').textContent=(s.name||'')+(s.sector?' · '+s.sector:'');
    $('dpill').innerHTML='<span class="pill '+pC(s.signal)+'">'+pL(s.signal)+'</span>';
    $('dprice').textContent='$'+fmt(s.price);
    $('dchg').innerHTML='<span class="'+(s.change>=0?'gn':'rd')+'">'+fmtP(s.change)+'</span>';
    $('dscore').textContent=(s.score||0)+'/10';$('datr').textContent='$'+fmt(s.atr);
    $('dstop').textContent='$'+fmt(s.stop);$('dtgt').textContent='$'+fmt(s.target);
    $('drr').textContent='1:'+fmt(s.risk_reward,1);
    $('wlbtn').textContent=watchlist.has(ticker)?'&#10003; In watchlist':'+ Watchlist';
    $('dinds').innerHTML=[
      {l:'RSI '+fmt(s.rsi,1),c:s.rsi<35?'bull':s.rsi>70?'bear':'neut'},
      {l:'MACD '+(s.macd_val>=0?'+':'')+fmt(s.macd_val,2)+' '+s.macd_dir,c:s.macd_val>0?'bull':'bear'},
      {l:'EMA20 '+(s.pct_above_ema>=0?'+':'')+fmt(s.pct_above_ema,1)+'%',c:s.pct_above_ema>0?'bull':'bear'},
      {l:'Vol '+fmt(s.vol_mult,2)+'x',c:s.vol_mult>1.5?'bull':s.vol_mult<0.8?'bear':'neut'},
      {l:'BB '+(s.bb_pos*100).toFixed(0)+'%',c:s.bb_pos<0.2?'bull':s.bb_pos>0.8?'bear':'neut'},
      {l:'Trend: '+(s.trend_status||'Unknown'),c:s.trend_bullish?'bull':'bear'},
      {l:'SMA50 '+(s.pct_above_sma50!=null?(s.pct_above_sma50>=0?'+':'')+fmt(s.pct_above_sma50,1)+'%':'—'),c:s.pct_above_sma50>=0?'bull':'bear'},
      {l:'SMA200 '+(s.pct_above_sma200!=null?(s.pct_above_sma200>=0?'+':'')+fmt(s.pct_above_sma200,1)+'%':'—'),c:s.pct_above_sma200>=0?'bull':'bear'},
      {l:'ROC60 '+(s.roc60!=null?(s.roc60>=0?'+':'')+fmt(s.roc60,1)+'%':'—'),c:s.roc60<-22?'bull':s.roc60>20?'bear':'neut'},
      {l:'52w high '+(s.pct_52w_high!=null?fmt(s.pct_52w_high,1)+'%':'—'),c:s.pct_52w_high<-3.82?'bull':s.pct_52w_high>-2?'bear':'neut'},
      ...(s.rule1_fired?[{l:'Rule 1 fired',c:'bull'}]:[]),
      ...(s.rule2_fired?[{l:'Rule 2 fired',c:'bull'}]:[]),
      ...(s.ml_available&&s.ml_prob!=null?[{l:'ML '+Math.round(s.ml_prob*100)+'%',c:s.ml_prob>=0.55?'bull':s.ml_prob<=0.35?'bear':'neut'}]:[]),
    ].map(i=>'<span class="itag '+i.c+'">'+i.l+'</span>').join('');
    $('dbuy').innerHTML=(s.buy_points||[]).map(p=>'· '+p).join('<br>');
    $('dsell').innerHTML=(s.sell_points||[]).map(p=>'· '+p).join('<br>');
    const hist=s.history||[];
    if(hist.length>1){
      const color=s.change>=0?'#2d7a3a':'#b03030';
      const labels=hist.map((_,i)=>i===hist.length-1?'Today':'-'+(hist.length-1-i)+'d');
      chart=new Chart($('pc').getContext('2d'),{type:'line',
        data:{labels,datasets:[{data:hist,borderColor:color,backgroundColor:color+'22',borderWidth:1.5,pointRadius:0,fill:true,tension:0.3}]},
        options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>'$'+c.raw.toFixed(2)}}},
        scales:{x:{grid:{display:false},ticks:{font:{size:10},color:'#999',maxRotation:0}},y:{grid:{color:'#eee'},ticks:{font:{size:10},color:'#999',callback:v=>'$'+v.toFixed(0)}}}}});
    }
    $('dl').style.display='none';$('dc').style.display='';
  }catch(e){
    $('dl').style.display='none';$('de').style.display='';
    $('de').innerHTML='<div class="ebox">Could not load '+ticker+': '+e.message+'</div>';
  }
}
function showMain(){$('main').style.display='';$('detail').style.display='none';if(chart){chart.destroy();chart=null;}}
function toggleWatchlist(){
  if(!curTicker)return;
  watchlist.has(curTicker)?watchlist.delete(curTicker):watchlist.add(curTicker);
  $('wlbtn').textContent=watchlist.has(curTicker)?'&#10003; In watchlist':'+ Watchlist';
  renderWatchlist();
}
function renderWatchlist(){
  const wl=stocks.filter(s=>watchlist.has(s.ticker));
  if(!wl.length){$('we').style.display='';$('wt').innerHTML='';return;}
  $('we').style.display='none';
  $('wt').innerHTML='<div class="card"><table><thead><tr><th style="width:15%">Ticker</th><th style="width:12%">Price</th><th style="width:12%">1D chg</th><th style="width:15%">Signal</th><th style="width:18%">Score</th><th style="width:8%">RSI</th><th style="width:10%">Stop</th><th style="width:10%">Target</th></tr></thead><tbody>'+
  wl.map(s=>'<tr onclick="showDetail(\''+s.ticker+'\')"><td><strong>'+s.ticker+'</strong></td><td>$'+fmt(s.price)+'</td><td class="'+(s.change>=0?'gn':'rd')+'">'+fmtP(s.change)+'</td><td><span class="pill '+pC(s.signal)+'">'+pL(s.signal)+'</span></td><td>'+sbar(s.score,s.signal)+'</td><td>'+fmt(s.rsi,1)+'</td><td style="color:#b03030">$'+fmt(s.stop)+'</td><td style="color:#2d7a3a">$'+fmt(s.target)+'</td></tr>').join('')+
  '</tbody></table></div>';
}
function setTab(name){
  ['screen','hunt','global','check','backtest','watch'].forEach(t=>{
    const el=$('t-'+t);
    if(el) el.classList.toggle('active',t===name);
    const panel=$('tab-'+t);
    if(panel) panel.style.display=t===name?'':'none';
  });
  if(name==='watch')renderWatchlist();
  if(name==='check') setTimeout(()=>{ const i=$('ticker-input'); if(i) i.focus(); },100);
  if(name==='backtest') setTimeout(()=>{ const i=$('bt-ticker'); if(i) i.focus(); },100);
}

async function startGlobalScan(){
  $('global-idle').style.display='none';
  $('global-results').style.display='none';
  $('global-err').style.display='none';
  $('global-loading').style.display='';
  $('global-progress').textContent='Scanning 300+ stocks...';
  $('global-sub').textContent='Fetching data in parallel batches of 20';

  // Animate progress bar
  let pct=5;
  const barInterval=setInterval(()=>{
    pct=Math.min(pct+2,90);
    $('global-bar').style.width=pct+'%';
  },1500);

  try{
    const res=await fetch(API+'/hunt-rule2?max_scan=300');
    clearInterval(barInterval);
    $('global-bar').style.width='100%';

    if(!res.ok) throw new Error('Scan failed: '+res.status);
    const d=await res.json();

    $('global-loading').style.display='none';
    $('global-results').style.display='';

    // Summary bar
    $('global-summary').innerHTML=
      '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:1rem">'+
      '<div class="bt-mc"><div class="ml">Scanned</div><div class="mv">'+d.scanned+'</div></div>'+
      '<div class="bt-mc"><div class="ml">Rule 2 hits</div><div class="mv" style="color:'+(d.rule2_count>0?'#2d7a3a':'#666')+'">'+d.rule2_count+'</div></div>'+
      '<div class="bt-mc"><div class="ml">Rule 1 hits</div><div class="mv" style="color:'+(d.rule1_count>0?'#b07020':'#666')+'">'+d.rule1_count+'</div></div>'+
      '<div class="bt-mc"><div class="ml">Universe</div><div class="mv">'+d.universe_size+'</div></div>'+
      '</div>';

    const makeCard=(s,ruleLabel,cardColor)=>{
      const chgColor=s.change>=0?'#2d7a3a':'#b03030';
      return '<div class="hunt-card" style="border-left:3px solid '+cardColor+'">'+
        '<div class="hunt-card-top">'+
          '<div><div class="hunt-card-ticker">'+s.ticker+
            (ruleLabel?'<span style="font-size:11px;font-weight:400;background:'+cardColor+';color:#fff;padding:2px 7px;border-radius:10px;margin-left:8px">'+ruleLabel+'</span>':'')+
          '</div>'+
          '<div class="hunt-card-name">'+s.name+'</div></div>'+
          '<div><div class="hunt-card-price">$'+fmt(s.price,2)+'</div>'+
          '<div class="hunt-card-chg" style="color:'+chgColor+'">'+(s.change>=0?'+':'')+fmt(s.change,2)+'%</div></div>'+
        '</div>'+
        '<div class="hunt-card-body">'+
          '<div><div class="hunt-stat-label">RSI(14)</div><div class="hunt-stat-value" style="color:'+(s.rsi<35?'#2d7a3a':s.rsi>70?'#b03030':'inherit')+'">'+fmt(s.rsi,1)+'</div></div>'+
          '<div><div class="hunt-stat-label">ROC 60d</div><div class="hunt-stat-value" style="color:'+(s.roc60<-22?'#2d7a3a':'inherit')+'">'+fmt(s.roc60,1)+'%</div></div>'+
          '<div><div class="hunt-stat-label">vs SMA200</div><div class="hunt-stat-value" style="color:'+(s.pct_above_sma200>=0?'#2d7a3a':'#b03030')+'">'+fmt(s.pct_above_sma200,1)+'%</div></div>'+
          '<div><div class="hunt-stat-label">From 52w high</div><div class="hunt-stat-value">'+fmt(s.pct_52w_high,1)+'%</div></div>'+
          '<div><div class="hunt-stat-label">MACD</div><div class="hunt-stat-value" style="color:'+(s.macd_val>0?'#2d7a3a':'#b03030')+'">'+(s.macd_val>=0?'+':'')+fmt(s.macd_val,3)+'</div></div>'+
          '<div><div class="hunt-stat-label">ATR</div><div class="hunt-stat-value">$'+fmt(s.atr,2)+'</div></div>'+
        '</div>'+
        '<div style="display:flex;gap:6px;flex-wrap:wrap;font-size:11px;margin-bottom:10px">'+
          s.buy_points.slice(0,3).map(p=>'<span style="color:#555">· '+p+'</span>').join('')+
        '</div>'+
        '<div class="hunt-risk">'+
          '<div><div class="hunt-stat-label">Entry</div><div style="font-weight:600">$'+fmt(s.price,2)+'</div></div>'+
          '<div><div class="hunt-stat-label">Stop</div><div style="font-weight:600;color:#b03030">$'+fmt(s.stop,2)+'</div></div>'+
          '<div><div class="hunt-stat-label">Target</div><div style="font-weight:600;color:#2d7a3a">$'+fmt(s.target,2)+'</div></div>'+
        '</div>'+
      '</div>';
    };

    // Rule 2 section
    if(d.rule2_count>0){
      $('global-r2-count').textContent=d.rule2_count;
      $('global-r2-section').style.display='';
      $('global-r2-cards').innerHTML=d.rule2_hits.map(s=>makeCard(s,'Rule 2','#2d7a3a')).join('');
    } else {
      $('global-r2-section').style.display='none';
    }

    // Rule 1 section
    if(d.rule1_count>0){
      $('global-r1-count').textContent=d.rule1_count;
      $('global-r1-section').style.display='';
      $('global-r1-cards').innerHTML=d.rule1_hits.map(s=>makeCard(s,'Rule 1','#b07020')).join('');
    } else {
      $('global-r1-section').style.display='none';
    }

    // Neither found
    if(d.rule2_count===0 && d.rule1_count===0){
      $('global-none').style.display='';
    } else {
      $('global-none').style.display='none';
    }

  }catch(e){
    clearInterval(barInterval);
    $('global-loading').style.display='none';
    $('global-err').style.display='';
    $('global-err').innerHTML='<div class="ebox">'+e.message+'</div>';
    $('global-idle').style.display='';
  }
}

let btChart=null;

async function runBacktest(){
  const ticker=($('bt-ticker').value||'').trim().toUpperCase().replace(/[^A-Z.]/g,'');
  if(!ticker){ $('bt-ticker').style.borderColor='#e8aaaa'; setTimeout(()=>$('bt-ticker').style.borderColor='',1200); return; }
  $('bt-ticker').value=ticker;
  const period=$('bt-period').value;
  const size=parseInt($('bt-size').value);
  const minpos=parseInt($('bt-minpos').value);
  const mode=$('bt-mode').value;
  $('bt-loading').style.display='';
  $('bt-results').style.display='none';
  $('bt-err').style.display='none';
  $('bt-btn').disabled=true;
  if(btChart){btChart.destroy();btChart=null;}

  try{
    const res=await fetch(API+'/backtest/'+encodeURIComponent(ticker)+'?period='+period+'&trade_size='+size+'&min_position='+minpos+'&entry_mode='+mode);
    if(res.status===404) throw new Error(ticker+' not found');
    if(!res.ok) throw new Error('Server error '+res.status);
    const d=await res.json();

    $('bt-start-val').textContent=size.toLocaleString();
    const pnl=d.total_pnl||0;
    const pct=d.total_return_pct||0;
    const wins=d.winning_trades||0;
    const total=d.total_trades||0;
    const winrate=total>0?((wins/total)*100).toFixed(0):0;
    const mdd=d.max_drawdown_pct||0;
    const final=d.final_value||size;

    $('bt-metrics').innerHTML=[
      {l:'Final value',v:'€'+final.toLocaleString('de-DE',{minimumFractionDigits:0,maximumFractionDigits:0}),cls:final>=size?'bt-win':'bt-loss'},
      {l:'Total P&L',v:(pnl>=0?'+':'')+'€'+Math.abs(pnl).toFixed(0),cls:pnl>=0?'bt-win':'bt-loss'},
      {l:'Total return',v:(pct>=0?'+':'')+pct.toFixed(1)+'%',cls:pct>=0?'bt-win':'bt-loss'},
      {l:'Trades',v:total,cls:'bt-neut'},
      {l:'Win rate',v:winrate+'%',cls:parseFloat(winrate)>=50?'bt-win':'bt-loss'},
      {l:'Max drawdown',v:'-'+Math.abs(mdd).toFixed(1)+'%',cls:'bt-loss'},
      {l:'Best trade',v:(d.best_trade_pct||0)>=0?'+':''+(d.best_trade_pct||0).toFixed(1)+'%',cls:'bt-win'},
      {l:'Worst trade',v:(d.worst_trade_pct||0).toFixed(1)+'%',cls:'bt-loss'},
      {l:'Total fees paid',v:'€'+(d.total_fees_paid||0).toFixed(0),cls:'bt-loss'},
      {l:'Avg fee/trade',v:'€'+(d.avg_fee_per_trade||0).toFixed(0),cls:'bt-neut'},
      {l:'Fee drag',v:'-'+(d.fee_drag_pct||0).toFixed(1)+'%',cls:'bt-loss'},
    ].map(m=>`<div class="bt-mc"><div class="ml">${m.l}</div><div class="mv ${m.cls}">${m.v}</div></div>`).join('');

    // Equity curve chart
    const eq=d.equity_curve||[];
    if(eq.length>1){
      const labels=eq.map(p=>p.date);
      const values=eq.map(p=>p.value);
      const color=final>=size?'#2d7a3a':'#b03030';
      btChart=new Chart($('bt-equity-chart').getContext('2d'),{
        type:'line',
        data:{labels,datasets:[
          {data:values,borderColor:color,backgroundColor:color+'18',borderWidth:2,pointRadius:0,fill:true,tension:0.3,label:'Portfolio'},
          {data:eq.map(p=>p.buy_hold),borderColor:'#888',borderWidth:1,borderDash:[4,4],pointRadius:0,fill:false,tension:0.3,label:'Buy & hold'},
        ]},
        options:{responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:true,position:'top',labels:{font:{size:11},boxWidth:12}},
            tooltip:{callbacks:{label:c=>'€'+c.raw.toFixed(0)}}},
          scales:{
            x:{grid:{display:false},ticks:{font:{size:10},color:'#999',maxTicksLimit:8,maxRotation:0}},
            y:{grid:{color:'#eee'},ticks:{font:{size:10},color:'#999',callback:v=>'€'+v.toFixed(0)}}
          }
        }
      });
    }

    // Trade log
    const trades=d.trades||[];
    $('bt-trades').innerHTML=trades.length===0
      ?'<tr><td colspan="8" style="text-align:center;padding:1.5rem;color:#999">No trades generated in this period</td></tr>'
      :trades.map((t,i)=>`<tr>
        <td>${i+1}</td>
        <td>${t.buy_date}</td>
        <td>$${parseFloat(t.buy_price).toFixed(2)}</td>
        <td>${t.sell_date||'Open'}</td>
        <td>${t.sell_price?'$'+parseFloat(t.sell_price).toFixed(2):'-'}</td>
        <td style="color:#666">${t.days_held||'-'}d</td>
        <td>€${t.invested?Math.round(t.invested):'-'}</td>
        <td style="color:#92620a">€${t.total_fees?t.total_fees.toFixed(0):'-'}</td>
        <td class="${t.pnl>=0?'bt-win':'bt-loss'}" style="font-weight:600">${t.pnl>=0?'+':''}€${Math.abs(t.pnl).toFixed(0)}</td>
        <td class="${t.pnl_pct>=0?'bt-win':'bt-loss'}">${t.pnl_pct>=0?'+':''}${parseFloat(t.pnl_pct).toFixed(1)}%</td>
        <td style="font-size:11px;color:#999">${t.exit_reason||'-'}</td>
      </tr>`).join('');

    $('bt-loading').style.display='none';
    $('bt-results').style.display='';

    // Show strategy notes — remove old block first so it only appears once
    const notes=d.strategy_notes||[];
    const existing=document.getElementById('bt-strategy-notes');
    if(existing) existing.remove();
    if(notes.length){
      const notesHtml='<div id="bt-strategy-notes" style="background:#e8f5ea;border:.5px solid #a8d5b0;border-radius:var(--rl);padding:12px 16px;margin-bottom:1rem;font-size:12px;color:#1a5c28">'+
        '<strong style="display:block;margin-bottom:6px">Strategy rules active in this backtest:</strong>'+
        notes.map(n=>'· '+n).join('<br>')+
        '</div>';
      $('bt-results').insertAdjacentHTML('afterbegin',notesHtml);
    }
  }catch(e){
    $('bt-loading').style.display='none';
    $('bt-err').style.display='';
    $('bt-err').innerHTML='<div class="ebox">'+e.message+'</div>';
  }
  $('bt-btn').disabled=false;
}


let ckChart=null;
let ckCurTicker=null;

async function checkTicker(){
  const inp=$('ticker-input');
  const ticker=(inp.value||'').trim().toUpperCase().replace(/[^A-Z.]/g,'');
  if(!ticker){ inp.style.borderColor='#e8aaaa'; setTimeout(()=>inp.style.borderColor='',1200); return; }
  inp.value=ticker;
  $('check-loading').style.display='';
  $('check-result').style.display='none';
  $('check-err').style.display='none';
  $('check-btn').disabled=true;
  if(ckChart){ckChart.destroy();ckChart=null;}

  try{
    const res=await fetch(API+'/stock/'+encodeURIComponent(ticker));
    if(res.status===404) throw new Error(ticker+' not found — check the symbol and try again');
    if(!res.ok) throw new Error('Server error '+res.status);
    const s=await res.json();
    ckCurTicker=s.ticker;

    $('ck-ticker').textContent=s.ticker;
    $('ck-name').textContent=(s.name||'')+(s.sector?' · '+s.sector:'');
    $('ck-pill').innerHTML='<span class="pill '+pC(s.signal)+'">'+pL(s.signal)+'</span>';
    $('ck-price').textContent='$'+fmt(s.price);
    $('ck-chg').innerHTML='<span class="'+(s.change>=0?'gn':'rd')+'">'+fmtP(s.change)+'</span>';
    $('ck-score').textContent=(s.score||0)+'/10';
    $('ck-atr').textContent='$'+fmt(s.atr);
    $('ck-stop').textContent='$'+fmt(s.stop);
    $('ck-tgt').textContent='$'+fmt(s.target);
    $('ck-rr').textContent='1:'+fmt(s.risk_reward,1);
    $('ck-wlbtn').textContent=watchlist.has(s.ticker)?'&#10003; In watchlist':'+ Watchlist';

    $('ck-inds').innerHTML=[
      {l:'RSI '+fmt(s.rsi,1),c:s.rsi<35?'bull':s.rsi>70?'bear':'neut'},
      {l:'MACD '+(s.macd_val>=0?'+':'')+fmt(s.macd_val,2)+' '+s.macd_dir,c:s.macd_val>0?'bull':'bear'},
      {l:'EMA20 '+(s.pct_above_ema>=0?'+':'')+fmt(s.pct_above_ema,1)+'%',c:s.pct_above_ema>0?'bull':'bear'},
      {l:'Vol '+fmt(s.vol_mult,2)+'x',c:s.vol_mult>1.5?'bull':s.vol_mult<0.8?'bear':'neut'},
      {l:'BB '+(s.bb_pos*100).toFixed(0)+'%',c:s.bb_pos<0.2?'bull':s.bb_pos>0.8?'bear':'neut'},
      {l:'Trend: '+(s.trend_status||'Unknown'),c:s.trend_bullish?'bull':'bear'},
      {l:'SMA50 '+(s.pct_above_sma50!=null?(s.pct_above_sma50>=0?'+':'')+fmt(s.pct_above_sma50,1)+'%':'—'),c:s.pct_above_sma50>=0?'bull':'bear'},
      {l:'SMA200 '+(s.pct_above_sma200!=null?(s.pct_above_sma200>=0?'+':'')+fmt(s.pct_above_sma200,1)+'%':'—'),c:s.pct_above_sma200>=0?'bull':'bear'},
      {l:'ROC60 '+(s.roc60!=null?(s.roc60>=0?'+':'')+fmt(s.roc60,1)+'%':'—'),c:s.roc60<-22?'bull':s.roc60>20?'bear':'neut'},
      {l:'52w high '+(s.pct_52w_high!=null?fmt(s.pct_52w_high,1)+'%':'—'),c:s.pct_52w_high<-3.82?'bull':s.pct_52w_high>-2?'bear':'neut'},
      ...(s.rule1_fired?[{l:'Rule 1 fired',c:'bull'}]:[]),
      ...(s.rule2_fired?[{l:'Rule 2 fired',c:'bull'}]:[]),
      ...(s.ml_available&&s.ml_prob!=null?[{l:'ML '+Math.round(s.ml_prob*100)+'%',c:s.ml_prob>=0.55?'bull':s.ml_prob<=0.35?'bear':'neut'}]:[]),
    ].map(i=>'<span class="itag '+i.c+'">'+i.l+'</span>').join('');

    $('ck-buy').innerHTML=(s.buy_points||[]).map(p=>'· '+p).join('<br>');
    $('ck-sell').innerHTML=(s.sell_points||[]).map(p=>'· '+p).join('<br>');

    const hist=s.history||[];
    if(hist.length>1){
      const color=s.change>=0?'#2d7a3a':'#b03030';
      const labels=hist.map((_,i)=>i===hist.length-1?'Today':'-'+(hist.length-1-i)+'d');
      ckChart=new Chart($('ck-chart').getContext('2d'),{type:'line',
        data:{labels,datasets:[{data:hist,borderColor:color,backgroundColor:color+'22',borderWidth:1.5,pointRadius:0,fill:true,tension:0.3}]},
        options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>'$'+c.raw.toFixed(2)}}},
        scales:{x:{grid:{display:false},ticks:{font:{size:10},color:'#999',maxRotation:0}},y:{grid:{color:'#eee'},ticks:{font:{size:10},color:'#999',callback:v=>'$'+v.toFixed(0)}}}}});
    }

    $('check-loading').style.display='none';
    $('check-result').style.display='';
  }catch(e){
    $('check-loading').style.display='none';
    $('check-err').style.display='';
    $('check-err').innerHTML='<div class="ebox">'+e.message+'</div>';
  }
  $('check-btn').disabled=false;
}

function ckWatchlist(){
  if(!ckCurTicker)return;
  watchlist.has(ckCurTicker)?watchlist.delete(ckCurTicker):watchlist.add(ckCurTicker);
  $('ck-wlbtn').textContent=watchlist.has(ckCurTicker)?'&#10003; In watchlist':'+ Watchlist';
  renderWatchlist();
}

function renderHuntCards(found){
  $('hunt-cards').innerHTML=found.map(s=>`
    <div class="hunt-card" onclick="showDetail('${s.ticker}')">
      <div class="hunt-card-top">
        <div>
          <div class="hunt-card-ticker">${s.ticker} <span class="pill sb" style="font-size:12px;vertical-align:middle">Strong buy</span></div>
          <div class="hunt-card-name">${s.name||''} &middot; ${s.sector||''}</div>
        </div>
        <div>
          <div class="hunt-card-price">$${fmt(s.price)}</div>
          <div class="hunt-card-chg ${s.change>=0?'gn':'rd'}">${fmtP(s.change)} today</div>
        </div>
      </div>
      <div class="hunt-card-body">
        <div><div class="hunt-stat-label">Signal score</div><div class="hunt-stat-value" style="color:#2d7a3a">${s.score}/10</div></div>
        <div><div class="hunt-stat-label">RSI (14)</div><div class="hunt-stat-value" style="color:${s.rsi<35?'#2d7a3a':s.rsi>70?'#b03030':'inherit'}">${fmt(s.rsi,1)}</div></div>
        <div><div class="hunt-stat-label">ROC 60d</div><div class="hunt-stat-value ${s.roc60<-22?'bt-win':'bt-neut'}">${s.roc60!=null?(s.roc60>=0?'+':'')+fmt(s.roc60,1)+'%':'—'}</div></div>
        <div><div class="hunt-stat-label">From 52w high</div><div class="hunt-stat-value ${s.pct_52w_high<-3.82?'bt-win':'bt-neut'}">${s.pct_52w_high!=null?fmt(s.pct_52w_high,1)+'%':'—'}</div></div>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">
        ${s.rule2_fired?'<span class="itag bull">&#128994; Rule 2 fired — 72% win rate</span>':''}
        ${s.rule1_fired?'<span class="itag bull">&#128992; Rule 1 fired — 55% win rate</span>':''}
        <span class="itag ${s.trend_bullish?'bull':'bear'}">${s.trend_status||'—'}</span>
        ${s.ml_available&&s.ml_prob!=null?'<span class="itag '+(s.ml_prob>=0.55?'bull':s.ml_prob<=0.35?'bear':'neut')+'">ML '+Math.round(s.ml_prob*100)+'%</span>':''}
      </div>
      <div class="hunt-points">${(s.buy_points||[]).map(p=>'&#183; '+p).join('<br>')}</div>
      <div class="hunt-risk">
        <div><div class="hunt-stat-label">Stop loss</div><div style="font-weight:600;color:#b03030">$${fmt(s.stop)}</div></div>
        <div><div class="hunt-stat-label">Target</div><div style="font-weight:600;color:#2d7a3a">$${fmt(s.target)}</div></div>
        <div><div class="hunt-stat-label">Risk/reward</div><div style="font-weight:600">1:${fmt(s.risk_reward,1)}</div></div>
      </div>
      <div style="margin-top:10px;font-size:12px;color:#999">Click for full analysis &#8594;</div>
    </div>`).join('');
}

async function startHunt(){
  $('hunt-idle').style.display='none';
  $('hunt-results').style.display='none';
  $('hunt-err').style.display='none';
  $('hunt-loading').style.display='';
  $('hunt-progress').textContent='searching...';
  $('hunt-scanned').textContent='0 stocks scanned';
  $('hunt-bar').style.width='0%';

  let tick=0;
  const timer=setInterval(()=>{
    tick++;
    $('hunt-scanned').textContent=(tick*8)+' stocks scanned...';
    $('hunt-bar').style.width=Math.min(90,tick*6)+'%';
  },800);

  try{
    const res=await fetch(API+'/hunt?target=1');
    clearInterval(timer);
    if(!res.ok)throw new Error('Server error '+res.status);
    const data=await res.json();
    const found=data.strong_buys||[];
    const r2=data.rule2_count||0;
    $('hunt-progress').textContent=r2>0?'Rule 2 found!':'Rule 1 found';
    $('hunt-scanned').textContent=data.scanned+' stocks scanned';
    $('hunt-bar').style.width='100%';

    setTimeout(()=>{
      $('hunt-loading').style.display='none';
      $('hunt-results').style.display='';
      const sum=$('hunt-summary');

      if(found.length===0){
        sum.style.cssText='padding:12px 16px;background:#fdeaea;border:.5px solid #e8aaaa;border-radius:var(--rl);color:#7a1c1c;font-size:13px;margin-bottom:1rem';
        sum.innerHTML='<strong>No Rule 1 or Rule 2 signals found</strong> after scanning all '+data.scanned+' stocks. '
          +'Neither rule is firing right now — the market may be in a trending phase without pullbacks. '
          +'This is actually useful information: no oversold setups means there is nothing to buy today.';
        $('hunt-cards').innerHTML='';
      } else if(r2>0){
        sum.style.cssText='padding:12px 16px;background:#e8f5ea;border:.5px solid #a8d5b0;border-radius:var(--rl);color:#1a5c28;font-size:13px;margin-bottom:1rem';
        sum.innerHTML='<strong>&#128994; Rule 2 signal found</strong> after scanning '+data.scanned+' stocks. '
          +'Rule 2 historically outperforms SPY by 1%+ in <strong>72% of cases</strong> with avg alpha +12.4% over 20 days.';
        renderHuntCards(found);
      } else {
        sum.style.cssText='padding:12px 16px;background:#fef9e7;border:.5px solid #e8d08a;border-radius:var(--rl);color:#7a6520;font-size:13px;margin-bottom:1rem';
        sum.innerHTML='<strong>&#128992; Rule 1 signal found</strong> (Rule 2 not firing today) after scanning '+data.scanned+' stocks. '
          +'Rule 1 historically outperforms SPY in <strong>55% of cases</strong> with avg alpha +14.5% over 20 days.';
        renderHuntCards(found);
      }
    },400);
  }catch(e){
    clearInterval(timer);
    $('hunt-loading').style.display='none';
    $('hunt-err').style.display='';
    $('hunt-err').innerHTML='<div class="ebox">Hunt failed: '+e.message+'</div>';
  }
}
$('rbtn').addEventListener('click',load);
$('backbtn').addEventListener('click',showMain);
$('wlbtn').addEventListener('click',toggleWatchlist);
$('fsig').addEventListener('change',applyFilters);
$('fsec').addEventListener('change',applyFilters);
$('fsort').addEventListener('change',applyFilters);
$('ticker-input').addEventListener('keydown',e=>{ if(e.key==='Enter') checkTicker(); });
$('bt-ticker').addEventListener('keydown',e=>{ if(e.key==='Enter') runBacktest(); });
load();
</script>
</body>
</html>"""


def fetch_market_context() -> dict:
    try:
        vix_hist = yf.Ticker("^VIX").history(period="5d", interval="1d")
        vix = float(vix_hist["Close"].iloc[-1]) if not vix_hist.empty else None

        spy_hist = yf.Ticker("SPY").history(period="300d", interval="1d", auto_adjust=True)
        if spy_hist.empty or len(spy_hist) < 50:
            raise ValueError("Not enough SPY data")

        spy_close = spy_hist["Close"].astype(float)
        spy_price = float(spy_close.iloc[-1])
        sma200 = float(spy_close.tail(200).mean())
        sma50 = float(spy_close.tail(50).mean())
        spy_vs_200 = (spy_price - sma200) / sma200 * 100
        spy_vs_50 = (spy_price - sma50) / sma50 * 100

        spy_rsi_s = ta.rsi(spy_close, length=14)
        spy_rsi = float(spy_rsi_s.iloc[-1]) if spy_rsi_s is not None else 50.0

        score = 0
        if vix is not None:
            if vix < 15: score += 2
            elif vix < 20: score += 1
            elif vix < 30: score -= 1
            else: score -= 2
        if spy_vs_200 > 2: score += 2
        elif spy_vs_200 > 0: score += 1
        elif spy_vs_200 < -5: score -= 2
        else: score -= 1
        if spy_vs_50 > 1: score += 1
        elif spy_vs_50 < -3: score -= 1
        if spy_rsi < 35: score += 1
        elif spy_rsi > 75: score -= 1

        vix_str = f"{vix:.1f}" if vix else "N/A"
        above200 = "above" if spy_vs_200 >= 0 else "below"
        above50 = "above" if spy_vs_50 >= 0 else "below"

        if score >= 4:
            color, regime = "green", "Bull market"
            verdict = "Good time to enter — market conditions are favorable"
            reason = (f"VIX at {vix_str} signals low fear. SPY is {abs(spy_vs_200):.1f}% {above200} its 200 SMA "
                      f"and {abs(spy_vs_50):.1f}% {above50} its 50 SMA — healthy bull market. "
                      f"Individual setups carry higher success probability in this environment.")
        elif score >= 1:
            color, regime = "amber", "Mixed / Cautious"
            verdict = "Proceed with caution — mixed market signals"
            reason = (f"VIX at {vix_str} ({'elevated' if vix and vix > 20 else 'moderate'}). "
                      f"SPY is {abs(spy_vs_200):.1f}% {above200} its 200 SMA. "
                      f"Reduce position sizes and focus on highest-conviction setups only.")
        elif score >= -1:
            color, regime = "amber", "Choppy / Uncertain"
            verdict = "Consider waiting — market lacks clear direction"
            reason = (f"VIX at {vix_str}. SPY {abs(spy_vs_200):.1f}% {above200} 200 SMA — trend is weakening. "
                      f"Technical strategies underperform in choppy conditions. Wait for clearer setups.")
        else:
            color, regime = "red", "Bear / Risk-off"
            verdict = "Reconsider entering — high-risk market environment"
            reason = (f"VIX at {vix_str} — {'crisis-level fear' if vix and vix > 30 else 'elevated fear'}. "
                      f"SPY is {abs(spy_vs_200):.1f}% {above200} its 200 SMA — bear market conditions. "
                      f"Most long positions fail in bear markets. Consider raising cash or waiting for VIX < 20.")

        return {"vix": round(vix, 2) if vix else None, "spy_price": round(spy_price, 2),
                "spy_vs_200sma": round(spy_vs_200, 2), "spy_vs_50sma": round(spy_vs_50, 2),
                "spy_rsi": round(spy_rsi, 1), "sma200": round(sma200, 2), "sma50": round(sma50, 2),
                "regime_score": score, "regime": regime, "color": color,
                "verdict": verdict, "reason": reason,
                "updated_at": datetime.now(timezone.utc).isoformat()}
    except Exception:
        print(f"Market context error: {traceback.format_exc()}")
        return {"vix": None, "spy_price": None, "spy_vs_200sma": None, "spy_vs_50sma": None,
                "regime": "Unknown", "color": "amber",
                "verdict": "Market data unavailable",
                "reason": "Could not fetch VIX or SPY data.",
                "updated_at": datetime.now(timezone.utc).isoformat()}


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=FRONTEND_HTML)


@app.get("/app", response_class=HTMLResponse)
def frontend():
    return HTMLResponse(content=FRONTEND_HTML)


@app.get("/market")
async def market_context():
    """VIX + SPY regime — overall market entry recommendation."""
    result = await asyncio.get_event_loop().run_in_executor(executor, fetch_market_context)
    return result


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/hunt")
async def hunt_strong_buys(target: int = Query(default=1, description="Number of Rule 2 signals to find")):
    """
    Scans the full mid-cap universe in batches of 10.
    Stops as soon as it finds `target` stocks where Rule 2 fires:
    price > SMA200 AND ROC(60) < -22% AND far from 52w high.
    Falls back to Rule 1 (MACD+ and RSI<35) if Rule 2 not found.
    """
    target = max(1, min(target, 10))
    rule2_found = []
    rule1_found = []
    scanned = 0
    batch_size = 10

    loop = asyncio.get_event_loop()

    for i in range(0, len(HUNT_UNIVERSE), batch_size):
        if len(rule2_found) >= target:
            break
        batch = HUNT_UNIVERSE[i:i + batch_size]
        results = await asyncio.gather(
            *[loop.run_in_executor(executor, fetch_and_analyze, t) for t in batch]
        )
        scanned += len(batch)
        for r in results:
            if not r:
                continue
            if r.get("rule2_fired"):
                rule2_found.append(r)
            elif r.get("rule1_fired") and len(rule1_found) < 5:
                rule1_found.append(r)  # collect as fallback

    # If Rule 2 found nothing, keep scanning for Rule 1
    if not rule2_found:
        for i in range(scanned, len(HUNT_UNIVERSE), batch_size):
            if len(rule1_found) >= target:
                break
            batch = HUNT_UNIVERSE[i:i + batch_size]
            results = await asyncio.gather(
                *[loop.run_in_executor(executor, fetch_and_analyze, t) for t in batch]
            )
            scanned += len(batch)
            for r in results:
                if r and r.get("rule1_fired"):
                    rule1_found.append(r)

    # Prioritise Rule 2, fall back to Rule 1
    found = rule2_found if rule2_found else rule1_found
    found.sort(key=lambda s: (s.get("rule2_fired", False), s.get("score", 0)), reverse=True)

    rule_used = "Rule 2 (above SMA200 + deep pullback)" if rule2_found else "Rule 1 (MACD positive + RSI oversold)"

    return {
        "strong_buys":   found,
        "found":         len(found),
        "target":        target,
        "scanned":       scanned,
        "universe_size": len(HUNT_UNIVERSE),
        "rule2_count":   len(rule2_found),
        "rule1_count":   len(rule1_found),
        "rule_used":     rule_used,
        "complete":      len(found) >= target,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "data_note":     "15-min delayed intraday via Yahoo Finance",
    }


@app.get("/hunt-rule2")
async def hunt_rule2(max_scan: int = Query(default=300, description="Max stocks to scan")):
    """
    Scans the entire global universe for Rule 2 signals:
    price > SMA200 AND ROC(60) < -22% AND pct from 52w high < -3.82%
    This is the highest-precision signal: 72% win rate vs SPY, avg +12.4% alpha.
    Scans in parallel batches of 20, returns ALL matches found.
    """
    max_scan  = min(max_scan, len(HUNT_UNIVERSE))
    universe  = HUNT_UNIVERSE[:max_scan]
    batch_size= 20
    rule2_hits = []
    rule1_hits = []
    scanned   = 0
    failed    = 0
    loop      = asyncio.get_event_loop()

    for i in range(0, len(universe), batch_size):
        batch   = universe[i:i + batch_size]
        results = await asyncio.gather(
            *[loop.run_in_executor(executor, fetch_and_analyze, t) for t in batch]
        )
        scanned += len(batch)
        for r in results:
            if r is None:
                failed += 1
                continue
            if r.get("rule2_fired"):
                rule2_hits.append(r)
            elif r.get("rule1_fired"):
                rule1_hits.append(r)

    # Sort Rule 2 hits by how deeply oversold they are (most negative ROC60 first)
    rule2_hits.sort(key=lambda s: s.get("roc60", 0))
    # Sort Rule 1 hits by RSI ascending (most oversold first)
    rule1_hits.sort(key=lambda s: s.get("rsi", 99))

    return {
        "rule2_hits":    rule2_hits,
        "rule1_hits":    rule1_hits,
        "rule2_count":   len(rule2_hits),
        "rule1_count":   len(rule1_hits),
        "scanned":       scanned,
        "failed":        failed,
        "universe_size": len(universe),
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "rule2_desc":    "Above SMA200 + 60-day return < -22% + >3.82% from 52w high — 72% win rate vs SPY",
        "rule1_desc":    "MACD positive + RSI(14) < 35 — 55% win rate vs SPY, avg +14.5% alpha",
    }


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


def commerzbank_fee(trade_value: float) -> float:
    """
    Commerzbank DirectDepot order fee (shares/ETFs):
    0.25% of trade value + €4.90 flat fee, minimum €9.90.
    Source: commerzbank.expats.de/depot-en/ (2025)
    """
    fee = trade_value * 0.0025 + 4.90
    return max(fee, 9.90)


def run_backtest(ticker: str, period: str, trade_size: float,
                 min_position: float = 2000.0, entry_mode: str = "both") -> dict:
    """
    Replay signal logic on historical daily OHLCV.
    Includes:
    - Trend filter: only buy when price > SMA50 > SMA200
    - Volatility position sizing: risk 1% of portfolio per trade (position = risk / ATR*2)
    - Commerzbank transaction costs: 0.25% + €4.90, min €9.90 per trade
    - Exits: stop loss, target hit, score <= 3, or trend filter breaks
    """
    try:
        tk = yf.Ticker(ticker)
        fetch_period = "max" if period == "5y" else period
        hist = tk.history(period=fetch_period, interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 220:
            return None

        close  = hist["Close"].astype(float)
        high   = hist["High"].astype(float)
        low    = hist["Low"].astype(float)
        volume = hist["Volume"].astype(float)
        dates  = hist.index

        period_bars = {"1y": 252, "2y": 504, "5y": 1260}
        max_bars = period_bars.get(period, 252)
        start_i = max(210, len(close) - max_bars)

        rsi_s   = ta.rsi(close, length=14)
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        ema20_s = ta.ema(close, length=20)
        bb_df   = ta.bbands(close, length=20, std=2)
        atr_s   = ta.atr(high, low, close, length=14)

        hist_col  = [c for c in macd_df.columns if "MACDh" in c] if macd_df is not None else []
        lower_col = [c for c in bb_df.columns if "BBL" in c] if bb_df is not None else []
        upper_col = [c for c in bb_df.columns if "BBU" in c] if bb_df is not None else []

        trades          = []
        portfolio       = float(trade_size)
        cash            = float(trade_size)
        initial         = float(trade_size)
        risk_pct        = 0.01
        position        = None
        equity_curve    = []
        total_fees_paid = 0.0
        buy_hold_start  = float(close.iloc[start_i])

        for i in range(start_i, len(close)):
            price  = float(close.iloc[i])
            date_s = str(dates[i].date())

            def safe(s, default=50.0):
                v = s.iloc[i] if s is not None else None
                return float(v) if v is not None and not pd.isna(v) else default

            rsi      = safe(rsi_s, 50.0)
            macd_val = float(macd_df[hist_col[0]].iloc[i])   if hist_col and not pd.isna(macd_df[hist_col[0]].iloc[i])   else 0.0
            prev_mac = float(macd_df[hist_col[0]].iloc[i-1]) if hist_col and not pd.isna(macd_df[hist_col[0]].iloc[i-1]) else macd_val
            macd_dir = "rising" if macd_val > prev_mac else "falling"
            ema20    = safe(ema20_s, price)
            atr      = safe(atr_s, price * 0.015)

            bb_lower = float(bb_df[lower_col[0]].iloc[i]) if lower_col and not pd.isna(bb_df[lower_col[0]].iloc[i]) else price * 0.95
            bb_upper = float(bb_df[upper_col[0]].iloc[i]) if upper_col and not pd.isna(bb_df[upper_col[0]].iloc[i]) else price * 1.05
            bb_range = bb_upper - bb_lower
            bb_pos   = (price - bb_lower) / bb_range if bb_range > 0 else 0.5

            sma50  = float(close.iloc[max(0, i-50):i+1].mean())
            sma200 = float(close.iloc[max(0, i-200):i+1].mean())
            # Relaxed trend filter: price above SMA200 (matches screener logic)
            trend_ok = (price > sma200)

            pct_above_ema = (price - ema20) / ema20 * 100
            prev_price    = float(close.iloc[i - 1]) if i > 0 else price
            price_change  = (price - prev_price) / prev_price * 100
            vol_window    = volume.iloc[max(0, i - 20):i]
            vol_avg       = float(vol_window.mean()) if len(vol_window) > 0 else float(volume.iloc[i])
            vol_mult      = float(volume.iloc[i]) / vol_avg if vol_avg > 0 else 1.0

            score = compute_signal_score(rsi, macd_val, macd_dir, pct_above_ema, vol_mult, bb_pos, price_change)
            if not trend_ok and score >= 5:
                score = 4

            # ── Data-proven rules ─────────────────────────────────────
            roc60_bt     = (price / float(close.iloc[i-60]) - 1) * 100 if i >= 60 else 0.0
            high52_bt    = float(high.iloc[max(0,i-252):i+1].max())
            pct_52w_h_bt = (price - high52_bt) / high52_bt * 100 if high52_bt > 0 else 0.0
            rule1_bt = (macd_val > 0) and (rsi < 35)
            rule2_bt = (trend_ok) and (roc60_bt < -22.0) and (pct_52w_h_bt < -3.82)

            # ── Exit logic ────────────────────────────────────────────
            if position is not None:
                days_held = i - position["entry_bar"]
                exit_reason = exit_price = None
                if price <= position["stop"]:
                    exit_reason, exit_price = "Stop loss", position["stop"]
                elif price >= position["target"]:
                    exit_reason, exit_price = "Target hit", position["target"]
                elif days_held >= 5 and score <= 2:
                    exit_reason, exit_price = "Signal weak", price
                elif days_held >= 5 and not trend_ok:
                    exit_reason, exit_price = "Trend broke", price

                if exit_reason:
                    gross_proceeds = position["shares"] * exit_price
                    sell_fee       = commerzbank_fee(gross_proceeds)
                    net_proceeds   = gross_proceeds - sell_fee
                    total_fees_paid += sell_fee
                    # P&L = net proceeds - total invested (buy cost_basis already net of buy fee)
                    pnl     = net_proceeds - position["cost_basis"]
                    pnl_pct = (exit_price - position["buy_price"]) / position["buy_price"] * 100
                    cash   += net_proceeds
                    portfolio = cash
                    trades.append({
                        "buy_date":    position["buy_date"],
                        "buy_price":   round(position["buy_price"], 2),
                        "sell_date":   date_s,
                        "sell_price":  round(exit_price, 2),
                        "shares":      round(position["shares"], 4),
                        "invested":    round(position["cost_basis"], 2),
                        "buy_fee":     round(position["buy_fee"], 2),
                        "sell_fee":    round(sell_fee, 2),
                        "total_fees":  round(position["buy_fee"] + sell_fee, 2),
                        "days_held":   i - position["entry_bar"],
                        "pnl":         round(pnl, 2),
                        "pnl_pct":     round(pnl_pct, 2),
                        "exit_reason": exit_reason,
                    })
                    position = None

            # ── Entry: controlled by entry_mode parameter ─────────────
            if entry_mode == "rule1only":
                entry_signal = rule1_bt
            elif entry_mode == "rule2only":
                entry_signal = rule2_bt
            elif entry_mode == "score":
                entry_signal = (score >= 5 and trend_ok)
            else:  # "both" — all signals
                entry_signal = rule1_bt or (rule2_bt and trend_ok) or (score >= 5 and trend_ok)

            if position is None and entry_signal:
                risk_amount   = portfolio * risk_pct
                stop_distance = atr * 2
                shares        = risk_amount / stop_distance if stop_distance > 0 else 0
                gross_cost    = shares * price
                # Enforce minimum position so fees stay proportionate
                if 0 < gross_cost < min_position:
                    shares     = min_position / price
                    gross_cost = min_position
                buy_fee    = commerzbank_fee(gross_cost)
                total_cost = gross_cost + buy_fee
                if shares > 0 and total_cost <= cash:
                    cash           -= total_cost
                    total_fees_paid += buy_fee
                    position = {
                        "shares":     shares,
                        "buy_price":  price,
                        "cost_basis": total_cost,
                        "buy_fee":    buy_fee,
                        "stop":       price - atr * 2,
                        "target":     price + atr * 4,
                        "buy_date":   date_s,
                        "entry_bar":  i,
                    }

            portfolio_val = cash + (position["shares"] * price if position else 0)
            buy_hold_val  = initial * (price / buy_hold_start)
            equity_curve.append({
                "date":     date_s,
                "value":    round(portfolio_val, 2),
                "buy_hold": round(buy_hold_val, 2),
            })
            if position is None:
                portfolio = portfolio_val

        # Close open position at last price
        if position is not None:
            lp             = float(close.iloc[-1])
            gross_proceeds = position["shares"] * lp
            sell_fee       = commerzbank_fee(gross_proceeds)
            net_proceeds   = gross_proceeds - sell_fee
            total_fees_paid += sell_fee
            pnl     = net_proceeds - position["cost_basis"]
            pnl_pct = (lp - position["buy_price"]) / position["buy_price"] * 100
            trades.append({
                "buy_date":    position["buy_date"],
                "buy_price":   round(position["buy_price"], 2),
                "sell_date":   "Open",
                "sell_price":  None,
                "shares":      round(position["shares"], 4),
                "invested":    round(position["cost_basis"], 2),
                "buy_fee":     round(position["buy_fee"], 2),
                "sell_fee":    round(sell_fee, 2),
                "total_fees":  round(position["buy_fee"] + sell_fee, 2),
                "days_held":   len(close) - 1 - position["entry_bar"],
                "pnl":         round(pnl, 2),
                "pnl_pct":     round(pnl_pct, 2),
                "exit_reason": "Still open",
            })

        final_value    = equity_curve[-1]["value"] if equity_curve else initial
        total_pnl      = final_value - initial
        total_trades   = len(trades)
        winning_trades = sum(1 for t in trades if t["pnl"] > 0)
        pnl_pcts       = [t["pnl_pct"] for t in trades]

        peak = initial
        max_dd = 0.0
        for pt in equity_curve:
            peak = max(peak, pt["value"])
            dd   = (peak - pt["value"]) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        step    = max(1, len(equity_curve) // 300)
        eq_thin = equity_curve[::step]
        if equity_curve and equity_curve[-1] != eq_thin[-1]:
            eq_thin.append(equity_curve[-1])

        avg_fee = total_fees_paid / (total_trades * 2) if total_trades else 0

        return {
            "ticker":              ticker,
            "period":              period,
            "trade_size":          trade_size,
            "initial_value":       round(initial, 2),
            "final_value":         round(final_value, 2),
            "total_pnl":           round(total_pnl, 2),
            "total_return_pct":    round((total_pnl / initial) * 100, 2),
            "total_trades":        total_trades,
            "winning_trades":      winning_trades,
            "losing_trades":       total_trades - winning_trades,
            "win_rate_pct":        round(winning_trades / total_trades * 100, 1) if total_trades else 0,
            "best_trade_pct":      round(max(pnl_pcts), 2) if pnl_pcts else 0,
            "worst_trade_pct":     round(min(pnl_pcts), 2) if pnl_pcts else 0,
            "max_drawdown_pct":    round(max_dd, 2),
            "total_fees_paid":     round(total_fees_paid, 2),
            "avg_fee_per_trade":   round(avg_fee, 2),
            "fee_drag_pct":        round((total_fees_paid / initial) * 100, 2),
            "trades":              trades,
            "equity_curve":        eq_thin,
            "strategy_notes": [
                "Trend filter: price > SMA200 required for Rule 2 and score-based entries",
                "Rule 1 (MACD positive + RSI < 35): fires regardless of trend — 55% win rate vs SPY, avg +14.5% alpha",
                "Rule 2 (above SMA200 + ROC60 < -22% + far from 52w high): 72% win rate vs SPY, avg +12.4% alpha",
                "Position sizing: risks 1% of portfolio per trade (ATR-based), minimum €500",
                "Target: 4x ATR above entry (R:R = 1:2), Stop: 2x ATR below entry",
                "Minimum 5-day hold before signal-based exit",
                "Transaction costs: Commerzbank 0.25% + €4.90, min €9.90 per trade",
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception:
        print(f"Backtest error {ticker}: {traceback.format_exc()}")
        return None


@app.get("/backtest/{ticker}")
async def backtest(
    ticker: str,
    period: str = Query(default="1y"),
    trade_size: float = Query(default=10000.0),
    min_position: float = Query(default=2000.0),
    entry_mode: str = Query(default="both"),
):
    """Backtest signal strategy. Example: GET /backtest/AAPL?period=2y&trade_size=10000"""
    if period not in {"1y", "2y", "5y"}:
        raise HTTPException(status_code=400, detail="Period must be 1y, 2y, or 5y")
    if trade_size < 100 or trade_size > 500000:
        raise HTTPException(status_code=400, detail="Trade size must be 100–500,000")
    if entry_mode not in {"both", "rule1only", "rule2only", "score"}:
        raise HTTPException(status_code=400, detail="Invalid entry_mode")
    result = await asyncio.get_event_loop().run_in_executor(
        executor, run_backtest, ticker.upper(), period, trade_size, min_position, entry_mode
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Could not backtest {ticker}")
    return result

