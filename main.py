"""
XAU/USD Live Signal Server  —  Gold Intelligence Score
======================================================
Combines free macro drivers, market/news context, event risk,
and 5M/15M/1H technical confirmation for XAU/USD research.

HOW IT WORKS:
  • Final trade checker runs every 15 minutes
  • It sends Telegram only when the weighted model has an executable trade
  • Telegram sends one alert per new 15M/5M sweep-triggered trade
  • Dashboard shows weighted decision, event risk, and validation
  • UptimeRobot ping at /health (HEAD + GET both supported)

ENVIRONMENT VARIABLES:
  TELEGRAM_BOT_TOKEN   → from @BotFather on Telegram
  TELEGRAM_CHAT_ID     → from @userinfobot on Telegram
  TRADE_CHECK_SECONDS  → optional; default 900
  TRADE_LOG_PATH       → optional; default trade_history.json
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager, suppress
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from html import escape
import hashlib
import json
import xml.etree.ElementTree as ET
import uvicorn, os, asyncio, httpx, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("xauusd")

TG_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID",   "")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
TRADE_CHECK_SECONDS = int(os.environ.get("TRADE_CHECK_SECONDS", "900"))
TRADE_LOG_PATH = os.environ.get("TRADE_LOG_PATH", "trade_history.json")
CONTEXT_TTL_SECONDS = int(os.environ.get("CONTEXT_TTL_SECONDS", "300"))
CONTEXT_CACHE = {"ts": None, "data": None}
EVENT_STUDY_TTL_SECONDS = int(os.environ.get("EVENT_STUDY_TTL_SECONDS", "21600"))
EVENT_STUDY_CACHE = {"ts": None, "data": None}
EXECUTABLE_TRADE_QUALITIES = {"CONFIRMED", "CAUTION", "COUNTERTREND"}

MODEL_WEIGHTS = {
    "technical": 0.50,
    "news": 0.20,
    "macro": 0.15,
    "sentiment": 0.15,
}

NEWS_QUERY = (
    '("gold" OR XAUUSD OR "gold futures" OR "Federal Reserve" OR '
    '"US dollar" OR Treasury yields OR inflation OR CPI OR PCE OR FOMC)'
)
PEOPLE_SENTIMENT_QUERY = (
    '("gold traders" OR "XAUUSD traders" OR "gold market sentiment" OR '
    '"gold bullish" OR "gold bearish" OR "safe haven demand")'
)

PAST_EVENTS = [
    {
        "name": "FOMC statement",
        "period": "April 29, 2026",
        "time_utc": "2026-04-29T18:00:00+00:00",
        "source": "Federal Reserve",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260429a.htm",
    },
    {
        "name": "CPI report",
        "period": "March 2026",
        "time_utc": "2026-04-10T12:30:00+00:00",
        "source": "BLS",
        "url": "https://www.bls.gov/news.release/archives/cpi_04102026.htm",
    },
    {
        "name": "Employment Situation",
        "period": "March 2026",
        "time_utc": "2026-04-03T12:30:00+00:00",
        "source": "BLS",
        "url": "https://www.bls.gov/news.release/empsit.htm",
    },
    {
        "name": "FOMC statement",
        "period": "March 18, 2026",
        "time_utc": "2026-03-18T18:00:00+00:00",
        "source": "Federal Reserve",
        "url": "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20260318.htm",
    },
]

UPCOMING_EVENTS = [
    {
        "name": "Employment Situation",
        "period": "April 2026",
        "time_utc": "2026-05-08T12:30:00+00:00",
        "source": "BLS",
        "url": "https://www.bls.gov/schedule/news_release/",
    },
    {
        "name": "CPI report",
        "period": "April 2026",
        "time_utc": "2026-05-12T12:30:00+00:00",
        "source": "BLS",
        "url": "https://www.bls.gov/schedule/news_release/",
    },
    {
        "name": "FOMC minutes",
        "period": "Apr. 28-29 meeting",
        "time_utc": "2026-05-20T18:00:00+00:00",
        "source": "Federal Reserve",
        "url": "https://www.federalreserve.gov/monetarypolicy.htm",
    },
    {
        "name": "FOMC meeting",
        "period": "June 16-17, 2026",
        "time_utc": "2026-06-17T18:00:00+00:00",
        "source": "Federal Reserve",
        "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    },
]


# ═══════════════════════════════════════════════════════════════════
#  DATA FETCHER
# ═══════════════════════════════════════════════════════════════════
def timeframe_label(interval: str) -> str:
    return {"5m": "5M", "15m": "15M", "1h": "1H"}.get(interval, interval.upper())


def fetch_data(interval: str) -> tuple:
    """interval = '1h', '15m', or '5m'. Returns (DataFrame, error_or_None)."""
    try:
        import yfinance as yf
        periods = {"1h": "60d", "15m": "30d", "5m": "30d"}
        min_bars_by_interval = {"1h": 220, "15m": 400, "5m": 800}
        period = periods.get(interval, "30d")
        min_bars = min_bars_by_interval.get(interval, 400)
        df = yf.download("GC=F", period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None, f"Empty data ({interval})"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df.dropna()
        if len(df) < min_bars:
            return None, f"Only {len(df)} bars ({interval}) — need {min_bars}+"
        log.info(f"yfinance {interval}: {len(df)} bars  "
                 f"{df.index[0].date()} → {df.index[-1].date()}")
        return df, None
    except Exception as e:
        return None, f"yfinance error ({interval}): {e}"


# ═══════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=7):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def atr(df, p=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()


def adx(df, p=14):
    high = df['high']
    low = df['low']
    close = df['close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_wilder = tr.ewm(alpha=1 / p, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / p, adjust=False).mean() / (atr_wilder + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / p, adjust=False).mean() / (atr_wilder + 1e-9)
    dx = ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)) * 100
    return dx.ewm(alpha=1 / p, adjust=False).mean(), plus_di, minus_di


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def pct_change_now(series: pd.Series, idx: int, lookback: int) -> float:
    if idx < lookback:
        return 0.0
    base = safe_float(series.iloc[idx - lookback])
    if base == 0:
        return 0.0
    return (safe_float(series.iloc[idx]) / base - 1) * 100


def candle_location(row: pd.Series) -> dict:
    high = safe_float(row['high'])
    low = safe_float(row['low'])
    open_price = safe_float(row['open'])
    close = safe_float(row['close'])
    candle_range = max(high - low, 1e-9)
    upper_wick = high - max(open_price, close)
    lower_wick = min(open_price, close) - low
    return {
        "close_position": (close - low) / candle_range,
        "upper_wick_pct": upper_wick / candle_range,
        "lower_wick_pct": lower_wick / candle_range,
        "body_pct": abs(close - open_price) / candle_range,
    }


def empty_sweep(lookback: int) -> dict:
    return {
        "confirmed": False,
        "direction": "NONE",
        "level": None,
        "extreme": None,
        "opposite_liquidity": None,
        "age_bars": None,
        "lookback_bars": lookback,
        "score": 0,
        "close_position": None,
        "wick_pct": None,
        "displacement_atr": None,
        "bar_time": None,
        "summary": "No confirmed sweep/reclaim in the scan window.",
    }


def detect_liquidity_sweep(df: pd.DataFrame, at: pd.Series, interval: str) -> dict:
    if interval == "5m":
        lookback = 72
        scan_bars = 9
    elif interval == "15m":
        lookback = 48
        scan_bars = 5
    else:
        lookback = 30
        scan_bars = 3
    idx_now = len(df) - 1
    best = empty_sweep(lookback)

    for age in range(scan_bars):
        idx = idx_now - age
        if idx < lookback + 2:
            continue

        row = df.iloc[idx]
        history = df.iloc[max(0, idx - lookback):idx - 1]
        if history.empty:
            continue

        atr_value = max(safe_float(at.iloc[idx], 1.0), 1e-9)
        high = safe_float(row['high'])
        low = safe_float(row['low'])
        close = safe_float(row['close'])
        open_price = safe_float(row['open'])
        swing_high = safe_float(history['high'].max())
        swing_low = safe_float(history['low'].min())
        sweep_buffer = max(atr_value * 0.035, close * 0.00008)
        candle = candle_location(row)
        displacement = abs(close - open_price) / atr_value

        candidates = []
        swept_high = high > swing_high + sweep_buffer and close < swing_high
        swept_low = low < swing_low - sweep_buffer and close > swing_low

        if swept_low and candle["close_position"] >= 0.55:
            score = 42 + min(24, displacement * 18) + min(20, candle["lower_wick_pct"] * 40) - age * 5
            candidates.append({
                "confirmed": True,
                "direction": "LONG",
                "level": round(swing_low, 2),
                "extreme": round(low, 2),
                "opposite_liquidity": round(swing_high, 2),
                "age_bars": age,
                "lookback_bars": lookback,
                "score": int(round(clamp(score, 0, 100))),
                "close_position": round(candle["close_position"], 2),
                "wick_pct": round(candle["lower_wick_pct"] * 100, 1),
                "displacement_atr": round(displacement, 2),
                "bar_time": str(df.index[idx]),
                "summary": f"Swept below {swing_low:.2f} and closed back inside.",
            })

        if swept_high and candle["close_position"] <= 0.45:
            score = 42 + min(24, displacement * 18) + min(20, candle["upper_wick_pct"] * 40) - age * 5
            candidates.append({
                "confirmed": True,
                "direction": "SHORT",
                "level": round(swing_high, 2),
                "extreme": round(high, 2),
                "opposite_liquidity": round(swing_low, 2),
                "age_bars": age,
                "lookback_bars": lookback,
                "score": int(round(clamp(score, 0, 100))),
                "close_position": round(candle["close_position"], 2),
                "wick_pct": round(candle["upper_wick_pct"] * 100, 1),
                "displacement_atr": round(displacement, 2),
                "bar_time": str(df.index[idx]),
                "summary": f"Swept above {swing_high:.2f} and closed back inside.",
            })

        for candidate in candidates:
            if candidate["score"] > best["score"]:
                best = candidate

    return best


def build_trade_plan(price: float, atr_value: float, setup_direction: str, sweep: dict) -> dict:
    if setup_direction not in ("LONG", "SHORT"):
        return {
            "entry": round(price, 2),
            "stop": None,
            "target": None,
            "target_2": None,
            "risk": None,
            "reward": None,
            "rr": None,
            "position_1pct_10k_oz": None,
            "invalidates": "No trade without aligned 1H regime and execution-timeframe sweep.",
        }

    buffer = max(atr_value * 0.18, price * 0.00015)
    if setup_direction == "LONG":
        stop = (sweep.get("extreme") or price - atr_value) - buffer
        if stop >= price:
            stop = price - atr_value
        risk = price - stop
        target = price + risk * 2.0
        target_2 = sweep.get("opposite_liquidity") or price + risk * 3.0
        invalidates = "Execution timeframe closes back below the sweep extreme or 1H EMA stack flips bearish."
    else:
        stop = (sweep.get("extreme") or price + atr_value) + buffer
        if stop <= price:
            stop = price + atr_value
        risk = stop - price
        target = price - risk * 2.0
        target_2 = sweep.get("opposite_liquidity") or price - risk * 3.0
        invalidates = "Execution timeframe closes back above the sweep extreme or 1H EMA stack flips bullish."

    reward = abs(target - price)
    return {
        "entry": round(price, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "target_2": round(target_2, 2),
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "rr": round(reward / max(risk, 1e-9), 2),
        "position_1pct_10k_oz": round(100 / max(risk, 1e-9), 4),
        "invalidates": invalidates,
    }


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL ENGINE  — same logic for any timeframe
# ═══════════════════════════════════════════════════════════════════
def compute_signal(df: pd.DataFrame, interval: str) -> dict:
    c = df['close']
    e9 = ema(c, 9)
    e21 = ema(c, 21)
    e50 = ema(c, 50)
    e200 = ema(c, 200)
    rv = rsi(c, 14)
    at = atr(df, 14)
    ax, plus_di, minus_di = adx(df, 14)

    i = len(df) - 1
    price = float(c.iloc[i])
    ema9v = float(e9.iloc[i])
    ema21v = float(e21.iloc[i])
    ema50v = float(e50.iloc[i])
    ema200v = float(e200.iloc[i])
    atrv = float(at.iloc[i])
    rsiv = float(rv.iloc[i])
    adxv = safe_float(ax.iloc[i])
    plus_div = safe_float(plus_di.iloc[i])
    minus_div = safe_float(minus_di.iloc[i])
    ema9_slope = pct_change_now(e9, i, 5)
    ema21_slope = pct_change_now(e21, i, 5)
    ema50_slope = pct_change_now(e50, i, 8)
    momentum_pct = pct_change_now(c, i, 5)
    atr_pct = atrv / (price + 1e-9) * 100
    sweep = detect_liquidity_sweep(df, at, interval)

    trend_score = 0
    bull_stack = ema21v > ema50v > ema200v and price > ema21v
    bear_stack = ema21v < ema50v < ema200v and price < ema21v
    fast_bull = ema9v > ema21v > ema50v and price > ema9v
    fast_bear = ema9v < ema21v < ema50v and price < ema9v
    if bull_stack and ema21_slope > 0 and ema50_slope >= 0:
        trend_score = 42
    elif bear_stack and ema21_slope < 0 and ema50_slope <= 0:
        trend_score = -42
    elif price > ema50v > ema200v:
        trend_score = 26
    elif price < ema50v < ema200v:
        trend_score = -26
    elif price > ema200v:
        trend_score = 14
    elif price < ema200v:
        trend_score = -14

    momentum_score = 0
    if momentum_pct > 0.08 and ema9_slope > 0 and rsiv >= 50:
        momentum_score = 18
    elif momentum_pct < -0.08 and ema9_slope < 0 and rsiv <= 50:
        momentum_score = -18
    elif momentum_pct > 0:
        momentum_score = 8
    elif momentum_pct < 0:
        momentum_score = -8

    distance_from_ema21 = (price - ema21v) / (atrv + 1e-9)
    reclaim_score = 0
    if trend_score > 0:
        if -0.45 <= distance_from_ema21 <= 0.95:
            reclaim_score = 14
        elif fast_bull:
            reclaim_score = 10
        elif distance_from_ema21 > 2.2:
            reclaim_score = -9
    elif trend_score < 0:
        if -0.95 <= distance_from_ema21 <= 0.45:
            reclaim_score = -14
        elif fast_bear:
            reclaim_score = -10
        elif distance_from_ema21 < -2.2:
            reclaim_score = 9

    sweep_score = 0
    if sweep["confirmed"]:
        sweep_weight = 44 if interval == "15m" else 34 if interval == "5m" else 24
        sweep_score = sweep_weight if sweep["direction"] == "LONG" else -sweep_weight

    confluence_penalty = 0
    if sweep["direction"] == "LONG" and trend_score < -25:
        confluence_penalty = -12
    elif sweep["direction"] == "SHORT" and trend_score > 25:
        confluence_penalty = 12

    extension_penalty = 0
    if rsiv >= 76:
        extension_penalty = -10
    elif rsiv <= 24:
        extension_penalty = 10

    technical_score = int(clamp(
        trend_score + momentum_score + reclaim_score + sweep_score + confluence_penalty + extension_penalty,
        -100,
        100,
    ))
    setup_direction = "LONG" if technical_score > 15 else "SHORT" if technical_score < -15 else "NONE"
    if interval in ("15m", "5m"):
        execution_threshold = 65 if interval == "15m" else 60
        signal = "LONG" if (
            technical_score >= execution_threshold and sweep["direction"] == "LONG" and sweep["confirmed"]
        ) else "SHORT" if (
            technical_score <= -execution_threshold and sweep["direction"] == "SHORT" and sweep["confirmed"]
        ) else "WAIT"
    else:
        signal = "LONG" if technical_score >= 62 else "SHORT" if technical_score <= -62 else "WAIT"

    trade_direction = setup_direction if sweep["confirmed"] and sweep["direction"] == setup_direction else "NONE"
    trade = build_trade_plan(price, atrv, trade_direction, sweep)
    stop = trade["stop"] if trade["stop"] is not None else round(price - atrv if setup_direction == "LONG" else price + atrv, 2)
    target = trade["target"] if trade["target"] is not None else round(price + atrv * 2 if setup_direction == "LONG" else price - atrv * 2, 2)
    sd = trade["risk"] or atrv
    td = trade["reward"] or atrv * 2

    lookback = 288 if interval == "5m" else 96 if interval == "15m" else 24
    prev_close = float(c.iloc[i - lookback]) if i >= lookback else price
    change = round(price - prev_close, 2)
    change_pct = round((price / prev_close - 1) * 100, 2)

    chart_bars = 576 if interval == "5m" else 288 if interval == "15m" else 72
    price_hist = [round(float(v), 2) for v in c.iloc[-chart_bars:] if not np.isnan(v)]
    ema21_hist = [round(float(v), 2) for v in e21.iloc[-chart_bars:] if not np.isnan(v)]
    ema50_hist = [round(float(v), 2) for v in e50.iloc[-chart_bars:] if not np.isnan(v)]
    ema200_hist = [round(float(v), 2) for v in e200.iloc[-chart_bars:] if not np.isnan(v)]

    conditions = {
        "trend_aligned": abs(trend_score) >= 26,
        "ema_stack": bull_stack or bear_stack,
        "adx_trending": adxv >= 18,
        "momentum_confirmed": abs(momentum_score) >= 18,
        "liquidity_sweep": bool(sweep["confirmed"]),
        "ema_reclaim": abs(reclaim_score) >= 10,
        "not_overextended": extension_penalty == 0,
    }
    conds_met = sum(conditions.values())
    tf_label = timeframe_label(interval)

    return {
        "timeframe": tf_label,
        "signal": signal,
        "setup_direction": setup_direction,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "bar_time": str(df.index[-1]),
        "price": round(price, 2),
        "latest_high": round(safe_float(df['high'].iloc[i]), 2),
        "latest_low": round(safe_float(df['low'].iloc[i]), 2),
        "change": change,
        "change_pct": change_pct,
        "stop": stop,
        "target": target,
        "stop_dist": round(sd, 2),
        "target_dist": round(td, 2),
        "data_bars": len(df),
        "conditions_met": conds_met,
        "max_conditions": 7,
        "technical_score": technical_score,
        "components": {
            "trend": int(trend_score),
            "momentum": int(momentum_score),
            "ema_reclaim": int(reclaim_score),
            "liquidity_sweep": int(sweep_score),
            "confluence": int(confluence_penalty),
            "extension": int(extension_penalty),
        },
        "indicators": {
            "ema9": round(ema9v, 2),
            "ema21": round(ema21v, 2),
            "ema50": round(ema50v, 2),
            "ema200": round(ema200v, 2),
            "rsi14": round(rsiv, 1),
            "rsi7": round(rsiv, 1),
            "adx14": round(adxv, 1),
            "plus_di14": round(plus_div, 1),
            "minus_di14": round(minus_div, 1),
            "atr14": round(atrv, 2),
            "atr_pct": round(atr_pct, 3),
            "momentum_pct": round(momentum_pct, 2),
            "ema9_slope": round(ema9_slope, 3),
            "ema21_slope": round(ema21_slope, 3),
            "ema50_slope": round(ema50_slope, 3),
            "distance_from_ema21_atr": round(distance_from_ema21, 2),
            "squeeze_pct": 0,
        },
        "conditions": conditions,
        "liquidity_sweep": sweep,
        "trade": trade,
        "chart": {"price": price_hist, "ema21": ema21_hist, "ema50": ema50_hist, "ema200": ema200_hist},
    }


def get_signal(interval: str = "1h") -> dict:
    df, err = fetch_data(interval)
    if err or df is None:
        log.warning(f"[{interval}] {err}")
        return {"error": err or "No data", "timeframe": interval.upper()}
    return compute_signal(df, interval)


def get_both_signals() -> dict:
    s1h  = get_signal("1h")
    s15m = get_signal("15m")
    s5m = get_signal("5m")
    price = s5m.get("price") or s15m.get("price") or s1h.get("price") or 0
    return {
        "h1":        s1h,
        "m15":       s15m,
        "m5":        s5m,
        "price":     price,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


# ═══════════════════════════════════════════════════════════════════
#  FREE MARKET CONTEXT
# ═══════════════════════════════════════════════════════════════════
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def macro_point(symbol: str, label: str, bullish_when_down: bool, unit: str = "") -> dict:
    try:
        import yfinance as yf
        df = yf.download(symbol, period="10d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("empty")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        closes = df["close"].dropna()
        if len(closes) < 2:
            raise ValueError("not enough data")

        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        change = last - prev
        change_pct = (last / prev - 1) * 100 if prev else 0
        move = "UP" if change > 0 else "DOWN" if change < 0 else "FLAT"
        score = 0
        if abs(change_pct) >= 0.05:
            score = 1 if (change < 0 and bullish_when_down) or (change > 0 and not bullish_when_down) else -1

        return {
            "label": label,
            "symbol": symbol,
            "value": round(last, 2),
            "unit": unit,
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "move": move,
            "score": score,
            "status": "ok",
        }
    except Exception as e:
        return {
            "label": label,
            "symbol": symbol,
            "value": None,
            "unit": unit,
            "change": None,
            "change_pct": None,
            "move": "NA",
            "score": 0,
            "status": f"offline: {e}",
        }


def get_macro_context() -> dict:
    drivers = [
        macro_point("DX-Y.NYB", "DXY", True, ""),
        macro_point("^TNX", "10Y", True, "%"),
        macro_point("^VIX", "VIX", False, ""),
    ]
    score = sum(d["score"] for d in drivers)
    bias = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "MIXED"
    return {
        "score": int(score),
        "bias": bias,
        "drivers": drivers,
        "status": "ok" if any(d["status"] == "ok" for d in drivers) else "offline",
    }


def headline_score(title: str) -> tuple:
    text = title.lower()
    bull_words = [
        "safe haven", "geopolitical", "war", "conflict", "recession",
        "rate cut", "dovish", "weaker dollar", "dollar falls",
        "yields fall", "yield falls", "treasury yields drop", "central bank buying",
        "soft landing fears", "debt ceiling", "tariff", "flight to safety",
    ]
    bear_words = [
        "hawkish", "rate hike", "higher for longer", "strong dollar",
        "dollar rises", "yields rise", "yield rises", "strong jobs",
        "hot inflation", "risk-on", "selloff in gold", "profit taking",
        "fed delay", "sticky inflation",
    ]
    risk_words = [
        "fomc", "powell", "federal reserve", "fed", "cpi", "pce",
        "payrolls", "nfp", "unemployment", "jobs report", "inflation",
        "tariff", "war", "conflict", "treasury", "auction",
    ]
    high_impact_words = [
        "fomc", "powell", "cpi", "pce", "payrolls", "nfp", "jobs report",
        "federal reserve", "rate decision", "inflation",
    ]
    score = 0
    if any(w in text for w in bull_words):
        score += 1
    if any(w in text for w in bear_words):
        score -= 1
    risk = any(w in text for w in risk_words)
    impact = "HIGH" if any(w in text for w in high_impact_words) else "MEDIUM" if risk else "LOW"
    bias = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
    reason = (
        "bullish gold impulse" if score > 0 else
        "bearish gold impulse" if score < 0 else
        "headline risk, direction unclear" if risk else
        "low directional content"
    )
    return score, risk, bias, impact, reason


def get_news_context() -> dict:
    params = {
        "query": NEWS_QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 18,
        "timespan": "12h",
        "sort": "hybridrel",
    }
    try:
        with httpx.Client(timeout=8) as client:
            res = client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
            res.raise_for_status()
            payload = res.json()
        articles = payload.get("articles", [])
        headlines = []
        raw_score = 0
        risk_count = 0
        impact_points = 0
        for item in articles[:12]:
            title = item.get("title", "").strip()
            if not title:
                continue
            score, is_risk, bias, impact, reason = headline_score(title)
            raw_score += score
            risk_count += 1 if is_risk else 0
            impact_points += 3 if impact == "HIGH" else 2 if impact == "MEDIUM" else 1
            headlines.append({
                "title": title,
                "url": item.get("url", ""),
                "source": item.get("sourceCountry", "") or item.get("domain", ""),
                "bias": bias,
                "risk": is_risk,
                "impact": impact,
                "reason": reason,
                "published": item.get("seendate", ""),
            })
        score = int(clamp(raw_score, -4, 4))
        bias = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "MIXED"
        risk = "HIGH" if risk_count >= 3 or impact_points >= 12 else "MEDIUM" if risk_count else "LOW"
        return {
            "score": score,
            "bias": bias,
            "risk": risk,
            "risk_count": risk_count,
            "impact_points": impact_points,
            "headlines": headlines[:8],
            "status": "ok",
            "source": "GDELT 12h",
        }
    except Exception as e:
        fallback = get_fed_rss_context(str(e))
        if fallback["status"] == "ok":
            return fallback
        return {
            "score": 0,
            "bias": "MIXED",
            "risk": "UNKNOWN",
            "risk_count": 0,
            "headlines": [],
            "status": f"offline: {e}",
            "source": "GDELT 12h",
        }


def get_fed_rss_context(reason: str = "") -> dict:
    feeds = [
        "https://www.federalreserve.gov/feeds/press_monetary.xml",
        "https://www.federalreserve.gov/feeds/speeches.xml",
    ]
    try:
        headlines = []
        raw_score = 0
        risk_count = 0
        with httpx.Client(timeout=8) as client:
            for feed in feeds:
                res = client.get(feed)
                res.raise_for_status()
                root = ET.fromstring(res.text)
                for item in root.findall("./channel/item")[:4]:
                    title = (item.findtext("title") or "").strip()
                    url = (item.findtext("link") or "").strip()
                    if not title:
                        continue
                    score, is_risk, bias, impact, reason = headline_score(title)
                    raw_score += score
                    risk_count += 1 if is_risk else 0
                    headlines.append({
                        "title": title,
                        "url": url,
                        "source": "Federal Reserve",
                        "bias": bias,
                        "risk": is_risk,
                        "impact": impact,
                        "reason": reason,
                        "published": "",
                    })
        score = int(clamp(raw_score, -2, 2))
        bias = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "MIXED"
        risk = "MEDIUM" if risk_count else "LOW"
        return {
            "score": score,
            "bias": bias,
            "risk": risk,
            "risk_count": risk_count,
            "headlines": headlines[:5],
            "status": "ok",
            "source": "Fed RSS fallback",
            "fallback_reason": reason,
        }
    except Exception as e:
        return {
            "score": 0,
            "bias": "MIXED",
            "risk": "UNKNOWN",
            "risk_count": 0,
            "headlines": [],
            "status": f"offline: {e}",
            "source": "Fed RSS fallback",
            "fallback_reason": reason,
        }


def get_people_sentiment() -> dict:
    params = {
        "query": PEOPLE_SENTIMENT_QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 24,
        "timespan": "24h",
        "sort": "hybridrel",
    }
    try:
        with httpx.Client(timeout=8) as client:
            res = client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
            res.raise_for_status()
            payload = res.json()

        bulls = bears = neutral = 0
        samples = []
        for item in payload.get("articles", [])[:16]:
            title = item.get("title", "").strip()
            if not title:
                continue
            score, _, bias, impact, reason = headline_score(title)
            if score > 0:
                bulls += 1
            elif score < 0:
                bears += 1
            else:
                neutral += 1
            samples.append({
                "title": title,
                "url": item.get("url", ""),
                "source": item.get("sourceCountry", "") or item.get("domain", ""),
                "bias": bias,
                "impact": impact,
                "reason": reason,
            })

        raw_score = bulls - bears
        score = int(clamp(raw_score, -3, 3))
        bias = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "MIXED"
        total = max(1, bulls + bears + neutral)
        return {
            "score": score,
            "bias": bias,
            "bullish_count": bulls,
            "bearish_count": bears,
            "neutral_count": neutral,
            "bullish_pct": round(bulls / total * 100, 1),
            "bearish_pct": round(bears / total * 100, 1),
            "samples": samples[:5],
            "source": "GDELT public-web sentiment 24h",
            "status": "ok",
        }
    except Exception as e:
        return {
            "score": 0,
            "bias": "MIXED",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "bullish_pct": 0,
            "bearish_pct": 0,
            "samples": [],
            "source": "GDELT public-web sentiment 24h",
            "status": f"offline: {e}",
        }


def derive_people_sentiment_from_news(news: dict, reason: str = "") -> dict:
    headlines = news.get("headlines", [])
    bulls = sum(1 for h in headlines if h.get("bias") == "bullish")
    bears = sum(1 for h in headlines if h.get("bias") == "bearish")
    neutral = max(0, len(headlines) - bulls - bears)
    raw_score = bulls - bears
    score = int(clamp(raw_score, -3, 3))
    bias = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "MIXED"
    total = max(1, bulls + bears + neutral)
    return {
        "score": score,
        "bias": bias,
        "bullish_count": bulls,
        "bearish_count": bears,
        "neutral_count": neutral,
        "bullish_pct": round(bulls / total * 100, 1),
        "bearish_pct": round(bears / total * 100, 1),
        "samples": headlines[:5],
        "source": "Derived from current GDELT headline tape",
        "status": f"fallback: {reason}" if reason else "fallback",
    }


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def event_eta(minutes_to: int) -> str:
    sign = "in " if minutes_to >= 0 else ""
    minutes = abs(minutes_to)
    if minutes < 60:
        text = f"{minutes}m"
    elif minutes < 24 * 60:
        text = f"{minutes // 60}h {minutes % 60}m"
    else:
        text = f"{minutes // (24 * 60)}d {(minutes % (24 * 60)) // 60}h"
    return f"{sign}{text}" if minutes_to >= 0 else f"{text} ago"


def scheduled_event_impact(name: str) -> dict:
    text = name.lower()
    if "cpi" in text or "pce" in text or "inflation" in text:
        return {
            "impact": "HIGH",
            "gold_playbook": "Hot inflation can lift yields/USD and pressure gold; soft inflation favors rate-cut repricing and gold upside.",
        }
    if "employment" in text or "payroll" in text or "jobs" in text:
        return {
            "impact": "HIGH",
            "gold_playbook": "Strong labor data can delay cuts and pressure gold; weak labor data can support safe-haven/rate-cut bids.",
        }
    if "fomc" in text or "fed" in text or "powell" in text:
        return {
            "impact": "HIGH",
            "gold_playbook": "Dovish Fed tone is gold-positive; hawkish tone is gold-negative through real rates and USD.",
        }
    return {
        "impact": "MEDIUM",
        "gold_playbook": "Treat as volatility risk until actual/forecast surprise is known.",
    }


def get_event_risk(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    events = []
    for event in UPCOMING_EVENTS:
        event_time = parse_utc(event["time_utc"])
        minutes_to = int((event_time - now).total_seconds() / 60)
        impact = scheduled_event_impact(event["name"])
        events.append({
            **event,
            **impact,
            "time_utc": event_time.strftime("%Y-%m-%d %H:%M UTC"),
            "minutes_to": minutes_to,
            "eta": event_eta(minutes_to),
        })

    current_window = [e for e in events if -30 <= e["minutes_to"] <= 60]
    if current_window:
        event = min(current_window, key=lambda e: abs(e["minutes_to"]))
        return {
            "level": "HIGH",
            "reason": "High-impact event window is active.",
            "next_event": event,
            "events": events,
        }

    medium_window = [e for e in events if -180 <= e["minutes_to"] <= 24 * 60]
    if medium_window:
        event = min(medium_window, key=lambda e: abs(e["minutes_to"]))
        return {
            "level": "MEDIUM",
            "reason": "High-impact event is close enough to distort spreads and volatility.",
            "next_event": event,
            "events": events,
        }

    watch_window = [e for e in events if 0 <= e["minutes_to"] <= 72 * 60]
    if watch_window:
        event = min(watch_window, key=lambda e: e["minutes_to"])
        return {
            "level": "WATCH",
            "reason": "High-impact event is on the calendar within 72 hours.",
            "next_event": event,
            "events": events,
        }

    future_events = [e for e in events if e["minutes_to"] > 0]
    event = min(future_events, key=lambda e: e["minutes_to"]) if future_events else None
    return {
        "level": "LOW",
        "reason": "No scheduled high-impact event within 72 hours.",
        "next_event": event,
        "events": events,
    }


def history_value_before(series: pd.Series, target: pd.Timestamp, max_hours: int = 8) -> tuple | None:
    previous = series.loc[series.index <= target]
    if previous.empty:
        return None
    ts = previous.index[-1]
    age = (target - ts).total_seconds() / 3600
    if age > max_hours:
        return None
    return ts, float(previous.iloc[-1])


def history_value_after(series: pd.Series, target: pd.Timestamp, max_hours: int = 8) -> tuple | None:
    after = series.loc[series.index >= target]
    if after.empty:
        return None
    ts = after.index[0]
    age = (ts - target).total_seconds() / 3600
    if age > max_hours:
        return None
    return ts, float(after.iloc[0])


def event_move(base: float | None, target_value: tuple | None) -> dict | None:
    if base is None or target_value is None:
        return None
    _, value = target_value
    dollars = value - base
    return {
        "dollars": round(dollars, 2),
        "pct": round((value / base - 1) * 100, 2) if base else 0,
        "direction": "UP" if dollars > 0 else "DOWN" if dollars < 0 else "FLAT",
        "close": round(value, 2),
    }


def get_event_study() -> dict:
    now = datetime.now(timezone.utc)
    cached_ts = EVENT_STUDY_CACHE["ts"]
    if cached_ts and EVENT_STUDY_CACHE["data"]:
        age = (now - cached_ts).total_seconds()
        if age < EVENT_STUDY_TTL_SECONDS:
            return EVENT_STUDY_CACHE["data"]

    try:
        import yfinance as yf
        df = yf.download("GC=F", period="90d", interval="1h",
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("empty GC=F history")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index, utc=True)
        closes = df["close"].dropna()
        rows = []

        for event in PAST_EVENTS:
            event_time = pd.Timestamp(parse_utc(event["time_utc"]))
            base_point = history_value_before(closes, event_time, max_hours=8)
            base = base_point[1] if base_point else None
            move_4h = event_move(base, history_value_after(
                closes, event_time + timedelta(hours=4), max_hours=8))
            move_24h = event_move(base, history_value_after(
                closes, event_time + timedelta(hours=24), max_hours=12))
            rows.append({
                **event,
                "time_utc": event_time.strftime("%Y-%m-%d %H:%M UTC"),
                "event_price": round(base, 2) if base else None,
                "move_4h": move_4h,
                "move_24h": move_24h,
                "status": "ok" if base else "missing price history",
            })

        moves = [abs(r["move_4h"]["dollars"]) for r in rows if r.get("move_4h")]
        avg_abs_4h = round(sum(moves) / len(moves), 2) if moves else None
        max_event = None
        if rows:
            valid = [r for r in rows if r.get("move_4h")]
            if valid:
                max_event = max(valid, key=lambda r: abs(r["move_4h"]["dollars"]))

        result = {
            "status": "ok",
            "symbol": "GC=F",
            "timeframe": "1h",
            "events": rows,
            "summary": {
                "events_measured": len([r for r in rows if r.get("move_4h")]),
                "avg_abs_4h": avg_abs_4h,
                "largest_4h_event": max_event["name"] if max_event else None,
                "largest_4h_move": max_event["move_4h"] if max_event else None,
            },
            "timestamp": now.strftime("%Y-%m-%d %H:%M UTC"),
        }
    except Exception as e:
        result = {
            "status": f"offline: {e}",
            "symbol": "GC=F",
            "timeframe": "1h",
            "events": [],
            "summary": {
                "events_measured": 0,
                "avg_abs_4h": None,
                "largest_4h_event": None,
                "largest_4h_move": None,
            },
            "timestamp": now.strftime("%Y-%m-%d %H:%M UTC"),
        }

    EVENT_STUDY_CACHE["ts"] = now
    EVENT_STUDY_CACHE["data"] = result
    return result


def get_market_context(force: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    cached_ts = CONTEXT_CACHE["ts"]
    if not force and cached_ts and CONTEXT_CACHE["data"]:
        age = (now - cached_ts).total_seconds()
        if age < CONTEXT_TTL_SECONDS:
            return CONTEXT_CACHE["data"]

    macro = get_macro_context()
    news = get_news_context()
    people = get_people_sentiment()
    if str(people.get("status", "")).startswith("offline") and news.get("headlines"):
        people = derive_people_sentiment_from_news(news, people.get("status", ""))
    event_risk = get_event_risk(now)
    score = int(clamp(macro["score"] + news["score"] + people["score"], -8, 8))
    bias = "BULLISH" if score >= 3 else "BEARISH" if score <= -3 else "NEUTRAL"
    risk_rank = {"LOW": 0, "WATCH": 1, "UNKNOWN": 1, "MEDIUM": 2, "HIGH": 3}
    news_risk = news.get("risk", "UNKNOWN")
    event_level = event_risk.get("level", "LOW")
    risk = news_risk if risk_rank.get(news_risk, 1) >= risk_rank.get(event_level, 0) else event_level
    context = {
        "score": score,
        "bias": bias,
        "risk": risk,
        "macro": macro,
        "news": news,
        "people": people,
        "event_risk": event_risk,
        "timestamp": now.strftime("%Y-%m-%d %H:%M UTC"),
        "summary": f"{bias} bias · score {score:+d} · risk {risk} · refreshed from live feeds",
    }
    CONTEXT_CACHE["ts"] = now
    CONTEXT_CACHE["data"] = context
    return context


def normalized_score(value: float, max_abs: float) -> int:
    if max_abs == 0:
        return 0
    return int(round(clamp((value / max_abs) * 100, -100, 100)))


def technical_composite(data: dict) -> dict:
    h1 = data.get("h1", {})
    m15 = data.get("m15", {})
    m5 = data.get("m5", {})
    parts = []
    if isinstance(h1.get("technical_score"), (int, float)):
        parts.append(("1H", 0.40, float(h1["technical_score"])))
    if isinstance(m15.get("technical_score"), (int, float)):
        parts.append(("15M", 0.40, float(m15["technical_score"])))
    if isinstance(m5.get("technical_score"), (int, float)):
        parts.append(("5M", 0.20, float(m5["technical_score"])))
    if not parts:
        return {"score": 0, "label": "WAIT", "parts": []}

    weight_sum = sum(weight for _, weight, _ in parts)
    score = sum(weight * value for _, weight, value in parts) / weight_sum
    label = "LONG" if score >= 55 else "SHORT" if score <= -55 else "LONG_BIAS" if score >= 20 else "SHORT_BIAS" if score <= -20 else "WAIT"
    return {
        "score": int(round(clamp(score, -100, 100))),
        "label": label,
        "parts": [{"timeframe": tf, "weight": weight, "score": int(round(value))} for tf, weight, value in parts],
    }


def execution_signal_candidates(data: dict) -> list[dict]:
    candidates = []
    for key, tf in (("h1", "1H"), ("m15", "15M"), ("m5", "5M")):
        signal_data = data.get(key, {})
        sweep = signal_data.get("liquidity_sweep", {})
        direction = sweep.get("direction", "NONE") if sweep.get("confirmed") else "NONE"
        trade = signal_data.get("trade", {})
        if direction not in ("LONG", "SHORT"):
            continue
        if trade.get("stop") is None or trade.get("target") is None:
            continue
        candidates.append({
            "key": key,
            "timeframe": tf,
            "direction": direction,
            "score": int(safe_float(signal_data.get("technical_score"))),
            "sweep": sweep,
            "trade": trade,
        })
    return candidates


def select_execution_signal(data: dict) -> dict:
    candidates = execution_signal_candidates(data)
    if not candidates:
        return {
            "key": None,
            "timeframe": None,
            "direction": "NONE",
            "score": 0,
            "sweep": {},
            "trade": {},
        }

    def rank(candidate: dict) -> float:
        timeframe_bonus = {"1H": 10, "15M": 12, "5M": 0}.get(candidate["timeframe"], 0)
        return abs(candidate["score"]) + safe_float(candidate["sweep"].get("score")) * 0.25 + timeframe_bonus

    return max(candidates, key=rank)


def micro_execution_aligned(execution: dict, m15_score: float) -> bool:
    if execution.get("timeframe") != "5M":
        return True
    if execution.get("direction") == "LONG":
        return m15_score >= 10
    if execution.get("direction") == "SHORT":
        return m15_score <= -10
    return False


def active_technical_signal(data: dict) -> str:
    return technical_composite(data)["label"]


def final_decision(data: dict, context: dict) -> dict:
    technical = technical_composite(data)
    macro_score = normalized_score(context.get("macro", {}).get("score", 0), 3)
    news_score = normalized_score(context.get("news", {}).get("score", 0), 4)
    sentiment_score = normalized_score(context.get("people", {}).get("score", 0), 3)
    technical_score = technical["score"]
    weighted_score = (
        MODEL_WEIGHTS["technical"] * technical_score
        + MODEL_WEIGHTS["news"] * news_score
        + MODEL_WEIGHTS["macro"] * macro_score
        + MODEL_WEIGHTS["sentiment"] * sentiment_score
    )
    weighted_score = int(round(clamp(weighted_score, -100, 100)))

    risk = context.get("risk", "UNKNOWN")
    event_risk = context.get("event_risk", {})
    event_level = event_risk.get("level", "LOW")
    h1 = data.get("h1", {})
    m15 = data.get("m15", {})
    m5 = data.get("m5", {})
    execution = select_execution_signal(data)
    sweep = execution.get("sweep", {})
    trade = execution.get("trade", {})
    h1_score = int(safe_float(h1.get("technical_score")))
    m15_score = int(safe_float(m15.get("technical_score")))
    m5_score = int(safe_float(m5.get("technical_score")))
    execution_score = execution.get("score", 0)
    execution_timeframe = execution.get("timeframe")
    sweep_direction = execution.get("direction", "NONE")
    execution_aligned = micro_execution_aligned(execution, m15_score)
    agreement = sum([
        1 if macro_score > 15 else -1 if macro_score < -15 else 0,
        1 if news_score > 15 else -1 if news_score < -15 else 0,
        1 if sentiment_score > 15 else -1 if sentiment_score < -15 else 0,
        1 if technical_score > 15 else -1 if technical_score < -15 else 0,
    ])
    confidence = abs(weighted_score) + (10 if abs(agreement) >= 2 else 0)
    if sweep_direction != "NONE":
        confidence += 8
    if (sweep_direction == "LONG" and h1_score < 0) or (sweep_direction == "SHORT" and h1_score > 0):
        confidence -= 12
    if risk == "MEDIUM":
        confidence -= 10
    if risk == "HIGH":
        confidence -= 25
    confidence = int(clamp(confidence, 5, 95))

    base = {
        "technical": technical["label"],
        "technical_detail": technical,
        "weights": {k: int(v * 100) for k, v in MODEL_WEIGHTS.items()},
        "weighted_scores": {
            "technical": technical_score,
            "macro": macro_score,
            "news": news_score,
            "sentiment": sentiment_score,
            "technical_weighted": round(MODEL_WEIGHTS["technical"] * technical_score, 1),
            "news_weighted": round(MODEL_WEIGHTS["news"] * news_score, 1),
            "macro_weighted": round(MODEL_WEIGHTS["macro"] * macro_score, 1),
            "sentiment_weighted": round(MODEL_WEIGHTS["sentiment"] * sentiment_score, 1),
        },
        "total_score": weighted_score,
        "confidence": confidence,
        "risk": risk,
        "event_level": event_level,
        "sweep_direction": sweep_direction,
        "sweep": sweep,
        "trade": trade,
        "h1_score": h1_score,
        "m15_score": m15_score,
        "m5_score": m5_score,
        "execution_timeframe": execution_timeframe,
        "execution_score": execution_score,
        "execution_aligned": execution_aligned,
    }

    if risk == "HIGH":
        return {
            **base,
            "direction": "WAIT",
            "quality": "BLOCKED",
            "reason": "High-impact news/event window is active. Stand down until spreads and first impulse settle.",
        }

    if sweep_direction == "NONE":
        if weighted_score >= 30:
            return {
                **base,
                "direction": "LONG_BIAS",
                "quality": "NO_TRIGGER",
                "reason": "Macro/news and EMA regime lean long, but no 1H/15M/5M execution trade has swept liquidity and reclaimed yet.",
            }
        if weighted_score <= -30:
            return {
                **base,
                "direction": "SHORT_BIAS",
                "quality": "NO_TRIGGER",
                "reason": "Macro/news and EMA regime lean short, but no 1H/15M/5M execution trade has swept liquidity and rejected yet.",
            }
        return {
            **base,
            "direction": "WAIT",
            "quality": "NO_TRIGGER",
            "reason": "No fresh 1H/15M/5M liquidity sweep. Wait for stop-run/reclaim before taking intraday risk.",
        }

    long_allowed = (
        sweep_direction == "LONG"
        and h1_score >= 10
        and execution_score >= 45
        and weighted_score >= 35
        and execution_aligned
    )
    short_allowed = (
        sweep_direction == "SHORT"
        and h1_score <= -10
        and execution_score <= -45
        and weighted_score <= -35
        and execution_aligned
    )
    countertrend_long = (
        sweep_direction == "LONG"
        and h1_score > -25
        and weighted_score >= 58
        and execution_score >= 45
        and execution_aligned
    )
    countertrend_short = (
        sweep_direction == "SHORT"
        and h1_score < 25
        and weighted_score <= -58
        and execution_score <= -45
        and execution_aligned
    )

    if long_allowed or countertrend_long:
        quality = "CAUTION" if risk == "MEDIUM" else "CONFIRMED"
        if countertrend_long and not long_allowed:
            quality = "COUNTERTREND"
        return {
            **base,
            "direction": "LONG",
            "quality": quality,
            "reason": f"{execution_timeframe} swept sell-side liquidity and reclaimed; 1H EMA regime/score supports a long. Weighted score {weighted_score:+d}.",
        }
    if short_allowed or countertrend_short:
        quality = "CAUTION" if risk == "MEDIUM" else "CONFIRMED"
        if countertrend_short and not short_allowed:
            quality = "COUNTERTREND"
        return {
            **base,
            "direction": "SHORT",
            "quality": quality,
            "reason": f"{execution_timeframe} swept buy-side liquidity and rejected; 1H EMA regime/score supports a short. Weighted score {weighted_score:+d}.",
        }

    alignment_note = " 5M execution also requires directional support from 15M." if execution_timeframe == "5M" and not execution_aligned else ""
    if sweep_direction == "LONG":
        return {
            **base,
            "direction": "LONG_BIAS",
            "quality": "MISMATCH",
            "reason": f"{execution_timeframe} liquidity sweep is long, but 1H/15M regime or macro/news score is not strong enough for execution.{alignment_note}",
        }
    if sweep_direction == "SHORT":
        return {
            **base,
            "direction": "SHORT_BIAS",
            "quality": "MISMATCH",
            "reason": f"{execution_timeframe} liquidity sweep is short, but 1H/15M regime or macro/news score is not strong enough for execution.{alignment_note}",
        }
    return {
        **base,
        "direction": "WAIT",
        "quality": "NEUTRAL",
        "reason": f"Weighted score {weighted_score:+d}: no aligned intraday edge yet.",
    }


def timeframe_trade_decisions(snapshot: dict) -> list[dict]:
    context = snapshot.get("context", {})
    final = snapshot.get("final", {})
    risk = context.get("risk", "UNKNOWN")
    event_level = context.get("event_risk", {}).get("level", "LOW")
    if risk == "HIGH":
        return []

    decisions = []
    scores = {
        "h1_score": snapshot.get("h1", {}).get("technical_score"),
        "m15_score": snapshot.get("m15", {}).get("technical_score"),
        "m5_score": snapshot.get("m5", {}).get("technical_score"),
    }
    for key, tf in (("h1", "1H"), ("m15", "15M"), ("m5", "5M")):
        signal_data = snapshot.get(key, {})
        if signal_data.get("error"):
            continue

        direction = signal_data.get("signal")
        trade = signal_data.get("trade", {})
        if direction not in ("LONG", "SHORT"):
            continue
        if trade.get("stop") is None or trade.get("target") is None:
            continue

        sweep = signal_data.get("liquidity_sweep", {})
        score = int(safe_float(signal_data.get("technical_score")))
        confidence = abs(score) + (10 if sweep.get("confirmed") else 0)
        if risk == "MEDIUM":
            confidence -= 10
        confidence = int(clamp(confidence, 5, 95))
        quality = "CAUTION" if risk == "MEDIUM" else "CONFIRMED"
        sweep_direction = sweep.get("direction", "NONE") if sweep.get("confirmed") else "NONE"
        reason = (
            f"{tf} standalone {direction} trade: score {score:+d}, "
            f"sweep {sweep_direction}, RR {trade.get('rr') or '--'}."
        )

        decisions.append({
            "technical": direction,
            "technical_detail": final.get("technical_detail", technical_composite(snapshot)),
            "weights": final.get("weights", {k: int(v * 100) for k, v in MODEL_WEIGHTS.items()}),
            "weighted_scores": final.get("weighted_scores", {}),
            "total_score": final.get("total_score"),
            "confidence": confidence,
            "risk": risk,
            "event_level": event_level,
            "sweep_direction": sweep_direction,
            "sweep": sweep,
            "trade": trade,
            **scores,
            "execution_timeframe": tf,
            "execution_score": score,
            "execution_aligned": True,
            "direction": direction,
            "quality": quality,
            "reason": reason,
        })

    return decisions


def grid_level_plan(price: float, direction: str, step: float, atr_value: float, anchor: float, account_risk: float = 10_000) -> dict:
    max_levels = 3
    per_level_risk_usd = account_risk * 0.0025
    levels = []

    if direction == "LONG":
        start = min(price, anchor)
        raw_levels = [start - step * idx for idx in range(max_levels)]
        stop = raw_levels[-1] - atr_value * 1.25
        average_entry = sum(raw_levels) / len(raw_levels)
        target = average_entry + abs(average_entry - stop) * 1.5
        invalidates = "Abort if 1H closes below EMA200, ADX drops under 16, or NFP/CPI/FOMC event window turns HIGH."
    else:
        start = max(price, anchor)
        raw_levels = [start + step * idx for idx in range(max_levels)]
        stop = raw_levels[-1] + atr_value * 1.25
        average_entry = sum(raw_levels) / len(raw_levels)
        target = average_entry - abs(stop - average_entry) * 1.5
        invalidates = "Abort if 1H closes above EMA200, ADX drops under 16, or NFP/CPI/FOMC event window turns HIGH."

    for idx, level in enumerate(raw_levels, start=1):
        distance_to_stop = abs(level - stop)
        levels.append({
            "level": idx,
            "price": round(level, 2),
            "size_oz_025pct_10k": round(per_level_risk_usd / max(distance_to_stop, 1e-9), 4),
            "risk_usd": round(per_level_risk_usd, 2),
        })

    basket_risk = per_level_risk_usd * max_levels
    basket_reward = abs(target - average_entry) * sum(level["size_oz_025pct_10k"] for level in levels)
    return {
        "max_levels": max_levels,
        "step": round(step, 2),
        "step_atr_multiple": round(step / max(atr_value, 1e-9), 2),
        "levels": levels,
        "average_entry": round(average_entry, 2),
        "stop": round(stop, 2),
        "basket_target": round(target, 2),
        "basket_risk_usd_10k": round(basket_risk, 2),
        "estimated_reward_usd_10k": round(basket_reward, 2),
        "estimated_rr": round(basket_reward / max(basket_risk, 1e-9), 2),
        "invalidates": invalidates,
        "sizing_note": "Equal risk per level; no martingale, no lot multiplier, max 0.75% account risk per full grid.",
    }


def trend_grid_bot_signal(data: dict, context: dict) -> dict:
    h1 = data.get("h1", {})
    m15 = data.get("m15", {})
    if h1.get("error") or m15.get("error"):
        return {
            "signal": "WAIT",
            "quality": "OFFLINE",
            "direction": "NONE",
            "reason": h1.get("error") or m15.get("error") or "Missing timeframe data.",
            "plan": None,
            "conditions": {},
        }

    price = safe_float(data.get("price") or h1.get("price") or m15.get("price"))
    h1_ind = h1.get("indicators", {})
    m15_ind = m15.get("indicators", {})
    h1_price = safe_float(h1.get("price"), price)
    h1_ema200 = safe_float(h1_ind.get("ema200"))
    h1_ema50 = safe_float(h1_ind.get("ema50"))
    h1_adx = safe_float(h1_ind.get("adx14"))
    h1_plus_di = safe_float(h1_ind.get("plus_di14"))
    h1_minus_di = safe_float(h1_ind.get("minus_di14"))
    m15_adx = safe_float(m15_ind.get("adx14"))
    m15_atr = max(safe_float(m15_ind.get("atr14"), 0), price * 0.001)
    m15_ema21 = safe_float(m15_ind.get("ema21"), price)
    distance_from_ema21 = safe_float(m15_ind.get("distance_from_ema21_atr"))
    atr_pct = safe_float(m15_ind.get("atr_pct"))

    bull_regime = h1_price > h1_ema200 and h1_ema50 > h1_ema200
    bear_regime = h1_price < h1_ema200 and h1_ema50 < h1_ema200
    trend_confirmed = h1_adx >= 18 or m15_adx >= 20
    volatility_ok = 0.12 <= atr_pct <= 0.95
    not_chasing = abs(distance_from_ema21) <= 1.8
    event_risk = context.get("risk", "UNKNOWN")
    event_blocked = event_risk == "HIGH"

    direction = "LONG" if bull_regime else "SHORT" if bear_regime else "NONE"
    conditions = {
        "h1_price_above_ema200": h1_price > h1_ema200,
        "h1_price_below_ema200": h1_price < h1_ema200,
        "ema50_confirms_ema200": h1_ema50 > h1_ema200 if direction == "LONG" else h1_ema50 < h1_ema200 if direction == "SHORT" else False,
        "di_confirms_direction": h1_plus_di >= h1_minus_di if direction == "LONG" else h1_minus_di >= h1_plus_di if direction == "SHORT" else False,
        "adx_trending": trend_confirmed,
        "atr_usable": volatility_ok,
        "not_chasing": not_chasing,
        "event_not_high": not event_blocked,
    }

    if event_blocked:
        return {
            "signal": "BLOCKED",
            "quality": "EVENT_RISK",
            "direction": direction,
            "reason": "High-impact event risk is active; trend grid is disabled until volatility normalizes.",
            "plan": None,
            "conditions": conditions,
            "inputs": {
                "h1_ema200": round(h1_ema200, 2),
                "h1_adx14": round(h1_adx, 1),
                "m15_adx14": round(m15_adx, 1),
                "m15_atr14": round(m15_atr, 2),
                "atr_pct": round(atr_pct, 3),
            },
        }

    if direction == "NONE":
        return {
            "signal": "WAIT",
            "quality": "NO_REGIME",
            "direction": "NONE",
            "reason": "H1 price/EMA50 are not cleanly aligned around EMA200.",
            "plan": None,
            "conditions": conditions,
            "inputs": {
                "h1_ema200": round(h1_ema200, 2),
                "h1_adx14": round(h1_adx, 1),
                "m15_adx14": round(m15_adx, 1),
                "m15_atr14": round(m15_atr, 2),
                "atr_pct": round(atr_pct, 3),
            },
        }

    if not trend_confirmed:
        return {
            "signal": "WAIT",
            "quality": "LOW_ADX",
            "direction": direction,
            "reason": "EMA200 regime exists, but ADX does not confirm enough trend strength for grid entries.",
            "plan": None,
            "conditions": conditions,
            "inputs": {
                "h1_ema200": round(h1_ema200, 2),
                "h1_adx14": round(h1_adx, 1),
                "m15_adx14": round(m15_adx, 1),
                "m15_atr14": round(m15_atr, 2),
                "atr_pct": round(atr_pct, 3),
            },
        }

    if not volatility_ok:
        return {
            "signal": "WAIT",
            "quality": "BAD_ATR",
            "direction": direction,
            "reason": "ATR regime is either too quiet or too stretched for controlled grid spacing.",
            "plan": None,
            "conditions": conditions,
            "inputs": {
                "h1_ema200": round(h1_ema200, 2),
                "h1_adx14": round(h1_adx, 1),
                "m15_adx14": round(m15_adx, 1),
                "m15_atr14": round(m15_atr, 2),
                "atr_pct": round(atr_pct, 3),
            },
        }

    step = max(m15_atr * 0.75, price * 0.0015)
    plan = grid_level_plan(price, direction, step, m15_atr, m15_ema21)
    quality = "CAUTION" if event_risk == "MEDIUM" or not not_chasing or not conditions["di_confirms_direction"] else "CONFIRMED"
    signal = "LONG_GRID" if direction == "LONG" else "SHORT_GRID"
    reason = (
        f"H1 is {direction.lower()} versus EMA200, ADX confirms trend, and ATR sets a {plan['step']:.2f} grid step. "
        "Use capped pullback entries only; no martingale."
    )
    if not not_chasing:
        reason = f"{reason} Current 15M price is extended from EMA21, so first fill should wait for pullback."

    return {
        "signal": signal,
        "quality": quality,
        "direction": direction,
        "reason": reason,
        "plan": plan,
        "conditions": conditions,
        "inputs": {
            "h1_ema200": round(h1_ema200, 2),
            "h1_ema50": round(h1_ema50, 2),
            "h1_adx14": round(h1_adx, 1),
            "h1_plus_di14": round(h1_plus_di, 1),
            "h1_minus_di14": round(h1_minus_di, 1),
            "m15_adx14": round(m15_adx, 1),
            "m15_atr14": round(m15_atr, 2),
            "atr_pct": round(atr_pct, 3),
            "distance_from_ema21_atr": round(distance_from_ema21, 2),
        },
    }


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def load_trade_history() -> dict:
    if not os.path.exists(TRADE_LOG_PATH):
        return {"version": 1, "trades": []}
    try:
        with open(TRADE_LOG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        trades = data.get("trades", []) if isinstance(data, dict) else []
        return {"version": 1, "trades": trades}
    except Exception as e:
        log.error(f"Trade history read error: {e}")
        return {"version": 1, "trades": []}


def save_trade_history(history: dict) -> bool:
    try:
        directory = os.path.dirname(TRADE_LOG_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = f"{TRADE_LOG_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        os.replace(tmp_path, TRADE_LOG_PATH)
        return True
    except Exception as e:
        log.error(f"Trade history write error: {e}")
        return False


def trade_record_id(alert_key: str) -> str:
    return hashlib.sha1(alert_key.encode("utf-8")).hexdigest()[:12]


def find_trade_record(alert_key: str) -> dict | None:
    for trade in load_trade_history()["trades"]:
        if trade.get("alert_key") == alert_key:
            return trade
    return None


def execution_key_for_timeframe(timeframe: str | None) -> str:
    if timeframe == "1H":
        return "h1"
    if timeframe == "5M":
        return "m5"
    return "m15"


def snapshot_execution_data(snapshot: dict, timeframe: str | None) -> dict:
    key = execution_key_for_timeframe(timeframe)
    execution_data = snapshot.get(key, {})
    if not execution_data or execution_data.get("error"):
        return snapshot.get("m15", {})
    return execution_data


def build_trade_record(snapshot: dict, alert_key: str, source: str = "telegram_alert") -> dict:
    final = snapshot.get("final", {})
    trade = final.get("trade", {})
    sweep = final.get("sweep", {})
    h1 = snapshot.get("h1", {})
    m15 = snapshot.get("m15", {})
    m5 = snapshot.get("m5", {})
    execution_timeframe = final.get("execution_timeframe") or "15M"
    execution_data = snapshot_execution_data(snapshot, execution_timeframe)
    return {
        "id": trade_record_id(alert_key),
        "alert_key": alert_key,
        "source": source,
        "status": "OPEN",
        "result": "OPEN",
        "direction": final.get("direction"),
        "quality": final.get("quality"),
        "execution_timeframe": execution_timeframe,
        "opened_at": snapshot.get("timestamp") or utc_now_text(),
        "opened_bar": sweep.get("bar_time") or execution_data.get("bar_time"),
        "entry": trade.get("entry"),
        "stop": trade.get("stop"),
        "target": trade.get("target"),
        "target_2": trade.get("target_2"),
        "risk": trade.get("risk"),
        "reward": trade.get("reward"),
        "rr": trade.get("rr"),
        "size_1pct_10k_oz": trade.get("position_1pct_10k_oz"),
        "weighted_score": final.get("total_score"),
        "confidence": final.get("confidence"),
        "h1_score": h1.get("technical_score"),
        "m15_score": m15.get("technical_score"),
        "m5_score": m5.get("technical_score"),
        "sweep_summary": sweep.get("summary"),
        "reason": final.get("reason"),
        "closed_at": None,
        "closed_bar": None,
        "closed_price": None,
        "observed_price": None,
        "observed_high": None,
        "observed_low": None,
        "r_multiple": None,
        "pnl_1pct_10k_usd": None,
    }


def record_trade_alert(snapshot: dict, source: str = "telegram_alert") -> tuple[dict | None, bool]:
    alert_key = trade_alert_key(snapshot)
    if not alert_key:
        return None, False

    history = load_trade_history()
    for trade in history["trades"]:
        if trade.get("alert_key") == alert_key:
            return trade, False

    record = build_trade_record(snapshot, alert_key, source=source)
    history["trades"].append(record)
    save_trade_history(history)
    return record, True


def close_trade_record(trade: dict, result: str, snapshot: dict) -> dict:
    execution_timeframe = trade.get("execution_timeframe") or "15M"
    execution_data = snapshot_execution_data(snapshot, execution_timeframe)
    price = safe_float(snapshot.get("price") or execution_data.get("price"))
    high = safe_float(execution_data.get("latest_high"), price)
    low = safe_float(execution_data.get("latest_low"), price)
    entry = safe_float(trade.get("entry"))
    stop = safe_float(trade.get("stop"))
    target = safe_float(trade.get("target"))
    risk = max(safe_float(trade.get("risk")), 1e-9)
    direction = trade.get("direction")
    closed_price = target if result == "TP" else stop
    pnl_per_oz = closed_price - entry if direction == "LONG" else entry - closed_price
    r_multiple = pnl_per_oz / risk

    trade.update({
        "status": "CLOSED",
        "result": result,
        "closed_at": snapshot.get("timestamp") or utc_now_text(),
        "closed_bar": execution_data.get("bar_time"),
        "closed_price": round(closed_price, 2),
        "observed_price": round(price, 2),
        "observed_high": round(high, 2),
        "observed_low": round(low, 2),
        "r_multiple": round(r_multiple, 2),
        "pnl_1pct_10k_usd": round(r_multiple * 100, 2),
    })
    return trade


def trade_hit_result(trade: dict, snapshot: dict) -> str | None:
    if trade.get("status") != "OPEN":
        return None

    execution_timeframe = trade.get("execution_timeframe") or "15M"
    execution_data = snapshot_execution_data(snapshot, execution_timeframe)
    current_bar = execution_data.get("bar_time")
    if current_bar and current_bar == trade.get("opened_bar"):
        return None

    price = safe_float(snapshot.get("price") or execution_data.get("price"))
    high = safe_float(execution_data.get("latest_high"), price)
    low = safe_float(execution_data.get("latest_low"), price)
    stop = safe_float(trade.get("stop"))
    target = safe_float(trade.get("target"))
    direction = trade.get("direction")

    if direction == "LONG":
        hit_sl = low <= stop
        hit_tp = high >= target
    elif direction == "SHORT":
        hit_sl = high >= stop
        hit_tp = low <= target
    else:
        return None

    if hit_sl and hit_tp:
        return "SL"
    if hit_tp:
        return "TP"
    if hit_sl:
        return "SL"
    return None


def update_trade_results(snapshot: dict) -> list[dict]:
    history = load_trade_history()
    updated = []
    for trade in history["trades"]:
        result = trade_hit_result(trade, snapshot)
        if result:
            updated.append(close_trade_record(trade, result, snapshot))

    if updated:
        save_trade_history(history)
    return updated


def trade_history_summary(trades: list[dict]) -> dict:
    closed = [trade for trade in trades if trade.get("status") == "CLOSED"]
    wins = [trade for trade in closed if trade.get("result") == "TP"]
    losses = [trade for trade in closed if trade.get("result") == "SL"]
    pnl = sum(safe_float(trade.get("pnl_1pct_10k_usd")) for trade in closed)
    r_total = sum(safe_float(trade.get("r_multiple")) for trade in closed)
    return {
        "total": len(trades),
        "open": len([trade for trade in trades if trade.get("status") == "OPEN"]),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "net_r": round(r_total, 2),
        "pnl_1pct_10k_usd": round(pnl, 2),
    }


def trade_history_payload(limit: int = 50) -> dict:
    trades = load_trade_history()["trades"]
    recent = list(reversed(trades))[:limit]
    return {
        "summary": trade_history_summary(trades),
        "trades": recent,
        "path": TRADE_LOG_PATH,
        "note": "Result tracking uses observed execution-timeframe candle high/low; if TP and SL are both touched in one bar, SL is recorded conservatively.",
    }


def get_trade_snapshot(force_context: bool = False) -> dict:
    data = get_both_signals()
    context = get_market_context(force=force_context)
    snapshot = {
        **data,
        "context": context,
        "final": final_decision(data, context),
    }
    snapshot["timeframe_trades"] = timeframe_trade_decisions(snapshot)
    return snapshot


def get_dashboard_snapshot(force: bool = False) -> dict:
    snapshot = get_trade_snapshot(force_context=force)
    return {
        **snapshot,
        "event_study": get_event_study(),
        "trade_history": trade_history_payload(),
        "trend_grid_bot": trend_grid_bot_signal(snapshot, snapshot["context"]),
    }


# ═══════════════════════════════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════════════════════════════
async def tg_send(text: str) -> bool:
    if not TG_TOKEN or not TG_CHAT_ID:
        log.warning("Telegram not configured")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.post(url, json={
                "chat_id":    TG_CHAT_ID,
                "text":       text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            })
        if r.status_code == 200:
            log.info("Telegram ✅")
            return True
        log.error(f"Telegram {r.status_code}: {r.text}")
        return False
    except Exception as e:
        log.error(f"Telegram error: {e}")
        return False


def html_text(value) -> str:
    return escape(str(value or ""), quote=False)


def fmt_money(value) -> str:
    if value is None:
        return "--"
    return f"${safe_float(value):,.2f}"


def fmt_signed_score(value) -> str:
    score = int(safe_float(value))
    return f"{score:+d}"


def fmt_trade_size(value) -> str:
    if value is None:
        return "--"
    return f"{safe_float(value):.4f} oz"


def is_executable_trade(decision: dict) -> bool:
    trade = decision.get("trade", {})
    return (
        decision.get("direction") in ("LONG", "SHORT")
        and decision.get("quality") in EXECUTABLE_TRADE_QUALITIES
        and trade.get("stop") is not None
        and trade.get("target") is not None
    )


def executable_trade_snapshots(snapshot: dict) -> list[dict]:
    trade_decisions = snapshot.get("timeframe_trades") or timeframe_trade_decisions(snapshot)
    snapshots = []
    for decision in trade_decisions:
        if is_executable_trade(decision):
            snapshots.append({**snapshot, "final": decision})
    return snapshots


def trade_alert_key(snapshot: dict) -> str | None:
    final = snapshot.get("final", {})
    if not is_executable_trade(final):
        return None

    sweep = final.get("sweep", {})
    execution_timeframe = final.get("execution_timeframe") or "15M"
    execution_data = snapshot_execution_data(snapshot, execution_timeframe)
    trigger_bar = sweep.get("bar_time") or execution_data.get("bar_time") or snapshot.get("timestamp")
    return f"{final.get('direction')}:{execution_timeframe}:{trigger_bar}"


def msg_trade_alert(snapshot: dict) -> str:
    final = snapshot["final"]
    trade = final.get("trade", {})
    context = snapshot.get("context", {})
    h1 = snapshot.get("h1", {})
    m15 = snapshot.get("m15", {})
    m5 = snapshot.get("m5", {})
    sweep = final.get("sweep", {})
    execution_timeframe = final.get("execution_timeframe") or "15M"
    execution_data = snapshot_execution_data(snapshot, execution_timeframe)
    direction = final.get("direction", "WAIT")
    quality = final.get("quality", "--")
    side = "🟢" if direction == "LONG" else "🔴"
    action = "BUY" if direction == "LONG" else "SELL"
    rr = trade.get("rr")
    rr_text = f"1 : {safe_float(rr):.2f}" if rr is not None else "--"
    target_2 = trade.get("target_2")
    target_2_line = f"🎯 <b>Target 2</b>     <code>{fmt_money(target_2)}</code>\n" if target_2 is not None else ""
    next_event = context.get("event_risk", {}).get("next_event") or {}

    return (
        f"{side} <b>XAU/USD TRADE — {html_text(execution_timeframe)} {html_text(direction)}</b>\n"
        f"<b>{action}</b> · Timeframe: <b>{html_text(execution_timeframe)}</b> · "
        f"Quality: <b>{html_text(quality)}</b> · Confidence: <b>{final.get('confidence', 0)}%</b>\n\n"
        f"💰 <b>Entry</b>        <code>{fmt_money(trade.get('entry'))}</code>\n"
        f"🛑 <b>Stop</b>         <code>{fmt_money(trade.get('stop'))}</code>  "
        f"Risk <code>{fmt_money(trade.get('risk'))}</code>\n"
        f"🎯 <b>Target 1</b>     <code>{fmt_money(trade.get('target'))}</code>  "
        f"Reward <code>{fmt_money(trade.get('reward'))}</code>\n"
        f"{target_2_line}"
        f"⚖️ <b>RR</b>           <code>{rr_text}</code>\n"
        f"💼 <b>Size</b>         <code>{fmt_trade_size(trade.get('position_1pct_10k_oz'))}</code> "
        f"for 1% risk on $10k\n\n"
        f"📊 <b>Model</b>\n"
        f"Weighted score: <code>{fmt_signed_score(final.get('total_score'))}</code> · "
        f"1H: <code>{fmt_signed_score(h1.get('technical_score'))}</code> · "
        f"15M: <code>{fmt_signed_score(m15.get('technical_score'))}</code> · "
        f"5M: <code>{fmt_signed_score(m5.get('technical_score'))}</code>\n"
        f"Risk: <b>{html_text(final.get('risk'))}</b> · Event: <b>{html_text(final.get('event_level'))}</b>\n\n"
        f"🧭 <b>Trigger</b>\n"
        f"{html_text(sweep.get('summary'))}\n"
        f"{html_text(execution_timeframe)} sweep bar: <code>{html_text(sweep.get('bar_time') or execution_data.get('bar_time'))}</code>\n\n"
        f"📋 <b>Reason</b>\n"
        f"{html_text(final.get('reason'))}\n\n"
        f"🚫 <b>Invalidation</b>\n"
        f"{html_text(trade.get('invalidates'))}\n\n"
        f"🗓 <b>Next event</b> {html_text(next_event.get('name') or 'None')} "
        f"{html_text(next_event.get('eta') or '')}\n"
        f"🕐 <i>{html_text(snapshot.get('timestamp'))}</i>\n"
        f"<i>Research only, not financial advice.</i>"
    )


def msg_trade_result(trade: dict) -> str:
    result = trade.get("result")
    direction = trade.get("direction")
    side = "✅" if result == "TP" else "🛑"
    label = "TAKE PROFIT HIT" if result == "TP" else "STOP LOSS HIT"
    pnl = safe_float(trade.get("pnl_1pct_10k_usd"))
    pnl_sign = "+" if pnl > 0 else ""
    execution_timeframe = trade.get("execution_timeframe") or "15M"
    return (
        f"{side} <b>XAU/USD RESULT — {label}</b>\n\n"
        f"Direction: <b>{html_text(execution_timeframe)} {html_text(direction)}</b>\n"
        f"Entry: <code>{fmt_money(trade.get('entry'))}</code>\n"
        f"Exit: <code>{fmt_money(trade.get('closed_price'))}</code>\n"
        f"Stop: <code>{fmt_money(trade.get('stop'))}</code> · "
        f"TP: <code>{fmt_money(trade.get('target'))}</code>\n\n"
        f"R multiple: <code>{safe_float(trade.get('r_multiple')):+.2f}R</code>\n"
        f"P/L on 1% risk / $10k: <code>{pnl_sign}${pnl:.2f}</code>\n\n"
        f"Observed {html_text(execution_timeframe)} range: <code>{fmt_money(trade.get('observed_low'))}</code> - "
        f"<code>{fmt_money(trade.get('observed_high'))}</code>\n"
        f"Closed: <i>{html_text(trade.get('closed_at'))}</i>"
    )


def msg_trade_cleared(prev_direction: str, snapshot: dict) -> str:
    final = snapshot.get("final", {})
    price = snapshot.get("price")
    reason = final.get("reason", "The weighted model no longer has an executable trade.")
    return (
        f"⚪ <b>XAU/USD TRADE CLEARED</b>\n\n"
        f"Previous <b>{html_text(prev_direction)}</b> setup is no longer executable.\n"
        f"Current decision: <b>{html_text(final.get('direction', 'WAIT'))}</b> · "
        f"Quality: <b>{html_text(final.get('quality', '--'))}</b>\n"
        f"Price: <code>{fmt_money(price)}</code>\n\n"
        f"{html_text(reason)}\n\n"
        f"🕐 <i>{html_text(snapshot.get('timestamp'))}</i>"
    )


def msg_signal(d: dict) -> str:
    sig   = d["signal"]
    tf    = d["timeframe"]
    price = d["price"];    stop = d["stop"];    target = d["target"]
    sd    = d["stop_dist"];    td = d["target_dist"]
    ind   = d["indicators"]
    rsi14 = ind["rsi14"];  e21   = ind["ema21"]
    e50   = ind["ema50"];  e200  = ind["ema200"]
    atr14 = ind["atr14"]
    ts    = d["timestamp"]
    chg   = d["change"];   chgp  = d["change_pct"]
    arrow = "▲" if chg >= 0 else "▼"
    pos   = round(100 / sd, 4) if sd > 0 else "—"
    tf_icon = "⏱" if tf == "5M" else "⚡" if tf == "15M" else "🕐"
    score = d.get("technical_score", 0)

    if sig == "LONG":
        head  = f"🟢 <b>LONG  {tf_icon} {tf}  —  XAU/USD GOLD</b>"
        act   = "📈 BUY at market price"
        plan1 = "Price drops below SL → exit"
        plan2 = "Price reaches TP → take profit"
    else:
        head  = f"🔴 <b>SHORT  {tf_icon} {tf}  —  XAU/USD GOLD</b>"
        act   = "📉 SELL at market price"
        plan1 = "Price rises above SL → exit"
        plan2 = "Price falls to TP → take profit"

    return (
        f"{head}\n{act}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 <b>ENTRY</b>        <code>${price:,.2f}</code>\n"
        f"🛑 <b>STOP LOSS</b>   <code>${stop:,.2f}</code>  <i>(−${sd:.2f})</i>\n"
        f"🎯 <b>TAKE PROFIT</b> <code>${target:,.2f}</code>  <i>(+${td:.2f})</i>\n"
        f"⚖️ <b>RISK/REWARD</b>  <code>1 : 2</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Technical confirmation</b>\n"
        f"  Score:   <code>{score:+d}/100</code>   RSI(14): <code>{rsi14:.1f}</code>\n"
        f"  EMA21/50/200: <code>${e21:,.2f}</code> / <code>${e50:,.2f}</code> / <code>${e200:,.2f}</code>\n"
        f"  ATR(14): <code>${atr14:.2f}</code>   24h: <code>{arrow} ${abs(chg):.2f} ({chgp:+.2f}%)</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 <b>Size</b>  (1% risk / $10k)\n"
        f"  Risk $100 → <code>{pos} oz gold</code>\n\n"
        f"📋 <b>Plan</b>\n"
        f"  • {plan1}\n"
        f"  • {plan2}\n"
        f"  • Opposite signal → exit early\n\n"
        f"🕐 <i>{ts}</i>"
    )


def msg_approaching(d: dict) -> str:
    tf    = d["timeframe"]
    cond  = d["conditions"];  ind = d["indicators"]
    price = d["price"];  stop = d["stop"];  target = d["target"]
    sd    = d["stop_dist"];   td  = d["target_dist"]
    tf_icon   = "⏱" if tf == "5M" else "⚡" if tf == "15M" else "🕐"
    direction = d.get("setup_direction", "NONE")
    rows = "\n".join([
        ("✅" if cond.get("trend_aligned")       else "❌") + " EMA regime aligned",
        ("✅" if cond.get("liquidity_sweep")     else "❌") + " Liquidity sweep confirmed",
        ("✅" if cond.get("ema_reclaim")         else "❌") + " EMA reclaim/rejection valid",
        ("✅" if cond.get("momentum_confirmed")  else "❌") + " Momentum confirmed",
        ("✅" if cond.get("not_overextended")    else "❌") + " Not RSI-overextended",
    ])
    return (
        f"⚡ <b>Forming  {tf_icon} {tf}  —  XAU/USD</b>\n\n"
        f"<b>{d['conditions_met']} of {d['max_conditions']} conditions met</b>  ·  Direction: <b>{direction}</b>\n\n"
        f"{rows}\n\n"
        f"💰 Price:  <code>${price:,.2f}</code>\n"
        f"🛑 Est SL: <code>${stop:,.2f}</code>  (−${sd:.2f})\n"
        f"🎯 Est TP: <code>${target:,.2f}</code>  (+${td:.2f})\n"
        f"📊 Score:  <code>{d.get('technical_score', 0):+d}/100</code>\n\n"
        f"<i>⚠️ Not confirmed — wait for full signal.</i>\n"
        f"🕐 <i>{d['timestamp']}</i>"
    )


def msg_cleared(tf: str, prev: str, price: float, ts: str) -> str:
    tf_icon = "⏱" if tf == "5M" else "⚡" if tf == "15M" else "🕐"
    return (
        f"⚪ <b>Cleared  {tf_icon} {tf}  —  XAU/USD</b>\n\n"
        f"The <b>{prev}</b> signal is no longer active.\n\n"
        f"💰 Current price: <code>${price:,.2f}</code>\n\n"
        f"Watching for next setup...\n"
        f"🕐 <i>{ts}</i>"
    )


def msg_startup() -> str:
    interval_minutes = max(1, TRADE_CHECK_SECONDS // 60)
    return (
        "✅ <b>XAU/USD Intelligence Bot Online</b>\n\n"
        f"Scanning for executable trades every <b>{interval_minutes} minutes</b>.\n"
        "Telegram alerts are sent for executable <b>1H</b>, <b>15M</b>, and <b>5M</b> trades.\n\n"
        "<b>Model:</b> Technical 50% · News 20% · Macro 15% · Sentiment 15%\n"
        "<b>Strategy:</b> 1H/15M/5M liquidity sweep/reclaim · 1:2 RR\n"
        "<b>Data:</b> yfinance + GDELT/Fed RSS + scheduled macro event seeds"
    )


# ═══════════════════════════════════════════════════════════════════
#  BACKGROUND CHECKERS
# ═══════════════════════════════════════════════════════════════════
async def trade_checker(check_secs: int):
    log.info(f"[TRADE] started — every {check_secs}s")
    last_alert_keys = set()

    while True:
        try:
            snapshot = get_trade_snapshot()
            for closed_trade in update_trade_results(snapshot):
                log.info(f"[TRADE] {closed_trade.get('result')} {closed_trade.get('direction')} {closed_trade.get('id')}")
                await tg_send(msg_trade_result(closed_trade))

            trade_snapshots = executable_trade_snapshots(snapshot)
            if trade_snapshots:
                for trade_snapshot in trade_snapshots:
                    final = trade_snapshot.get("final", {})
                    key = trade_alert_key(trade_snapshot)
                    if not key:
                        continue

                    direction = final.get("direction")
                    timeframe = final.get("execution_timeframe")
                    existing_trade = find_trade_record(key)
                    if existing_trade:
                        last_alert_keys.add(key)
                        log.info(f"[TRADE] {timeframe} {direction} already recorded as {existing_trade.get('id')}")
                    elif key not in last_alert_keys:
                        log.info(f"[TRADE] 🚨 {timeframe} {direction} @ ${final.get('trade', {}).get('entry')}")
                        sent = await tg_send(msg_trade_alert(trade_snapshot))
                        if sent:
                            record_trade_alert(trade_snapshot)
                            last_alert_keys.add(key)
                    else:
                        log.info(f"[TRADE] active {timeframe} {direction}; duplicate alert suppressed")
            else:
                final = snapshot.get("final", {})
                log.info(f"[TRADE] no executable timeframe trade: {final.get('direction')} {final.get('quality')}")

        except Exception as e:
            log.error(f"[TRADE] error: {e}")

        await asyncio.sleep(check_secs)


async def checker(interval: str, check_secs: int):
    tf = timeframe_label(interval)
    log.info(f"[{tf}] started — every {check_secs}s")
    prev_signal   = None
    prev_bar_time = None
    warned_4of5   = False

    while True:
        await asyncio.sleep(check_secs)
        try:
            log.info(f"[{tf}] checking…")
            data = get_signal(interval)
            if "error" in data:
                log.warning(f"[{tf}] {data['error']}")
                continue

            sig      = data["signal"]
            bar_time = data["bar_time"]
            conds    = data["conditions_met"]
            context  = get_market_context()
            full_data = get_both_signals()
            full_data["h1" if interval == "1h" else "m5" if interval == "5m" else "m15"] = data
            decision = final_decision(full_data, context)
            approved_signal = (
                interval in ("15m", "5m")
                and decision.get("direction") == sig
                and decision.get("quality") in ("CONFIRMED", "CAUTION", "COUNTERTREND")
            )

            if sig in ("LONG", "SHORT"):
                if not approved_signal:
                    log.info(f"[{tf}] technical {sig} blocked by weighted model: {decision.get('direction')} {decision.get('total_score')}")
                    if prev_signal in ("LONG", "SHORT"):
                        await tg_send(msg_cleared(tf, prev_signal, data["price"], data["timestamp"]))
                    prev_signal = None; prev_bar_time = None; warned_4of5 = False
                    continue
                is_new = (sig != prev_signal) or (bar_time != prev_bar_time)
                if is_new:
                    log.info(f"[{tf}] 🚨 {sig} @ ${data['price']}")
                    await tg_send(msg_signal(data))
                    prev_signal   = sig
                    prev_bar_time = bar_time
                    warned_4of5   = False

            elif sig == "WAIT" and prev_signal in ("LONG", "SHORT"):
                await tg_send(msg_cleared(tf, prev_signal, data["price"], data["timestamp"]))
                prev_signal = None;  prev_bar_time = None;  warned_4of5 = False

            elif sig == "WAIT" and conds >= 4 and not warned_4of5:
                await tg_send(msg_approaching(data))
                warned_4of5 = True

            elif conds < 4:
                warned_4of5 = False

        except Exception as e:
            log.error(f"[{tf}] error: {e}")


# ═══════════════════════════════════════════════════════════════════
#  APP LIFESPAN
# ═══════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    await tg_send(msg_startup())
    trade_task = asyncio.create_task(trade_checker(TRADE_CHECK_SECONDS))
    log.info("Trade checker running")
    yield
    trade_task.cancel()
    with suppress(asyncio.CancelledError):
        await trade_task

app = FastAPI(title="XAU/USD Intelligence Signal Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════
@app.get("/api/signals")
def api_signals():
    return JSONResponse(get_both_signals())

@app.get("/api/dashboard")
def api_dashboard(refresh: bool = False):
    return JSONResponse(get_dashboard_snapshot(force=refresh))

@app.get("/api/trend-grid-bot")
def api_trend_grid_bot(refresh: bool = False):
    data = get_both_signals()
    context = get_market_context(force=refresh)
    return JSONResponse(trend_grid_bot_signal(data, context))

@app.get("/api/signal")
def api_signal():
    return JSONResponse(get_signal("1h"))

@app.get("/health")
@app.head("/health")
def health():
    return {
        "status":   "ok",
        "time":     datetime.now(timezone.utc).isoformat(),
        "mode":     "EMA regime + 1H/15M/5M liquidity sweep + live news/event sentiment",
        "data":     "yfinance GC=F + DXY/10Y/VIX + GDELT/Fed RSS + scheduled macro events",
        "telegram": "configured ✅" if TG_TOKEN and TG_CHAT_ID else "NOT configured ❌",
        "trade_check_seconds": TRADE_CHECK_SECONDS,
    }

@app.get("/api/test-notification")
async def test_notification(token: str = ""):
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return JSONResponse({"error": "admin token required"}, status_code=403)
    if not TG_TOKEN or not TG_CHAT_ID:
        return JSONResponse({"error": "Telegram not configured"}, status_code=400)
    ok = await tg_send(
        "🧪 <b>Test — XAU/USD Intelligence Bot</b>\n\n"
        "✅ Telegram working!\n\n"
        f"Trade scanner interval: {max(1, TRADE_CHECK_SECONDS // 60)} minutes\n"
        "Alerts send when 1H, 15M, or 5M has an executable LONG/SHORT trade.\n\n"
        "<i>Test only.</i>"
    )
    return {"sent": ok}

@app.get("/api/check-now")
async def check_now(token: str = ""):
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return JSONResponse({"error": "admin token required"}, status_code=403)
    snapshot = get_trade_snapshot(force_context=True)
    trade_snapshots = executable_trade_snapshots(snapshot)
    if trade_snapshots:
        notifications = []
        for trade_snapshot in trade_snapshots:
            decision = trade_snapshot.get("final", {})
            alert_key = trade_alert_key(trade_snapshot)
            existing_trade = find_trade_record(alert_key) if alert_key else None
            if existing_trade:
                notifications.append({
                    "sent": False,
                    "already_recorded": True,
                    "timeframe": decision.get("execution_timeframe"),
                    "direction": decision.get("direction"),
                    "trade_record": existing_trade,
                    "trade_alert_key": alert_key,
                })
                continue

            ok = await tg_send(msg_trade_alert(trade_snapshot))
            record, recorded = record_trade_alert(trade_snapshot, source="check_now") if ok else (None, False)
            notifications.append({
                "sent": ok,
                "recorded": recorded,
                "timeframe": decision.get("execution_timeframe"),
                "direction": decision.get("direction"),
                "trade_record": record,
                "trade_alert_key": alert_key,
            })

        sent_any = any(item["sent"] for item in notifications)
        return {
            **snapshot,
            "notification_sent": sent_any,
            "notifications": notifications,
        }
    decision = snapshot.get("final", {})
    return {
        **snapshot,
        "notification_sent": False,
        "reason": decision.get("reason"),
    }


@app.get("/api/trades")
def api_trades(refresh: bool = False, limit: int = 100):
    if refresh:
        data = get_both_signals()
        update_trade_results(data)
    return JSONResponse(trade_history_payload(limit=max(1, min(limit, 500))))


# ═══════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0b0b09">
<title>AURUM Risk OS</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b0b09;color:#ece7dc;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1440px;margin:0 auto;padding:16px}.top{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:start;margin-bottom:12px}
h1{font-size:18px;margin:0;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.sub{font-size:12px;color:#9f9a8f;margin-top:4px}.stamp{font-size:12px;color:#9f9a8f;text-align:right}
button,.navlink,.tablink{border:1px solid #494534;background:#171712;color:#ece7dc;border-radius:6px;padding:8px 12px;cursor:pointer;font-weight:700;text-decoration:none;display:inline-block;margin-left:6px}button:hover,.navlink:hover,.tablink:hover{border-color:#b99036}
.tabs-nav{display:flex;gap:8px;margin:12px 0}.tablink{margin-left:0}.tablink.active{background:#2a2518;border-color:#b99036;color:#f2c76b}
.grid{display:grid;gap:10px}.hero{grid-template-columns:1.25fr .75fr}.four{grid-template-columns:repeat(4,1fr)}.three{grid-template-columns:repeat(3,1fr)}.two{grid-template-columns:repeat(2,1fr)}
.panel{background:#141410;border:1px solid #2d2b22;border-radius:8px;padding:13px;box-shadow:0 1px 0 rgba(255,255,255,.03) inset}.panel.tight{padding:10px}
.label{font-size:10px;color:#9f9a8f;text-transform:uppercase;letter-spacing:.13em}.muted{color:#9f9a8f}.small{font-size:12px}.tiny{font-size:11px}.long{color:#4ecb71}.short{color:#ef6a5b}.wait{color:#d7a93f}.blue{color:#6bb7d7}.violet{color:#b999e6}
.decision{min-height:255px;display:flex;flex-direction:column;justify-content:space-between}.direction{font-size:58px;line-height:.92;font-weight:900;letter-spacing:-.03em;margin:8px 0 5px}.reason{font-size:13px;color:#c9c0ae;line-height:1.45;max-width:900px}
.scoreline{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.pill{border:1px solid #393528;background:#1b1a14;border-radius:999px;padding:6px 9px;font-size:12px;color:#d8d0bf}
.barrow{display:grid;grid-template-columns:92px 1fr 80px;gap:9px;align-items:center;margin-top:9px}.bar{height:8px;background:#2a281f;border-radius:999px;overflow:hidden}.bar span{display:block;height:100%;width:0;border-radius:999px}
.price{font-size:38px;font-weight:900;margin:4px 0}.metric{background:#10100c;border:1px solid #28251d;border-radius:7px;padding:9px}.metric b{display:block;font-size:18px;margin-top:4px}.metric .label{font-size:9px}
.section-title{display:flex;justify-content:space-between;gap:10px;align-items:center;margin:16px 0 8px}.section-title h2{font-size:12px;margin:0;text-transform:uppercase;letter-spacing:.14em;color:#d8d0bf}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;border-bottom:1px solid #2c2a21;padding:8px 6px;vertical-align:top}th{color:#9f9a8f;font-size:10px;text-transform:uppercase;letter-spacing:.1em}tr:last-child td{border-bottom:0}
canvas{width:100%;height:92px;display:block;margin-top:8px}.tf-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.tf-title{font-size:20px;font-weight:850}.tf-score{font-size:28px;font-weight:900}
.kv{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:10px}.kv div{background:#10100c;border:1px solid #28251d;border-radius:7px;padding:8px}.kv span{display:block;font-size:9px;color:#9f9a8f;text-transform:uppercase;letter-spacing:.1em}.kv b{font-size:13px;margin-top:3px;display:block}
.headline{display:grid;grid-template-columns:8px 1fr auto;gap:9px;border-bottom:1px solid #2c2a21;padding:8px 0;font-size:12px;line-height:1.35}.headline:last-child{border-bottom:0}.dot{width:8px;height:8px;border-radius:99px;background:#9f9a8f;margin-top:5px}
.playbook{line-height:1.45;color:#c9c0ae}.split{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:start}.risk{font-size:25px;font-weight:900}.foot{font-size:11px;color:#817b70;margin-top:14px;line-height:1.45}
@media(max-width:920px){main{padding:11px}.top,.hero,.four,.three,.two{grid-template-columns:1fr}.direction{font-size:42px}.kv{grid-template-columns:repeat(2,1fr)}.stamp{text-align:left}.barrow{grid-template-columns:76px 1fr 62px}}
</style>
</head>
<body>
<main>
  <div class="top">
    <div>
      <h1>AURUM Risk OS</h1>
      <div class="sub">XAU/USD intraday cockpit: live news search, scheduled event risk, public-web sentiment, 1H EMA regime, 15M/5M liquidity sweep execution.</div>
      <div class="tabs-nav"><a class="tablink active" href="/">Cockpit</a><a class="tablink" href="/trades">Journal</a></div>
    </div>
    <div class="stamp"><div id="status">loading...</div><button onclick="loadDashboard(true)">Refresh Feed</button><a class="navlink" href="/trades">Journal</a></div>
  </div>
  <div id="app"><div class="panel">Loading market state...</div></div>
  <div class="foot">Research only, not financial advice. Free feeds can lag or fail; liquidity-sweep signals are rules-based approximations, not broker-grade order-flow data.</div>
</main>
<script>
const money=v=>v==null?'--':'$'+Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const signed=v=>v==null?'--':(v>0?'+':'')+Number(v).toFixed(2);
const pct=v=>v==null?'--':(v>0?'+':'')+Number(v).toFixed(2)+'%';
const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const cls=s=>s>20?'long':s<-20?'short':'wait';
const dirCls=d=>(d||'').includes('LONG')?'long':(d||'').includes('SHORT')?'short':'wait';
const biasCls=d=>(d||'').toUpperCase().includes('BULL')?'long':(d||'').toUpperCase().includes('BEAR')?'short':'wait';
const impactCls=v=>v==='HIGH'?'short':v==='MEDIUM'||v==='WATCH'?'wait':'long';

function moveCell(move){
  if(!move)return '<span class="muted">closed/no data</span>';
  const c=move.dollars>0?'long':move.dollars<0?'short':'muted';
  return `<span class="${c}">${move.dollars>0?'+':''}${money(move.dollars)} (${pct(move.pct)})</span>`;
}

function scoreBar(label, weight, raw, weighted){
  const abs=Math.min(100,Math.abs(raw||0));
  const color=(raw||0)>20?'#4ecb71':(raw||0)<-20?'#ef6a5b':'#d7a93f';
  return `<div class="barrow">
    <div class="tiny">${label} <span class="muted">${weight}%</span></div>
    <div class="bar"><span style="width:${abs}%;background:${color}"></span></div>
    <div class="tiny ${cls(raw||0)}">${raw>0?'+':''}${raw||0} <span class="muted">${weighted>0?'+':''}${weighted}</span></div>
  </div>`;
}

function draw(canvas, chart, score){
  if(!canvas||!chart?.price?.length)return;
  const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=92;
  canvas.width=w*dpr;canvas.height=h*dpr;
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
  const price=chart.price||[],ema21=chart.ema21||[],ema200=chart.ema200||[];
  const all=[...price,...ema21,...ema200].filter(Number.isFinite);
  const min=Math.min(...all),max=Math.max(...all),range=max-min||1;
  const x=i=>price.length<=1?0:(i/(price.length-1))*w;
  const y=v=>h-7-((v-min)/range)*(h-14);
  ctx.strokeStyle='#2d2b22';ctx.lineWidth=1;
  for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo(0,h*i/4);ctx.lineTo(w,h*i/4);ctx.stroke();}
  [[ema200,'#756f64',[4,4]],[ema21,'#d7a93f',[2,3]],[price,score>20?'#4ecb71':score<-20?'#ef6a5b':'#ece7dc',[]]].forEach(([series,color,dash])=>{
    if(!series.length)return;
    ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=series===price?2:1.2;ctx.beginPath();
    series.forEach((v,i)=>i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v)));
    ctx.stroke();ctx.setLineDash([]);
  });
}

function metrics(items){
  return items.map(([label,value,klass,hint])=>`<div class="metric"><span class="label">${esc(label)}</span><b class="${klass||''}">${value}</b><div class="tiny muted">${esc(hint||'')}</div></div>`).join('');
}

function renderTf(key,data){
  const d=data?.[key]||{}, ind=d.indicators||{}, score=d.technical_score||0, comp=d.components||{}, sweep=d.liquidity_sweep||{};
  const id='chart-'+key;
  const label=key==='h1'?'1H EMA regime':key==='m15'?'15M execution tape':'5M micro execution tape';
  return `<section class="panel">
    <div class="tf-head">
      <div><div class="label">${label}</div><div class="tf-title ${dirCls(d.setup_direction)}">${esc(d.setup_direction||'NONE')}</div></div>
      <div class="tf-score ${cls(score)}">${score>0?'+':''}${score}</div>
    </div>
    <canvas id="${id}"></canvas>
    <div class="kv">
      <div><span>EMA 9/21</span><b>${money(ind.ema9)} / ${money(ind.ema21)}</b></div>
      <div><span>EMA 50/200</span><b>${money(ind.ema50)} / ${money(ind.ema200)}</b></div>
      <div><span>Momentum</span><b class="${cls(ind.momentum_pct||0)}">${pct(ind.momentum_pct)}</b></div>
      <div><span>Sweep</span><b class="${dirCls(sweep.direction)}">${esc(sweep.direction||'NONE')}</b></div>
    </div>
    <div class="scoreline">
      <span class="pill">Trend <b class="${cls(comp.trend||0)}">${comp.trend>0?'+':''}${comp.trend??0}</b></span>
      <span class="pill">ADX <b>${ind.adx14??'--'}</b></span>
      <span class="pill">Sweep <b class="${cls(comp.liquidity_sweep||0)}">${comp.liquidity_sweep>0?'+':''}${comp.liquidity_sweep??0}</b></span>
      <span class="pill">RSI <b>${ind.rsi14??'--'}</b></span>
      <span class="pill">ATR <b>${money(ind.atr14)}</b></span>
    </div>
  </section>`;
}

function renderGridBot(bot){
  const plan=bot.plan||{}, inputs=bot.inputs||{}, conditions=bot.conditions||{};
  const rows=(plan.levels||[]).map(l=>`<tr><td>${l.level}</td><td>${money(l.price)}</td><td>${l.size_oz_025pct_10k} oz</td><td>${money(l.risk_usd)}</td></tr>`).join('')||'<tr><td colspan="4" class="muted">No active grid levels.</td></tr>';
  const conditionRows=Object.entries(conditions).map(([key,value])=>`<span class="pill">${esc(key.replaceAll('_',' '))} <b class="${value?'long':'short'}">${value?'YES':'NO'}</b></span>`).join('');
  return `<section class="panel">
    <div class="split">
      <div>
        <div class="label">Trend-following grid bot</div>
        <div class="tf-title ${dirCls(bot.signal)}">${esc(bot.signal||'WAIT')}</div>
        <div class="reason">${esc(bot.reason||'')}</div>
      </div>
      <div class="risk ${dirCls(bot.direction)}">${esc(bot.quality||'--')}</div>
    </div>
    <div class="grid four" style="margin-top:12px">${metrics([
      ['EMA200 H1',money(inputs.h1_ema200),'',`EMA50 ${money(inputs.h1_ema50)}`],
      ['ADX H1/M15',`${inputs.h1_adx14??'--'} / ${inputs.m15_adx14??'--'}`,'wait','trend strength'],
      ['ATR Step',money(plan.step),'blue',`${plan.step_atr_multiple??'--'}x M15 ATR`],
      ['Basket RR',plan.estimated_rr??'--',plan.estimated_rr>=1?'long':'wait',`risk ${money(plan.basket_risk_usd_10k)}`],
    ])}</div>
    <div class="scoreline">${conditionRows}</div>
    <table style="margin-top:10px"><thead><tr><th>Level</th><th>Limit price</th><th>Size</th><th>Risk</th></tr></thead><tbody>${rows}</tbody></table>
    <div class="scoreline">
      <span class="pill">Avg <b>${money(plan.average_entry)}</b></span>
      <span class="pill">Stop <b class="short">${money(plan.stop)}</b></span>
      <span class="pill">Basket TP <b class="long">${money(plan.basket_target)}</b></span>
      <span class="pill">Max levels <b>${plan.max_levels??'--'}</b></span>
    </div>
    <div class="tiny muted" style="margin-top:8px">${esc(plan.sizing_note||'No martingale. Signal only.')}</div>
  </section>`;
}

function render(data){
  const final=data.final||{}, context=data.context||{}, macro=context.macro||{}, news=context.news||{}, people=context.people||{}, event=context.event_risk||{};
  const next=event.next_event||{}, trade=final.trade||{}, sweep=final.sweep||{}, ws=final.weighted_scores||{}, weights=final.weights||{}, study=data.event_study||{}, gridBot=data.trend_grid_bot||{};
  const direction=final.direction||'WAIT', riskColor=impactCls(context.risk);
  const drivers=metrics((macro.drivers||[]).map(d=>[d.label,d.value==null?'--':d.value+(d.unit||''),d.score>0?'long':d.score<0?'short':'muted',`${d.move} ${pct(d.change_pct)}`]));
  const tradeMetrics=metrics([
    ['Entry',money(trade.entry),'',sweep.summary||'No sweep yet'],
    ['Stop',money(trade.stop),'short',trade.invalidates||'No active plan'],
    ['Target 1',money(trade.target),'long',`RR ${trade.rr??'--'}`],
    ['Size',trade.position_1pct_10k_oz==null?'--':trade.position_1pct_10k_oz+' oz','wait','1% risk on $10k'],
  ]);
  const newsRows=(news.headlines||[]).slice(0,7).map(h=>`<div class="headline">
    <span class="dot" style="background:${h.bias==='bullish'?'#4ecb71':h.bias==='bearish'?'#ef6a5b':'#9f9a8f'}"></span>
    <div>${esc(h.title)}<div class="tiny muted">${esc(h.source||news.source||'news')} · ${esc(h.reason||'')}</div></div>
    <div class="tiny ${impactCls(h.impact)}">${esc(h.impact||'LOW')}</div>
  </div>`).join('')||'<div class="muted small">Current headline feed unavailable.</div>';
  const peopleRows=(people.samples||[]).slice(0,4).map(s=>`<div class="headline">
    <span class="dot" style="background:${s.bias==='bullish'?'#4ecb71':s.bias==='bearish'?'#ef6a5b':'#9f9a8f'}"></span>
    <div>${esc(s.title)}<div class="tiny muted">${esc(s.source||people.source||'public web')} · ${esc(s.reason||'')}</div></div>
    <div class="tiny ${biasCls(s.bias)}">${esc(s.bias||'neutral')}</div>
  </div>`).join('')||'<div class="muted small">Public-web sentiment feed unavailable.</div>';
  const eventRows=(event.events||[]).map(e=>`<tr><td>${esc(e.name)}</td><td>${esc(e.period)}</td><td>${esc(e.eta)}</td><td class="${impactCls(e.impact)}">${esc(e.impact)}</td><td>${esc(e.gold_playbook)}</td></tr>`).join('');
  const studyRows=(study.events||[]).map(e=>`<tr><td>${esc(e.name)}</td><td>${esc(e.period)}</td><td>${money(e.event_price)}</td><td>${moveCell(e.move_4h)}</td><td>${moveCell(e.move_24h)}</td></tr>`).join('')||'<tr><td colspan="5" class="muted">Event study unavailable.</td></tr>';

  document.getElementById('app').innerHTML=`<div class="grid hero">
    <section class="panel decision">
      <div>
        <div class="label">Investment command decision</div>
        <div class="direction ${dirCls(direction)}">${esc(direction).replace('_',' ')}</div>
        <div class="reason">${esc(final.reason)}</div>
        <div class="scoreline">
          <span class="pill">Score <b class="${cls(final.total_score||0)}">${final.total_score>0?'+':''}${final.total_score??0}</b></span>
          <span class="pill">Confidence <b>${final.confidence??0}%</b></span>
          <span class="pill">Quality <b>${esc(final.quality||'--')}</b></span>
          <span class="pill">Risk <b class="${riskColor}">${esc(context.risk||'--')}</b></span>
          <span class="pill">Exec sweep <b class="${dirCls(final.sweep_direction)}">${esc(final.execution_timeframe||'--')} ${esc(final.sweep_direction||'NONE')}</b></span>
        </div>
      </div>
      <div>
        ${scoreBar('Technical',weights.technical||50,ws.technical||0,ws.technical_weighted||0)}
        ${scoreBar('News',weights.news||20,ws.news||0,ws.news_weighted||0)}
        ${scoreBar('Macro',weights.macro||15,ws.macro||0,ws.macro_weighted||0)}
        ${scoreBar('Sentiment',weights.sentiment||15,ws.sentiment||0,ws.sentiment_weighted||0)}
      </div>
    </section>
    <section class="panel">
      <div class="split"><div><div class="label">Live gold futures proxy</div><div class="price">${money(data.price)}</div><div class="${(data.h1?.change||0)>=0?'long':'short'}">${signed(data.h1?.change)} (${pct(data.h1?.change_pct)}) 24h</div></div><div class="risk ${impactCls(event.level)}">${esc(event.level||'LOW')}</div></div>
      <div class="grid three" style="margin-top:12px">${drivers}</div>
      <div class="scoreline">
        <span class="pill">News <b class="${biasCls(news.bias)}">${esc(news.bias||'MIXED')}</b></span>
        <span class="pill">Public sentiment <b class="${biasCls(people.bias)}">${esc(people.bias||'MIXED')}</b></span>
        <span class="pill">Bull/Bear <b>${people.bullish_count??0}/${people.bearish_count??0}</b></span>
      </div>
    </section>
  </div>

  <div class="section-title"><h2>Execution Playbook</h2><span class="muted small">${esc(sweep.summary||'Waiting for sweep')}</span></div>
  <div class="grid two">
    <section class="panel"><div class="grid four">${tradeMetrics}</div></section>
    <section class="panel"><div class="label">Next scheduled catalyst</div><div class="split"><div><b>${esc(next.name||'None')}</b><div class="small muted">${esc(next.period||'')} ${next.time_utc?'. '+esc(next.time_utc):''}</div><div class="playbook tiny">${esc(next.gold_playbook||'')}</div></div><div class="risk ${impactCls(event.level)}">${esc(next.eta||'')}</div></div></section>
  </div>

  <div class="section-title"><h2>EMA200 ADX ATR Grid Signal</h2><span class="muted small">trend-following, ATR-spaced, capped, non-martingale</span></div>
  ${renderGridBot(gridBot)}

  <div class="section-title"><h2>Technical State</h2><span class="muted small">1H regime + 15M/5M liquidity sweep/reclaim</span></div>
  <div class="grid three">${renderTf('h1',data)}${renderTf('m15',data)}${renderTf('m5',data)}</div>

  <div class="section-title"><h2>Live News And Public Sentiment</h2><span class="muted small">${esc(context.timestamp||'')}</span></div>
  <div class="grid two">
    <section class="panel"><div class="label">Current market news impact</div>${newsRows}</section>
    <section class="panel"><div class="label">People/public-web sentiment proxy</div>${peopleRows}</section>
  </div>

  <div class="section-title"><h2>Forward Event Tape</h2><span class="muted small">${esc(event.reason||'')}</span></div>
  <section class="panel"><table><thead><tr><th>Event</th><th>Period</th><th>ETA</th><th>Impact</th><th>Gold impact map</th></tr></thead><tbody>${eventRows}</tbody></table></section>

  <div class="section-title"><h2>Stress And Event Verification</h2><span class="muted small">Hourly GC=F reaction after recent catalysts</span></div>
  <section class="panel">
    <table><thead><tr><th>Event</th><th>Period</th><th>Event price</th><th>+4h</th><th>+24h</th></tr></thead><tbody>${studyRows}</tbody></table>
    <div class="scoreline"><span class="pill">Measured <b>${study.summary?.events_measured??0}</b></span><span class="pill">Avg abs +4h <b>${money(study.summary?.avg_abs_4h)}</b></span><span class="pill">Largest +4h <b>${moveCell(study.summary?.largest_4h_move)}</b></span></div>
  </section>`;
  requestAnimationFrame(()=>{
    draw(document.getElementById('chart-h1'),data.h1?.chart,data.h1?.technical_score||0);
    draw(document.getElementById('chart-m15'),data.m15?.chart,data.m15?.technical_score||0);
    draw(document.getElementById('chart-m5'),data.m5?.chart,data.m5?.technical_score||0);
  });
}

async function loadDashboard(force=false){
  document.getElementById('status').textContent='searching live feeds...';
  try{
    const response=await fetch(`/api/dashboard?refresh=${force ? '1' : '0'}&ts=${Date.now()}`);
    if(!response.ok)throw new Error('HTTP '+response.status);
    const data=await response.json();
    render(data);
    document.getElementById('status').textContent='live '+new Date().toLocaleTimeString();
  }catch(error){
    document.getElementById('status').textContent='offline';
    document.getElementById('app').innerHTML=`<div class="panel short">Dashboard error: ${esc(error.message)}</div>`;
  }
}
loadDashboard(true);
setInterval(()=>loadDashboard(true),300000);
</script>
</body>
</html>"""

TRADES_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0b0b09">
<title>XAU/USD Trade Journal</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b0b09;color:#ece7dc;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:16px}.top{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:start;margin-bottom:12px}
h1{font-size:18px;margin:0;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.sub{font-size:12px;color:#9f9a8f;margin-top:4px}.stamp{font-size:12px;color:#9f9a8f;text-align:right}
button,a.btn,.tablink{border:1px solid #494534;background:#171712;color:#ece7dc;border-radius:6px;padding:8px 12px;cursor:pointer;font-weight:700;text-decoration:none;display:inline-block;margin-left:6px}button:hover,a.btn:hover,.tablink:hover{border-color:#b99036}
.tabs-nav{display:flex;gap:8px;margin:12px 0}.tablink{margin-left:0}.tablink.active{background:#2a2518;border-color:#b99036;color:#f2c76b}
.panel{background:#141410;border:1px solid #2d2b22;border-radius:8px;padding:13px;box-shadow:0 1px 0 rgba(255,255,255,.03) inset}.grid{display:grid;gap:10px}.four{grid-template-columns:repeat(4,1fr)}
.metric{background:#10100c;border:1px solid #28251d;border-radius:7px;padding:10px}.metric span{display:block;font-size:10px;color:#9f9a8f;text-transform:uppercase;letter-spacing:.13em}.metric b{display:block;font-size:22px;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:12px}th,td{text-align:left;border-bottom:1px solid #2c2a21;padding:8px 6px;vertical-align:top}th{color:#9f9a8f;font-size:10px;text-transform:uppercase;letter-spacing:.1em}tr:last-child td{border-bottom:0}
.long{color:#4ecb71}.short{color:#ef6a5b}.wait{color:#d7a93f}.muted{color:#9f9a8f}.tiny{font-size:11px}.pill{border:1px solid #393528;background:#1b1a14;border-radius:999px;padding:5px 8px;font-size:11px;display:inline-block}
.note{font-size:11px;color:#817b70;margin-top:10px;line-height:1.45}
@media(max-width:900px){main{padding:11px}.top,.four{grid-template-columns:1fr}.stamp{text-align:left}table{font-size:11px;display:block;overflow-x:auto;white-space:nowrap}}
</style>
</head>
<body>
<main>
  <div class="top">
    <div>
      <h1>XAU/USD Trade Journal</h1>
      <div class="sub">Executed alert record with TP/SL outcome tracking from the alert execution timeframe.</div>
      <div class="tabs-nav"><a class="tablink" href="/">Cockpit</a><a class="tablink active" href="/trades">Journal</a></div>
    </div>
    <div class="stamp"><div id="status">loading...</div><button onclick="loadTrades(true)">Refresh</button><a class="btn" href="/">Cockpit</a></div>
  </div>
  <div id="app" class="panel">Loading trade journal...</div>
</main>
<script>
const money=v=>v==null?'--':'$'+Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const signed=v=>v==null?'--':(Number(v)>0?'+':'')+Number(v).toFixed(2);
const cls=v=>v==='TP'||Number(v)>0?'long':v==='SL'||Number(v)<0?'short':'wait';
const dirCls=d=>(d||'').includes('LONG')?'long':(d||'').includes('SHORT')?'short':'wait';

function metrics(summary){
  return `<div class="grid four">
    <div class="metric"><span>Total</span><b>${summary.total||0}</b></div>
    <div class="metric"><span>Open</span><b class="wait">${summary.open||0}</b></div>
    <div class="metric"><span>Win rate</span><b class="${summary.win_rate>=50?'long':'short'}">${summary.win_rate||0}%</b></div>
    <div class="metric"><span>Net R / $10k</span><b class="${cls(summary.net_r)}">${signed(summary.net_r)}R · ${money(summary.pnl_1pct_10k_usd)}</b></div>
  </div>`;
}

function render(data){
  const summary=data.summary||{}, trades=data.trades||[];
  const rows=trades.map(t=>`<tr>
    <td><span class="pill ${cls(t.result)}">${esc(t.result||'OPEN')}</span><div class="tiny muted">${esc(t.id)}</div></td>
    <td><b class="${dirCls(t.direction)}">${esc(t.direction)}</b><div class="tiny muted">${esc(t.quality||'')}</div></td>
    <td>${money(t.entry)}<div class="tiny muted">${esc(t.opened_at||'')}</div></td>
    <td class="short">${money(t.stop)}</td>
    <td class="long">${money(t.target)}</td>
    <td>${money(t.closed_price)}<div class="tiny muted">${esc(t.closed_at||'open')}</div></td>
    <td class="${cls(t.r_multiple)}">${t.r_multiple==null?'--':signed(t.r_multiple)+'R'}<div class="tiny">${money(t.pnl_1pct_10k_usd)}</div></td>
    <td>${esc(t.sweep_summary||'')}<div class="tiny muted">${esc(t.execution_timeframe||'15M')} ${esc(t.opened_bar||'')}</div></td>
  </tr>`).join('')||'<tr><td colspan="8" class="muted">No executed trade alerts recorded yet.</td></tr>';
  document.getElementById('app').innerHTML=`${metrics(summary)}
    <table><thead><tr><th>Result</th><th>Side</th><th>Entry</th><th>SL</th><th>TP</th><th>Exit</th><th>R / P&L</th><th>Trigger</th></tr></thead><tbody>${rows}</tbody></table>
    <div class="note">${esc(data.note||'')}</div>`;
}

async function loadTrades(refresh=false){
  document.getElementById('status').textContent='loading...';
  try{
    const response=await fetch(`/api/trades?refresh=${refresh?'1':'0'}&limit=200&ts=${Date.now()}`);
    if(!response.ok)throw new Error('HTTP '+response.status);
    const data=await response.json();
    render(data);
    document.getElementById('status').textContent='updated '+new Date().toLocaleTimeString();
  }catch(error){
    document.getElementById('status').textContent='offline';
    document.getElementById('app').innerHTML=`<span class="short">Trade journal error: ${esc(error.message)}</span>`;
  }
}
loadTrades(false);
setInterval(()=>loadTrades(false),300000);
</script>
</body>
</html>"""

@app.get("/trades", response_class=HTMLResponse)
def trades_page():
    return HTMLResponse(TRADES_HTML)

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
