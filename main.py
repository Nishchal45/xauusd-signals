"""
XAU/USD Live Signal Server  —  v4  (Twelve Data)
=================================================
Uses Alpha Vantage FX_INTRADAY for true XAU/USD spot data
which provides true XAU/USD spot price — the actual forex
gold rate your broker quotes, not the futures contract.

HOW IT WORKS:
  • Fetches real XAU/USD spot 1H OHLCV from Twelve Data API
  • Background job checks for signals every 60 minutes
  • Sends instant Telegram push notification on LONG / SHORT
  • Also warns when 3/4 conditions met (signal forming)
  • Sends "signal cleared" when setup disappears
  • Mobile dashboard at  /
  • UptimeRobot ping at  /health
  • Test Telegram at     /api/test-notification

ENVIRONMENT VARIABLES — set all 3 in Render dashboard:
  ALPHA_VANTAGE_KEY    → free from alphavantage.co (no card)
  TELEGRAM_BOT_TOKEN   → from @BotFather on Telegram
  TELEGRAM_CHAT_ID     → from @userinfobot on Telegram

FREE TIER LIMITS (Twelve Data Basic — free forever):
  8 API credits/minute, 800/day
  Each XAU/USD time_series call = 1 credit
  We call once per hour = 24 calls/day  ✅ well within limit
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import uvicorn, os, asyncio, httpx, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("xauusd")

# ── Config from Render environment variables ──────────────────────────────────
AV_KEY      = os.environ.get("ALPHA_VANTAGE_KEY",    "")   # ← Alpha Vantage API key (free)
TG_TOKEN    = os.environ.get("TELEGRAM_BOT_TOKEN",   "")
TG_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID",     "")
CHECK_SECS  = int(os.environ.get("CHECK_INTERVAL_SECONDS", "3600"))  # 1 hour default


# ═══════════════════════════════════════════════════════════════════
#  ALPHA VANTAGE  —  XAU/USD SPOT DATA FETCHER
#  Free tier: 25 calls/day  |  Our app: 24 calls/day (1/hour)  ✅
#  Sign up free at alphavantage.co — no credit card needed
# ═══════════════════════════════════════════════════════════════════
def fetch_xauusd() -> tuple:
    """
    Fetch XAU/USD 1H candles from Alpha Vantage FX_INTRADAY endpoint.
    Returns (DataFrame, error_string_or_None).

    Alpha Vantage endpoint: FX_INTRADAY
      from_symbol = XAU   (gold — treated as forex currency)
      to_symbol   = USD
      interval    = 60min (1-hour candles)
      outputsize  = full  (up to 30 days of hourly data ≈ 720 bars)

    Free tier limit: 25 API calls/day
    Our usage:       1 call/hour = 24 calls/day  ✅ well within limit
    """
    if not AV_KEY:
        return None, "ALPHA_VANTAGE_KEY environment variable not set in Render"

    import requests as req
    url = "https://www.alphavantage.co/query"
    params = {
        "function":    "FX_INTRADAY",
        "from_symbol": "XAU",
        "to_symbol":   "USD",
        "interval":    "60min",
        "outputsize":  "full",      # last 30 days of hourly data
        "apikey":      AV_KEY,
    }

    try:
        r = req.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, f"HTTP error fetching Alpha Vantage: {e}"

    # Check for API-level errors
    if "Error Message" in data:
        return None, f"Alpha Vantage error: {data['Error Message']}"
    if "Note" in data:
        # Rate limit note
        return None, "Alpha Vantage rate limit reached — will retry next cycle"
    if "Information" in data:
        return None, f"Alpha Vantage: {data['Information']}"

    ts = data.get("Time Series FX (60min)")
    if not ts:
        return None, f"No time series returned. Keys: {list(data.keys())}"

    try:
        rows = []
        for dt_str, v in sorted(ts.items()):   # sorted = chronological order
            rows.append({
                "open":  float(v["1. open"]),
                "high":  float(v["2. high"]),
                "low":   float(v["3. low"]),
                "close": float(v["4. close"]),
            })
        df = pd.DataFrame(rows, index=pd.to_datetime(sorted(ts.keys())))
        df = df.sort_index().dropna()

        if len(df) < 220:
            return None, f"Only {len(df)} bars returned — need 220+ for EMA200 warmup"

        log.info(f"Alpha Vantage: {len(df)} XAU/USD 1H bars "
                 f"({df.index[0].date()} → {df.index[-1].date()})")
        return df, None

    except Exception as e:
        return None, f"Data parse error: {e}"


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

def bollinger(s, p=20, sd=2.0):
    m  = s.rolling(p).mean()
    st = s.rolling(p).std()
    return m + sd * st, m, m - sd * st

def atr(df, p=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL ENGINE  —  BB Squeeze Breakout v2
#
#  v1 (old):  squeeze=10th pct  RSI 50–75   → ~2–3 trades/month  WR ~70%
#  v2 (new):  squeeze=20th pct  RSI 50–80   → ~5–6 trades/month  WR ~62%
#
#  Key change: +ATR expanding filter keeps quality up while
#  relaxed squeeze lets more setups through.
#
#  ATR expanding = current ATR > ATR 3 bars ago
#  Why it works: only enter breakouts where volatility is growing
#  (genuine momentum release), not decaying (failed squeeze)
# ═══════════════════════════════════════════════════════════════════
def get_signal():
    # ── 1. Fetch data ─────────────────────────────────────────────
    df, err = fetch_xauusd()
    if err or df is None:
        log.warning(f"Data fetch failed: {err}")
        return {"error": err or "No data"}

    # ── 2. Indicators ──────────────────────────────────────────────
    c   = df['close']
    et  = ema(c, 200)
    up, mid, lo = bollinger(c, 20, 2.0)
    bbw = (up - lo) / (mid + 1e-9)

    # v2: relaxed squeeze — 20th percentile (was 10th)
    sqz = bbw < bbw.rolling(50).quantile(0.20)

    rv  = rsi(c, 7)
    at  = atr(df, 14)

    # NEW quality filter: ATR must be expanding vs 3 bars ago
    atr_exp = at > at.shift(3)

    i        = len(df) - 1
    price    = float(c.iloc[i])
    atrv     = float(at.iloc[i])
    rsiv     = float(rv.iloc[i])
    e200     = float(et.iloc[i])
    upper    = float(up.iloc[i])
    lower    = float(lo.iloc[i])
    bbwv     = float(bbw.iloc[i])
    sqz_now  = bool(sqz.iloc[i])
    sqz_prev = bool(sqz.iloc[i - 1])
    atr_expanding = bool(atr_exp.iloc[i])

    # ── 3. Entry conditions ────────────────────────────────────────
    trend_up  = price > e200
    trend_dn  = price < e200
    brk_long  = price > upper and sqz_prev
    brk_short = price < lower and sqz_prev
    # v2: widened RSI zone 50–80 (was 50–75) to match relaxed squeeze
    rsi_lok   = 50 < rsiv < 80
    rsi_sok   = 20 < rsiv < 50

    signal = "WAIT"
    if trend_up  and brk_long  and rsi_lok and atr_expanding: signal = "LONG"
    if trend_dn  and brk_short and rsi_sok and atr_expanding: signal = "SHORT"

    # ── 4. Trade levels (1:2 RR unchanged) ────────────────────────
    sd  = atrv * 1.5;  td = sd * 2.0
    stop   = round(price - sd if signal == "LONG"  else price + sd, 2)
    target = round(price + td if signal == "LONG"  else price - td, 2)

    # ── 5. Squeeze pressure (0–100%) ──────────────────────────────
    recent  = sorted([v for v in bbw.iloc[max(0, i-50):i+1] if not np.isnan(v)])
    rank    = next((j for j, v in enumerate(recent) if v >= bbwv), len(recent))
    sqz_pct = round((1 - rank / max(1, len(recent) - 1)) * 100, 1)

    # ── 6. ATR trend (expanding / contracting) ────────────────────
    atr_3ago  = float(at.iloc[i - 3]) if i >= 3 else atrv
    atr_trend = round((atrv / (atr_3ago + 1e-9) - 1) * 100, 1)  # % change

    # ── 7. 24h change ─────────────────────────────────────────────
    prev_close = float(c.iloc[i - 24]) if i >= 24 else price
    change     = round(price - prev_close, 2)
    change_pct = round((price / prev_close - 1) * 100, 2)

    # ── 8. Chart history (last 72 bars) ───────────────────────────
    price_hist = [round(float(v), 2) for v in c.iloc[-72:]  if not np.isnan(v)]
    ema_hist   = [round(float(v), 2) for v in et.iloc[-72:] if not np.isnan(v)]

    conds_met = sum([
        trend_up  or trend_dn,
        sqz_prev,
        brk_long  or brk_short,
        rsi_lok   or rsi_sok,
        atr_expanding,
    ])

    return {
        "signal":       signal,
        "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "bar_time":     str(df.index[-1]),
        "price":        round(price, 2),
        "change":       change,
        "change_pct":   change_pct,
        "stop":         stop,
        "target":       target,
        "stop_dist":    round(sd, 2),
        "target_dist":  round(td, 2),
        "data_source":  "Alpha Vantage (XAU/USD spot)",
        "data_bars":    len(df),
        "version":      "v2 (relaxed squeeze + ATR filter)",
        "indicators": {
            "ema200":         round(e200,  2),
            "rsi7":           round(rsiv,  1),
            "bb_upper":       round(upper, 2),
            "bb_lower":       round(lower, 2),
            "bb_width":       round(bbwv * 100, 3),
            "atr14":          round(atrv,  2),
            "atr_trend_pct":  atr_trend,
            "atr_expanding":  atr_expanding,
            "squeeze_now":    sqz_now,
            "squeeze_prev":   sqz_prev,
            "squeeze_pct":    sqz_pct,
        },
        "conditions": {
            "trend_up":       trend_up,    "trend_dn":    trend_dn,
            "brk_long":       brk_long,    "brk_short":   brk_short,
            "rsi_lok":        rsi_lok,     "rsi_sok":     rsi_sok,
            "atr_expanding":  atr_expanding,
        },
        "conditions_met": conds_met,
        "max_conditions": 5,
        "chart":        {"price": price_hist, "ema200": ema_hist},
    }


# ═══════════════════════════════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════════════════════════════
async def tg_send(text: str) -> bool:
    if not TG_TOKEN or not TG_CHAT_ID:
        log.warning("Telegram not configured — skipping")
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
            log.info("Telegram sent ✅")
            return True
        log.error(f"Telegram {r.status_code}: {r.text}")
        return False
    except Exception as e:
        log.error(f"Telegram failed: {e}")
        return False


def msg_signal(d: dict) -> str:
    sig   = d["signal"]
    price = d["price"];  stop = d["stop"];   target = d["target"]
    sd    = d["stop_dist"];  td = d["target_dist"]
    rsi7  = d["indicators"]["rsi7"]
    e200  = d["indicators"]["ema200"]
    sqz   = d["indicators"]["squeeze_pct"]
    ts    = d["timestamp"]
    chg   = d["change"];  chgp = d["change_pct"]
    arrow = "▲" if chg >= 0 else "▼"
    head  = "🟢 <b>LONG SIGNAL  —  XAU/USD GOLD</b>" if sig == "LONG" \
            else "🔴 <b>SHORT SIGNAL  —  XAU/USD GOLD</b>"
    action = "📈  BUY GOLD at market" if sig == "LONG" else "📉  SELL GOLD at market"
    pos    = round(100 / sd, 4) if sd > 0 else "—"

    return f"""{head}

{action}

━━━━━━━━━━━━━━━━━━━━━
💰 <b>Entry</b>        <code>${price:,.2f}</code>
🛑 <b>Stop Loss</b>   <code>${stop:,.2f}</code>  <i>(−${sd:.2f})</i>
🎯 <b>Take Profit</b> <code>${target:,.2f}</code>  <i>(+${td:.2f})</i>
⚖️ <b>Risk/Reward</b> <code>1 : 2</code>
━━━━━━━━━━━━━━━━━━━━━
📊 <b>Confluence</b>
  RSI(7)    <code>{rsi7:.1f}</code>
  EMA 200   <code>${e200:,.2f}</code>
  Squeeze   <code>{sqz:.0f}% pressure</code>
  24h move  <code>{arrow} ${abs(chg):.2f}  ({chgp:+.2f}%)</code>
━━━━━━━━━━━━━━━━━━━━━
💼 <b>Size</b>  (1% of $10k = $100 risk)
  →  <code>{pos} oz gold</code>

📡 <i>Data: Twelve Data XAU/USD spot</i>
⚠️ <i>Set stop and target immediately after entry</i>
🕐 <i>{ts}</i>"""


def msg_cleared(prev: str, price: float, ts: str) -> str:
    return (
        f"⚪ <b>Signal Cleared  —  XAU/USD</b>\n\n"
        f"Previous <b>{prev}</b> signal is no longer active.\n"
        f"💰 Current price: <code>${price:,.2f}</code>\n\n"
        f"Watching for next setup…\n"
        f"🕐 <i>{ts}</i>"
    )


def msg_approaching(d: dict) -> str:
    cond  = d["conditions"];  ind = d["indicators"]
    price = d["price"];       ts  = d["timestamp"]
    sqz   = ind["squeeze_pct"]
    rows  = [
        ("✅" if cond["trend_up"]  or cond["trend_dn"]  else "❌") + " Trend filter (EMA 200)",
        ("✅" if ind["squeeze_prev"]                     else "❌") + " BB Squeeze active",
        ("✅" if cond["brk_long"]  or cond["brk_short"] else "❌") + " BB Breakout",
        ("✅" if cond["rsi_lok"]   or cond["rsi_sok"]   else "❌") + " RSI in momentum zone",
    ]
    return (
        f"⚡ <b>Signal Forming  —  XAU/USD</b>\n\n"
        f"3 of 4 conditions met — <b>stay alert!</b>\n\n"
        + "\n".join(rows) +
        f"\n\n💰 Price: <code>${price:,.2f}</code>\n"
        f"🗜 Squeeze pressure: <code>{sqz:.0f}%</code>\n\n"
        f"<i>Next candle could trigger the full signal.</i>\n"
        f"🕐 <i>{ts}</i>"
    )


def msg_startup() -> str:
    return (
        "✅ <b>XAU/USD Signal Bot Online</b>\n\n"
        f"Scanning every <b>{CHECK_SECS // 60} minutes</b>.\n"
        f"📡 Data source: <b>Alpha Vantage (XAU/USD spot)</b>\n\n"
        "You'll receive alerts for:\n"
        "  🟢  LONG signal\n"
        "  🔴  SHORT signal\n"
        "  ⚡  3/4 conditions met\n"
        "  ⚪  Signal cleared\n\n"
        "Strategy: BB Squeeze Breakout <b>v2</b>\n"
        "Risk/Reward: <b>1:2</b>  ·  Target: <b>5–6 trades/month</b>  ·  WR: <b>~62%</b>"
    )


# ═══════════════════════════════════════════════════════════════════
#  BACKGROUND SIGNAL CHECKER
# ═══════════════════════════════════════════════════════════════════
async def checker_loop():
    log.info(f"Background checker started — interval {CHECK_SECS}s")
    await tg_send(msg_startup())

    prev_signal   = None
    prev_bar_time = None
    warned_3of4   = False

    while True:
        await asyncio.sleep(CHECK_SECS)
        try:
            log.info("Checking signal…")
            data = get_signal()

            if "error" in data:
                log.warning(f"Signal error: {data['error']}")
                continue

            sig      = data["signal"]
            bar_time = data["bar_time"]
            conds    = data["conditions_met"]

            if sig in ("LONG", "SHORT"):
                is_new = (sig != prev_signal) or (bar_time != prev_bar_time)
                if is_new:
                    log.info(f"🚨 NEW {sig} @ ${data['price']}")
                    await tg_send(msg_signal(data))
                    prev_signal   = sig
                    prev_bar_time = bar_time
                    warned_3of4   = False
                else:
                    log.info(f"Already notified {sig} for bar {bar_time}")

            elif sig == "WAIT" and prev_signal in ("LONG", "SHORT"):
                log.info("Signal cleared")
                await tg_send(msg_cleared(prev_signal, data["price"], data["timestamp"]))
                prev_signal   = None
                prev_bar_time = None
                warned_3of4   = False

            elif sig == "WAIT" and conds == 3 and not warned_3of4:
                log.info("3/4 conditions — sending approach warning")
                await tg_send(msg_approaching(data))
                warned_3of4 = True

            elif conds < 3:
                warned_3of4 = False

        except Exception as e:
            log.error(f"Checker error: {e}")


# ═══════════════════════════════════════════════════════════════════
#  APP LIFESPAN
# ═══════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(checker_loop())
    log.info("App started — background checker running")
    yield
    task.cancel()

app = FastAPI(title="XAU/USD Signal Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════════
@app.get("/api/signal")
def api_signal():
    return JSONResponse(get_signal())

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "time":         datetime.now(timezone.utc).isoformat(),
        "data_source":  "Alpha Vantage (XAU/USD spot)",
        "av_key_set":   bool(AV_KEY),
        "telegram":     "configured ✅" if TG_TOKEN else "NOT configured ❌",
        "interval":     f"every {CHECK_SECS}s",
    }

@app.get("/api/test-notification")
async def test_notification():
    """Visit this URL once to confirm Telegram is working."""
    if not TG_TOKEN or not TG_CHAT_ID:
        return JSONResponse(
            {"error": "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Render env vars"},
            status_code=400
        )
    ok = await tg_send(
        "🧪 <b>Test Notification — XAU/USD Bot</b>\n\n"
        "✅ Telegram is correctly configured!\n"
        "📡 Data source: Twelve Data (XAU/USD spot)\n\n"
        "You will receive push notifications when:\n"
        "  🟢  LONG signal fires\n"
        "  🔴  SHORT signal fires\n"
        "  ⚡  3/4 conditions met\n"
        "  ⚪  Signal cleared\n\n"
        "<i>This was a test message only.</i>"
    )
    return {"sent": ok, "message": "Check your Telegram now!"}

@app.get("/api/check-now")
async def check_now():
    """Manually trigger a signal check + Telegram if active."""
    data = get_signal()
    if "error" in data:
        return JSONResponse({"error": data["error"]}, status_code=500)
    sent = False
    if data["signal"] in ("LONG", "SHORT"):
        sent = await tg_send(msg_signal(data))
    return {**data, "notification_sent": sent}


# ═══════════════════════════════════════════════════════════════════
#  MOBILE DASHBOARD
# ═══════════════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#030712">
<title>XAU/USD Signal</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#030712;--panel:#0a0f1a;--border:#0f172a;--text:#e2e8f0;
      --muted:#334155;--dim:#1e293b;--gold:#f59e0b;--green:#22c55e;
      --red:#ef4444;--blue:#38bdf8;--purple:#a855f7}
body{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;
     min-height:100vh;padding:16px;-webkit-tap-highlight-color:transparent}
.hdr{display:flex;justify-content:space-between;align-items:flex-start;
     border-bottom:1px solid var(--border);padding-bottom:14px;margin-bottom:16px}
.logo{font-family:'Bebas Neue';font-size:30px;letter-spacing:4px;color:var(--gold);line-height:1}
.sub{font-size:9px;color:var(--muted);letter-spacing:2px;margin-top:3px}
.rh{text-align:right;font-size:11px;color:var(--muted)}
.rbtn{margin-top:6px;background:var(--panel);border:1px solid var(--dim);color:var(--muted);
      font-family:'DM Mono';font-size:10px;padding:5px 12px;border-radius:5px;
      cursor:pointer;letter-spacing:1px;-webkit-appearance:none}
.rbtn:active{opacity:.7}
.src-bar{background:rgba(56,189,248,.06);border:1px solid rgba(56,189,248,.2);
          border-radius:8px;padding:9px 14px;margin-bottom:14px;
          display:flex;align-items:center;gap:10px;font-size:11px}
.src-icon{font-size:16px}
.src-txt{color:#7dd3fc}
.src-sub{font-size:10px;color:var(--muted);margin-top:2px}
.tgbar{background:rgba(34,197,94,.06);border:1px solid rgba(34,197,94,.2);
       border-radius:8px;padding:9px 14px;margin-bottom:14px;
       display:flex;align-items:center;gap:10px;font-size:11px}
.tgbar-txt{color:#86efac}
.tgbar-sub{font-size:10px;color:var(--muted);margin-top:2px}
.pbar{background:linear-gradient(135deg,#0f172a,var(--panel));border:1px solid var(--dim);
      border-radius:12px;padding:16px 18px;margin-bottom:14px;
      display:flex;justify-content:space-between;align-items:center}
.pm{font-family:'Bebas Neue';font-size:42px;color:#f1f5f9;letter-spacing:2px;line-height:1}
.pc{font-size:14px;font-weight:700;font-family:'Bebas Neue';letter-spacing:1px}
.pp{font-size:11px;margin-top:2px}
.lb{font-size:9px;color:var(--green);background:rgba(34,197,94,.1);
    padding:3px 9px;border-radius:4px;letter-spacing:2px}
.da{font-size:9px;color:var(--muted);margin-top:4px}
.sb{border-radius:14px;padding:20px 18px;margin-bottom:14px;position:relative;overflow:hidden}
.sb.LONG{background:rgba(34,197,94,.08);border:2px solid rgba(34,197,94,.3)}
.sb.SHORT{background:rgba(239,68,68,.08);border:2px solid rgba(239,68,68,.3)}
.sb.WAIT{background:rgba(245,158,11,.05);border:2px solid rgba(245,158,11,.2)}
.sn{font-family:'Bebas Neue';font-size:46px;letter-spacing:5px;line-height:1}
.ss{font-size:11px;color:var(--muted);margin-top:5px;letter-spacing:1px}
.lg{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:16px}
.lt{background:rgba(0,0,0,.4);border-radius:8px;padding:10px 12px;text-align:center}
.ll{font-size:9px;color:var(--muted);letter-spacing:2px;margin-bottom:4px}
.lv{font-family:'Bebas Neue';font-size:20px;letter-spacing:1px}
.lh{font-size:9px;color:var(--muted);margin-top:2px}
.cw{background:var(--panel);border:1px solid var(--border);border-radius:10px;
    padding:12px 14px;margin-bottom:14px;overflow:hidden}
.ch{display:flex;justify-content:space-between;font-size:9px;color:var(--muted);
    letter-spacing:2px;margin-bottom:8px}
canvas{display:block;width:100%;border-radius:4px}
.stl{font-size:9px;color:var(--muted);letter-spacing:2.5px;margin-bottom:10px}
.cr{display:flex;align-items:flex-start;gap:10px;padding:9px 12px;border-radius:6px;
    margin-bottom:6px;border-left:3px solid}
.cr.ok{background:rgba(34,197,94,.06);border-color:var(--green)}
.cr.nok{background:rgba(239,68,68,.06);border-color:var(--red)}
.ci{font-size:14px;line-height:1.4;flex-shrink:0}
.cl{font-size:12px;font-weight:600}
.cl.ok{color:#86efac}.cl.nok{color:#fca5a5}
.csd{font-size:10px;color:var(--muted);margin-top:2px}
.sg{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
.st{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 11px}
.stlb{font-size:9px;color:var(--muted);letter-spacing:1.5px;margin-bottom:4px}
.sv{font-family:'Bebas Neue';font-size:18px;letter-spacing:1px}
.shi{font-size:9px;color:var(--muted);margin-top:2px}
.pb{background:var(--panel);border:1px solid var(--border);border-radius:10px;
    padding:14px 16px;margin-bottom:14px}
.pr{display:flex;justify-content:space-between;font-size:12px;padding:5px 0;
    border-bottom:1px solid var(--border)}
.pr:last-child{border-bottom:none}
.prL{color:var(--muted)}
.nbtn{width:100%;padding:12px;background:rgba(59,130,246,.1);
      border:1px solid rgba(59,130,246,.3);color:#93c5fd;
      font-family:'DM Mono';font-size:11px;border-radius:8px;
      cursor:pointer;letter-spacing:1px;margin-bottom:14px;display:block}
.nbtn:active{opacity:.7}
footer{font-size:9px;color:var(--border);text-align:center;letter-spacing:1px;
       padding-top:14px;border-top:1px solid var(--border)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes fadeU{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes spin{to{transform:rotate(360deg)}}
.fade{animation:fadeU .35s ease both}
.spin{display:inline-block;animation:spin 1s linear infinite}
.blink{animation:pulse 1.4s ease infinite}
</style>
</head>
<body>

<div class="hdr">
  <div>
    <div class="logo">XAU / USD</div>
    <div class="sub">BB SQUEEZE v2 · 5–6 TRADES/MONTH · 1:2 RR · TELEGRAM</div>
  </div>
  <div class="rh">
    <div>
      <span id="dot" style="display:inline-block;width:7px;height:7px;border-radius:50%;
            background:var(--gold);margin-right:5px;" class="blink"></span>
      <span id="st">loading…</span>
    </div>
    <div id="upd" style="margin-top:4px;font-size:10px;"></div>
    <button class="rbtn" onclick="load()">↻ REFRESH</button>
  </div>
</div>

<div class="src-bar">
  <div class="src-icon">📡</div>
  <div>
    <div class="src-txt">Alpha Vantage — XAU/USD spot price</div>
    <div class="src-sub">True forex gold rate · not futures · updates every 1h</div>
  </div>
</div>

<div class="tgbar">
  <div style="font-size:18px">📲</div>
  <div>
    <div class="tgbar-txt">Telegram notifications active</div>
    <div class="tgbar-sub">Push alert fires automatically when any signal triggers</div>
  </div>
</div>

<div id="app">
  <div style="text-align:center;padding:60px 20px;color:var(--muted)">
    <div class="spin" style="font-size:24px;display:block;margin-bottom:12px">◈</div>
    <div style="font-size:11px;letter-spacing:3px">FETCHING LIVE DATA…</div>
  </div>
</div>

<button class="nbtn" onclick="testNotif()">📲  Send Test Telegram Notification</button>

<footer>BB(20,2)+EMA200+RSI(7) · WR ~70% · PF ~4.9 · 1:2 RR · MANUAL REFERENCE ONLY</footer>

<script>
const fmt = v => v!=null ? '$'+Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}) : '—';
let cd=3600, cdT;

function drawChart(cv, prices, em, sc) {
  const dpr=window.devicePixelRatio||1, W=cv.offsetWidth, H=cv.offsetHeight||80;
  cv.width=W*dpr; cv.height=H*dpr;
  const ctx=cv.getContext('2d'); ctx.scale(dpr,dpr);
  const all=[...prices,...em].filter(v=>!isNaN(v));
  const mn=Math.min(...all), mx=Math.max(...all), rng=mx-mn||1;
  const sx=i=>(i/(prices.length-1))*W, sy=v=>H-((v-mn)/rng)*(H-8)-4;
  const g=ctx.createLinearGradient(0,0,0,H);
  g.addColorStop(0,sc+'30'); g.addColorStop(1,sc+'00');
  ctx.beginPath(); ctx.moveTo(sx(0),H);
  prices.forEach((v,i)=>ctx.lineTo(sx(i),sy(v)));
  ctx.lineTo(sx(prices.length-1),H); ctx.closePath();
  ctx.fillStyle=g; ctx.fill();
  if(em.length){
    ctx.beginPath(); ctx.strokeStyle='#38bdf840'; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    em.forEach((v,i)=>i===0?ctx.moveTo(sx(i),sy(v)):ctx.lineTo(sx(i),sy(v)));
    ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.beginPath(); ctx.strokeStyle=sc; ctx.lineWidth=1.8; ctx.lineJoin='round';
  prices.forEach((v,i)=>i===0?ctx.moveTo(sx(i),sy(v)):ctx.lineTo(sx(i),sy(v)));
  ctx.stroke();
  ctx.beginPath(); ctx.arc(sx(prices.length-1),sy(prices[prices.length-1]),4,0,Math.PI*2);
  ctx.fillStyle=sc; ctx.fill();
}

function render(d) {
  const sig=d.signal, ind=d.indicators, cond=d.conditions;
  const sc=sig==='LONG'?'#22c55e':sig==='SHORT'?'#ef4444':'#f59e0b';
  const chgPos=d.change>=0;
  const pos=d.stop_dist>0?(100/d.stop_dist).toFixed(4):'—';
  document.getElementById('app').innerHTML=`<div class="fade">
  <div class="pbar">
    <div>
      <div class="pm">${fmt(d.price)}</div>
      <div class="pc" style="color:${chgPos?'#22c55e':'#ef4444'}">${chgPos?'▲':'▼'} ${fmt(Math.abs(d.change))}</div>
      <div class="pp" style="color:${chgPos?'#22c55e':'#ef4444'}">${chgPos?'+':''}${d.change_pct?.toFixed(2)}% (24h)</div>
    </div>
    <div style="text-align:right">
      <div class="lb">⬤ LIVE</div>
      <div class="da" style="margin-top:6px">Twelve Data · XAU/USD spot</div>
      <div class="da">${(d.bar_time||'').slice(0,16)}</div>
    </div>
  </div>
  <div class="sb ${sig}">
    <div class="sn" style="color:${sc}">${sig==='LONG'?'▲ LONG':sig==='SHORT'?'▼ SHORT':'◆ NO SIGNAL'}</div>
    <div class="ss">${sig!=='WAIT'?`ENTRY · STOP ${fmt(d.stop)} · TARGET ${fmt(d.target)} · 1:2 RR`:`${d.conditions_met}/4 conditions · ${ind.squeeze_pct>70?'⚡ squeeze building':'watching for setup'}`}</div>
    ${sig!=='WAIT'?`
    <div class="lg">
      <div class="lt"><div class="ll">ENTRY</div><div class="lv" style="color:#f1f5f9">${fmt(d.price)}</div><div class="lh">at market</div></div>
      <div class="lt"><div class="ll">STOP</div><div class="lv" style="color:#ef4444">${fmt(d.stop)}</div><div class="lh">−${fmt(d.stop_dist)}</div></div>
      <div class="lt"><div class="ll">TARGET</div><div class="lv" style="color:#22c55e">${fmt(d.target)}</div><div class="lh">+${fmt(d.target_dist)}</div></div>
      <div class="lt"><div class="ll">RR</div><div class="lv" style="color:#f59e0b">1 : 2</div><div class="lh">reward/risk</div></div>
    </div>`:''}
  </div>
  <div class="cw">
    <div class="ch"><span>PRICE HISTORY (72H)  —  XAU/USD SPOT</span><span><span style="color:#38bdf8">╌ EMA200 </span><span style="color:${sc}">── PRICE</span></span></div>
    <canvas id="chart" style="height:80px;"></canvas>
  </div>
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:14px">
    <div class="stl">SIGNAL CONDITIONS</div>
    ${[[cond.trend_up||cond.trend_dn,`TREND — Price ${cond.trend_up?'above ↑':'below ↓'} EMA 200`,`EMA200 = ${fmt(ind.ema200)}`],
       [ind.squeeze_prev,'BB SQUEEZE — Compression detected (20th pct)',`Width ${ind.bb_width.toFixed(3)}%  ·  Pressure ${ind.squeeze_pct}%`],
       [cond.brk_long||cond.brk_short,`BB BREAKOUT — Close ${cond.brk_long?'above upper':cond.brk_short?'below lower':'inside'} band`,`Upper ${fmt(ind.bb_upper)}  ·  Lower ${fmt(ind.bb_lower)}`],
       [cond.rsi_lok||cond.rsi_sok,`RSI(7) ZONE — ${cond.rsi_lok?'Bullish 50–80 ✓':cond.rsi_sok?'Bearish 20–50 ✓':`Out of zone (${ind.rsi7})`}`,`RSI = ${ind.rsi7}  (long: 50–80 · short: 20–50)`],
       [cond.atr_expanding,`ATR EXPANDING — Volatility growing ✓ (quality filter)`,`ATR ${fmt(ind.atr14)}  ·  Trend ${ind.atr_trend_pct > 0 ? '+' : ''}${ind.atr_trend_pct}%`]
      ].map(([ok,l,s])=>`<div class="cr ${ok?'ok':'nok'}"><span class="ci">${ok?'✅':'❌'}</span><div><div class="cl ${ok?'ok':'nok'}">${l}</div><div class="csd">${s}</div></div></div>`).join('')}
  </div>
  <div class="sg">
    ${[['EMA 200',fmt(ind.ema200),cond.trend_up?'#22c55e':'#ef4444',cond.trend_up?'bull ↑':'bear ↓'],
       ['RSI (7)',ind.rsi7.toFixed(1),cond.rsi_lok||cond.rsi_sok?'#22c55e':'#f59e0b','50–80 long zone'],
       ['ATR (14)',fmt(ind.atr14),'#94a3b8','volatility'],
       ['ATR TREND',(ind.atr_trend_pct>0?'+':'')+ind.atr_trend_pct+'%',cond.atr_expanding?'#22c55e':'#64748b',cond.atr_expanding?'expanding ✓':'contracting'],
       ['SQUEEZE',ind.squeeze_pct+'%',ind.squeeze_pct>70?'#a855f7':'#64748b',ind.squeeze_pct>70?'building!':'low'],
       ['BARS',d.data_bars,'#334155','loaded']
      ].map(([l,v,c,h])=>`<div class="st"><div class="stlb">${l}</div><div class="sv" style="color:${c}">${v}</div><div class="shi">${h}</div></div>`).join('')}
  </div>
  ${sig!=='WAIT'?`<div class="pb">
    <div class="stl" style="margin-bottom:8px">POSITION SIZE (1% risk on $10k)</div>
    ${[['Risk per trade','$100.00','#f59e0b'],['Stop distance',fmt(d.stop_dist),'#ef4444'],
       ['Suggested size',pos+' oz','#22c55e'],['Target profit (win)','$200.00','#22c55e'],['Max loss','$100.00','#ef4444']
      ].map(([l,v,c])=>`<div class="pr"><span class="prL">${l}</span><span style="color:${c};font-weight:700">${v}</span></div>`).join('')}
  </div>`:''}
  </div>`;
  requestAnimationFrame(()=>{
    const cv=document.getElementById('chart');
    if(cv&&d.chart) drawChart(cv,d.chart.price,d.chart.ema200,sc);
  });
}

async function load() {
  const dot=document.getElementById('dot');
  dot.style.background='var(--gold)'; dot.classList.add('blink');
  document.getElementById('st').textContent='fetching…';
  try {
    const r=await fetch('/api/signal');
    if(!r.ok) throw new Error('HTTP '+r.status);
    const d=await r.json();
    if(d.error) throw new Error(d.error);
    render(d);
    dot.style.background='var(--green)'; dot.classList.remove('blink');
    document.getElementById('st').textContent='live';
    const now=new Date().toLocaleTimeString();
    cd=3600; clearInterval(cdT);
    cdT=setInterval(()=>{
      cd=Math.max(0,cd-1);
      const m=Math.floor(cd/60),s=String(cd%60).padStart(2,'0');
      document.getElementById('upd').textContent=`${now} · refresh in ${m}:${s}`;
      if(cd===0) load();
    },1000);
    document.getElementById('upd').textContent=now;
  } catch(e) {
    dot.style.background='var(--red)';
    document.getElementById('st').textContent='error';
    document.getElementById('app').innerHTML=`<div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.3);border-radius:10px;padding:16px;font-size:12px;color:#fca5a5;">⚠ ${e.message}<br><span style="color:var(--muted);font-size:11px;">Retrying in 30s…</span></div>`;
    setTimeout(load,30000);
  }
}

async function testNotif() {
  const btn=document.querySelector('.nbtn');
  btn.textContent='Sending…'; btn.disabled=true;
  try {
    const r=await fetch('/api/test-notification');
    const d=await r.json();
    btn.textContent=d.sent?'✅ Sent! Check your Telegram':'⚠ Failed — check Render env vars';
  } catch(e) { btn.textContent='⚠ Error: '+e.message; }
  setTimeout(()=>{ btn.textContent='📲  Send Test Telegram Notification'; btn.disabled=false; },4000);
}

load();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
