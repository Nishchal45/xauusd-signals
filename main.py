"""
XAU/USD Live Signal Server  —  v5  (Dual Timeframe: 15M + 1H)
==============================================================
Runs BB Squeeze Breakout strategy on BOTH 15-minute and 1-hour
timeframes simultaneously. Each fires its own Telegram alerts.

HOW IT WORKS:
  • 15M checker runs every 15 minutes
  • 1H  checker runs every 60 minutes
  • Each timeframe sends its own Telegram notification
  • Dashboard shows both signals side by side with conditions
  • UptimeRobot ping at /health (HEAD + GET both supported)

ENVIRONMENT VARIABLES:
  TELEGRAM_BOT_TOKEN   → from @BotFather on Telegram
  TELEGRAM_CHAT_ID     → from @userinfobot on Telegram
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

TG_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID",   "")


# ═══════════════════════════════════════════════════════════════════
#  DATA FETCHER
# ═══════════════════════════════════════════════════════════════════
def fetch_data(interval: str) -> tuple:
    """interval = '1h' or '15m'. Returns (DataFrame, error_or_None)."""
    try:
        import yfinance as yf
        period   = "60d" if interval == "1h" else "30d"
        min_bars = 220   if interval == "1h" else 400
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


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL ENGINE  — same logic for any timeframe
# ═══════════════════════════════════════════════════════════════════
def compute_signal(df: pd.DataFrame, interval: str) -> dict:
    c   = df['close']
    et  = ema(c, 200)
    bbm = c.rolling(20).mean()
    bbs = c.rolling(20).std()
    bbu = bbm + 2 * bbs
    bbl = bbm - 2 * bbs
    bbw = (bbu - bbl) / (bbm + 1e-9)
    sqz = bbw < bbw.rolling(50).quantile(0.20)
    rv  = rsi(c, 7)
    at  = atr(df, 14)
    atr_exp = at > at.shift(3)

    i             = len(df) - 1
    price         = float(c.iloc[i])
    atrv          = float(at.iloc[i])
    rsiv          = float(rv.iloc[i])
    e200          = float(et.iloc[i])
    upper         = float(bbu.iloc[i])
    lower         = float(bbl.iloc[i])
    bbwv          = float(bbw.iloc[i])
    sqz_now       = bool(sqz.iloc[i])
    sqz_prev      = bool(sqz.iloc[i - 1])
    atr_expanding = bool(atr_exp.iloc[i])

    trend_up  = price > e200
    trend_dn  = price < e200
    brk_long  = price > upper and sqz_prev
    brk_short = price < lower and sqz_prev
    rsi_lok   = 50 < rsiv < 80
    rsi_sok   = 20 < rsiv < 50

    signal = "WAIT"
    if trend_up  and brk_long  and rsi_lok and atr_expanding: signal = "LONG"
    if trend_dn  and brk_short and rsi_sok and atr_expanding: signal = "SHORT"

    sd     = atrv * 1.5
    td     = sd   * 2.0
    stop   = round(price - sd if signal == "LONG"  else price + sd, 2)
    target = round(price + td if signal == "LONG"  else price - td, 2)

    recent  = sorted([v for v in bbw.iloc[max(0, i-50):i+1] if not np.isnan(v)])
    rank    = next((j for j, v in enumerate(recent) if v >= bbwv), len(recent))
    sqz_pct = round((1 - rank / max(1, len(recent)-1)) * 100, 1)

    atr_3ago  = float(at.iloc[i-3]) if i >= 3 else atrv
    atr_trend = round((atrv / (atr_3ago + 1e-9) - 1) * 100, 1)

    lookback   = 96 if interval == "15m" else 24
    prev_close = float(c.iloc[i - lookback]) if i >= lookback else price
    change     = round(price - prev_close, 2)
    change_pct = round((price / prev_close - 1) * 100, 2)

    chart_bars = 288 if interval == "15m" else 72
    price_hist = [round(float(v), 2) for v in c.iloc[-chart_bars:] if not np.isnan(v)]
    ema_hist   = [round(float(v), 2) for v in et.iloc[-chart_bars:] if not np.isnan(v)]

    conds_met = sum([
        trend_up or trend_dn,
        sqz_prev,
        brk_long or brk_short,
        rsi_lok  or rsi_sok,
        atr_expanding,
    ])

    tf_label = "15M" if interval == "15m" else "1H"

    return {
        "timeframe":      tf_label,
        "signal":         signal,
        "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "bar_time":       str(df.index[-1]),
        "price":          round(price, 2),
        "change":         change,
        "change_pct":     change_pct,
        "stop":           stop,
        "target":         target,
        "stop_dist":      round(sd, 2),
        "target_dist":    round(td, 2),
        "data_bars":      len(df),
        "conditions_met": conds_met,
        "max_conditions": 5,
        "indicators": {
            "ema200":        round(e200,  2),
            "rsi7":          round(rsiv,  1),
            "bb_upper":      round(upper, 2),
            "bb_lower":      round(lower, 2),
            "bb_width":      round(bbwv * 100, 3),
            "atr14":         round(atrv,  2),
            "atr_trend_pct": atr_trend,
            "atr_expanding": atr_expanding,
            "squeeze_now":   sqz_now,
            "squeeze_prev":  sqz_prev,
            "squeeze_pct":   sqz_pct,
        },
        "conditions": {
            "trend_up":      trend_up,
            "trend_dn":      trend_dn,
            "brk_long":      brk_long,
            "brk_short":     brk_short,
            "rsi_lok":       rsi_lok,
            "rsi_sok":       rsi_sok,
            "atr_expanding": atr_expanding,
        },
        "chart": {"price": price_hist, "ema200": ema_hist},
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
    price = s1h.get("price") or s15m.get("price") or 0
    return {
        "h1":        s1h,
        "m15":       s15m,
        "price":     price,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
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


def msg_signal(d: dict) -> str:
    sig   = d["signal"]
    tf    = d["timeframe"]
    price = d["price"];    stop = d["stop"];    target = d["target"]
    sd    = d["stop_dist"];    td = d["target_dist"]
    ind   = d["indicators"]
    rsi7  = ind["rsi7"];   e200  = ind["ema200"]
    atr14 = ind["atr14"];  sqz   = ind["squeeze_pct"]
    ts    = d["timestamp"]
    chg   = d["change"];   chgp  = d["change_pct"]
    arrow = "▲" if chg >= 0 else "▼"
    pos   = round(100 / sd, 4) if sd > 0 else "—"
    tf_icon = "⚡" if tf == "15M" else "🕐"

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
        f"📊 <b>Indicators</b>\n"
        f"  RSI(7):  <code>{rsi7:.1f}</code>   EMA200: <code>${e200:,.2f}</code>\n"
        f"  ATR(14): <code>${atr14:.2f}</code>   Squeeze: <code>{sqz:.0f}%</code>\n"
        f"  24h:     <code>{arrow} ${abs(chg):.2f} ({chgp:+.2f}%)</code>\n"
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
    sqz   = ind["squeeze_pct"]
    tf_icon   = "⚡" if tf == "15M" else "🕐"
    direction = "LONG 📈" if cond["trend_up"] else "SHORT 📉" if cond["trend_dn"] else "—"
    rows = "\n".join([
        ("✅" if cond["trend_up"]  or cond["trend_dn"]  else "❌") + " Trend (EMA 200)",
        ("✅" if ind["squeeze_prev"]                     else "❌") + " BB Squeeze",
        ("✅" if cond["brk_long"]  or cond["brk_short"] else "❌") + " BB Breakout",
        ("✅" if cond["rsi_lok"]   or cond["rsi_sok"]   else "❌") + " RSI zone (50–80)",
        ("✅" if cond["atr_expanding"]                  else "❌") + " ATR expanding",
    ])
    return (
        f"⚡ <b>Forming  {tf_icon} {tf}  —  XAU/USD</b>\n\n"
        f"<b>4 of 5 conditions met</b>  ·  Direction: <b>{direction}</b>\n\n"
        f"{rows}\n\n"
        f"💰 Price:  <code>${price:,.2f}</code>\n"
        f"🛑 Est SL: <code>${stop:,.2f}</code>  (−${sd:.2f})\n"
        f"🎯 Est TP: <code>${target:,.2f}</code>  (+${td:.2f})\n"
        f"🗜 Squeeze: <code>{sqz:.0f}%</code>\n\n"
        f"<i>⚠️ Not confirmed — wait for full signal.</i>\n"
        f"🕐 <i>{d['timestamp']}</i>"
    )


def msg_cleared(tf: str, prev: str, price: float, ts: str) -> str:
    tf_icon = "⚡" if tf == "15M" else "🕐"
    return (
        f"⚪ <b>Cleared  {tf_icon} {tf}  —  XAU/USD</b>\n\n"
        f"The <b>{prev}</b> signal is no longer active.\n\n"
        f"💰 Current price: <code>${price:,.2f}</code>\n\n"
        f"Watching for next setup...\n"
        f"🕐 <i>{ts}</i>"
    )


def msg_startup() -> str:
    return (
        "✅ <b>XAU/USD Dual-TF Bot Online</b>\n\n"
        "Running <b>TWO timeframes simultaneously:</b>\n"
        "  ⚡ <b>15M</b> — scans every 15 minutes\n"
        "  🕐 <b>1H</b>  — scans every 60 minutes\n\n"
        "<b>Alerts include Entry / Stop Loss / Take Profit</b>\n\n"
        "<b>Strategy:</b> BB Squeeze Breakout v2\n"
        "<b>Risk/Reward:</b> 1:2  ·  <b>Data:</b> yfinance GC=F (free)"
    )


# ═══════════════════════════════════════════════════════════════════
#  BACKGROUND CHECKERS
# ═══════════════════════════════════════════════════════════════════
async def checker(interval: str, check_secs: int):
    tf = "15M" if interval == "15m" else "1H"
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

            if sig in ("LONG", "SHORT"):
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
    t1 = asyncio.create_task(checker("1h",  3600))
    t2 = asyncio.create_task(checker("15m",  900))
    log.info("Both timeframe checkers running")
    yield
    t1.cancel(); t2.cancel()

app = FastAPI(title="XAU/USD Dual-TF Signal Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════
@app.get("/api/signals")
def api_signals():
    return JSONResponse(get_both_signals())

@app.get("/api/signal")
def api_signal():
    return JSONResponse(get_signal("1h"))

@app.get("/health")
@app.head("/health")
def health():
    return {
        "status":   "ok",
        "time":     datetime.now(timezone.utc).isoformat(),
        "mode":     "dual TF: 15M + 1H",
        "data":     "yfinance GC=F (free)",
        "telegram": "configured ✅" if TG_TOKEN else "NOT configured ❌",
    }

@app.get("/api/test-notification")
async def test_notification():
    if not TG_TOKEN or not TG_CHAT_ID:
        return JSONResponse({"error": "Telegram not configured"}, status_code=400)
    ok = await tg_send(
        "🧪 <b>Test — XAU/USD Dual-TF Bot</b>\n\n"
        "✅ Telegram working!\n\n"
        "  ⚡ 15M scans every 15 minutes\n"
        "  🕐 1H  scans every 60 minutes\n\n"
        "<i>Test only.</i>"
    )
    return {"sent": ok}

@app.get("/api/check-now")
async def check_now():
    data = get_both_signals()
    sent = []
    for key in ("h1", "m15"):
        d = data[key]
        if d.get("signal") in ("LONG", "SHORT"):
            ok = await tg_send(msg_signal(d))
            sent.append({"tf": d["timeframe"], "sent": ok})
    return {**data, "notifications_sent": sent}


# ═══════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#030712">
<title>XAU/USD Signals</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#030712;--panel:#0a0f1a;--border:#0f172a;--text:#e2e8f0;
      --muted:#334155;--dim:#1e293b;--gold:#f59e0b;--green:#22c55e;
      --red:#ef4444;--blue:#38bdf8;--purple:#a855f7;--orange:#f0883e}
body{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;
     min-height:100vh;padding:14px;-webkit-tap-highlight-color:transparent}
.hdr{display:flex;justify-content:space-between;align-items:flex-start;
     border-bottom:1px solid var(--border);padding-bottom:12px;margin-bottom:14px}
.logo{font-family:'Bebas Neue';font-size:28px;letter-spacing:4px;color:var(--gold);line-height:1}
.sub{font-size:9px;color:var(--muted);letter-spacing:2px;margin-top:3px}
.rh{text-align:right;font-size:11px;color:var(--muted)}
.rbtn{margin-top:6px;background:var(--panel);border:1px solid var(--dim);color:var(--muted);
      font-family:'DM Mono';font-size:10px;padding:5px 12px;border-radius:5px;
      cursor:pointer;-webkit-appearance:none}
.rbtn:active{opacity:.7}
.pbar{background:linear-gradient(135deg,#0f172a,var(--panel));border:1px solid var(--dim);
      border-radius:12px;padding:14px 16px;margin-bottom:12px;
      display:flex;justify-content:space-between;align-items:center}
.pm{font-family:'Bebas Neue';font-size:40px;color:#f1f5f9;letter-spacing:2px;line-height:1}
.plive{font-size:9px;color:var(--green);background:rgba(34,197,94,.1);
       padding:3px 9px;border-radius:4px;letter-spacing:2px}
.tf-overview{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.tfc{background:var(--panel);border:2px solid var(--border);border-radius:10px;
     padding:12px;cursor:pointer;transition:border-color .2s}
.tfc:active{opacity:.7}
.tfc.LONG {border-color:rgba(34,197,94,.5)}
.tfc.SHORT{border-color:rgba(239,68,68,.5)}
.tfc-lbl{font-size:9px;color:var(--muted);letter-spacing:2px;margin-bottom:5px}
.tfc-sig{font-family:'Bebas Neue';font-size:22px;letter-spacing:3px}
.tfc-cond{font-size:10px;margin-top:3px}
.tfc-dots{display:flex;gap:4px;margin-top:7px}
.dot{width:10px;height:10px;border-radius:50%;background:var(--dim);transition:background .3s}
.dot.on{background:var(--green)}
.dot.on.short{background:var(--red)}
.tabs{display:flex;gap:8px;margin-bottom:12px}
.tab{flex:1;padding:10px 0;text-align:center;border-radius:8px;cursor:pointer;
     font-family:'Bebas Neue';font-size:16px;letter-spacing:3px;
     border:2px solid var(--dim);color:var(--muted);background:var(--panel);transition:all .2s}
.tab.act15{border-color:var(--orange);color:var(--orange);background:rgba(240,136,62,.08)}
.tab.act1h{border-color:var(--blue);  color:var(--blue);  background:rgba(56,189,248,.08)}
.tab:active{opacity:.7}
.sb{border-radius:12px;padding:16px;margin-bottom:12px}
.sb.LONG {background:rgba(34,197,94,.08);border:2px solid rgba(34,197,94,.3)}
.sb.SHORT{background:rgba(239,68,68,.08);border:2px solid rgba(239,68,68,.3)}
.sb.WAIT {background:rgba(245,158,11,.05);border:2px solid rgba(245,158,11,.15)}
.sn{font-family:'Bebas Neue';font-size:40px;letter-spacing:4px;line-height:1}
.ss{font-size:11px;color:var(--muted);margin-top:4px}
.lg{display:grid;grid-template-columns:repeat(2,1fr);gap:7px;margin-top:12px}
.lt{background:rgba(0,0,0,.4);border-radius:8px;padding:9px 11px;text-align:center}
.ll{font-size:9px;color:var(--muted);letter-spacing:2px;margin-bottom:3px}
.lv{font-family:'Bebas Neue';font-size:18px;letter-spacing:1px}
.lh{font-size:9px;color:var(--muted);margin-top:2px}
.cp-wrap{background:var(--panel);border:1px solid var(--border);border-radius:10px;
         padding:12px 14px;margin-bottom:12px}
.cp-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.cp-ttl{font-size:9px;color:var(--muted);letter-spacing:2px}
.cp-score{font-family:'Bebas Neue';font-size:22px;letter-spacing:2px}
.cp-bar{height:5px;border-radius:3px;background:var(--dim);margin-bottom:10px;overflow:hidden}
.cp-fill{height:100%;border-radius:3px;transition:width .5s}
.cr{display:flex;align-items:center;gap:9px;padding:7px 0;
    border-bottom:1px solid var(--border)}
.cr:last-child{border-bottom:none}
.ci{font-size:13px;flex-shrink:0;width:18px;text-align:center}
.clbl{flex:1;font-size:11px}
.clbl.ok{color:#86efac}.clbl.nok{color:#fca5a5}
.cv{font-size:10px;color:var(--muted);text-align:right}
.sg{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-bottom:12px}
.st{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:9px 10px}
.stlb{font-size:9px;color:var(--muted);letter-spacing:1.5px;margin-bottom:3px}
.sv{font-family:'Bebas Neue';font-size:16px}
.shi{font-size:9px;color:var(--muted);margin-top:2px}
.cw{background:var(--panel);border:1px solid var(--border);border-radius:10px;
    padding:10px 12px;margin-bottom:12px;overflow:hidden}
.ch{display:flex;justify-content:space-between;font-size:9px;color:var(--muted);
    letter-spacing:2px;margin-bottom:7px}
canvas{display:block;width:100%;border-radius:4px}
.pb{background:var(--panel);border:1px solid var(--border);border-radius:10px;
    padding:12px 14px;margin-bottom:12px}
.pr{display:flex;justify-content:space-between;font-size:12px;padding:5px 0;
    border-bottom:1px solid var(--border)}
.pr:last-child{border-bottom:none}
.prL{color:var(--muted)}
.nbtn{width:100%;padding:11px;background:rgba(59,130,246,.1);
      border:1px solid rgba(59,130,246,.3);color:#93c5fd;
      font-family:'DM Mono';font-size:11px;border-radius:8px;
      cursor:pointer;letter-spacing:1px;margin-bottom:12px;display:block}
.nbtn:active{opacity:.7}
footer{font-size:9px;color:var(--border);text-align:center;letter-spacing:1px;
       padding-top:12px;border-top:1px solid var(--border)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes fadeU{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes spin{to{transform:rotate(360deg)}}
.fade{animation:fadeU .3s ease both}
.spin{display:inline-block;animation:spin 1s linear infinite}
.blink{animation:pulse 1.4s ease infinite}
</style>
</head>
<body>

<div class="hdr">
  <div>
    <div class="logo">XAU / USD</div>
    <div class="sub">DUAL TF · ⚡15M + 🕐1H · BB SQUEEZE · 1:2 RR</div>
  </div>
  <div class="rh">
    <div>
      <span id="dot" style="display:inline-block;width:7px;height:7px;border-radius:50%;
            background:var(--gold);margin-right:5px" class="blink"></span>
      <span id="st">loading…</span>
    </div>
    <div id="upd" style="margin-top:4px;font-size:10px"></div>
    <button class="rbtn" onclick="loadAll()">↻ REFRESH</button>
  </div>
</div>

<div id="app">
  <div style="text-align:center;padding:50px 0;color:var(--muted)">
    <div class="spin" style="font-size:22px;display:block;margin-bottom:10px">◈</div>
    <div style="font-size:11px;letter-spacing:3px">LOADING BOTH TIMEFRAMES…</div>
  </div>
</div>

<button class="nbtn" onclick="testNotif()">📲  Send Test Telegram Notification</button>
<footer>BB(20,2)+EMA200+RSI(7)+ATR · ⚡15M ~22/mo · 🕐1H ~6/mo · 1:2 RR · REFERENCE ONLY</footer>

<script>
const fmt=v=>v!=null?'$'+Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}):'—';
let activeTF='h1', allData=null, cd=900, cdT;

function drawChart(cv,prices,em,sc){
  if(!cv||!prices||!prices.length)return;
  const dpr=window.devicePixelRatio||1,W=cv.offsetWidth,H=cv.offsetHeight||70;
  cv.width=W*dpr;cv.height=H*dpr;
  const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
  const all=[...prices,...em].filter(v=>!isNaN(v));
  const mn=Math.min(...all),mx=Math.max(...all),rng=mx-mn||1;
  const sx=i=>(i/(prices.length-1))*W,sy=v=>H-((v-mn)/rng)*(H-8)-4;
  const g=ctx.createLinearGradient(0,0,0,H);
  g.addColorStop(0,sc+'28');g.addColorStop(1,sc+'00');
  ctx.beginPath();ctx.moveTo(sx(0),H);
  prices.forEach((v,i)=>ctx.lineTo(sx(i),sy(v)));
  ctx.lineTo(sx(prices.length-1),H);ctx.closePath();
  ctx.fillStyle=g;ctx.fill();
  if(em.length){
    ctx.beginPath();ctx.strokeStyle='#38bdf835';ctx.lineWidth=1;ctx.setLineDash([3,3]);
    em.forEach((v,i)=>i===0?ctx.moveTo(sx(i),sy(v)):ctx.lineTo(sx(i),sy(v)));
    ctx.stroke();ctx.setLineDash([]);
  }
  ctx.beginPath();ctx.strokeStyle=sc;ctx.lineWidth=1.8;ctx.lineJoin='round';
  prices.forEach((v,i)=>i===0?ctx.moveTo(sx(i),sy(v)):ctx.lineTo(sx(i),sy(v)));
  ctx.stroke();
  ctx.beginPath();ctx.arc(sx(prices.length-1),sy(prices[prices.length-1]),4,0,Math.PI*2);
  ctx.fillStyle=sc;ctx.fill();
}

function overviewCard(d,key){
  const sig=d.signal||'WAIT',cm=d.conditions_met||0;
  const sc=sig==='LONG'?'var(--green)':sig==='SHORT'?'var(--red)':'var(--gold)';
  const icon=key==='m15'?'⚡':'🕐',lbl=key==='m15'?'15M':'1H';
  const isShort=sig==='SHORT';
  const dots=Array.from({length:5},(_,i)=>{
    const on=i<cm;
    return `<div class="dot${on?' on':''}${on&&isShort?' short':''}"></div>`;
  }).join('');
  return `<div class="tfc ${sig}" onclick="switchTF('${key}')">
    <div class="tfc-lbl">${icon} ${lbl} TIMEFRAME</div>
    <div class="tfc-sig" style="color:${sc}">${sig==='LONG'?'▲ LONG':sig==='SHORT'?'▼ SHORT':'◆ WAIT'}</div>
    <div class="tfc-cond" style="color:${sc}">${cm}/5 conditions met</div>
    <div class="tfc-dots">${dots}</div>
  </div>`;
}

function renderDetail(d,key){
  if(!d||d.error)return`<div style="color:var(--red);padding:12px;font-size:11px">⚠ ${d?.error||'No data'}</div>`;
  const sig=d.signal,ind=d.indicators,cond=d.conditions;
  const sc=sig==='LONG'?'#22c55e':sig==='SHORT'?'#ef4444':'#f59e0b';
  const pos=d.stop_dist>0?(100/d.stop_dist).toFixed(4):'—';
  const cm=d.conditions_met||0;
  const fillPct=Math.round(cm/5*100);
  const fillCol=cm>=5?'#22c55e':cm>=4?'#f59e0b':cm>=3?'#38bdf8':'#475569';
  const tf=key==='m15'?'15M':'1H',icon=key==='m15'?'⚡':'🕐';

  const crows=[
    [cond.trend_up||cond.trend_dn,'Trend (EMA 200)',`${cond.trend_up?'above ↑':'below ↓'} ${fmt(ind.ema200)}`],
    [ind.squeeze_prev,'BB Squeeze',`${ind.bb_width?.toFixed(3)}%  ·  ${ind.squeeze_pct}% pressure`],
    [cond.brk_long||cond.brk_short,'BB Breakout',`${cond.brk_long?'Above '+fmt(ind.bb_upper):cond.brk_short?'Below '+fmt(ind.bb_lower):'Inside bands'}`],
    [cond.rsi_lok||cond.rsi_sok,'RSI(7) Zone',`${ind.rsi7}  (50–80 long · 20–50 short)`],
    [cond.atr_expanding,'ATR Expanding',`${fmt(ind.atr14)} · ${ind.atr_trend_pct>0?'+':''}${ind.atr_trend_pct}%`],
  ].map(([ok,l,v])=>`<div class="cr">
    <div class="ci">${ok?'✅':'❌'}</div>
    <div class="clbl ${ok?'ok':'nok'}">${l}</div>
    <div class="cv">${v}</div>
  </div>`).join('');

  return `
  <div class="cp-wrap">
    <div class="cp-hdr">
      <div class="cp-ttl">CONDITIONS  ${icon} ${tf}</div>
      <div class="cp-score" style="color:${fillCol}">${cm} / 5</div>
    </div>
    <div class="cp-bar"><div class="cp-fill" style="width:${fillPct}%;background:${fillCol}"></div></div>
    ${crows}
  </div>
  <div class="sb ${sig}">
    <div class="sn" style="color:${sc}">${sig==='LONG'?'▲ LONG':sig==='SHORT'?'▼ SHORT':'◆ NO SIGNAL'}</div>
    <div class="ss">${sig!=='WAIT'
      ?`${icon} ${tf}  ·  ENTRY ${fmt(d.price)}  ·  STOP ${fmt(d.stop)}  ·  TARGET ${fmt(d.target)}`
      :`${cm}/5  ·  ${ind.squeeze_pct>60?'⚡ squeeze building':'watching…'}`
    }</div>
    ${sig!=='WAIT'?`
    <div class="lg">
      <div class="lt"><div class="ll">ENTRY</div><div class="lv" style="color:#f1f5f9">${fmt(d.price)}</div><div class="lh">at market</div></div>
      <div class="lt"><div class="ll">STOP LOSS</div><div class="lv" style="color:#ef4444">${fmt(d.stop)}</div><div class="lh">−${fmt(d.stop_dist)}</div></div>
      <div class="lt"><div class="ll">TAKE PROFIT</div><div class="lv" style="color:#22c55e">${fmt(d.target)}</div><div class="lh">+${fmt(d.target_dist)}</div></div>
      <div class="lt"><div class="ll">RISK/REWARD</div><div class="lv" style="color:#f59e0b">1 : 2</div><div class="lh">ratio</div></div>
    </div>`:''}
  </div>
  <div class="cw">
    <div class="ch">
      <span>PRICE CHART  ${icon} ${tf}</span>
      <span><span style="color:#38bdf8">╌EMA200 </span><span style="color:${sc}">─PRICE</span></span>
    </div>
    <canvas id="chart-${key}" style="height:70px"></canvas>
  </div>
  <div class="sg">
    ${[['EMA200',fmt(ind.ema200),cond.trend_up?'#22c55e':'#ef4444',cond.trend_up?'bull':'bear'],
       ['RSI(7)',(ind.rsi7||0).toFixed(1),cond.rsi_lok||cond.rsi_sok?'#22c55e':'#f59e0b','zone ok?'],
       ['ATR(14)',fmt(ind.atr14),'#94a3b8','volatility'],
       ['ATR%',(ind.atr_trend_pct>0?'+':'')+ind.atr_trend_pct+'%',cond.atr_expanding?'#22c55e':'#64748b',cond.atr_expanding?'growing':'shrinking'],
       ['SQUEEZE',ind.squeeze_pct+'%',ind.squeeze_pct>70?'#a855f7':'#64748b',ind.squeeze_pct>70?'building!':'low'],
       ['BARS',d.data_bars,'#334155','loaded']
      ].map(([l,v,c,h])=>`<div class="st"><div class="stlb">${l}</div><div class="sv" style="color:${c}">${v}</div><div class="shi">${h}</div></div>`).join('')}
  </div>
  ${sig!=='WAIT'?`<div class="pb">
    <div style="font-size:9px;color:var(--muted);letter-spacing:2px;margin-bottom:8px">POSITION SIZE  (1% risk / $10k)</div>
    ${[['Risk','$100.00','#f59e0b'],['Stop dist',fmt(d.stop_dist),'#ef4444'],
       ['Size',pos+' oz','#22c55e'],['Target P&L','$200.00','#22c55e'],['Max loss','$100.00','#ef4444']
      ].map(([l,v,c])=>`<div class="pr"><span class="prL">${l}</span><span style="color:${c};font-weight:700">${v}</span></div>`).join('')}
  </div>`:''}`;
}

function render(data){
  const h1=data.h1||{},m15=data.m15||{};
  const price=data.price||0;
  const chgPos=(h1.change||0)>=0,chg=h1.change||0,chgp=h1.change_pct||0;

  document.getElementById('app').innerHTML=`<div class="fade">
  <div class="pbar">
    <div>
      <div class="pm">${fmt(price)}</div>
      <div style="font-size:13px;font-weight:700;font-family:'Bebas Neue';color:${chgPos?'#22c55e':'#ef4444'}">${chgPos?'▲':'▼'} ${fmt(Math.abs(chg))}</div>
      <div style="font-size:11px;margin-top:2px;color:${chgPos?'#22c55e':'#ef4444'}">${chgPos?'+':''}${chgp?.toFixed(2)}% (24h)</div>
    </div>
    <div style="text-align:right">
      <div class="plive">⬤ LIVE</div>
      <div style="font-size:9px;color:var(--muted);margin-top:6px">yfinance · GC=F</div>
      <div style="font-size:9px;color:var(--muted)">${(data.timestamp||'').slice(0,16)}</div>
    </div>
  </div>
  <div style="font-size:9px;color:var(--muted);letter-spacing:2px;margin-bottom:8px">BOTH TIMEFRAMES — TAP TO SWITCH</div>
  <div class="tf-overview">${overviewCard(m15,'m15')}${overviewCard(h1,'h1')}</div>
  <div class="tabs">
    <div class="tab ${activeTF==='m15'?'act15':''}" id="tab-m15" onclick="switchTF('m15')">⚡ 15M</div>
    <div class="tab ${activeTF==='h1'?'act1h':''}"  id="tab-h1"  onclick="switchTF('h1')">🕐 1H</div>
  </div>
  <div id="tf-content">${renderDetail(activeTF==='m15'?m15:h1,activeTF)}</div>
  </div>`;

  requestAnimationFrame(()=>{
    const d=activeTF==='m15'?m15:h1;
    const sc=(d.signal==='LONG')?'#22c55e':(d.signal==='SHORT')?'#ef4444':'#f59e0b';
    const cv=document.getElementById('chart-'+activeTF);
    if(cv&&d.chart) drawChart(cv,d.chart.price,d.chart.ema200,sc);
  });
}

function switchTF(tf){
  activeTF=tf;
  document.getElementById('tab-m15')?.classList.toggle('act15',tf==='m15');
  document.getElementById('tab-h1')?.classList.toggle('act1h',tf==='h1');
  if(!allData)return;
  const d=tf==='m15'?allData.m15:allData.h1;
  const cont=document.getElementById('tf-content');
  if(cont){
    cont.innerHTML=renderDetail(d,tf);
    requestAnimationFrame(()=>{
      const sc=(d.signal==='LONG')?'#22c55e':(d.signal==='SHORT')?'#ef4444':'#f59e0b';
      const cv=document.getElementById('chart-'+tf);
      if(cv&&d.chart) drawChart(cv,d.chart.price,d.chart.ema200,sc);
    });
  }
}

async function loadAll(){
  const dot=document.getElementById('dot');
  dot.style.background='var(--gold)';dot.classList.add('blink');
  document.getElementById('st').textContent='fetching…';
  try{
    const r=await fetch('/api/signals');
    if(!r.ok) throw new Error('HTTP '+r.status);
    const data=await r.json();
    allData=data;
    render(data);
    dot.style.background='var(--green)';dot.classList.remove('blink');
    document.getElementById('st').textContent='live';
    const now=new Date().toLocaleTimeString();
    cd=900;clearInterval(cdT);
    cdT=setInterval(()=>{
      cd=Math.max(0,cd-1);
      const m=Math.floor(cd/60),s=String(cd%60).padStart(2,'0');
      document.getElementById('upd').textContent=`${now} · refresh ${m}:${s}`;
      if(cd===0) loadAll();
    },1000);
    document.getElementById('upd').textContent=now;
  }catch(e){
    dot.style.background='var(--red)';
    document.getElementById('st').textContent='error';
    document.getElementById('app').innerHTML=`<div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.3);border-radius:10px;padding:14px;font-size:12px;color:#fca5a5">⚠ ${e.message}<br><span style="color:var(--muted);font-size:11px">Retrying in 30s…</span></div>`;
    setTimeout(loadAll,30000);
  }
}

async function testNotif(){
  const btn=document.querySelector('.nbtn');
  btn.textContent='Sending…';btn.disabled=true;
  try{
    const r=await fetch('/api/test-notification');
    const d=await r.json();
    btn.textContent=d.sent?'✅ Sent! Check Telegram':'⚠ Failed — check env vars';
  }catch(e){btn.textContent='⚠ '+e.message;}
  setTimeout(()=>{btn.textContent='📲  Send Test Telegram Notification';btn.disabled=false;},4000);
}

loadAll();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
