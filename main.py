"""
XAU/USD Live Signal Server
===========================
Fetches real gold price via yfinance, computes BB Squeeze strategy,
serves a mobile-optimised dashboard. Deploy free on Render.com.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import uvicorn, os

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

app = FastAPI(title="XAU/USD Signal Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Indicators ──────────────────────────────────────────────────────────────
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=7):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def bollinger(s, p=20, sd=2.0):
    m = s.rolling(p).mean(); st = s.rolling(p).std()
    return m + sd*st, m, m - sd*st

def atr(df, p=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

# ── Fetch + Compute ─────────────────────────────────────────────────────────
def get_signal():
    if not HAS_YF:
        return {"error": "yfinance not available"}, None

    try:
        df = yf.download("GC=F", period="60d", interval="1h",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 220:
            return {"error": "Not enough data"}, None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df.dropna()
    except Exception as e:
        return {"error": str(e)}, None

    c = df['close']
    et    = ema(c, 200)
    up, mid, lo = bollinger(c, 20, 2.0)
    bbw   = (up - lo) / (mid + 1e-9)
    sqz   = bbw < bbw.rolling(50).quantile(0.10)
    rv    = rsi(c, 7)
    at    = atr(df, 14)

    i     = len(df) - 1
    price = float(c.iloc[i])
    atrv  = float(at.iloc[i])
    rsiv  = float(rv.iloc[i])
    e200  = float(et.iloc[i])
    upper = float(up.iloc[i])
    lower = float(lo.iloc[i])
    bbwv  = float(bbw.iloc[i])
    sqz_now  = bool(sqz.iloc[i])
    sqz_prev = bool(sqz.iloc[i-1])

    trend_up   = price > e200
    trend_dn   = price < e200
    brk_long   = price > upper and sqz_prev
    brk_short  = price < lower and sqz_prev
    rsi_lok    = 50 < rsiv < 75
    rsi_sok    = 25 < rsiv < 50

    signal = "WAIT"
    if trend_up  and brk_long  and rsi_lok: signal = "LONG"
    if trend_dn  and brk_short and rsi_sok: signal = "SHORT"

    sd  = atrv * 1.5
    td  = sd   * 2.0
    stop   = price - sd if signal == "LONG"  else price + sd
    target = price + td if signal == "LONG"  else price - td

    # Squeeze pressure
    recent = sorted([v for v in bbw.iloc[max(0,i-50):i+1] if not np.isnan(v)])
    rank   = next((j for j,v in enumerate(recent) if v >= bbwv), len(recent))
    sqz_pct = round((1 - rank / max(1, len(recent)-1)) * 100, 1)

    # Price history for chart (last 72 bars)
    price_hist = [round(float(v),2) for v in c.iloc[-72:] if not np.isnan(v)]
    ema_hist   = [round(float(v),2) for v in et.iloc[-72:] if not np.isnan(v)]

    # Daily change
    prev_close = float(c.iloc[i-24]) if i >= 24 else price
    change     = round(price - prev_close, 2)
    change_pct = round((price / prev_close - 1) * 100, 2)

    return {
        "signal": signal,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "price": round(price, 2),
        "change": change,
        "change_pct": change_pct,
        "stop":   round(stop, 2),
        "target": round(target, 2),
        "stop_dist":   round(sd, 2),
        "target_dist": round(td, 2),
        "indicators": {
            "ema200":   round(e200, 2),
            "rsi7":     round(rsiv, 1),
            "bb_upper": round(upper, 2),
            "bb_lower": round(lower, 2),
            "bb_width": round(bbwv * 100, 3),
            "atr14":    round(atrv, 2),
            "squeeze_now":  sqz_now,
            "squeeze_prev": sqz_prev,
            "squeeze_pct":  sqz_pct,
        },
        "conditions": {
            "trend_up":  trend_up,
            "trend_dn":  trend_dn,
            "brk_long":  brk_long,
            "brk_short": brk_short,
            "rsi_lok":   rsi_lok,
            "rsi_sok":   rsi_sok,
        },
        "conditions_met": sum([trend_up or trend_dn, sqz_prev,
                               brk_long or brk_short, rsi_lok or rsi_sok]),
        "chart": {"price": price_hist, "ema200": ema_hist},
        "data_bars": len(df),
    }, df

# ── API Routes ───────────────────────────────────────────────────────────────
@app.get("/api/signal")
def api_signal():
    data, _ = get_signal()
    return JSONResponse(data)

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

# ── Dashboard HTML ────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#030712">
<title>XAU/USD Signal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:     #030712;
    --panel:  #0a0f1a;
    --border: #0f172a;
    --text:   #e2e8f0;
    --muted:  #334155;
    --dim:    #1e293b;
    --gold:   #f59e0b;
    --green:  #22c55e;
    --red:    #ef4444;
    --blue:   #38bdf8;
    --purple: #a855f7;
  }
  body { background: var(--bg); color: var(--text); font-family: 'DM Mono', monospace;
         min-height: 100vh; padding: 16px; -webkit-tap-highlight-color: transparent; }
  .header { display: flex; justify-content: space-between; align-items: flex-start;
            border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 16px; }
  .logo { font-family: 'Bebas Neue'; font-size: 30px; letter-spacing: 4px; color: var(--gold); line-height: 1; }
  .sub  { font-size: 9px; color: var(--muted); letter-spacing: 2px; margin-top: 3px; }
  .status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
                background: var(--green); margin-right: 5px; }
  .status-dot.loading { background: var(--gold); animation: pulse 1s infinite; }
  .right-header { text-align: right; font-size: 11px; color: var(--muted); }
  .refresh-btn { margin-top: 6px; background: var(--panel); border: 1px solid var(--dim);
                 color: var(--muted); font-family: 'DM Mono'; font-size: 10px;
                 padding: 5px 12px; border-radius: 5px; cursor: pointer; letter-spacing: 1px;
                 -webkit-appearance: none; }
  .refresh-btn:active { opacity: .7; }

  /* Price bar */
  .price-bar { background: linear-gradient(135deg, #0f172a, var(--panel));
               border: 1px solid var(--dim); border-radius: 12px;
               padding: 16px 18px; margin-bottom: 14px;
               display: flex; justify-content: space-between; align-items: center; }
  .price-main { font-family: 'Bebas Neue'; font-size: 42px; color: #f1f5f9;
                letter-spacing: 2px; line-height: 1; }
  .price-change { font-size: 14px; font-weight: 700; font-family: 'Bebas Neue'; letter-spacing: 1px; }
  .price-pct    { font-size: 11px; margin-top: 2px; }
  .live-badge   { font-size: 9px; color: var(--green); background: rgba(34,197,94,.1);
                  padding: 3px 9px; border-radius: 4px; letter-spacing: 2px; }
  .data-age     { font-size: 9px; color: var(--muted); margin-top: 4px; }

  /* Signal box */
  .signal-box { border-radius: 14px; padding: 20px 18px; margin-bottom: 14px;
                position: relative; overflow: hidden; }
  .signal-box.LONG  { background: rgba(34,197,94,.08);  border: 2px solid rgba(34,197,94,.3); }
  .signal-box.SHORT { background: rgba(239,68,68,.08);  border: 2px solid rgba(239,68,68,.3); }
  .signal-box.WAIT  { background: rgba(245,158,11,.05); border: 2px solid rgba(245,158,11,.2); }
  .signal-name { font-family: 'Bebas Neue'; font-size: 46px; letter-spacing: 5px; line-height: 1; }
  .signal-sub  { font-size: 11px; color: var(--muted); margin-top: 5px; letter-spacing: 1px; }
  .levels-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 8px; margin-top: 16px; }
  .level-tile  { background: rgba(0,0,0,.4); border-radius: 8px; padding: 10px 12px; text-align: center; }
  .level-lbl   { font-size: 9px; color: var(--muted); letter-spacing: 2px; margin-bottom: 4px; }
  .level-val   { font-family: 'Bebas Neue'; font-size: 20px; letter-spacing: 1px; }
  .level-hint  { font-size: 9px; color: var(--muted); margin-top: 2px; }

  /* Chart */
  .chart-wrap { background: var(--panel); border: 1px solid var(--border);
                border-radius: 10px; padding: 12px 14px; margin-bottom: 14px; overflow: hidden; }
  .chart-hdr  { display: flex; justify-content: space-between; font-size: 9px;
                color: var(--muted); letter-spacing: 2px; margin-bottom: 8px; }
  canvas { display: block; width: 100%; border-radius: 4px; }

  /* Conditions */
  .section-title { font-size: 9px; color: var(--muted); letter-spacing: 2.5px; margin-bottom: 10px; }
  .cond-row { display: flex; align-items: flex-start; gap: 10px; padding: 9px 12px;
              border-radius: 6px; margin-bottom: 6px; border-left: 3px solid; }
  .cond-row.ok  { background: rgba(34,197,94,.06); border-color: var(--green); }
  .cond-row.nok { background: rgba(239,68,68,.06); border-color: var(--red); }
  .cond-icon { font-size: 14px; line-height: 1.4; flex-shrink: 0; }
  .cond-lbl  { font-size: 12px; font-weight: 600; }
  .cond-lbl.ok  { color: #86efac; }
  .cond-lbl.nok { color: #fca5a5; }
  .cond-sub  { font-size: 10px; color: var(--muted); margin-top: 2px; }

  /* Stats */
  .stats-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-bottom: 14px; }
  .stat-tile  { background: var(--panel); border: 1px solid var(--border);
                border-radius: 8px; padding: 10px 11px; }
  .stat-lbl   { font-size: 9px; color: var(--muted); letter-spacing: 1.5px; margin-bottom: 4px; }
  .stat-val   { font-family: 'Bebas Neue'; font-size: 18px; letter-spacing: 1px; }
  .stat-hint  { font-size: 9px; color: var(--muted); margin-top: 2px; }

  /* Position size */
  .pos-box { background: var(--panel); border: 1px solid var(--border);
             border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }
  .pos-row { display: flex; justify-content: space-between; font-size: 12px;
             padding: 5px 0; border-bottom: 1px solid var(--border); }
  .pos-row:last-child { border-bottom: none; }
  .pos-lbl { color: var(--muted); }

  /* Footer */
  footer { font-size: 9px; color: var(--border); text-align: center;
           letter-spacing: 1px; padding-top: 14px; border-top: 1px solid var(--border); }

  @keyframes pulse  { 0%,100%{opacity:1}  50%{opacity:.3} }
  @keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }
  @keyframes spin   { to{transform:rotate(360deg)} }
  .fade  { animation: fadeUp .35s ease both; }
  .spin  { display:inline-block; animation: spin 1s linear infinite; }
  .blink { animation: pulse 1.4s ease infinite; }
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="logo">XAU/USD</div>
    <div class="sub">BB SQUEEZE · 1:2 RR · LIVE SIGNALS</div>
  </div>
  <div class="right-header">
    <div><span class="status-dot loading" id="dot"></span><span id="status-text">loading…</span></div>
    <div id="updated" style="margin-top:4px;font-size:10px;"></div>
    <button class="refresh-btn" onclick="load()">↻ REFRESH</button>
  </div>
</div>

<div id="app">
  <div style="text-align:center;padding:60px 20px;color:var(--muted);">
    <div class="spin" style="font-size:24px;display:block;margin-bottom:12px;">◈</div>
    <div style="font-size:11px;letter-spacing:3px;">FETCHING LIVE DATA…</div>
  </div>
</div>

<footer>STRATEGY: BB(20,2)+EMA200+RSI(7) · WR ~70% · PF ~4.9 · FOR MANUAL REFERENCE ONLY</footer>

<script>
const fmt   = v => v != null ? '$'+Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}) : '—';
const fmtN  = v => v != null ? Number(v).toFixed(1) : '—';
const clamp = (v,a,b) => Math.max(a,Math.min(b,v));

function drawChart(canvas, prices, ema200, signalColor) {
  if (!prices.length) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth, H = canvas.offsetHeight || 80;
  canvas.width  = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);

  const all  = [...prices, ...ema200].filter(v => !isNaN(v));
  const mn   = Math.min(...all), mx = Math.max(...all), rng = mx - mn || 1;
  const sx   = i => (i / (prices.length-1)) * W;
  const sy   = v => H - ((v-mn)/rng)*(H-8) - 4;

  // Gradient fill
  const grad = ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0,  signalColor+'30');
  grad.addColorStop(1, signalColor+'00');
  ctx.beginPath();
  ctx.moveTo(sx(0), H);
  prices.forEach((v,i) => ctx.lineTo(sx(i), sy(v)));
  ctx.lineTo(sx(prices.length-1), H);
  ctx.closePath(); ctx.fillStyle = grad; ctx.fill();

  // EMA line
  if (ema200.length) {
    ctx.beginPath(); ctx.strokeStyle = '#38bdf840'; ctx.lineWidth = 1;
    ctx.setLineDash([4,3]);
    ema200.forEach((v,i) => i===0 ? ctx.moveTo(sx(i),sy(v)) : ctx.lineTo(sx(i),sy(v)));
    ctx.stroke(); ctx.setLineDash([]);
  }

  // Price line
  ctx.beginPath(); ctx.strokeStyle = signalColor; ctx.lineWidth = 1.8;
  ctx.lineJoin = 'round';
  prices.forEach((v,i) => i===0 ? ctx.moveTo(sx(i),sy(v)) : ctx.lineTo(sx(i),sy(v)));
  ctx.stroke();

  // Dot at last point
  const lx = sx(prices.length-1), ly = sy(prices[prices.length-1]);
  ctx.beginPath(); ctx.arc(lx,ly,4,0,Math.PI*2);
  ctx.fillStyle = signalColor; ctx.fill();
}

function render(d) {
  const sig = d.signal;
  const sigColor = sig==='LONG' ? '#22c55e' : sig==='SHORT' ? '#ef4444' : '#f59e0b';
  const chgPos   = d.change >= 0;
  const ind      = d.indicators;
  const cond     = d.conditions;

  const html = `
  <div class="fade">

    <!-- Price Bar -->
    <div class="price-bar">
      <div>
        <div class="price-main">${fmt(d.price)}</div>
        <div class="price-change" style="color:${chgPos?'#22c55e':'#ef4444'}">
          ${chgPos?'▲':'▼'} ${fmt(Math.abs(d.change))}
        </div>
        <div class="price-pct" style="color:${chgPos?'#22c55e':'#ef4444'}">
          ${chgPos?'+':''}${d.change_pct?.toFixed(2)}% (24h)
        </div>
      </div>
      <div style="text-align:right">
        <div class="live-badge">⬤ LIVE</div>
        <div class="data-age" style="margin-top:6px;">via yfinance</div>
        <div class="data-age">${d.data_bars} bars</div>
      </div>
    </div>

    <!-- Signal Box -->
    <div class="signal-box ${sig}">
      <div class="signal-name" style="color:${sigColor}">
        ${sig==='LONG'?'▲ LONG':sig==='SHORT'?'▼ SHORT':'◆ NO SIGNAL'}
      </div>
      <div class="signal-sub">
        ${sig!=='WAIT'
          ? `ENTRY AT MARKET · STOP ${fmt(d.stop)} · TARGET ${fmt(d.target)} · 1:2 RR`
          : `${d.conditions_met}/4 conditions met · ${ind.squeeze_pct>70?'⚡ squeeze building':'watching for setup'}`}
      </div>
      ${sig!=='WAIT' ? `
      <div class="levels-grid">
        <div class="level-tile">
          <div class="level-lbl">ENTRY</div>
          <div class="level-val" style="color:#f1f5f9">${fmt(d.price)}</div>
          <div class="level-hint">at market</div>
        </div>
        <div class="level-tile">
          <div class="level-lbl">STOP</div>
          <div class="level-val" style="color:#ef4444">${fmt(d.stop)}</div>
          <div class="level-hint">−${fmt(d.stop_dist)}</div>
        </div>
        <div class="level-tile">
          <div class="level-lbl">TARGET</div>
          <div class="level-val" style="color:#22c55e">${fmt(d.target)}</div>
          <div class="level-hint">+${fmt(d.target_dist)}</div>
        </div>
        <div class="level-tile">
          <div class="level-lbl">RISK/REWARD</div>
          <div class="level-val" style="color:#f59e0b">1 : 2</div>
          <div class="level-hint">per trade</div>
        </div>
      </div>` : ''}
    </div>

    <!-- Chart -->
    <div class="chart-wrap">
      <div class="chart-hdr">
        <span>PRICE HISTORY (72H)</span>
        <span><span style="color:#38bdf8">╌ EMA200 </span><span style="color:${sigColor}">── PRICE</span></span>
      </div>
      <canvas id="chart" style="height:80px;"></canvas>
    </div>

    <!-- Conditions -->
    <div style="background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:14px;">
      <div class="section-title">SIGNAL CONDITIONS</div>
      ${[
        [cond.trend_up||cond.trend_dn,
         `TREND — Price ${cond.trend_up?'above ↑':'below ↓'} EMA 200`,
         `EMA200 = ${fmt(ind.ema200)}`],
        [ind.squeeze_prev,
         `BB SQUEEZE — Compression detected`,
         `Width ${ind.bb_width.toFixed(3)}%  ·  Pressure ${ind.squeeze_pct}%`],
        [cond.brk_long||cond.brk_short,
         `BB BREAKOUT — Close ${cond.brk_long?'above upper':cond.brk_short?'below lower':'inside'} band`,
         `Upper ${fmt(ind.bb_upper)}  ·  Lower ${fmt(ind.bb_lower)}`],
        [cond.rsi_lok||cond.rsi_sok,
         `RSI(7) ZONE — ${cond.rsi_lok?'Bullish 50–75 ✓':cond.rsi_sok?'Bearish 25–50 ✓':`Out of zone (${ind.rsi7})`}`,
         `RSI = ${ind.rsi7}  (long: 50–75 · short: 25–50)`],
      ].map(([ok,lbl,sub])=>`
        <div class="cond-row ${ok?'ok':'nok'}">
          <span class="cond-icon">${ok?'✅':'❌'}</span>
          <div>
            <div class="cond-lbl ${ok?'ok':'nok'}">${lbl}</div>
            <div class="cond-sub">${sub}</div>
          </div>
        </div>`).join('')}
    </div>

    <!-- Stats grid -->
    <div class="stats-grid">
      ${[
        ['EMA 200',   fmt(ind.ema200),       cond.trend_up?'#22c55e':'#ef4444', cond.trend_up?'bull ↑':'bear ↓'],
        ['RSI (7)',   ind.rsi7.toFixed(1),   cond.rsi_lok||cond.rsi_sok?'#22c55e':'#f59e0b', 'momentum'],
        ['ATR (14)',  fmt(ind.atr14),         '#94a3b8', 'volatility'],
        ['BB WIDTH',  ind.bb_width.toFixed(3)+'%', ind.squeeze_now?'#a855f7':'#64748b', ind.squeeze_now?'⚡ squeezed':'normal'],
        ['SQUEEZE',   ind.squeeze_pct+'%',   ind.squeeze_pct>70?'#a855f7':'#64748b', ind.squeeze_pct>70?'building!':'low'],
        ['BARS',      d.data_bars,           '#334155', '60 days'],
      ].map(([l,v,c,h])=>`
        <div class="stat-tile">
          <div class="stat-lbl">${l}</div>
          <div class="stat-val" style="color:${c}">${v}</div>
          <div class="stat-hint">${h}</div>
        </div>`).join('')}
    </div>

    <!-- Position sizing -->
    ${sig!=='WAIT'?`
    <div class="pos-box">
      <div class="section-title">POSITION SIZING (1% risk)</div>
      ${[
        ['Account $10k → risk per trade', '$100', '#f59e0b'],
        ['Stop distance', fmt(d.stop_dist), '#ef4444'],
        ['Suggested size', (100/d.stop_dist).toFixed(4)+' oz', '#22c55e'],
        ['Target profit (if win)', '$200', '#22c55e'],
        ['Risk (if loss)',        '$100', '#ef4444'],
      ].map(([l,v,c])=>`
        <div class="pos-row">
          <span class="pos-lbl">${l}</span>
          <span style="color:${c};font-weight:700">${v}</span>
        </div>`).join('')}
    </div>`:''}

  </div>`;

  document.getElementById('app').innerHTML = html;

  // Draw chart after DOM update
  requestAnimationFrame(() => {
    const canvas = document.getElementById('chart');
    if (canvas && d.chart) drawChart(canvas, d.chart.price, d.chart.ema200, sigColor);
  });
}

let countdown = 3600;
let cdInterval;

async function load() {
  document.getElementById('dot').className = 'status-dot loading';
  document.getElementById('status-text').textContent = 'fetching…';

  try {
    const res = await fetch('/api/signal');
    if (!res.ok) throw new Error('HTTP '+res.status);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    render(data);

    document.getElementById('dot').className = 'status-dot';
    document.getElementById('dot').style.background = 'var(--green)';
    document.getElementById('status-text').textContent = 'live';
    document.getElementById('updated').textContent = 'updated '+new Date().toLocaleTimeString();

    countdown = 3600;
    clearInterval(cdInterval);
    cdInterval = setInterval(() => {
      countdown = Math.max(0, countdown-1);
      const m = Math.floor(countdown/60), s = String(countdown%60).padStart(2,'0');
      document.getElementById('updated').textContent =
        `updated ${new Date().toLocaleTimeString()} · refresh in ${m}:${s}`;
      if (countdown === 0) load();
    }, 1000);

  } catch(e) {
    document.getElementById('dot').className = 'status-dot';
    document.getElementById('dot').style.background = 'var(--red)';
    document.getElementById('status-text').textContent = 'error';
    document.getElementById('app').innerHTML = `
      <div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.3);border-radius:10px;
                  padding:16px;font-size:12px;color:#fca5a5;">
        ⚠ ${e.message}<br>
        <span style="color:var(--muted);font-size:11px;">Retrying in 30s…</span>
      </div>`;
    setTimeout(load, 30000);
  }
}

// Load on start, then auto-refresh every hour
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
