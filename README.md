# xauusd-signals

XAUUSD signal generator with FastAPI, Telegram alerts, Supabase persistence, optional paid-access integrations, macro feature storage, and Render deployment.

## Backend Setup

1. Create a Supabase project and run `supabase_schema.sql` in the SQL editor.
2. Add the Render environment variables listed below.
3. Optional later: turn on Stripe/PostHog only if you move from private trading to paid subscribers.

Optional Stripe webhook endpoint when `PAYMENTS_ENABLED=true`:

```text
https://YOUR_RENDER_URL/api/webhooks/stripe
```

Recommended Stripe webhook events:

```text
checkout.session.completed
customer.subscription.created
customer.subscription.updated
customer.subscription.deleted
invoice.payment_succeeded
invoice.payment_failed
```

## Required Environment Variables

```bash
APP_BASE_URL=
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_ANON_KEY=
PAYMENTS_ENABLED=false
PRODUCT_ANALYTICS_ENABLED=false
OANDA_API_KEY=
OANDA_ACCOUNT_ID=
OANDA_ENV=practice
OANDA_INSTRUMENT=XAU_USD
OANDA_GRANULARITIES=M1,M5,M15,H1
OANDA_INGEST_ENABLED=false
OANDA_INGEST_SECONDS=300
MARKET_DATA_SOURCE=auto
SUPABASE_OHLCV_SOURCE=oanda
SUPABASE_OHLCV_LIMIT=5000
MACRO_FEATURES_ENABLED=true
MACRO_FEATURES_INGEST_SECONDS=21600
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EMAIL_ENABLED=false
EMAIL_BACKUP_MODE=failover
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_USE_TLS=true
SMTP_USE_SSL=false
EMAIL_FROM=
EMAIL_TO=
ADMIN_TOKEN=
TRADE_CHECK_SECONDS=900
HARD_FILTERS_ENABLED=true
RECENT_SIGNAL_COOLDOWN_MINUTES=120
MAX_SPREAD_CENTS=30
XAUUSD_SPREAD_CENTS=
SIGNAL_VALID_MINUTES=90
SIGNAL_INVALIDATION_ATR_BUFFER=0.20
GO_LIVE_MIN_CLOSED_SIGNALS=50
GO_LIVE_MIN_WIN_RATE=50
GO_LIVE_MIN_AVG_R=0.40
GO_LIVE_MIN_PROFIT_FACTOR=1.50
GO_LIVE_MAX_LOSING_STREAK=10
```

Optional monetization/product analytics variables:

```bash
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=
STRIPE_PRICE_ID=
POSTHOG_API_KEY=
POSTHOG_HOST=https://us.i.posthog.com
```

## Phase 1 Hard Filters

The live alert path now blocks executable alerts when any mandatory filter fails:

- London/NY session only: `07:00-20:00 UTC`
- high-impact event blackout: `-30m` to `+30m`
- no new signals after `18:00 UTC` Friday
- no signals during first two hours after Sunday open
- no signals during daily rollover: `21:55-22:10 UTC`
- no same-direction signal inside `RECENT_SIGNAL_COOLDOWN_MINUTES`
- skip high-volatility choppy regimes
- spread check when `XAUUSD_SPREAD_CENTS` is configured

The spread check is not broker-grade until OANDA or another bid/ask feed is wired. If `XAUUSD_SPREAD_CENTS` is blank, the app reports the check as not configured and does not block.

## Phase 2 Analytics and Rule Replay

Performance analytics are available at:

```text
GET /api/analytics/performance?source=auto&limit=500
```

`source=auto` reads Supabase first when configured and falls back to local JSON trade history. The response includes win rate, average R, net R, profit factor, max losing streak, max drawdown, monthly signal count, and breakdowns by timeframe/direction/quality.

Protected rule-only replay is available at:

```text
GET /api/backtest/rule-replay?token=ADMIN_TOKEN&interval=15m&days=30&include_filters=true
```

This is a baseline replay of the current single-timeframe rules. It is not the production multi-timeframe replay.

## Phase 3 Multi-Timeframe Execution Stack

Executable alerts now require the full intraday stack:

1. `1H` directional bias agrees with trade direction.
2. `15M` price is testing a scored order-block zone.
3. `5M` liquidity sweep confirms in the same direction.
4. `5M` EMA/RSI trigger checks pass.
5. Phase 1 hard filters pass.

Standalone `1H` or `15M` signals can still appear as context, but Telegram/Supabase executable alerts are only generated from the `15M_ORDER_BLOCK_5M_SWEEP` stack.

## Phase 4 OANDA Data Layer

OANDA candle ingestion is available but disabled by default:

```text
POST /api/data/ingest?token=ADMIN_TOKEN&granularities=M1,M5,M15,H1&count=500
GET  /api/data/status
GET  /api/data/spread?token=ADMIN_TOKEN
```

When `OANDA_INGEST_ENABLED=true`, the app starts a background worker that upserts recent OANDA candles into `ohlcv_xauusd` every `OANDA_INGEST_SECONDS`.

Required for candles:

```bash
OANDA_API_KEY=
OANDA_ENV=practice
OANDA_INSTRUMENT=XAU_USD
```

Required for live spread checks:

```bash
OANDA_ACCOUNT_ID=
```

The hard-filter spread check uses OANDA account pricing when `OANDA_ACCOUNT_ID` is configured. Until then, `XAUUSD_SPREAD_CENTS` can be set manually or left blank.

## Phase 5 Market Data Source Priority

The signal engine now uses `MARKET_DATA_SOURCE`:

```text
auto      -> Supabase OHLCV, then direct OANDA, then yfinance fallback
supabase  -> Supabase OHLCV only
oanda     -> direct OANDA candles only
yfinance  -> yfinance only
```

Use `auto` during rollout. After OANDA ingestion has populated enough candles in Supabase, switch to `supabase` for production.

## Phase 6 Supabase Full-Stack Backtest

After OANDA ingestion has populated `ohlcv_xauusd`, run the protected Supabase replay:

```text
GET /api/backtest/supabase-stack?token=ADMIN_TOKEN&days=30&max_checks=2000&include_filters=true
```

This replays the production stack on stored candles:

1. `1H` directional bias.
2. `15M` scored order-block zone.
3. `5M` liquidity sweep and EMA/RSI trigger.
4. Session, rollover, Friday close, Sunday open, choppy-regime, and same-direction cooldown filters.
5. Conservative TP/SL outcome tracking on future `5M` candles.

The response includes `metrics.monthly_signal_count`, `metrics.estimated_trades_per_month`, win rate, average R, net R, profit factor, max drawdown, skipped-condition counts, and the latest 200 replayed trades.

## Phase 7 Feedback Analytics and Readiness

User feedback analytics are available for subscribed Supabase users:

```text
GET /api/analytics/feedback?limit=500
Authorization: Bearer SUPABASE_USER_JWT
```

This compares `TAKEN`, `SKIPPED`, and `PENDING` signals, including:

- average R on taken trades
- would-have average R on skipped trades
- win rate by setup type, confidence tier, direction, timeframe, and session
- sample-size warnings before tuning too early
- setup-level recommendations when taken trades underperform or skipped trades would have worked

Admin launch readiness is available at:

```text
GET /api/admin/readiness?token=ADMIN_TOKEN&backtest_days=30
```

This reports critical blockers and warnings for Supabase, OANDA candles, OANDA pricing/spread checks, Telegram, optional Stripe/PostHog modes, hard filters, macro feature history, and whether `ohlcv_xauusd` has enough `H1/M15/M5` coverage for the full-stack backtest.

## Phase 8 Signal Expiration and Invalidation

Live trade records now include lifecycle fields:

- `valid_until`
- `invalidated_at`
- `invalidation_reason`
- `entry_low`
- `entry_high`

Open signals are automatically closed as:

- `TP` when target is touched first
- `SL` when stop is touched first
- `INVALIDATED` when the 15M/5M price breaks the stored order-block zone by `SIGNAL_INVALIDATION_ATR_BUFFER * ATR`
- `EXPIRED` when the signal is still open after `SIGNAL_VALID_MINUTES`

Telegram alerts include the valid-until timestamp, and result messages distinguish TP, SL, invalidated, and expired outcomes.

## Phase 9 Manual Take/Skip Feedback

The journal page now supports human-in-the-loop feedback:

```text
GET /trades
POST /api/trades/{trade_id}/action?token=ADMIN_TOKEN
```

Supported actions:

- `TAKEN`
- `SKIPPED`
- `PENDING`

The `/trades` UI has an `ADMIN_TOKEN` field that is stored in browser local storage and used by the action buttons. Decisions are written to local trade history and synced into Supabase `signals.user_action`, `signals.user_action_at`, and `signals.user_notes`.

## Phase 10 Email Backup Notifications

Telegram remains the primary alert channel. Email can now be configured as a backup or parallel channel:

```text
GET /api/test-notification?token=ADMIN_TOKEN
GET /api/test-email?token=ADMIN_TOKEN
```

Modes:

- `EMAIL_BACKUP_MODE=failover` sends email only when Telegram fails.
- `EMAIL_BACKUP_MODE=all` sends both Telegram and email.
- `EMAIL_BACKUP_MODE=off` disables email sends even if SMTP is configured.

Trade alerts, trade result messages, and startup notifications use this notification router. Readiness and health endpoints now report email backup status.

## Phase 11 Model Health and Go-Live Gates

Protected model health is available at:

```text
GET /api/model-health?token=ADMIN_TOKEN&source=auto&limit=1000&recent_days=30
```

Optional:

```text
refresh_current=true
```

The endpoint checks whether the system is statistically ready for live capital:

- closed signal sample size
- win rate
- average R per closed signal
- profit factor
- max losing streak
- recent 30-day degradation
- stale open signals past `valid_until`
- expired/invalidated ratio
- confidence, direction, result, and score distribution

Status values:

- `ready` means all critical gates pass.
- `watch` means critical gates pass but warnings remain.
- `blocked` means at least one critical go-live gate fails.

## Phase 12 Ops Console

The browser ops console is available at:

```text
GET /ops
```

It uses `ADMIN_TOKEN` from browser local storage and renders:

- launch readiness checks
- model-health go-live gates
- recent performance window
- result/confidence/direction distributions
- current protected maintenance actions

Ops actions available from the page:

- test notification router
- test email backup
- run check-now
- run 30-day Supabase stack backtest
- snapshot macro features into Supabase

## Phase 13 Admin Token Hardening

Protected admin endpoints now accept:

```text
Authorization: Bearer ADMIN_TOKEN
```

Query-string `?token=ADMIN_TOKEN` still works for simple CLI testing, but the browser Journal and Ops pages now use the `Authorization` header so the admin token is not placed in URLs, server access logs, or proxy logs.

## Phase 14 Private Mode and Macro Feature Store

Stripe and PostHog are skipped by default:

```bash
PAYMENTS_ENABLED=false
PRODUCT_ANALYTICS_ENABLED=false
```

Readiness reports them as disabled, not missing. Billing endpoints return `payments_disabled` until `PAYMENTS_ENABLED=true`.

Macro feature storage is enabled by default. It stores DXY, 10Y yield, VIX, macro score, macro bias, and raw macro drivers in Supabase `macro_features` every `MACRO_FEATURES_INGEST_SECONDS`.

Protected macro endpoints:

```text
GET  /api/macro/features
POST /api/macro/ingest
Authorization: Bearer ADMIN_TOKEN
```

After updating this phase, rerun `supabase_schema.sql` in Supabase so the `macro_features` table and policies exist.

The `/ops` page includes a `Snapshot Macro` action and shows the latest stored macro bias/score beside model-health state.

## Phase 15 Global Macro Catalyst Tape

The live news model now tracks country and macro catalysts that can move USD and gold, including U.S./China trade, Russia/Ukraine, Middle East/oil, Japan/BOJ/yen, Europe/UK rates, Fed/USD/yields, and global trade/sanctions. It queries GDELT over the recent multi-day window, scores each headline for gold and USD impact, and merges the aggregate gold score into the existing `news` score used by the weighted model.

Configuration:

```bash
GLOBAL_CATALYST_ENABLED=true
GLOBAL_CATALYST_TIMESPAN=7d
GLOBAL_CATALYST_MIN_HEADLINES=2
```

Legacy `TRADE_CATALYST_*` environment variables still work as fallbacks.

The dashboard surfaces `Global catalysts` in the headline area and adds a `Global country and macro catalyst impact` section with:

- model gold impact score and USD impact score
- region/catalyst label
- HIGH/MEDIUM impact rating
- per-headline playbook text so the trader can manually override the model

This only biases direction; executable alerts still require the 1H/15M/5M execution stack and hard filters to pass.
