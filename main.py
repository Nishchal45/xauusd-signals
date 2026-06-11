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
  TRADE_CHECK_SECONDS  → optional; default 300
  TRADE_LOG_PATH       → optional; default trade_history.json
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager, suppress
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from html import escape
from email.message import EmailMessage
import copy
import hashlib
import json
import hmac
import re
import smtplib
import ssl
import threading
import xml.etree.ElementTree as ET
import uvicorn, os, asyncio, httpx, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("xauusd")

CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def env_value(name: str, default: str = "") -> str:
    raw = os.environ.get(name, default)
    if raw is None:
        return ""
    value = str(raw)
    cleaned = CONTROL_CHAR_RE.sub("", value).strip()
    if cleaned.startswith(f"{name}="):
        cleaned = cleaned.split("=", 1)[1].strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    if cleaned != value:
        log.warning(f"{name} contained hidden whitespace/control characters; using sanitized value.")
    return cleaned


def env_url(name: str, default: str = "") -> str:
    value = env_value(name, default).rstrip("/")
    if value and not value.startswith(("http://", "https://")):
        log.warning(f"{name} is missing a URL scheme; assuming https://.")
        value = f"https://{value}"
    return value


TG_TOKEN   = env_value("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = env_value("TELEGRAM_CHAT_ID")
ADMIN_TOKEN = env_value("ADMIN_TOKEN")
EMAIL_ENABLED = os.environ.get("EMAIL_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
EMAIL_BACKUP_MODE = os.environ.get("EMAIL_BACKUP_MODE", "failover").lower()
SMTP_HOST = env_value("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = env_value("SMTP_USER")
SMTP_PASSWORD = env_value("SMTP_PASSWORD")
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes", "on"}
SMTP_USE_SSL = os.environ.get("SMTP_USE_SSL", "false").lower() in {"1", "true", "yes", "on"}
EMAIL_FROM = env_value("EMAIL_FROM", SMTP_USER)
EMAIL_TO = env_value("EMAIL_TO")
APP_BASE_URL = env_url("APP_BASE_URL", "http://localhost:8000")
SUPABASE_URL = env_url("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = env_value("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ANON_KEY = env_value("SUPABASE_ANON_KEY")
PAYMENTS_ENABLED = os.environ.get("PAYMENTS_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
STRIPE_SECRET_KEY = env_value("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = env_value("STRIPE_WEBHOOK_SECRET")
STRIPE_PRICE_ID = env_value("STRIPE_PRICE_ID")
PRODUCT_ANALYTICS_ENABLED = os.environ.get("PRODUCT_ANALYTICS_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
POSTHOG_API_KEY = env_value("POSTHOG_API_KEY")
POSTHOG_HOST = env_url("POSTHOG_HOST", "https://us.i.posthog.com")
OANDA_API_KEY = env_value("OANDA_API_KEY")
OANDA_ACCOUNT_ID = env_value("OANDA_ACCOUNT_ID")
OANDA_ENV = os.environ.get("OANDA_ENV", "practice").lower()
OANDA_API_BASE_URL = env_url("OANDA_API_BASE_URL")
OANDA_INSTRUMENT = env_value("OANDA_INSTRUMENT", "XAU_USD")
OANDA_GRANULARITIES = env_value("OANDA_GRANULARITIES", "M1,M5,M15,H1")
OANDA_INGEST_ENABLED = os.environ.get("OANDA_INGEST_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
OANDA_INGEST_SECONDS = int(os.environ.get("OANDA_INGEST_SECONDS", "300"))
MARKET_DATA_SOURCE = os.environ.get("MARKET_DATA_SOURCE", "auto").lower()
SUPABASE_OHLCV_SOURCE = os.environ.get("SUPABASE_OHLCV_SOURCE", "oanda")
SUPABASE_OHLCV_LIMIT = int(os.environ.get("SUPABASE_OHLCV_LIMIT", "5000"))
MACRO_FEATURES_ENABLED = os.environ.get("MACRO_FEATURES_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
MACRO_FEATURES_INGEST_SECONDS = int(os.environ.get("MACRO_FEATURES_INGEST_SECONDS", "21600"))
TRADE_CHECK_SECONDS = int(os.environ.get("TRADE_CHECK_SECONDS", "300"))
TRADE_STARTUP_DELAY_SECONDS = int(os.environ.get("TRADE_STARTUP_DELAY_SECONDS", "5"))
TRADE_LOG_PATH = os.environ.get("TRADE_LOG_PATH", "trade_history.json")
HARD_FILTERS_ENABLED = os.environ.get("HARD_FILTERS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
RECENT_SIGNAL_COOLDOWN_MINUTES = int(os.environ.get("RECENT_SIGNAL_COOLDOWN_MINUTES", "60"))
MAX_SPREAD_CENTS = float(os.environ.get("MAX_SPREAD_CENTS", "30"))
XAUUSD_SPREAD_CENTS = os.environ.get("XAUUSD_SPREAD_CENTS", "")
SPREAD_CACHE_TTL_SECONDS = int(os.environ.get("SPREAD_CACHE_TTL_SECONDS", "60"))
SIGNAL_VALID_MINUTES = int(os.environ.get("SIGNAL_VALID_MINUTES", "90"))
SIGNAL_INVALIDATION_ATR_BUFFER = float(os.environ.get("SIGNAL_INVALIDATION_ATR_BUFFER", "0.20"))
WATCH_ALERTS_ENABLED = os.environ.get("WATCH_ALERTS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
WATCH_CHECK_SECONDS = int(os.environ.get("WATCH_CHECK_SECONDS", str(TRADE_CHECK_SECONDS)))
WATCH_STARTUP_DELAY_SECONDS = int(os.environ.get("WATCH_STARTUP_DELAY_SECONDS", "15"))
WATCH_ALERT_COOLDOWN_MINUTES = int(os.environ.get("WATCH_ALERT_COOLDOWN_MINUTES", "30"))
WATCH_MIN_CONDITIONS = int(os.environ.get("WATCH_MIN_CONDITIONS", "4"))
LONDON_ALERTS_ENABLED = os.environ.get("LONDON_ALERTS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
LONDON_ALERT_CHECK_SECONDS = int(os.environ.get("LONDON_ALERT_CHECK_SECONDS", str(TRADE_CHECK_SECONDS)))
LONDON_ALERT_STARTUP_DELAY_SECONDS = int(os.environ.get("LONDON_ALERT_STARTUP_DELAY_SECONDS", "20"))
PROP_MODE_ENABLED = os.environ.get("PROP_MODE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
PROP_FIRM_NAME = env_value("PROP_FIRM_NAME", "Generic Prop Firm")
PROP_ACCOUNT_SIZE = float(os.environ.get("PROP_ACCOUNT_SIZE", "100000"))
PROP_PROFIT_TARGET_PCT = float(os.environ.get("PROP_PROFIT_TARGET_PCT", "10"))
PROP_MAX_DAILY_LOSS_PCT = float(os.environ.get("PROP_MAX_DAILY_LOSS_PCT", "5"))
PROP_MAX_TOTAL_DRAWDOWN_PCT = float(os.environ.get("PROP_MAX_TOTAL_DRAWDOWN_PCT", "10"))
PROP_RISK_PER_TRADE_PCT = float(os.environ.get("PROP_RISK_PER_TRADE_PCT", "0.50"))
PROP_PROTECT_RISK_PER_TRADE_PCT = float(os.environ.get("PROP_PROTECT_RISK_PER_TRADE_PCT", "0.25"))
PROP_MAX_DAILY_RISK_PCT = float(os.environ.get("PROP_MAX_DAILY_RISK_PCT", "1.00"))
PROP_MIN_RR = float(os.environ.get("PROP_MIN_RR", "2.00"))
PROP_MIN_CONFIDENCE = int(os.environ.get("PROP_MIN_CONFIDENCE", "70"))
PROP_MAX_TRADES_PER_DAY = int(os.environ.get("PROP_MAX_TRADES_PER_DAY", "3"))
PROP_MAX_LOSSES_PER_DAY = int(os.environ.get("PROP_MAX_LOSSES_PER_DAY", "2"))
PROP_ALLOW_COUNTERTREND = os.environ.get("PROP_ALLOW_COUNTERTREND", "false").lower() in {"1", "true", "yes", "on"}
LIQUIDITY_SWEEP_SCAN_BARS_5M = int(os.environ.get("LIQUIDITY_SWEEP_SCAN_BARS_5M", "12"))
LIQUIDITY_SWEEP_SCAN_BARS_15M = int(os.environ.get("LIQUIDITY_SWEEP_SCAN_BARS_15M", "6"))
LIQUIDITY_SWEEP_SCAN_BARS_1H = int(os.environ.get("LIQUIDITY_SWEEP_SCAN_BARS_1H", "3"))
ORDER_BLOCK_IMPULSE_ATR = float(os.environ.get("ORDER_BLOCK_IMPULSE_ATR", "1.90"))
ORDER_BLOCK_MIN_SCORE = int(os.environ.get("ORDER_BLOCK_MIN_SCORE", "55"))
ORDER_BLOCK_TESTED_MIN_SCORE = int(os.environ.get("ORDER_BLOCK_TESTED_MIN_SCORE", "55"))
ORDER_BLOCK_TEST_TOLERANCE_ATR = float(os.environ.get("ORDER_BLOCK_TEST_TOLERANCE_ATR", "0.30"))
TECHNICAL_SIGNAL_THRESHOLD_5M = int(os.environ.get("TECHNICAL_SIGNAL_THRESHOLD_5M", "58"))
TECHNICAL_SIGNAL_THRESHOLD_15M = int(os.environ.get("TECHNICAL_SIGNAL_THRESHOLD_15M", "62"))
TECHNICAL_SIGNAL_THRESHOLD_1H = int(os.environ.get("TECHNICAL_SIGNAL_THRESHOLD_1H", "62"))
STACK_H1_BIAS_MIN_SCORE = int(os.environ.get("STACK_H1_BIAS_MIN_SCORE", "10"))
FIB_EMA_LOOKBACK_BARS = int(os.environ.get("FIB_EMA_LOOKBACK_BARS", "192"))
FIB_EMA_MIN_IMPULSE_ATR = float(os.environ.get("FIB_EMA_MIN_IMPULSE_ATR", "1.50"))
FIB_EMA_ENTRY_TOLERANCE_ATR = float(os.environ.get("FIB_EMA_ENTRY_TOLERANCE_ATR", "0.15"))
FIB_EMA_STOP_BUFFER_ATR = float(os.environ.get("FIB_EMA_STOP_BUFFER_ATR", "0.25"))
ICT_LOOKBACK_M15_BARS = int(os.environ.get("ICT_LOOKBACK_M15_BARS", "96"))
ICT_BOS_LOOKBACK_5M_BARS = int(os.environ.get("ICT_BOS_LOOKBACK_5M_BARS", "36"))
ICT_MIN_IMPULSE_ATR = float(os.environ.get("ICT_MIN_IMPULSE_ATR", "1.20"))
ICT_ZONE_TOUCH_TOLERANCE_ATR = float(os.environ.get("ICT_ZONE_TOUCH_TOLERANCE_ATR", "0.20"))
ICT_STOP_BUFFER_ATR = float(os.environ.get("ICT_STOP_BUFFER_ATR", "0.20"))
ICT_MIN_RR = float(os.environ.get("ICT_MIN_RR", "2.00"))
ICT_MAX_TRADES_PER_DAY = int(os.environ.get("ICT_MAX_TRADES_PER_DAY", "3"))
ICT_MAX_CONSECUTIVE_LOSSES = int(os.environ.get("ICT_MAX_CONSECUTIVE_LOSSES", "2"))
ICT_DEFAULT_SESSIONS = {"LONDON_OPEN_CT", "NY_OPEN_OVERLAP_CT"}
ICT_LONDON_ONLY_SESSIONS = {"LONDON_OPEN_CT"}
MT5_AUTOTRADE_ENABLED = os.environ.get("MT5_AUTOTRADE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
MT5_MANUAL_EXECUTION_ENABLED = os.environ.get("MT5_MANUAL_EXECUTION_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
MT5_DEMO_MODE = os.environ.get("MT5_DEMO_MODE", "true").lower() in {"1", "true", "yes", "on"}
MT5_SYMBOL = env_value("MT5_SYMBOL", "XAUUSD")
MT5_LOGIN = env_value("MT5_LOGIN")
MT5_PASSWORD = env_value("MT5_PASSWORD")
MT5_SERVER = env_value("MT5_SERVER")
MT5_TERMINAL_PATH = env_value("MT5_TERMINAL_PATH")
MT5_RISK_PER_TRADE_PCT = float(os.environ.get("MT5_RISK_PER_TRADE_PCT", "1.0"))
MT5_MAX_VOLUME = float(os.environ.get("MT5_MAX_VOLUME", "5.0"))
MT5_DEVIATION_POINTS = int(os.environ.get("MT5_DEVIATION_POINTS", "20"))
MT5_MAGIC = int(os.environ.get("MT5_MAGIC", "26060801"))
MT5_CHECK_SECONDS = int(os.environ.get("MT5_CHECK_SECONDS", "60"))
MT5_AUTO_STARTUP_DELAY_SECONDS = int(os.environ.get("MT5_AUTO_STARTUP_DELAY_SECONDS", "20"))
MT5_MAX_ENTRY_DRIFT_USD = float(os.environ.get("MT5_MAX_ENTRY_DRIFT_USD", "2.50"))
MT5_ORDER_COMMENT = env_value("MT5_ORDER_COMMENT", "AURUM London BOS")
GO_LIVE_MIN_CLOSED_SIGNALS = int(os.environ.get("GO_LIVE_MIN_CLOSED_SIGNALS", "50"))
GO_LIVE_MIN_WIN_RATE = float(os.environ.get("GO_LIVE_MIN_WIN_RATE", "50"))
GO_LIVE_MIN_AVG_R = float(os.environ.get("GO_LIVE_MIN_AVG_R", "0.40"))
GO_LIVE_MIN_PROFIT_FACTOR = float(os.environ.get("GO_LIVE_MIN_PROFIT_FACTOR", "1.50"))
GO_LIVE_MAX_LOSING_STREAK = int(os.environ.get("GO_LIVE_MAX_LOSING_STREAK", "10"))
CONTEXT_TTL_SECONDS = int(os.environ.get("CONTEXT_TTL_SECONDS", "300"))
CONTEXT_CACHE = {"ts": None, "data": None}
CONTEXT_CACHE_LOCK = threading.Lock()
SIGNAL_CACHE_TTL_SECONDS = int(os.environ.get("SIGNAL_CACHE_TTL_SECONDS", "60"))
SIGNALS_CACHE = {"ts": None, "data": None}
SIGNALS_CACHE_LOCK = threading.Lock()
FIB_EMA_CACHE = {"ts": None, "data": None}
FIB_EMA_CACHE_LOCK = threading.Lock()
ICT_BOS_CACHE = {"ts": None, "data": None}
ICT_BOS_CACHE_LOCK = threading.Lock()
ICT_LONDON_BOS_CACHE = {"ts": None, "data": None}
ICT_LONDON_BOS_CACHE_LOCK = threading.Lock()
MT5_ORDER_LOCK = threading.Lock()
MT5_LAST_ORDER_KEY = {"key": None, "ts": None, "ticket": None}
SPREAD_CACHE = {"ts": None, "data": None}
EVENT_STUDY_TTL_SECONDS = int(os.environ.get("EVENT_STUDY_TTL_SECONDS", "21600"))
EVENT_STUDY_CACHE = {"ts": None, "data": None}
EXECUTABLE_TRADE_QUALITIES = {"CONFIRMED", "CAUTION", "COUNTERTREND"}
ACTIVE_SUBSCRIPTION_STATUSES = {"active", "trialing"}

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
TRADE_CATALYST_ENABLED = os.environ.get("TRADE_CATALYST_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
TRADE_CATALYST_TIMESPAN = os.environ.get("TRADE_CATALYST_TIMESPAN", "7d")
TRADE_CATALYST_MIN_HEADLINES = int(os.environ.get("TRADE_CATALYST_MIN_HEADLINES", "2"))
TRADE_CATALYST_QUERY = (
    '("Trump" AND ("China" OR "Xi Jinping") AND '
    '("trade deal" OR "trade talks" OR summit OR tariff OR tariffs OR '
    '"business executives" OR CEOs OR "CEO delegation" OR Boeing OR Nvidia OR Tesla OR Apple))'
)
GLOBAL_CATALYST_ENABLED = os.environ.get(
    "GLOBAL_CATALYST_ENABLED",
    os.environ.get("TRADE_CATALYST_ENABLED", "true"),
).lower() in {"1", "true", "yes", "on"}
GLOBAL_CATALYST_TIMESPAN = os.environ.get(
    "GLOBAL_CATALYST_TIMESPAN",
    os.environ.get("TRADE_CATALYST_TIMESPAN", "7d"),
)
GLOBAL_CATALYST_MIN_HEADLINES = int(os.environ.get(
    "GLOBAL_CATALYST_MIN_HEADLINES",
    os.environ.get("TRADE_CATALYST_MIN_HEADLINES", "2"),
))
GLOBAL_CATALYST_QUERY = (
    '("gold" OR XAUUSD OR "US dollar" OR DXY OR Treasury OR yields OR yen OR oil OR '
    'tariff OR tariffs OR sanctions OR "central bank" OR inflation OR CPI OR PCE OR FOMC OR '
    'Russia OR Ukraine OR China OR Taiwan OR Japan OR "Bank of Japan" OR BOJ OR '
    '"Middle East" OR Iran OR Israel OR Gaza OR "Red Sea" OR OPEC OR Saudi OR '
    'ECB OR Europe OR "Bank of England" OR BOE OR "United Kingdom")'
)


# ═══════════════════════════════════════════════════════════════════
#  SAAS INTEGRATIONS — SUPABASE / STRIPE / POSTHOG
# ═══════════════════════════════════════════════════════════════════
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def unix_to_utc_iso(value) -> str | None:
    try:
        if value is None:
            return None
        return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def bearer_token_from_request(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    return token or None


def admin_authorized(request: Request | None = None, token: str = "") -> bool:
    if not ADMIN_TOKEN:
        return False
    supplied = token or ""
    if not supplied and request is not None:
        supplied = bearer_token_from_request(request) or request.headers.get("x-admin-token", "")
    return hmac.compare_digest(str(supplied), ADMIN_TOKEN)


def service_status() -> dict:
    stripe_ready = bool(STRIPE_SECRET_KEY and STRIPE_PRICE_ID)
    stripe_webhook_ready = bool(STRIPE_WEBHOOK_SECRET)
    posthog_ready = bool(POSTHOG_API_KEY)
    return {
        "mode": "private_trading" if not PAYMENTS_ENABLED else "paid_subscription",
        "supabase": "configured" if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY else "missing",
        "stripe": "configured" if PAYMENTS_ENABLED and stripe_ready else "disabled" if not PAYMENTS_ENABLED else "missing",
        "stripe_webhook": "configured" if PAYMENTS_ENABLED and stripe_webhook_ready else "disabled" if not PAYMENTS_ENABLED else "missing",
        "posthog": "configured" if PRODUCT_ANALYTICS_ENABLED and posthog_ready else "disabled" if not PRODUCT_ANALYTICS_ENABLED else "missing",
        "telegram": "configured" if TG_TOKEN and TG_CHAT_ID else "missing",
        "email": "configured" if email_configured() else "missing",
        "oanda": "configured" if OANDA_API_KEY else "missing",
        "oanda_pricing": "configured" if OANDA_API_KEY and OANDA_ACCOUNT_ID else "missing",
        "macro_features": "enabled" if MACRO_FEATURES_ENABLED else "disabled",
    }


def error_response(code: str, message: str, status_code: int = 400, details: dict | None = None) -> JSONResponse:
    payload = {"error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return JSONResponse(payload, status_code=status_code)


def supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


def supabase_headers(use_service_role: bool = True, user_token: str | None = None) -> dict:
    key = SUPABASE_SERVICE_ROLE_KEY if use_service_role else SUPABASE_ANON_KEY
    auth_token = user_token or key
    return {
        "apikey": key,
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }


def supabase_table_url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


def supabase_request(
    method: str,
    table: str,
    *,
    params: dict | None = None,
    json_payload=None,
    prefer: str | None = None,
    use_service_role: bool = True,
):
    if not supabase_configured():
        return None
    headers = supabase_headers(use_service_role=use_service_role)
    if prefer:
        headers["Prefer"] = prefer
    try:
        with httpx.Client(timeout=12) as client:
            res = client.request(
                method,
                supabase_table_url(table),
                params=params,
                json=json_payload,
                headers=headers,
            )
        if res.status_code >= 400:
            log.error(f"Supabase {method} {table} {res.status_code}: {res.text[:500]}")
            return None
        if not res.text:
            return True
        return res.json()
    except Exception as e:
        log.error(f"Supabase {method} {table} error: {e}")
        return None


async def supabase_auth_user(request: Request) -> dict | None:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    token = bearer_token_from_request(request)
    if not token:
        return None
    try:
        headers = supabase_headers(use_service_role=False, user_token=token)
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers)
        if res.status_code != 200:
            return None
        user = res.json()
        return {
            "id": user.get("id"),
            "email": user.get("email"),
            "raw": user,
        }
    except Exception as e:
        log.error(f"Supabase auth error: {e}")
        return None


def get_user_subscription(user_id: str) -> dict | None:
    rows = supabase_request(
        "GET",
        "subscriptions",
        params={
            "user_id": f"eq.{user_id}",
            "select": "*",
            "order": "updated_at.desc",
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


def get_subscription_by_stripe_id(subscription_id: str | None) -> dict | None:
    if not subscription_id:
        return None
    rows = supabase_request(
        "GET",
        "subscriptions",
        params={
            "stripe_subscription_id": f"eq.{subscription_id}",
            "select": "*",
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


def get_subscription_by_customer(customer_id: str | None) -> dict | None:
    if not customer_id:
        return None
    rows = supabase_request(
        "GET",
        "subscriptions",
        params={
            "stripe_customer_id": f"eq.{customer_id}",
            "select": "*",
            "order": "updated_at.desc",
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


def subscription_is_active(subscription: dict | None) -> bool:
    if not subscription:
        return False
    status = str(subscription.get("status", "")).lower()
    if status not in ACTIVE_SUBSCRIPTION_STATUSES:
        return False
    current_period_end = subscription.get("current_period_end")
    if not current_period_end:
        return True
    try:
        end = parse_utc(str(current_period_end))
        return end > datetime.now(timezone.utc)
    except Exception:
        return True


async def entitlement_for_request(request: Request) -> dict:
    user = await supabase_auth_user(request)
    if not user:
        return {
            "authenticated": False,
            "active": False,
            "user": None,
            "subscription": None,
        }
    subscription = get_user_subscription(user["id"])
    return {
        "authenticated": True,
        "active": subscription_is_active(subscription),
        "user": {"id": user["id"], "email": user.get("email")},
        "subscription": subscription,
    }


def signal_record_to_supabase(record: dict) -> dict:
    return {
        "id": record.get("id"),
        "alert_key": record.get("alert_key"),
        "source": record.get("source"),
        "status": record.get("status"),
        "result": record.get("result"),
        "direction": record.get("direction"),
        "setup_type": record.get("setup_type"),
        "quality": record.get("quality"),
        "execution_timeframe": record.get("execution_timeframe"),
        "generated_at": record.get("opened_at"),
        "opened_bar": record.get("opened_bar"),
        "valid_until": record.get("valid_until"),
        "invalidated_at": record.get("invalidated_at"),
        "invalidation_reason": record.get("invalidation_reason"),
        "user_action": record.get("user_action"),
        "user_action_at": record.get("user_action_at"),
        "user_notes": record.get("user_notes"),
        "entry_low": record.get("entry_low"),
        "entry_high": record.get("entry_high"),
        "entry": record.get("entry"),
        "stop_loss": record.get("stop"),
        "take_profit_1": record.get("target"),
        "take_profit_2": record.get("target_2"),
        "risk": record.get("risk"),
        "reward": record.get("reward"),
        "risk_reward": record.get("rr"),
        "position_size_1pct_10k_oz": record.get("size_1pct_10k_oz"),
        "weighted_score": record.get("weighted_score"),
        "confidence": record.get("confidence"),
        "h1_score": record.get("h1_score"),
        "m15_score": record.get("m15_score"),
        "m5_score": record.get("m5_score"),
        "sweep_summary": record.get("sweep_summary"),
        "zone_low": record.get("zone_low"),
        "zone_high": record.get("zone_high"),
        "zone_score": record.get("zone_score"),
        "reason": record.get("reason"),
        "closed_at": record.get("closed_at"),
        "closed_bar": record.get("closed_bar"),
        "closed_price": record.get("closed_price"),
        "observed_price": record.get("observed_price"),
        "observed_high": record.get("observed_high"),
        "observed_low": record.get("observed_low"),
        "r_multiple": record.get("r_multiple"),
        "pnl_1pct_10k_usd": record.get("pnl_1pct_10k_usd"),
        "reasoning": record,
        "updated_at": utc_now_iso(),
    }


def upsert_signal_to_supabase(record: dict) -> bool:
    if not supabase_configured() or not record.get("id"):
        return False
    result = supabase_request(
        "POST",
        "signals",
        params={"on_conflict": "id"},
        json_payload=signal_record_to_supabase(record),
        prefer="resolution=merge-duplicates,return=minimal",
    )
    return result is not None


def insert_signal_outcome_to_supabase(record: dict) -> bool:
    if not supabase_configured() or not record.get("id") or record.get("status") != "CLOSED":
        return False
    result = supabase_request(
        "POST",
        "signal_outcomes",
        params={"on_conflict": "signal_id"},
        json_payload={
            "signal_id": record.get("id"),
            "outcome": record.get("result"),
            "outcome_price": record.get("closed_price"),
            "outcome_at": record.get("closed_at") or utc_now_iso(),
            "pnl_r": record.get("r_multiple"),
            "raw": record,
        },
        prefer="resolution=merge-duplicates,return=minimal",
    )
    return result is not None


def fetch_supabase_signals(limit: int = 500) -> list[dict]:
    rows = supabase_request(
        "GET",
        "signals",
        params={
            "select": "*",
            "order": "generated_at.desc",
            "limit": str(max(1, min(limit, 2000))),
        },
    )
    return rows if isinstance(rows, list) else []


def fetch_supabase_user_actions(user_id: str, limit: int = 500) -> list[dict]:
    rows = supabase_request(
        "GET",
        "user_signal_actions",
        params={
            "user_id": f"eq.{user_id}",
            "select": "*",
            "order": "acted_at.desc",
            "limit": str(max(1, min(limit, 5000))),
        },
    )
    return rows if isinstance(rows, list) else []


def fetch_supabase_outcomes(limit: int = 500) -> list[dict]:
    rows = supabase_request(
        "GET",
        "signal_outcomes",
        params={
            "select": "*",
            "order": "outcome_at.desc",
            "limit": str(max(1, min(limit, 5000))),
        },
    )
    return rows if isinstance(rows, list) else []


def upsert_subscription_from_stripe(subscription: dict, user_id: str | None = None) -> bool:
    if not supabase_configured() or not subscription:
        return False
    items = subscription.get("items", {}).get("data", [])
    price_id = None
    if items:
        price_id = (items[0].get("price") or {}).get("id")
    payload = {
        "stripe_customer_id": subscription.get("customer"),
        "stripe_subscription_id": subscription.get("id"),
        "status": subscription.get("status"),
        "price_id": price_id,
        "current_period_start": unix_to_utc_iso(subscription.get("current_period_start")),
        "current_period_end": unix_to_utc_iso(subscription.get("current_period_end")),
        "cancel_at_period_end": bool(subscription.get("cancel_at_period_end")),
        "updated_at": utc_now_iso(),
        "raw": subscription,
    }
    if user_id:
        payload["user_id"] = user_id
    result = supabase_request(
        "POST",
        "subscriptions",
        params={"on_conflict": "stripe_subscription_id"},
        json_payload=payload,
        prefer="resolution=merge-duplicates,return=minimal",
    )
    return result is not None


def stripe_configured() -> bool:
    return PAYMENTS_ENABLED and bool(STRIPE_SECRET_KEY)


def stripe_request(method: str, endpoint: str, *, data: dict | None = None):
    if not stripe_configured():
        return None
    try:
        with httpx.Client(timeout=15) as client:
            res = client.request(
                method,
                f"https://api.stripe.com/v1/{endpoint.lstrip('/')}",
                data=data,
                auth=(STRIPE_SECRET_KEY, ""),
            )
        if res.status_code >= 400:
            log.error(f"Stripe {method} {endpoint} {res.status_code}: {res.text[:500]}")
            return None
        return res.json()
    except Exception as e:
        log.error(f"Stripe {method} {endpoint} error: {e}")
        return None


def stripe_verify_signature(payload: bytes, signature_header: str | None) -> bool:
    if not PAYMENTS_ENABLED or not STRIPE_WEBHOOK_SECRET or not signature_header:
        return False
    parts = {}
    for item in signature_header.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts.setdefault(key, []).append(value)
    timestamp = parts.get("t", [None])[0]
    signatures = parts.get("v1", [])
    if not timestamp or not signatures:
        return False
    try:
        age = abs(datetime.now(timezone.utc).timestamp() - int(timestamp))
        if age > 300:
            return False
    except Exception:
        return False
    signed_payload = f"{timestamp}.".encode("utf-8") + payload
    expected = hmac.new(
        STRIPE_WEBHOOK_SECRET.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()
    return any(hmac.compare_digest(expected, signature) for signature in signatures)


def posthog_capture(distinct_id: str, event: str, properties: dict | None = None) -> bool:
    if not PRODUCT_ANALYTICS_ENABLED or not POSTHOG_API_KEY:
        return False
    payload = {
        "api_key": POSTHOG_API_KEY,
        "event": event,
        "distinct_id": distinct_id or "anonymous",
        "properties": properties or {},
        "timestamp": utc_now_iso(),
    }
    try:
        with httpx.Client(timeout=5) as client:
            res = client.post(f"{POSTHOG_HOST}/capture/", json=payload)
        if res.status_code >= 400:
            log.warning(f"PostHog {res.status_code}: {res.text[:300]}")
            return False
        return True
    except Exception as e:
        log.warning(f"PostHog error: {e}")
        return False


async def track_request_event(request: Request, event: str, properties: dict | None = None) -> None:
    if not PRODUCT_ANALYTICS_ENABLED:
        return
    user = await supabase_auth_user(request)
    distinct_id = user["id"] if user else request.headers.get("x-forwarded-for", "anonymous").split(",")[0]
    props = {
        "path": request.url.path,
        "method": request.method,
        **(properties or {}),
    }
    posthog_capture(distinct_id, event, props)


# ═══════════════════════════════════════════════════════════════════
#  OANDA DATA INGESTION
# ═══════════════════════════════════════════════════════════════════
def oanda_base_url() -> str:
    if OANDA_API_BASE_URL:
        return OANDA_API_BASE_URL
    if OANDA_ENV == "live":
        return "https://api-fxtrade.oanda.com"
    return "https://api-fxpractice.oanda.com"


def oanda_configured() -> bool:
    return bool(OANDA_API_KEY)


def oanda_headers() -> dict:
    return {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "RFC3339",
        "Content-Type": "application/json",
    }


def oanda_granularity_list(value: str | None = None) -> list[str]:
    raw = value or OANDA_GRANULARITIES
    allowed = {"M1", "M5", "M15", "H1"}
    granularities = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return [item for item in granularities if item in allowed] or ["M5", "M15", "H1"]


def oanda_get(path: str, params: dict | None = None) -> dict | None:
    if not oanda_configured():
        return None
    try:
        with httpx.Client(timeout=15) as client:
            res = client.get(f"{oanda_base_url()}{path}", params=params, headers=oanda_headers())
        if res.status_code >= 400:
            log.error(f"OANDA GET {path} {res.status_code}: {res.text[:500]}")
            return None
        return res.json()
    except Exception as e:
        log.error(f"OANDA GET {path} error: {e}")
        return None


def oanda_time_to_iso(value: str) -> str:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except Exception:
        return value


def oanda_candle_to_ohlcv(candle: dict, granularity: str) -> dict | None:
    mid = candle.get("mid") or {}
    if not mid:
        return None
    try:
        return {
            "source": "oanda",
            "instrument": OANDA_INSTRUMENT,
            "timeframe": granularity,
            "ts": oanda_time_to_iso(candle.get("time", "")),
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": int(candle.get("volume") or 0),
            "complete": bool(candle.get("complete")),
            "raw": candle,
        }
    except Exception as e:
        log.warning(f"OANDA candle parse error: {e}")
        return None


def fetch_oanda_candles(granularity: str, count: int = 500) -> list[dict]:
    count = int(clamp(count, 1, 5000))
    payload = oanda_get(
        f"/v3/instruments/{OANDA_INSTRUMENT}/candles",
        params={
            "price": "M",
            "granularity": granularity,
            "count": str(count),
        },
    )
    if not payload:
        return []
    records = []
    for candle in payload.get("candles", []):
        record = oanda_candle_to_ohlcv(candle, granularity)
        if record:
            records.append(record)
    return records


def upsert_ohlcv_to_supabase(records: list[dict]) -> int:
    if not records or not supabase_configured():
        return 0
    inserted = 0
    chunk_size = 500
    for idx in range(0, len(records), chunk_size):
        chunk = records[idx:idx + chunk_size]
        result = supabase_request(
            "POST",
            "ohlcv_xauusd",
            params={"on_conflict": "source,instrument,timeframe,ts"},
            json_payload=chunk,
            prefer="resolution=merge-duplicates,return=minimal",
        )
        if result is not None:
            inserted += len(chunk)
    return inserted


def ingest_oanda_ohlcv(granularities: list[str] | None = None, count: int = 500) -> dict:
    if not oanda_configured():
        return {"ok": False, "error": "OANDA_API_KEY is not configured.", "ingested": {}}
    if not supabase_configured():
        return {"ok": False, "error": "Supabase is not configured.", "ingested": {}}
    granularities = granularities or oanda_granularity_list()
    ingested = {}
    fetched = {}
    for granularity in granularities:
        records = fetch_oanda_candles(granularity, count=count)
        fetched[granularity] = len(records)
        ingested[granularity] = upsert_ohlcv_to_supabase(records)
    return {
        "ok": True,
        "source": "oanda",
        "instrument": OANDA_INSTRUMENT,
        "granularities": granularities,
        "fetched": fetched,
        "ingested": ingested,
        "timestamp": utc_now_iso(),
    }


def fetch_oanda_pricing() -> dict | None:
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        return None
    payload = oanda_get(
        f"/v3/accounts/{OANDA_ACCOUNT_ID}/pricing",
        params={"instruments": OANDA_INSTRUMENT},
    )
    prices = (payload or {}).get("prices", [])
    if not prices:
        return None
    price = prices[0]
    try:
        bid = float((price.get("bids") or [{}])[0].get("price"))
        ask = float((price.get("asks") or [{}])[0].get("price"))
        spread_cents = (ask - bid) * 100
        result = {
            "source": "oanda",
            "instrument": OANDA_INSTRUMENT,
            "time": oanda_time_to_iso(price.get("time", utc_now_iso())),
            "bid": round(bid, 3),
            "ask": round(ask, 3),
            "mid": round((bid + ask) / 2, 3),
            "spread_cents": round(spread_cents, 2),
            "raw": price,
        }
        if supabase_configured():
            supabase_request(
                "POST",
                "market_spreads",
                json_payload=result,
                prefer="return=minimal",
            )
        return result
    except Exception as e:
        log.warning(f"OANDA pricing parse error: {e}")
        return None


def get_cached_spread() -> dict | None:
    now = datetime.now(timezone.utc)
    cached_ts = SPREAD_CACHE.get("ts")
    if cached_ts and SPREAD_CACHE.get("data"):
        if (now - cached_ts).total_seconds() < SPREAD_CACHE_TTL_SECONDS:
            return SPREAD_CACHE["data"]
    data = fetch_oanda_pricing()
    if data:
        SPREAD_CACHE["ts"] = now
        SPREAD_CACHE["data"] = data
    return data


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


def interval_granularity(interval: str) -> str:
    return {"5m": "M5", "15m": "M15", "1h": "H1"}.get(interval.lower(), interval.upper())


def min_bars_for_interval(interval: str) -> int:
    return {"1h": 220, "15m": 400, "5m": 800}.get(interval.lower(), 400)


def fetch_limit_for_interval(interval: str) -> int:
    default_limit = {"1h": 1500, "15m": 3000, "5m": 5000}.get(interval.lower(), 3000)
    return int(clamp(SUPABASE_OHLCV_LIMIT, min_bars_for_interval(interval), max(default_limit, SUPABASE_OHLCV_LIMIT)))


def ohlcv_rows_to_df(rows: list[dict], source: str) -> tuple[pd.DataFrame | None, str | None]:
    if not rows:
        return None, f"No {source} OHLCV rows"
    try:
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
        df = df.set_index("ts")
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df = df[["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])].dropna()
        if df.empty:
            return None, f"{source} OHLCV rows parsed empty"
        df.attrs["source"] = source
        return df, None
    except Exception as e:
        return None, f"{source} OHLCV parse error: {e}"


def fetch_supabase_ohlcv(interval: str) -> tuple[pd.DataFrame | None, str | None]:
    if not supabase_configured():
        return None, "Supabase not configured"
    granularity = interval_granularity(interval)
    rows = supabase_request(
        "GET",
        "ohlcv_xauusd",
        params={
            "source": f"eq.{SUPABASE_OHLCV_SOURCE}",
            "instrument": f"eq.{OANDA_INSTRUMENT}",
            "timeframe": f"eq.{granularity}",
            "complete": "eq.true",
            "select": "ts,open,high,low,close,volume",
            "order": "ts.desc",
            "limit": str(fetch_limit_for_interval(interval)),
        },
    )
    df, err = ohlcv_rows_to_df(rows if isinstance(rows, list) else [], f"supabase:{SUPABASE_OHLCV_SOURCE}:{granularity}")
    if err:
        return None, err
    min_bars = min_bars_for_interval(interval)
    if len(df) < min_bars:
        return None, f"Supabase {granularity} has {len(df)} bars — need {min_bars}+"
    log.info(f"supabase {granularity}: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")
    return df, None


def supabase_ohlcv_rows(interval: str, limit: int, *, complete_only: bool = True) -> tuple[list[dict], str | None]:
    if not supabase_configured():
        return [], "Supabase not configured"

    granularity = interval_granularity(interval)
    target = int(clamp(limit, 1, 50000))
    page_size = 1000
    rows: list[dict] = []
    offset = 0

    while len(rows) < target:
        batch_limit = min(page_size, target - len(rows))
        params = {
            "source": f"eq.{SUPABASE_OHLCV_SOURCE}",
            "instrument": f"eq.{OANDA_INSTRUMENT}",
            "timeframe": f"eq.{granularity}",
            "select": "ts,open,high,low,close,volume",
            "order": "ts.desc",
            "limit": str(batch_limit),
            "offset": str(offset),
        }
        if complete_only:
            params["complete"] = "eq.true"
        batch = supabase_request("GET", "ohlcv_xauusd", params=params)
        if not isinstance(batch, list):
            return rows, f"Supabase {granularity} OHLCV query failed"
        rows.extend(batch)
        if len(batch) < batch_limit:
            break
        offset += batch_limit

    return rows, None


def fetch_supabase_ohlcv_history(interval: str, bars_needed: int) -> tuple[pd.DataFrame | None, str | None]:
    rows, err = supabase_ohlcv_rows(interval, bars_needed)
    if err:
        return None, err
    granularity = interval_granularity(interval)
    df, parse_err = ohlcv_rows_to_df(rows, f"supabase:{SUPABASE_OHLCV_SOURCE}:{granularity}:history")
    if parse_err:
        return None, parse_err
    if len(df) < min_bars_for_interval(interval):
        return None, f"Supabase {granularity} has {len(df)} bars — need {min_bars_for_interval(interval)}+"
    return df, None


def fetch_oanda_data(interval: str) -> tuple[pd.DataFrame | None, str | None]:
    if not oanda_configured():
        return None, "OANDA not configured"
    granularity = interval_granularity(interval)
    records = fetch_oanda_candles(granularity, count=min(fetch_limit_for_interval(interval), 5000))
    df, err = ohlcv_rows_to_df(records, f"oanda:{granularity}")
    if err:
        return None, err
    min_bars = min_bars_for_interval(interval)
    if len(df) < min_bars:
        return None, f"OANDA {granularity} has {len(df)} bars — need {min_bars}+"
    log.info(f"oanda {granularity}: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")
    return df, None


def prepare_yfinance_cache(yf) -> None:
    cache_dir = os.environ.get("YFINANCE_CACHE_DIR", "/tmp/yfinance-cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(cache_dir)
    except Exception as e:
        log.warning(f"yfinance cache setup skipped: {e}")


def fetch_yfinance_data(interval: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        import yfinance as yf
        prepare_yfinance_cache(yf)
        periods = {"1h": "60d", "15m": "30d", "5m": "30d"}
        period = periods.get(interval, "30d")
        min_bars = min_bars_for_interval(interval)
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
        df.attrs["source"] = "yfinance"
        return df, None
    except Exception as e:
        return None, f"yfinance error ({interval}): {e}"


def fetch_data(interval: str) -> tuple:
    """interval = '1h', '15m', or '5m'. Returns (DataFrame, error_or_None)."""
    source = MARKET_DATA_SOURCE
    errors = []
    fetchers = []
    if source == "supabase":
        fetchers = [fetch_supabase_ohlcv]
    elif source == "oanda":
        fetchers = [fetch_oanda_data]
    elif source == "yfinance":
        fetchers = [fetch_yfinance_data]
    else:
        fetchers = [fetch_supabase_ohlcv, fetch_oanda_data, fetch_yfinance_data]

    for fetcher in fetchers:
        df, err = fetcher(interval)
        if err or df is None:
            errors.append(err or f"{fetcher.__name__} returned no data")
            continue
        return df, None
    return None, " | ".join(errors) if errors else "No data source available"


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


def compact_error(error, limit: int = 160) -> str:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    reason = getattr(response, "reason_phrase", "")
    if status_code:
        return f"{status_code} {reason}".strip()

    text = str(error).splitlines()[0].strip() or error.__class__.__name__
    text = re.sub(r"https?://\S+", "<url>", text)
    if len(text) > limit:
        text = f"{text[:limit - 3].rstrip()}..."
    return text


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
        scan_bars = LIQUIDITY_SWEEP_SCAN_BARS_5M
    elif interval == "15m":
        lookback = 48
        scan_bars = LIQUIDITY_SWEEP_SCAN_BARS_15M
    else:
        lookback = 30
        scan_bars = LIQUIDITY_SWEEP_SCAN_BARS_1H
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


def empty_zone_context() -> dict:
    return {
        "active": False,
        "tested_zone": None,
        "zones": [],
        "liquidity_pools": [],
        "session_levels": {},
        "summary": "No high-quality 15M execution zone is being tested.",
    }


def price_in_zone(price: float, zone: dict, atr_value: float, tolerance_atr: float = ORDER_BLOCK_TEST_TOLERANCE_ATR) -> bool:
    tolerance = max(atr_value * tolerance_atr, price * 0.00005)
    return safe_float(zone.get("low")) - tolerance <= price <= safe_float(zone.get("high")) + tolerance


def detect_session_levels(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    try:
        work = df.copy()
        if work.index.tz is None:
            work.index = work.index.tz_localize("UTC")
        else:
            work.index = work.index.tz_convert("UTC")
        latest_day = work.index[-1].date()
        previous_days = [day for day in sorted(set(work.index.date)) if day < latest_day]
        prev_day = previous_days[-1] if previous_days else None
        current = work[work.index.date == latest_day]
        previous = work[work.index.date == prev_day] if prev_day else pd.DataFrame()

        def level_window(source: pd.DataFrame, start_hour: int, end_hour: int) -> dict:
            if source.empty:
                return {"high": None, "low": None}
            window = source[(source.index.hour >= start_hour) & (source.index.hour < end_hour)]
            if window.empty:
                return {"high": None, "low": None}
            return {"high": round(safe_float(window["high"].max()), 2), "low": round(safe_float(window["low"].min()), 2)}

        return {
            "previous_day": {
                "high": round(safe_float(previous["high"].max()), 2) if not previous.empty else None,
                "low": round(safe_float(previous["low"].min()), 2) if not previous.empty else None,
            },
            "asian": level_window(current, 0, 7),
            "london": level_window(current, 7, 12),
        }
    except Exception:
        return {}


def detect_liquidity_pools(df: pd.DataFrame, at: pd.Series, lookback: int = 96) -> list[dict]:
    if len(df) < 20:
        return []
    recent = df.iloc[-lookback:].copy()
    atr_value = max(safe_float(at.iloc[-1], 1.0), 1e-9)
    tolerance = atr_value * 0.50
    pools = []
    for side, column in (("BUY_SIDE", "high"), ("SELL_SIDE", "low")):
        values = recent[column].dropna().sort_values()
        if values.empty:
            continue
        clusters = []
        for value in values:
            if not clusters or abs(value - clusters[-1][-1]) > tolerance:
                clusters.append([safe_float(value)])
            else:
                clusters[-1].append(safe_float(value))
        for cluster in clusters:
            if len(cluster) >= 3:
                level = sum(cluster) / len(cluster)
                distance = abs(safe_float(df["close"].iloc[-1]) - level) / atr_value
                pools.append({
                    "side": side,
                    "level": round(level, 2),
                    "touches": len(cluster),
                    "distance_atr": round(distance, 2),
                })
    return sorted(pools, key=lambda item: (item["distance_atr"], -item["touches"]))[:6]


def detect_order_block_zones(df: pd.DataFrame, at: pd.Series, interval: str) -> dict:
    if interval != "15m" or len(df) < 80:
        return empty_zone_context()

    price = safe_float(df["close"].iloc[-1])
    current_atr = max(safe_float(at.iloc[-1], 1.0), 1e-9)
    start = max(5, len(df) - 180)
    end = len(df) - 4
    zones = []

    for idx in range(start, end):
        candle = df.iloc[idx]
        open_price = safe_float(candle["open"])
        close = safe_float(candle["close"])
        high = safe_float(candle["high"])
        low = safe_float(candle["low"])
        atr_value = max(safe_float(at.iloc[idx], current_atr), 1e-9)
        future = df.iloc[idx + 1:idx + 4]
        if future.empty:
            continue

        future_high_close = safe_float(future["close"].max())
        future_low_close = safe_float(future["close"].min())
        impulse_up = (future_high_close - close) / atr_value
        impulse_down = (close - future_low_close) / atr_value
        direction = None
        displacement = 0.0

        if close < open_price and impulse_up >= ORDER_BLOCK_IMPULSE_ATR:
            direction = "LONG"
            displacement = impulse_up
        elif close > open_price and impulse_down >= ORDER_BLOCK_IMPULSE_ATR:
            direction = "SHORT"
            displacement = impulse_down
        if not direction:
            continue

        zone_low = low
        zone_high = high
        formed_at = df.index[idx]
        after = df.iloc[idx + 4:]
        if direction == "LONG":
            invalidated = not after.empty and bool((after["close"] < zone_low - atr_value * 0.20).any())
        else:
            invalidated = not after.empty and bool((after["close"] > zone_high + atr_value * 0.20).any())
        if invalidated:
            continue

        retests = 0
        if not after.empty:
            touches = (after["low"] <= zone_high) & (after["high"] >= zone_low)
            retests = int(touches.sum())
        age_bars = len(df) - 1 - idx
        zone_mid = (zone_low + zone_high) / 2
        distance_atr = abs(price - zone_mid) / current_atr
        tested = price_in_zone(price, {"low": zone_low, "high": zone_high}, current_atr)
        score = (
            45
            + min(28, displacement * 8)
            + (14 if retests == 0 else 0)
            + (10 if tested else 0)
            - min(18, age_bars * 0.12)
            - min(18, retests * 5)
            - min(16, distance_atr * 4)
        )
        score = int(round(clamp(score, 0, 100)))
        if score < ORDER_BLOCK_MIN_SCORE:
            continue

        zones.append({
            "type": "ORDER_BLOCK",
            "direction": direction,
            "low": round(zone_low, 2),
            "high": round(zone_high, 2),
            "mid": round(zone_mid, 2),
            "score": score,
            "formed_at": str(formed_at),
            "age_bars": age_bars,
            "displacement_atr": round(displacement, 2),
            "retests": retests,
            "distance_atr": round(distance_atr, 2),
            "tested": tested,
            "summary": (
                f"{direction} 15M order block {zone_low:.2f}-{zone_high:.2f}; "
                f"{displacement:.1f} ATR displacement, {retests} retests."
            ),
        })

    zones = sorted(zones, key=lambda item: (item["tested"], item["score"], -item["distance_atr"]), reverse=True)[:8]
    tested_zones = [zone for zone in zones if zone["tested"] and zone["score"] >= ORDER_BLOCK_TESTED_MIN_SCORE]
    tested_zone = max(tested_zones, key=lambda item: item["score"]) if tested_zones else None
    liquidity_pools = detect_liquidity_pools(df, at)
    session_levels = detect_session_levels(df)
    return {
        "active": bool(tested_zone),
        "tested_zone": tested_zone,
        "zones": zones,
        "liquidity_pools": liquidity_pools,
        "session_levels": session_levels,
        "summary": tested_zone["summary"] if tested_zone else "No high-quality 15M order block is currently being tested.",
    }


def build_zone_trade_plan(price: float, atr_value: float, direction: str, zone: dict | None) -> dict:
    if direction not in ("LONG", "SHORT") or not zone:
        return build_trade_plan(price, atr_value, "NONE", {})
    zone_low = safe_float(zone.get("low"), price)
    zone_high = safe_float(zone.get("high"), price)
    if direction == "LONG":
        entry_low = zone_low
        entry_high = zone_high
        stop = zone_low - atr_value * 0.50
        risk = max(entry_high - stop, 1e-9)
        tp1 = entry_high + risk * 1.50
        tp2 = entry_high + risk * 3.00
        invalidates = "15M closes below the demand/order-block low or 5M sweep low fails."
        entry = min(max(price, entry_low), entry_high)
    else:
        entry_low = zone_low
        entry_high = zone_high
        stop = zone_high + atr_value * 0.50
        risk = max(stop - entry_low, 1e-9)
        tp1 = entry_low - risk * 1.50
        tp2 = entry_low - risk * 3.00
        invalidates = "15M closes above the supply/order-block high or 5M sweep high fails."
        entry = max(min(price, entry_high), entry_low)

    return {
        "entry": round(entry, 2),
        "entry_low": round(entry_low, 2),
        "entry_high": round(entry_high, 2),
        "stop": round(stop, 2),
        "target": round(tp1, 2),
        "target_2": round(tp2, 2),
        "risk": round(risk, 2),
        "reward": round(abs(tp1 - entry), 2),
        "rr": round(abs(tp1 - entry) / max(abs(entry - stop), 1e-9), 2),
        "rr_tp1": 1.5,
        "rr_tp2": 3.0,
        "position_1pct_10k_oz": round(100 / max(abs(entry - stop), 1e-9), 4),
        "invalidates": invalidates,
    }


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
    zone_context = detect_order_block_zones(df, at, interval)

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
        execution_threshold = TECHNICAL_SIGNAL_THRESHOLD_15M if interval == "15m" else TECHNICAL_SIGNAL_THRESHOLD_5M
        signal = "LONG" if (
            technical_score >= execution_threshold and sweep["direction"] == "LONG" and sweep["confirmed"]
        ) else "SHORT" if (
            technical_score <= -execution_threshold and sweep["direction"] == "SHORT" and sweep["confirmed"]
        ) else "WAIT"
    else:
        signal = (
            "LONG" if technical_score >= TECHNICAL_SIGNAL_THRESHOLD_1H
            else "SHORT" if technical_score <= -TECHNICAL_SIGNAL_THRESHOLD_1H
            else "WAIT"
        )

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
        "zone_context": zone_context,
        "trade": trade,
        "chart": {"price": price_hist, "ema21": ema21_hist, "ema50": ema50_hist, "ema200": ema200_hist},
    }


def get_signal(interval: str = "1h") -> dict:
    df, err = fetch_data(interval)
    if err or df is None:
        log.warning(f"[{interval}] {err}")
        return {"error": err or "No data", "timeframe": interval.upper()}
    signal = compute_signal(df, interval)
    signal["data_source"] = df.attrs.get("source", MARKET_DATA_SOURCE)
    signal["data_granularity"] = interval_granularity(interval)
    return signal


def get_both_signals(force: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    cached_at = SIGNALS_CACHE.get("ts")
    if (
        not force
        and SIGNAL_CACHE_TTL_SECONDS > 0
        and SIGNALS_CACHE.get("data") is not None
        and cached_at is not None
        and (now - cached_at).total_seconds() <= SIGNAL_CACHE_TTL_SECONDS
    ):
        return copy.deepcopy(SIGNALS_CACHE["data"])

    with SIGNALS_CACHE_LOCK:
        now = datetime.now(timezone.utc)
        cached_at = SIGNALS_CACHE.get("ts")
        if (
            not force
            and SIGNAL_CACHE_TTL_SECONDS > 0
            and SIGNALS_CACHE.get("data") is not None
            and cached_at is not None
            and (now - cached_at).total_seconds() <= SIGNAL_CACHE_TTL_SECONDS
        ):
            return copy.deepcopy(SIGNALS_CACHE["data"])

        s1h  = get_signal("1h")
        s15m = get_signal("15m")
        s5m = get_signal("5m")
        price = s5m.get("price") or s15m.get("price") or s1h.get("price") or 0
        data = {
            "h1":        s1h,
            "m15":       s15m,
            "m5":        s5m,
            "price":     price,
            "timestamp": now.strftime("%Y-%m-%d %H:%M UTC"),
        }
        SIGNALS_CACHE["ts"] = now
        SIGNALS_CACHE["data"] = copy.deepcopy(data)
        return data


# ═══════════════════════════════════════════════════════════════════
#  FREE MARKET CONTEXT
# ═══════════════════════════════════════════════════════════════════
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def macro_point(symbol: str, label: str, bullish_when_down: bool, unit: str = "") -> dict:
    try:
        import yfinance as yf
        prepare_yfinance_cache(yf)
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
            "status": f"offline: {compact_error(e)}",
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


def macro_driver(macro: dict, label: str) -> dict:
    for driver in macro.get("drivers", []):
        if driver.get("label") == label:
            return driver
    return {}


def macro_feature_record(macro: dict | None = None) -> dict:
    macro = macro or get_macro_context()
    dxy = macro_driver(macro, "DXY")
    us10y = macro_driver(macro, "10Y")
    vix = macro_driver(macro, "VIX")
    return {
        "ts": utc_now_iso(),
        "source": "yfinance",
        "status": macro.get("status", "unknown"),
        "macro_score": macro.get("score"),
        "macro_bias": macro.get("bias"),
        "dxy": dxy.get("value"),
        "dxy_change": dxy.get("change"),
        "dxy_change_pct": dxy.get("change_pct"),
        "us10y": us10y.get("value"),
        "us10y_change": us10y.get("change"),
        "us10y_change_pct": us10y.get("change_pct"),
        "vix": vix.get("value"),
        "vix_change": vix.get("change"),
        "vix_change_pct": vix.get("change_pct"),
        "raw": macro,
        "created_at": utc_now_iso(),
    }


def insert_macro_features_to_supabase(macro: dict | None = None) -> dict:
    if not supabase_configured():
        return {"ok": False, "error": "supabase_missing"}
    record = macro_feature_record(macro)
    result = supabase_request(
        "POST",
        "macro_features",
        json_payload=record,
        prefer="return=representation",
    )
    if result is None:
        return {"ok": False, "error": "macro_insert_failed", "record": record}
    return {
        "ok": True,
        "record": result[0] if isinstance(result, list) and result else record,
    }


def fetch_macro_features(limit: int = 50) -> list:
    if not supabase_configured():
        return []
    rows = supabase_request(
        "GET",
        "macro_features",
        params={
            "select": "*",
            "order": "ts.desc",
            "limit": str(max(1, min(int(limit), 500))),
        },
    )
    return rows if isinstance(rows, list) else []


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


def phrase_matches(text: str, phrase: str) -> bool:
    phrase = phrase.lower()
    if " " in phrase or "-" in phrase or "." in phrase:
        return phrase in text
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def has_phrase(text: str, phrases: list[str]) -> bool:
    return any(phrase_matches(text, phrase) for phrase in phrases)


def trade_catalyst_score(title: str) -> tuple:
    text = title.lower()
    has_us_side = has_phrase(text, ["trump", "white house", "u.s.", "us "])
    has_china_side = has_phrase(text, ["china", "xi", "beijing"])
    has_trade_angle = has_phrase(text, [
        "trade", "tariff", "tariffs", "summit", "deal", "talks",
        "delegation", "ceo", "ceos", "business executives",
        "boeing", "nvidia", "tesla", "apple", "jensen huang", "tim cook", "musk",
    ])
    if not (has_us_side and has_china_side and has_trade_angle):
        return 0, False, "neutral", "LOW", "not a U.S.-China trade catalyst", []

    deal_optimism = [
        "trade deal", "trade deals", "deal with china", "strike a deal",
        "deal optimism", "tariff relief", "tariff truce", "tariffs lowered",
        "cuts tariffs", "lower tariffs", "de-escalation", "stability",
        "open up china", "opening markets", "boeing deal", "invest",
        "hundreds of billions",
    ]
    disappointment = [
        "no deal", "no major deal", "no major trade", "few deals",
        "no big wins", "failed", "fell short", "underwhelmed",
        "nothing of real substance", "empty", "stalemate", "no breakthrough", "no breakthroughs",
        "sell off", "selling off", "tanked", "tariffs loom", "trade war",
        "retaliation", "taiwan", "rare earth", "leverage",
    ]
    delegation_optimism = [
        "business executives", "ceos", "ceo delegation", "accompanying trump",
        "join trade talks", "billionaire", "billionaires",
    ]
    ceo_delegation = [
        "ceos", "ceo", "business executives", "billionaire", "billionaires",
        "musk", "tim cook", "jensen huang", "nvidia", "tesla", "apple", "boeing",
    ]
    escalation = [
        "trade war", "retaliation", "new tariffs", "higher tariffs",
        "taiwan", "rare earth", "sanctions", "export controls",
    ]

    score = 0
    tags = []
    if has_phrase(text, disappointment):
        score += 1
        tags.append("summit disappointment/risk-off")
    if has_phrase(text, escalation):
        score += 1
        tags.append("trade tension safe-haven risk")
    has_disappointment = has_phrase(text, disappointment)
    if has_phrase(text, deal_optimism) and not has_disappointment:
        score -= 1
        tags.append("trade-deal optimism/risk-on")
    if (has_phrase(text, delegation_optimism) or has_phrase(text, ceo_delegation)) and not has_disappointment:
        score -= 1
        tags.append("CEO delegation risk-on")

    score = int(clamp(score, -2, 2))
    bias = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
    impact = "HIGH" if abs(score) >= 2 else "MEDIUM"
    reason = (
        "risk-on U.S.-China deal catalyst pressures gold" if score < 0 else
        "summit disappointment/tension can support safe-haven gold" if score > 0 else
        "U.S.-China trade catalyst, direction mixed"
    )
    return score, True, bias, impact, reason, tags


def trade_catalyst_offline(status: str) -> dict:
    return {
        "active": False,
        "score": 0,
        "bias": "MIXED",
        "risk": "UNKNOWN",
        "headline_count": 0,
        "directional_count": 0,
        "headlines": [],
        "status": status,
        "source": f"GDELT trade catalyst {TRADE_CATALYST_TIMESPAN}",
        "query": TRADE_CATALYST_QUERY,
        "reason": "trade catalyst feed unavailable",
    }


def get_trade_catalyst_context() -> dict:
    if not TRADE_CATALYST_ENABLED:
        return {
            "active": False,
            "score": 0,
            "bias": "MIXED",
            "risk": "LOW",
            "headline_count": 0,
            "directional_count": 0,
            "headlines": [],
            "status": "disabled",
            "source": f"GDELT trade catalyst {TRADE_CATALYST_TIMESPAN}",
            "query": TRADE_CATALYST_QUERY,
            "reason": "trade catalyst detection disabled by env",
        }

    params = {
        "query": TRADE_CATALYST_QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 24,
        "timespan": TRADE_CATALYST_TIMESPAN,
        "sort": "hybridrel",
    }
    try:
        with httpx.Client(timeout=8) as client:
            res = client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
            res.raise_for_status()
            payload = res.json()

        headlines = []
        seen_titles = set()
        raw_score = 0
        directional_count = 0
        impact_points = 0
        for item in payload.get("articles", [])[:20]:
            title = item.get("title", "").strip()
            title_key = title.lower()
            if not title or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            score, is_risk, bias, impact, reason, tags = trade_catalyst_score(title)
            if not is_risk:
                continue
            raw_score += score
            directional_count += 1 if score else 0
            impact_points += 3 if impact == "HIGH" else 2 if impact == "MEDIUM" else 1
            headlines.append({
                "title": title,
                "url": item.get("url", ""),
                "source": item.get("sourceCountry", "") or item.get("domain", ""),
                "bias": bias,
                "risk": True,
                "impact": impact,
                "reason": reason,
                "published": item.get("seendate", ""),
                "tags": tags,
                "category": "us_china_trade_catalyst",
            })

        active = directional_count >= max(1, TRADE_CATALYST_MIN_HEADLINES) or abs(raw_score) >= 2
        score = int(clamp(raw_score, -3, 3)) if active else 0
        bias = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "MIXED"
        risk = "MEDIUM" if active or impact_points >= 6 else "LOW"
        reason = (
            "Trump/China trade-deal or CEO-delegation trend is active; bearish score means risk-on pressure for gold."
            if active else
            "No directional U.S.-China trade catalyst trend detected."
        )
        return {
            "active": active,
            "score": score,
            "raw_score": raw_score,
            "bias": bias,
            "risk": risk,
            "headline_count": len(headlines),
            "directional_count": directional_count,
            "headlines": headlines[:8],
            "status": "ok",
            "source": f"GDELT trade catalyst {TRADE_CATALYST_TIMESPAN}",
            "query": TRADE_CATALYST_QUERY,
            "reason": reason,
        }
    except Exception as e:
        return trade_catalyst_offline(f"offline: {compact_error(e)}")


def merge_trade_catalyst_into_news(news: dict, catalyst: dict) -> dict:
    merged = {**news, "trade_catalyst": catalyst}
    if not catalyst.get("active"):
        return merged

    base_score = int(safe_float(news.get("score")))
    catalyst_score = int(safe_float(catalyst.get("score")))
    adjusted_score = int(clamp(base_score + catalyst_score, -4, 4))
    risk_rank = {"LOW": 0, "WATCH": 1, "UNKNOWN": 1, "MEDIUM": 2, "HIGH": 3}
    current_risk = news.get("risk", "UNKNOWN")
    catalyst_risk = catalyst.get("risk", "UNKNOWN")
    merged["base_score"] = base_score
    merged["score"] = adjusted_score
    merged["bias"] = "BULLISH" if adjusted_score >= 2 else "BEARISH" if adjusted_score <= -2 else "MIXED"
    merged["risk"] = catalyst_risk if risk_rank.get(catalyst_risk, 1) > risk_rank.get(current_risk, 1) else current_risk
    merged["source"] = f"{news.get('source', 'news')} + {catalyst.get('source', 'trade catalyst')}"
    merged["reason"] = catalyst.get("reason")

    headlines = []
    seen_titles = set()
    for item in (catalyst.get("headlines") or []) + (news.get("headlines") or []):
        title = item.get("title", "")
        title_key = title.lower()
        if not title or title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        headlines.append(item)
    merged["headlines"] = headlines[:8]
    return merged


GLOBAL_CATALYST_RULES = [
    {
        "id": "us_china_trade",
        "label": "U.S./China trade",
        "region": "U.S./China",
        "keywords": [
            "trump", "china", "xi", "beijing", "taiwan", "tariff", "tariffs",
            "trade deal", "trade talks", "ceo", "ceos", "business executives",
            "nvidia", "tesla", "apple", "boeing", "rare earth",
        ],
        "gold_bullish": [
            "no deal", "few deals", "no big wins", "trade war", "retaliation",
            "new tariffs", "higher tariffs", "taiwan", "rare earth", "sanctions",
            "export controls", "no breakthrough", "tension", "escalation",
        ],
        "gold_bearish": [
            "trade deal", "tariff relief", "tariff truce", "cuts tariffs",
            "lower tariffs", "de-escalation", "open up china", "opening markets",
            "ceo delegation", "business executives", "invest", "stability",
        ],
        "risk": "Risk-on trade de-escalation pressures gold; trade stress supports safe-haven demand.",
    },
    {
        "id": "russia_ukraine",
        "label": "Russia/Ukraine",
        "region": "Europe",
        "keywords": ["russia", "ukraine", "putin", "kremlin", "kyiv", "nato"],
        "gold_bullish": [
            "war", "attack", "missile", "drone", "invasion", "nuclear",
            "sanctions", "escalation", "strike", "mobilization",
        ],
        "gold_bearish": [
            "ceasefire", "peace deal", "peace talks", "truce", "de-escalation",
            "withdrawal", "settlement",
        ],
        "risk": "Escalation is gold-positive through safe-haven demand; ceasefire headlines reduce haven premium.",
    },
    {
        "id": "middle_east_oil",
        "label": "Middle East/Oil",
        "region": "Middle East",
        "keywords": [
            "middle east", "iran", "israel", "gaza", "hamas", "hezbollah",
            "houthi", "red sea", "saudi", "opec", "oil", "brent", "wti",
        ],
        "gold_bullish": [
            "attack", "missile", "war", "strike", "red sea", "supply disruption",
            "oil surges", "oil jumps", "sanctions", "nuclear", "escalation",
        ],
        "gold_bearish": [
            "ceasefire", "truce", "peace talks", "de-escalation", "oil falls",
            "oil drops", "supply restored",
        ],
        "risk": "Regional escalation and oil shocks support gold; ceasefire/oil relief removes risk premium.",
    },
    {
        "id": "japan_boj_yen",
        "label": "Japan/BOJ/Yen",
        "region": "Japan",
        "keywords": ["japan", "boj", "bank of japan", "yen", "tokyo", "ueda"],
        "gold_bullish": [
            "yen strengthens", "yen rises", "yen jumps", "intervention",
            "boj hikes", "rate hike", "dollar falls", "dollar weakens",
        ],
        "gold_bearish": [
            "yen weakens", "yen falls", "boj dovish", "holds rates",
            "dollar rises", "dollar strengthens", "carry trade",
        ],
        "risk": "Yen strength can weaken USD and support gold; yen weakness/carry trade usually supports USD and pressures gold.",
    },
    {
        "id": "europe_uk_rates",
        "label": "Europe/UK rates",
        "region": "Europe/UK",
        "keywords": [
            "ecb", "euro", "eurozone", "lagarde", "bank of england", "boe",
            "united kingdom", "uk inflation", "sterling", "pound",
        ],
        "gold_bullish": [
            "hawkish", "rate hike", "euro rises", "sterling rises", "dollar falls",
            "dollar weakens", "hot inflation",
        ],
        "gold_bearish": [
            "rate cut", "cuts rates", "dovish", "euro falls", "sterling falls",
            "dollar rises", "dollar strengthens", "recession",
        ],
        "risk": "Non-U.S. hawkish surprise can weaken USD and support gold; foreign easing can lift USD and pressure gold.",
    },
    {
        "id": "fed_usd_yields",
        "label": "Fed/USD/Yields",
        "region": "United States",
        "keywords": [
            "federal reserve", "fed", "powell", "fomc", "treasury", "yields",
            "dxy", "dollar", "cpi", "pce", "inflation", "payrolls", "nfp",
            "jobs report",
        ],
        "gold_bullish": [
            "dovish", "rate cut", "cuts rates", "soft inflation", "weak jobs",
            "unemployment rises", "dollar falls", "dollar weakens", "yields fall",
            "yield falls", "treasury yields drop", "recession",
        ],
        "gold_bearish": [
            "hawkish", "rate hike", "higher for longer", "hot inflation",
            "strong jobs", "dollar rises", "dollar strengthens", "yields rise",
            "yield rises", "sticky inflation",
        ],
        "risk": "Gold is most sensitive to real-rate/USD repricing from Fed, inflation, labor, and Treasury-yield headlines.",
    },
    {
        "id": "global_trade_sanctions",
        "label": "Global trade/sanctions",
        "region": "Global",
        "keywords": [
            "tariff", "tariffs", "sanctions", "export controls", "trade war",
            "supply chain", "north korea", "india", "south korea", "mexico",
            "canada", "brazil",
        ],
        "gold_bullish": [
            "new tariffs", "higher tariffs", "sanctions", "export controls",
            "trade war", "retaliation", "supply chain disruption", "escalation",
        ],
        "gold_bearish": [
            "trade deal", "tariff relief", "cuts tariffs", "lower tariffs",
            "de-escalation", "agreement", "truce",
        ],
        "risk": "Trade or sanctions escalation raises uncertainty; de-escalation usually lowers gold safe-haven demand.",
    },
]


def catalyst_rule_matches(text: str, rule: dict) -> bool:
    keywords = rule.get("keywords", [])
    if rule["id"] == "us_china_trade":
        return has_phrase(text, ["china", "xi", "beijing", "taiwan"]) and has_phrase(text, [
            "trump", "u.s.", "us ", "white house", "tariff", "trade", "ceo",
            "nvidia", "tesla", "apple", "boeing", "rare earth",
        ])
    return has_phrase(text, keywords)


def score_global_catalyst_headline(title: str) -> dict | None:
    text = title.lower()
    matched = []
    for rule in GLOBAL_CATALYST_RULES:
        if not catalyst_rule_matches(text, rule):
            continue
        bull_hits = [phrase for phrase in rule.get("gold_bullish", []) if phrase_matches(text, phrase)]
        bear_hits = [phrase for phrase in rule.get("gold_bearish", []) if phrase_matches(text, phrase)]
        if not bull_hits and not bear_hits:
            continue
        gold_score = int(clamp(len(bull_hits) - len(bear_hits), -2, 2))
        if gold_score == 0:
            continue
        usd_score = -gold_score
        matched.append({
            "id": rule["id"],
            "label": rule["label"],
            "region": rule["region"],
            "gold_score": gold_score,
            "usd_score": usd_score,
            "risk": rule["risk"],
            "drivers": bull_hits + bear_hits,
        })

    if not matched:
        return None

    primary = max(matched, key=lambda item: abs(item["gold_score"]))
    total_gold_score = int(clamp(sum(item["gold_score"] for item in matched), -3, 3))
    total_usd_score = int(clamp(sum(item["usd_score"] for item in matched), -3, 3))
    impact = "HIGH" if abs(total_gold_score) >= 2 or len(matched) >= 2 else "MEDIUM"
    gold_bias = "bullish" if total_gold_score > 0 else "bearish" if total_gold_score < 0 else "neutral"
    usd_bias = "bullish" if total_usd_score > 0 else "bearish" if total_usd_score < 0 else "neutral"
    if total_gold_score > 0:
        reason = "gold-positive safe-haven/USD-weakness catalyst"
    elif total_gold_score < 0:
        reason = "gold-negative risk-on/USD-strength catalyst"
    else:
        reason = "mixed catalyst"
    return {
        "category": "global_macro_catalyst",
        "catalyst": primary["label"],
        "region": primary["region"],
        "gold_score": total_gold_score,
        "usd_score": total_usd_score,
        "bias": gold_bias,
        "gold_bias": gold_bias,
        "usd_bias": usd_bias,
        "impact": impact,
        "reason": reason,
        "playbook": primary["risk"],
        "matched_rules": matched,
    }


def global_catalysts_offline(status: str) -> dict:
    return {
        "active": False,
        "score": 0,
        "usd_score": 0,
        "raw_score": 0,
        "raw_usd_score": 0,
        "bias": "MIXED",
        "usd_bias": "MIXED",
        "risk": "UNKNOWN",
        "headline_count": 0,
        "directional_count": 0,
        "headlines": [],
        "regions": {},
        "status": status,
        "source": f"GDELT global catalysts {GLOBAL_CATALYST_TIMESPAN}",
        "query": GLOBAL_CATALYST_QUERY,
        "reason": "global catalyst feed unavailable",
        "assessment": "No live catalyst evaluation available.",
    }


def get_global_catalyst_context() -> dict:
    if not GLOBAL_CATALYST_ENABLED:
        return {
            **global_catalysts_offline("disabled"),
            "risk": "LOW",
            "reason": "global catalyst detection disabled by env",
            "assessment": "Global catalyst detection is disabled.",
        }

    params = {
        "query": GLOBAL_CATALYST_QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 36,
        "timespan": GLOBAL_CATALYST_TIMESPAN,
        "sort": "hybridrel",
    }
    try:
        with httpx.Client(timeout=10) as client:
            res = client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
            res.raise_for_status()
            payload = res.json()

        headlines = []
        seen_titles = set()
        raw_score = 0
        raw_usd_score = 0
        region_scores = {}
        for item in payload.get("articles", [])[:30]:
            title = item.get("title", "").strip()
            title_key = title.lower()
            if not title or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            scored = score_global_catalyst_headline(title)
            if not scored:
                continue
            raw_score += scored["gold_score"]
            raw_usd_score += scored["usd_score"]
            region = scored["region"]
            region_scores[region] = region_scores.get(region, 0) + scored["gold_score"]
            headlines.append({
                **scored,
                "title": title,
                "url": item.get("url", ""),
                "source": item.get("sourceCountry", "") or item.get("domain", ""),
                "risk": True,
                "published": item.get("seendate", ""),
            })

        active = len(headlines) >= max(1, GLOBAL_CATALYST_MIN_HEADLINES) or abs(raw_score) >= 2
        score = int(clamp(raw_score, -4, 4)) if active else 0
        usd_score = int(clamp(raw_usd_score, -4, 4)) if active else 0
        bias = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "MIXED"
        usd_bias = "BULLISH" if usd_score >= 2 else "BEARISH" if usd_score <= -2 else "MIXED"
        max_abs_region = max([abs(value) for value in region_scores.values()] or [0])
        risk = "HIGH" if active and (abs(score) >= 3 or max_abs_region >= 3) else "MEDIUM" if active else "LOW"
        assessment = (
            f"Model read: {bias.lower()} gold catalyst, {usd_bias.lower()} USD catalyst, score {score:+d}."
            if active else
            "No active global macro/geopolitical catalyst trend detected."
        )
        return {
            "active": active,
            "score": score,
            "usd_score": usd_score,
            "raw_score": raw_score,
            "raw_usd_score": raw_usd_score,
            "bias": bias,
            "usd_bias": usd_bias,
            "risk": risk,
            "headline_count": len(headlines),
            "directional_count": len([headline for headline in headlines if headline.get("gold_score")]),
            "headlines": headlines[:10],
            "regions": {key: int(clamp(value, -4, 4)) for key, value in sorted(region_scores.items())},
            "status": "ok",
            "source": f"GDELT global catalysts {GLOBAL_CATALYST_TIMESPAN}",
            "query": GLOBAL_CATALYST_QUERY,
            "reason": assessment,
            "assessment": assessment,
        }
    except Exception as e:
        return global_catalysts_offline(f"offline: {compact_error(e)}")


def merge_global_catalysts_into_news(news: dict, catalysts: dict) -> dict:
    merged = {**news, "global_catalysts": catalysts, "trade_catalyst": catalysts}
    if not catalysts.get("active"):
        return merged

    base_score = int(safe_float(news.get("score")))
    catalyst_score = int(safe_float(catalysts.get("score")))
    adjusted_score = int(clamp(base_score + catalyst_score, -4, 4))
    risk_rank = {"LOW": 0, "WATCH": 1, "UNKNOWN": 1, "MEDIUM": 2, "HIGH": 3}
    current_risk = news.get("risk", "UNKNOWN")
    catalyst_risk = catalysts.get("risk", "UNKNOWN")
    merged["base_score"] = base_score
    merged["score"] = adjusted_score
    merged["bias"] = "BULLISH" if adjusted_score >= 2 else "BEARISH" if adjusted_score <= -2 else "MIXED"
    merged["risk"] = catalyst_risk if risk_rank.get(catalyst_risk, 1) > risk_rank.get(current_risk, 1) else current_risk
    merged["source"] = f"{news.get('source', 'news')} + {catalysts.get('source', 'global catalysts')}"
    merged["reason"] = catalysts.get("assessment")

    headlines = []
    seen_titles = set()
    for item in (catalysts.get("headlines") or []) + (news.get("headlines") or []):
        title = item.get("title", "")
        title_key = title.lower()
        if not title or title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        headlines.append(item)
    merged["headlines"] = headlines[:10]
    merged.update(headline_bias_distribution(merged["headlines"]))
    return merged


def headline_bias_distribution(headlines: list[dict]) -> dict:
    bulls = bears = neutral = 0
    for item in headlines:
        bias = str(item.get("gold_bias") or item.get("bias") or "").lower()
        score = safe_float(item.get("gold_score"))
        if "bull" in bias or score > 0:
            bulls += 1
        elif "bear" in bias or score < 0:
            bears += 1
        else:
            neutral += 1
    total = max(1, bulls + bears + neutral)
    return {
        "bullish_count": bulls,
        "bearish_count": bears,
        "neutral_count": neutral,
        "bullish_pct": round(bulls / total * 100, 1),
        "bearish_pct": round(bears / total * 100, 1),
        "neutral_pct": round(neutral / total * 100, 1),
    }


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
            **headline_bias_distribution(headlines[:8]),
            "status": "ok",
            "source": "GDELT 12h",
        }
    except Exception as e:
        fallback = get_fed_rss_context(compact_error(e))
        if fallback["status"] == "ok":
            return fallback
        return {
            "score": 0,
            "bias": "MIXED",
            "risk": "UNKNOWN",
            "risk_count": 0,
            "headlines": [],
            **headline_bias_distribution([]),
            "status": f"offline: {compact_error(e)}",
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
            **headline_bias_distribution(headlines[:5]),
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
            **headline_bias_distribution([]),
            "status": f"offline: {compact_error(e)}",
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
            "neutral_pct": round(neutral / total * 100, 1),
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
            "neutral_pct": 0,
            "samples": [],
            "source": "GDELT public-web sentiment 24h",
            "status": f"offline: {compact_error(e)}",
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
        "neutral_pct": round(neutral / total * 100, 1),
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
            "status": f"offline: {compact_error(e)}",
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

    with CONTEXT_CACHE_LOCK:
        now = datetime.now(timezone.utc)
        cached_ts = CONTEXT_CACHE["ts"]
        if not force and cached_ts and CONTEXT_CACHE["data"]:
            age = (now - cached_ts).total_seconds()
            if age < CONTEXT_TTL_SECONDS:
                return CONTEXT_CACHE["data"]

        macro = get_macro_context()
        news = get_news_context()
        global_catalysts = get_global_catalyst_context()
        news = merge_global_catalysts_into_news(news, global_catalysts)
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
        catalyst_note = (
            f" · global catalysts {global_catalysts.get('bias')}"
            if global_catalysts.get("active") else
            ""
        )
        context = {
            "score": score,
            "bias": bias,
            "risk": risk,
            "macro": macro,
            "news": news,
            "global_catalysts": global_catalysts,
            "trade_catalyst": global_catalysts,
            "people": people,
            "event_risk": event_risk,
            "timestamp": now.strftime("%Y-%m-%d %H:%M UTC"),
            "summary": f"{bias} bias · score {score:+d} · risk {risk}{catalyst_note} · refreshed from live feeds",
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


def direction_from_score(score: int, long_threshold: int = 10, short_threshold: int = -10) -> str:
    if score >= long_threshold:
        return "LONG"
    if score <= short_threshold:
        return "SHORT"
    return "FLAT"


def stacked_execution_setup(data: dict, context: dict | None = None) -> dict:
    h1 = data.get("h1", {})
    m15 = data.get("m15", {})
    m5 = data.get("m5", {})
    price = safe_float(data.get("price") or m5.get("price") or m15.get("price") or h1.get("price"))
    h1_score = int(safe_float(h1.get("technical_score")))
    m15_score = int(safe_float(m15.get("technical_score")))
    m5_score = int(safe_float(m5.get("technical_score")))
    h1_bias = direction_from_score(h1_score, STACK_H1_BIAS_MIN_SCORE, -STACK_H1_BIAS_MIN_SCORE)
    zone_context = m15.get("zone_context") or empty_zone_context()
    zone = zone_context.get("tested_zone")
    sweep = m5.get("liquidity_sweep", {})
    sweep_direction = sweep.get("direction", "NONE") if sweep.get("confirmed") else "NONE"
    direction = sweep_direction if sweep_direction in ("LONG", "SHORT") else "NONE"
    m5_ind = m5.get("indicators", {})
    ema9 = safe_float(m5_ind.get("ema9"))
    ema21 = safe_float(m5_ind.get("ema21"))
    ema9_slope = safe_float(m5_ind.get("ema9_slope"))
    rsi14 = safe_float(m5_ind.get("rsi14"))
    atr5 = max(safe_float(m5_ind.get("atr14"), safe_float(m15.get("indicators", {}).get("atr14"), 1.0)), 1e-9)

    if direction == "LONG":
        ema_ok = ema9 >= ema21 or ema9_slope > 0
        rsi_ok = rsi14 > 30
        m15_ok = m15_score >= 0
    elif direction == "SHORT":
        ema_ok = ema9 <= ema21 or ema9_slope < 0
        rsi_ok = rsi14 < 70
        m15_ok = m15_score <= 0
    else:
        ema_ok = False
        rsi_ok = False
        m15_ok = False

    zone_ok = bool(zone and zone.get("direction") == direction and zone.get("score", 0) >= ORDER_BLOCK_TESTED_MIN_SCORE)
    h1_ok = h1_bias == direction
    sweep_ok = direction in ("LONG", "SHORT") and bool(sweep.get("confirmed"))
    evaluation_time = snapshot_time(data)
    checks = {
        "h1_bias_agrees": h1_ok,
        "m15_zone_active": zone_ok,
        "m15_score_supports": m15_ok,
        "m5_sweep_confirmed": sweep_ok,
        "m5_ema_trigger": ema_ok,
        "m5_rsi_not_opposite_extreme": rsi_ok,
    }
    executable = all(checks.values())
    zone_score = safe_float((zone or {}).get("score"))
    confidence = int(clamp(
        0.30 * min(100, abs(h1_score))
        + 0.25 * zone_score
        + 0.15 * min(100, safe_float(m5_ind.get("adx14")) * 3)
        + 0.15 * min(100, abs(m5_score))
        + 0.10 * min(100, abs(m15_score))
        + 0.05 * (100 if utc_session_label(evaluation_time) != "OFF_SESSION" else 40),
        5,
        95,
    ))
    trade = build_zone_trade_plan(price, atr5, direction, zone) if executable else build_trade_plan(price, atr5, "NONE", {})

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "setup_type": "15M_ORDER_BLOCK_5M_SWEEP",
        "executable": executable,
        "direction": direction,
        "confidence": confidence,
        "h1_bias": h1_bias,
        "h1_score": h1_score,
        "m15_score": m15_score,
        "m5_score": m5_score,
        "zone": zone,
        "zone_context": zone_context,
        "sweep": sweep,
        "trade": trade,
        "checks": checks,
        "failed_checks": failed,
        "reason": (
            f"H1 {h1_bias}; 15M zone score {zone_score:.0f}; 5M {direction} sweep confirmed."
            if executable else
            f"Stack waiting: {', '.join(failed) if failed else 'no executable direction'}."
        ),
    }


def active_technical_signal(data: dict) -> str:
    return technical_composite(data)["label"]


def snapshot_time(snapshot: dict | None = None) -> datetime:
    value = (snapshot or {}).get("timestamp")
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d %H:%M UTC", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                parsed = datetime.strptime(value, fmt)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def utc_session_label(now: datetime) -> str:
    hour = now.hour + now.minute / 60
    if 12 <= hour < 16:
        return "LONDON_NY_OVERLAP"
    if 7 <= hour < 12:
        return "LONDON"
    if 16 <= hour < 20:
        return "NY"
    return "OFF_SESSION"


def is_rollover_window(now: datetime) -> bool:
    minutes = now.hour * 60 + now.minute
    return 21 * 60 + 55 <= minutes < 22 * 60 + 10


def is_friday_close_window(now: datetime) -> bool:
    return now.weekday() == 4 and (now.hour + now.minute / 60) >= 18


def is_sunday_open_window(now: datetime) -> bool:
    return now.weekday() == 6 and (now.hour + now.minute / 60) >= 22


def high_impact_news_blackout(context: dict) -> tuple[bool, dict | None]:
    events = context.get("event_risk", {}).get("events", [])
    for event in events:
        if event.get("impact") != "HIGH":
            continue
        minutes_to = event.get("minutes_to")
        if isinstance(minutes_to, (int, float)) and -30 <= minutes_to <= 30:
            return True, event
    event_risk = context.get("event_risk", {})
    next_event = event_risk.get("next_event") or {}
    minutes_to = next_event.get("minutes_to")
    if event_risk.get("level") == "HIGH" and isinstance(minutes_to, (int, float)) and -30 <= minutes_to <= 30:
        return True, next_event
    return False, None


def configured_spread_check() -> tuple[bool, float | None, str]:
    if XAUUSD_SPREAD_CENTS == "":
        pricing = get_cached_spread()
        if not pricing:
            return True, None, "Spread feed not configured; check is marked pass until broker pricing is wired."
        spread = safe_float(pricing.get("spread_cents"))
    else:
        try:
            spread = float(XAUUSD_SPREAD_CENTS)
        except Exception:
            return False, None, "Configured spread value is invalid."
    if spread > MAX_SPREAD_CENTS:
        return False, spread, f"Spread {spread:.1f}c is above max {MAX_SPREAD_CENTS:.1f}c."
    return True, spread, f"Spread {spread:.1f}c is inside max {MAX_SPREAD_CENTS:.1f}c."


def timeframe_snapshot_key(timeframe: str | None) -> str:
    if timeframe == "1H":
        return "h1"
    if timeframe == "5M":
        return "m5"
    return "m15"


def volatility_regime_for_decision(snapshot: dict, decision: dict | None = None) -> dict:
    timeframe = (decision or {}).get("execution_timeframe") or (snapshot.get("final") or {}).get("execution_timeframe")
    signal_data = snapshot.get(timeframe_snapshot_key(timeframe), {})
    indicators = signal_data.get("indicators", {})
    adx_value = safe_float(indicators.get("adx14"))
    atr_pct = safe_float(indicators.get("atr_pct"))
    if atr_pct <= 0:
        label = "UNKNOWN"
        passed = True
    elif atr_pct < 0.04:
        label = "LOW_VOL"
        passed = True
    elif atr_pct > 0.95 and adx_value < 18:
        label = "HIGH_VOL_CHOPPY"
        passed = False
    elif adx_value >= 18:
        label = "TRENDING"
        passed = True
    else:
        label = "NORMAL"
        passed = True
    return {
        "label": label,
        "passed": passed,
        "adx14": round(adx_value, 1),
        "atr_pct": round(atr_pct, 3),
    }


def recent_same_direction_signal(direction: str | None, now: datetime) -> dict | None:
    if direction not in ("LONG", "SHORT"):
        return None
    cutoff = now - timedelta(minutes=RECENT_SIGNAL_COOLDOWN_MINUTES)
    for trade in reversed(load_trade_history().get("trades", [])):
        if trade.get("direction") != direction:
            continue
        opened_at = trade.get("opened_at")
        if not opened_at:
            continue
        try:
            opened = parse_utc(str(opened_at).replace(" UTC", "+00:00"))
        except Exception:
            try:
                opened = datetime.strptime(str(opened_at), "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
            except Exception:
                continue
        if opened >= cutoff:
            return {
                "id": trade.get("id"),
                "direction": direction,
                "opened_at": opened.strftime("%Y-%m-%d %H:%M UTC"),
            }
    return None


def hard_filter_status(snapshot: dict, decision: dict | None = None) -> dict:
    now = snapshot_time(snapshot)
    context = snapshot.get("context", {})
    direction = (decision or snapshot.get("final") or {}).get("direction")
    session = utc_session_label(now)
    spread_passed, spread, spread_reason = configured_spread_check()
    news_blocked, news_event = high_impact_news_blackout(context)
    regime = volatility_regime_for_decision(snapshot, decision)
    recent_signal = recent_same_direction_signal(direction, now)

    checks = [
        {
            "name": "hard_filters_enabled",
            "passed": HARD_FILTERS_ENABLED,
            "blocking": False,
            "detail": "Hard filters are active." if HARD_FILTERS_ENABLED else "Hard filters are disabled by env.",
        },
        {
            "name": "session",
            "passed": session != "OFF_SESSION",
            "blocking": True,
            "detail": f"UTC session is {session}; allowed sessions are London/NY only.",
        },
        {
            "name": "news_blackout",
            "passed": not news_blocked,
            "blocking": True,
            "detail": (
                f"High-impact event blackout active: {news_event.get('name')} {news_event.get('eta')}."
                if news_event else
                "No high-impact event inside +/-30 minutes."
            ),
        },
        {
            "name": "friday_close",
            "passed": not is_friday_close_window(now),
            "blocking": True,
            "detail": "No new signals after 18:00 UTC Friday.",
        },
        {
            "name": "sunday_open",
            "passed": not is_sunday_open_window(now),
            "blocking": True,
            "detail": "No signals during the first two hours after Sunday open.",
        },
        {
            "name": "daily_rollover",
            "passed": not is_rollover_window(now),
            "blocking": True,
            "detail": "No signals during 21:55-22:10 UTC rollover.",
        },
        {
            "name": "spread",
            "passed": spread_passed,
            "blocking": True,
            "detail": spread_reason,
            "value_cents": spread,
        },
        {
            "name": "regime",
            "passed": regime["passed"],
            "blocking": True,
            "detail": f"Regime {regime['label']} from ADX {regime['adx14']} and ATR% {regime['atr_pct']}.",
        },
        {
            "name": "recent_signal",
            "passed": recent_signal is None,
            "blocking": True,
            "detail": (
                f"Recent same-direction signal {recent_signal['id']} opened at {recent_signal['opened_at']}."
                if recent_signal else
                f"No same-direction signal inside {RECENT_SIGNAL_COOLDOWN_MINUTES} minutes."
            ),
        },
    ]
    blocking_failures = [check for check in checks if check["blocking"] and not check["passed"]]
    allowed = (not HARD_FILTERS_ENABLED) or not blocking_failures
    return {
        "enabled": HARD_FILTERS_ENABLED,
        "allowed": allowed,
        "session": session,
        "timestamp": now.strftime("%Y-%m-%d %H:%M UTC"),
        "regime": regime,
        "blocked_reasons": [check["detail"] for check in blocking_failures],
        "checks": checks,
    }


def apply_hard_filters(data: dict, context: dict, decision: dict) -> dict:
    snapshot = {**data, "context": context, "final": decision}
    status = hard_filter_status(snapshot, decision)
    if decision.get("direction") not in ("LONG", "SHORT"):
        return {**decision, "hard_filters": status}
    if status["allowed"]:
        return {**decision, "hard_filters": status}
    return {
        **decision,
        "proposed_direction": decision.get("direction"),
        "direction": "WAIT",
        "quality": "BLOCKED",
        "hard_filters": status,
        "reason": f"Hard filter blocked execution: {'; '.join(status['blocked_reasons'])}",
    }


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
    stacked_setup = stacked_execution_setup(data, context)
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
        "stacked_setup": stacked_setup,
        "setup_type": stacked_setup.get("setup_type"),
    }

    def finalize(decision: dict) -> dict:
        return apply_hard_filters(data, context, decision)

    if stacked_setup.get("executable"):
        stack_direction = stacked_setup.get("direction")
        stack_confidence = max(confidence, int(stacked_setup.get("confidence", confidence)))
        return finalize({
            **base,
            "direction": stack_direction,
            "quality": "CONFIRMED" if risk != "MEDIUM" else "CAUTION",
            "confidence": stack_confidence,
            "trade": stacked_setup.get("trade"),
            "sweep": stacked_setup.get("sweep"),
            "sweep_direction": stack_direction,
            "execution_timeframe": "5M",
            "execution_score": m5_score,
            "zone": stacked_setup.get("zone"),
            "reason": (
                f"Full stack confirmed: H1 bias {stack_direction}, 15M order-block zone is being tested, "
                f"and 5M swept liquidity/reclaimed. Zone: {(stacked_setup.get('zone') or {}).get('low')} - "
                f"{(stacked_setup.get('zone') or {}).get('high')}. Weighted score {weighted_score:+d}."
            ),
        })

    if risk == "HIGH":
        return finalize({
            **base,
            "direction": "WAIT",
            "quality": "BLOCKED",
            "reason": "High-impact news/event window is active. Stand down until spreads and first impulse settle.",
        })

    if sweep_direction == "NONE":
        if weighted_score >= 30:
            return finalize({
                **base,
                "direction": "LONG_BIAS",
                "quality": "NO_TRIGGER",
                "reason": "Macro/news and EMA regime lean long, but no 1H/15M/5M execution trade has swept liquidity and reclaimed yet.",
            })
        if weighted_score <= -30:
            return finalize({
                **base,
                "direction": "SHORT_BIAS",
                "quality": "NO_TRIGGER",
                "reason": "Macro/news and EMA regime lean short, but no 1H/15M/5M execution trade has swept liquidity and rejected yet.",
            })
        return finalize({
            **base,
            "direction": "WAIT",
            "quality": "NO_TRIGGER",
            "reason": "No fresh 1H/15M/5M liquidity sweep. Wait for stop-run/reclaim before taking intraday risk.",
        })

    long_allowed = (
        sweep_direction == "LONG"
        and h1_score >= 10
        and execution_score >= 45
        and weighted_score >= 35
        and execution_aligned
        and stacked_setup.get("executable")
    )
    short_allowed = (
        sweep_direction == "SHORT"
        and h1_score <= -10
        and execution_score <= -45
        and weighted_score <= -35
        and execution_aligned
        and stacked_setup.get("executable")
    )
    countertrend_long = (
        sweep_direction == "LONG"
        and h1_score > -25
        and weighted_score >= 58
        and execution_score >= 45
        and execution_aligned
        and stacked_setup.get("executable")
    )
    countertrend_short = (
        sweep_direction == "SHORT"
        and h1_score < 25
        and weighted_score <= -58
        and execution_score <= -45
        and execution_aligned
        and stacked_setup.get("executable")
    )

    if long_allowed or countertrend_long:
        quality = "CAUTION" if risk == "MEDIUM" else "CONFIRMED"
        if countertrend_long and not long_allowed:
            quality = "COUNTERTREND"
        return finalize({
            **base,
            "direction": "LONG",
            "quality": quality,
            "reason": f"{execution_timeframe} swept sell-side liquidity and reclaimed; 1H EMA regime/score supports a long. Weighted score {weighted_score:+d}.",
        })
    if short_allowed or countertrend_short:
        quality = "CAUTION" if risk == "MEDIUM" else "CONFIRMED"
        if countertrend_short and not short_allowed:
            quality = "COUNTERTREND"
        return finalize({
            **base,
            "direction": "SHORT",
            "quality": quality,
            "reason": f"{execution_timeframe} swept buy-side liquidity and rejected; 1H EMA regime/score supports a short. Weighted score {weighted_score:+d}.",
        })

    alignment_note = " 5M execution also requires directional support from 15M." if execution_timeframe == "5M" and not execution_aligned else ""
    if sweep_direction == "LONG":
        return finalize({
            **base,
            "direction": "LONG_BIAS",
            "quality": "MISMATCH",
            "reason": f"{execution_timeframe} liquidity sweep is long, but 1H/15M regime or macro/news score is not strong enough for execution.{alignment_note}",
        })
    if sweep_direction == "SHORT":
        return finalize({
            **base,
            "direction": "SHORT_BIAS",
            "quality": "MISMATCH",
            "reason": f"{execution_timeframe} liquidity sweep is short, but 1H/15M regime or macro/news score is not strong enough for execution.{alignment_note}",
        })
    return finalize({
        **base,
        "direction": "WAIT",
        "quality": "NEUTRAL",
        "reason": f"Weighted score {weighted_score:+d}: no aligned intraday edge yet.",
    })


def timeframe_trade_decisions(snapshot: dict) -> list[dict]:
    context = snapshot.get("context", {})
    final = snapshot.get("final", {})
    risk = context.get("risk", "UNKNOWN")
    event_level = context.get("event_risk", {}).get("level", "LOW")
    if risk == "HIGH":
        return []

    decisions = []
    stacked = final.get("stacked_setup") or stacked_execution_setup(snapshot, context)
    scores = {
        "h1_score": snapshot.get("h1", {}).get("technical_score"),
        "m15_score": snapshot.get("m15", {}).get("technical_score"),
        "m5_score": snapshot.get("m5", {}).get("technical_score"),
    }
    for key, tf in (("m5", "5M"),):
        signal_data = snapshot.get(key, {})
        if signal_data.get("error"):
            continue

        direction = signal_data.get("signal")
        trade = signal_data.get("trade", {})
        if direction not in ("LONG", "SHORT"):
            continue
        if stacked.get("executable") and stacked.get("direction") == direction:
            trade = stacked.get("trade") or trade
        if trade.get("stop") is None or trade.get("target") is None:
            continue

        sweep = signal_data.get("liquidity_sweep", {})
        score = int(safe_float(signal_data.get("technical_score")))
        confidence = int(stacked.get("confidence") or abs(score) + (10 if sweep.get("confirmed") else 0))
        if risk == "MEDIUM":
            confidence -= 10
        confidence = int(clamp(confidence, 5, 95))
        quality = "CAUTION" if risk == "MEDIUM" else "CONFIRMED"
        if not stacked.get("executable"):
            quality = "STACK_WAIT"
        sweep_direction = sweep.get("direction", "NONE") if sweep.get("confirmed") else "NONE"
        reason = (
            f"Full-stack {tf} {direction}: H1 bias + 15M order block + 5M sweep; RR {trade.get('rr') or '--'}."
            if stacked.get("executable") else
            f"5M technical {direction} exists, but full stack is not complete: {', '.join(stacked.get('failed_checks') or [])}."
        )

        decision = {
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
            "zone": stacked.get("zone"),
            "stacked_setup": stacked,
            "setup_type": stacked.get("setup_type"),
            "trade": trade,
            **scores,
            "execution_timeframe": tf,
            "execution_score": score,
            "execution_aligned": True,
            "direction": direction,
            "quality": quality,
            "reason": reason,
        }
        filter_status = hard_filter_status(snapshot, decision)
        if not filter_status["allowed"]:
            decision = {
                **decision,
                "quality": "BLOCKED",
                "hard_filters": filter_status,
                "reason": f"Hard filter blocked {tf} {direction}: {'; '.join(filter_status['blocked_reasons'])}",
            }
        else:
            decision["hard_filters"] = filter_status
        decisions.append(decision)

    return decisions


def prop_firm_rules() -> dict:
    account_size = max(PROP_ACCOUNT_SIZE, 1.0)
    return {
        "enabled": PROP_MODE_ENABLED,
        "firm_name": PROP_FIRM_NAME,
        "account_size": round(account_size, 2),
        "profit_target_pct": PROP_PROFIT_TARGET_PCT,
        "profit_target_usd": round(account_size * PROP_PROFIT_TARGET_PCT / 100, 2),
        "max_daily_loss_pct": PROP_MAX_DAILY_LOSS_PCT,
        "max_daily_loss_usd": round(account_size * PROP_MAX_DAILY_LOSS_PCT / 100, 2),
        "max_total_drawdown_pct": PROP_MAX_TOTAL_DRAWDOWN_PCT,
        "max_total_drawdown_usd": round(account_size * PROP_MAX_TOTAL_DRAWDOWN_PCT / 100, 2),
        "base_risk_pct": PROP_RISK_PER_TRADE_PCT,
        "protect_risk_pct": PROP_PROTECT_RISK_PER_TRADE_PCT,
        "max_daily_risk_pct": PROP_MAX_DAILY_RISK_PCT,
        "max_daily_risk_usd": round(account_size * PROP_MAX_DAILY_RISK_PCT / 100, 2),
        "min_rr": PROP_MIN_RR,
        "min_confidence": PROP_MIN_CONFIDENCE,
        "max_trades_per_day": PROP_MAX_TRADES_PER_DAY,
        "max_losses_per_day": PROP_MAX_LOSSES_PER_DAY,
        "allow_countertrend": PROP_ALLOW_COUNTERTREND,
    }


def prop_counted_trades() -> list[dict]:
    return [
        trade for trade in load_trade_history().get("trades", [])
        if str(trade.get("user_action") or "").upper() == "TAKEN"
    ]


def prop_account_state(rules: dict) -> dict:
    trades = prop_counted_trades()
    closed = [
        trade for trade in trades
        if trade.get("status") == "CLOSED" and trade.get("r_multiple") is not None
    ]
    today = datetime.now(timezone.utc).date()
    today_trades = [trade for trade in trades if trade_time_value(trade).date() == today]
    today_closed = [
        trade for trade in today_trades
        if trade.get("status") == "CLOSED" and trade.get("r_multiple") is not None
    ]
    today_losses = [trade for trade in today_closed if safe_float(trade.get("r_multiple")) < 0]
    open_trades = [trade for trade in trades if trade.get("status") == "OPEN"]

    risk_unit_usd = rules["account_size"] * rules["base_risk_pct"] / 100
    net_r = sum(safe_float(trade.get("r_multiple")) for trade in closed)
    today_r = sum(safe_float(trade.get("r_multiple")) for trade in today_closed)
    realized_pnl_usd = net_r * risk_unit_usd
    today_pnl_usd = today_r * risk_unit_usd
    target_remaining_usd = max(0.0, rules["profit_target_usd"] - realized_pnl_usd)
    target_progress_pct = (
        clamp(realized_pnl_usd / rules["profit_target_usd"] * 100, -100, 150)
        if rules["profit_target_usd"] else 0
    )
    account_pnl_pct = realized_pnl_usd / rules["account_size"] * 100
    daily_loss_used_usd = max(0.0, -today_pnl_usd)
    daily_risk_left_usd = max(0.0, rules["max_daily_risk_usd"] - daily_loss_used_usd)

    if target_remaining_usd <= 0:
        phase = "TARGET_REACHED"
        phase_risk_pct = 0.0
    elif target_progress_pct >= 75:
        phase = "PROTECT"
        phase_risk_pct = min(rules["base_risk_pct"], rules["protect_risk_pct"])
    elif account_pnl_pct <= -1:
        phase = "DEFENSIVE"
        phase_risk_pct = min(rules["base_risk_pct"], rules["protect_risk_pct"])
    elif account_pnl_pct >= 2:
        phase = "PUSH"
        phase_risk_pct = rules["base_risk_pct"]
    else:
        phase = "BUILD_CUSHION"
        phase_risk_pct = rules["base_risk_pct"]

    return {
        "source": "local trade history marked TAKEN",
        "phase": phase,
        "phase_risk_pct": round(phase_risk_pct, 3),
        "risk_unit_usd": round(risk_unit_usd, 2),
        "counted_trades": len(trades),
        "closed_trades": len(closed),
        "open_trades": len(open_trades),
        "trades_today": len(today_trades),
        "closed_today": len(today_closed),
        "losses_today": len(today_losses),
        "net_r": round(net_r, 2),
        "today_r": round(today_r, 2),
        "realized_pnl_usd": round(realized_pnl_usd, 2),
        "today_pnl_usd": round(today_pnl_usd, 2),
        "daily_loss_used_usd": round(daily_loss_used_usd, 2),
        "daily_risk_left_usd": round(daily_risk_left_usd, 2),
        "target_remaining_usd": round(target_remaining_usd, 2),
        "target_progress_pct": round(target_progress_pct, 1),
        "account_pnl_pct": round(account_pnl_pct, 2),
        "note": "Mark executed Journal trades as TAKEN so challenge progress and daily limits stay accurate.",
    }


def prop_effective_risk_pct(state: dict, rules: dict, context: dict) -> float:
    risk_pct = safe_float(state.get("phase_risk_pct"))
    if context.get("risk") == "MEDIUM":
        risk_pct = min(risk_pct, rules["protect_risk_pct"])
    daily_left_pct = state.get("daily_risk_left_usd", 0) / max(rules["account_size"], 1e-9) * 100
    return round(max(0.0, min(risk_pct, daily_left_pct)), 3)


def prop_trade_sizing(trade: dict, risk_pct: float, rules: dict) -> dict:
    entry = safe_float(trade.get("entry"))
    stop = safe_float(trade.get("stop"))
    target = safe_float(trade.get("target"))
    target_2 = trade.get("target_2")
    stop_distance = abs(entry - stop)
    risk_usd = rules["account_size"] * risk_pct / 100
    size_oz = risk_usd / max(stop_distance, 1e-9) if stop_distance else 0.0
    reward_usd = abs(target - entry) * size_oz if target else 0.0
    target_2_reward_usd = abs(safe_float(target_2) - entry) * size_oz if target_2 is not None else None
    return {
        "risk_pct": round(risk_pct, 3),
        "risk_usd": round(risk_usd, 2),
        "size_oz": round(size_oz, 4),
        "stop_distance": round(stop_distance, 2),
        "target_profit_usd": round(reward_usd, 2),
        "target_2_profit_usd": round(target_2_reward_usd, 2) if target_2_reward_usd is not None else None,
        "position_note": "XAUUSD sizing assumes 1 oz P/L changes by $1 per $1 move in gold.",
    }


def prop_trade_candidate(snapshot: dict, rules: dict, state: dict) -> dict:
    final = snapshot.get("final", {})
    context = snapshot.get("context", {})
    trade = final.get("trade", {}) or {}
    hard_filters = final.get("hard_filters") or snapshot.get("hard_filters") or {}
    risk_pct = prop_effective_risk_pct(state, rules, context)
    sizing = prop_trade_sizing(trade, risk_pct, rules)
    blockers = []

    if not rules["enabled"]:
        blockers.append("Prop mode is disabled.")
    if not is_executable_trade(final):
        blockers.append("No executable full-stack trade is active.")
    if state.get("target_remaining_usd", 0) <= 0:
        blockers.append("Profit target is reached; stop trading and request review/payout.")
    if hard_filters and not hard_filters.get("allowed", True):
        blockers.extend(hard_filters.get("blocked_reasons") or ["Hard filter is blocking execution."])
    if context.get("risk") == "HIGH":
        blockers.append("High-impact event risk is active.")
    if final.get("quality") == "COUNTERTREND" and not rules["allow_countertrend"]:
        blockers.append("Countertrend setups are disabled for prop challenge mode.")
    if int(safe_float(final.get("confidence"))) < rules["min_confidence"]:
        blockers.append(f"Confidence below prop threshold {rules['min_confidence']}%.")
    if safe_float(trade.get("rr")) < rules["min_rr"]:
        blockers.append(f"Risk/reward below required {rules['min_rr']:.2f}R.")
    if state.get("trades_today", 0) >= rules["max_trades_per_day"]:
        blockers.append(f"Max trades per day reached: {rules['max_trades_per_day']}.")
    if state.get("losses_today", 0) >= rules["max_losses_per_day"]:
        blockers.append(f"Max losing trades per day reached: {rules['max_losses_per_day']}.")
    if state.get("daily_loss_used_usd", 0) >= rules["max_daily_risk_usd"]:
        blockers.append("Self-imposed daily risk stop is reached.")
    if sizing["risk_usd"] <= 0:
        blockers.append("No daily risk budget left.")

    target_remaining = safe_float(state.get("target_remaining_usd"))
    r_to_target = target_remaining / max(sizing["risk_usd"], 1e-9) if sizing["risk_usd"] > 0 else None
    ready = not blockers
    return {
        "status": "TRADE_READY" if ready else "WAIT",
        "ready": ready,
        "action": "BUY" if final.get("direction") == "LONG" and ready else "SELL" if final.get("direction") == "SHORT" and ready else "NO_TRADE",
        "direction": final.get("direction"),
        "quality": final.get("quality"),
        "confidence": final.get("confidence"),
        "execution_timeframe": final.get("execution_timeframe"),
        "entry": trade.get("entry"),
        "entry_low": trade.get("entry_low"),
        "entry_high": trade.get("entry_high"),
        "stop": trade.get("stop"),
        "target": trade.get("target"),
        "target_2": trade.get("target_2"),
        "rr": trade.get("rr"),
        "sizing": sizing,
        "blockers": blockers,
        "pass_math": {
            "target_remaining_usd": round(target_remaining, 2),
            "r_to_target": round(r_to_target, 2) if r_to_target is not None else None,
            "two_r_wins_to_target": int(np.ceil(target_remaining / max(sizing["target_profit_usd"], 1e-9))) if sizing["target_profit_usd"] > 0 else None,
            "daily_risk_left_usd": state.get("daily_risk_left_usd"),
        },
        "reason": final.get("reason"),
        "disclaimer": "Research signal only. Confirm your prop firm rules before execution.",
    }


def prop_firm_snapshot(snapshot: dict) -> dict:
    rules = prop_firm_rules()
    state = prop_account_state(rules)
    candidate = prop_trade_candidate(snapshot, rules, state)
    return {
        "rules": rules,
        "state": state,
        "candidate": candidate,
        "guards": [
            f"Risk {rules['base_risk_pct']}% max per A+ trade; protect phase risk {rules['protect_risk_pct']}%.",
            f"Stop after {rules['max_losses_per_day']} losses or {rules['max_trades_per_day']} trades in a day.",
            f"Only execute trades with at least {rules['min_rr']:.2f}R and confidence >= {rules['min_confidence']}%.",
            "Do not trade high-impact news windows or blocked hard-filter states.",
        ],
    }


def watch_signal_candidates(snapshot: dict) -> list[dict]:
    if not WATCH_ALERTS_ENABLED:
        return []

    context = snapshot.get("context", {})
    final = snapshot.get("final", {})
    stacked = final.get("stacked_setup") or stacked_execution_setup(snapshot, context)
    candidates = []

    for key, tf in (("m15", "15M"), ("m5", "5M")):
        signal_data = snapshot.get(key, {})
        if signal_data.get("error"):
            continue

        signal = signal_data.get("signal")
        setup_direction = signal_data.get("setup_direction")
        conditions_met = int(safe_float(signal_data.get("conditions_met")))
        direction = None
        quality = None
        reason = None

        if signal in ("LONG", "SHORT"):
            direction = signal
            quality = "WATCH_TRIGGER"
            reason = f"{tf} technical trigger exists, but full execution stack is not complete."
        elif conditions_met >= WATCH_MIN_CONDITIONS and setup_direction in ("LONG", "SHORT"):
            direction = setup_direction
            quality = "WATCH_FORMING"
            reason = f"{tf} setup is forming with {conditions_met}/{signal_data.get('max_conditions', 7)} conditions met."

        if direction not in ("LONG", "SHORT"):
            continue

        sweep = signal_data.get("liquidity_sweep", {})
        zone_context = signal_data.get("zone_context") or {}
        zone = zone_context.get("tested_zone")
        decision_stub = {
            "direction": direction,
            "quality": quality,
            "execution_timeframe": tf,
            "sweep": sweep,
            "trade": signal_data.get("trade", {}),
        }
        filter_status = hard_filter_status(snapshot, decision_stub)

        if stacked.get("executable") and stacked.get("direction") == direction:
            if filter_status.get("allowed"):
                continue
            if tf != "5M":
                continue
            quality = "WATCH_BLOCKED"
            reason = "Full stack is present, but a hard filter is blocking trade execution."

        failed_checks = stacked.get("failed_checks") or []
        candidates.append({
            "timeframe": tf,
            "direction": direction,
            "quality": quality,
            "bar_time": signal_data.get("bar_time"),
            "price": signal_data.get("price"),
            "technical_score": signal_data.get("technical_score"),
            "conditions_met": conditions_met,
            "max_conditions": signal_data.get("max_conditions", 7),
            "conditions": signal_data.get("conditions", {}),
            "trade": signal_data.get("trade", {}),
            "stop": signal_data.get("stop"),
            "target": signal_data.get("target"),
            "liquidity_sweep": sweep,
            "zone": zone,
            "zone_context": zone_context,
            "failed_stack_checks": failed_checks,
            "hard_filters": filter_status,
            "reason": reason,
        })

    return candidates


def watch_alert_key(candidate: dict) -> str:
    return ":".join([
        "WATCH",
        str(candidate.get("quality") or "UNKNOWN"),
        str(candidate.get("timeframe") or "TF"),
        str(candidate.get("direction") or "NONE"),
        str(candidate.get("bar_time") or "NO_BAR"),
    ])


def watch_alert_bucket(candidate: dict) -> tuple:
    return (
        candidate.get("quality"),
        candidate.get("timeframe"),
        candidate.get("direction"),
    )


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


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    if work.index.tz is None:
        work.index = work.index.tz_localize("UTC")
    else:
        work.index = work.index.tz_convert("UTC")
    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in work.columns:
        aggregation["volume"] = "sum"
    return work.resample(rule, label="right", closed="right").agg(aggregation).dropna()


def ema_alignment_snapshot(df: pd.DataFrame, label: str) -> dict:
    if df is None or len(df) < 205:
        return {
            "timeframe": label,
            "available": False,
            "direction": "NONE",
            "price": None,
            "ema9": None,
            "ema21": None,
            "ema50": None,
            "ema200": None,
            "ema21_slope": None,
        }
    close = df["close"]
    e9 = ema(close, 9)
    e21 = ema(close, 21)
    e50 = ema(close, 50)
    e200 = ema(close, 200)
    idx = len(df) - 1
    price = safe_float(close.iloc[idx])
    ema9v = safe_float(e9.iloc[idx])
    ema21v = safe_float(e21.iloc[idx])
    ema50v = safe_float(e50.iloc[idx])
    ema200v = safe_float(e200.iloc[idx])
    slope = pct_change_now(e21, idx, 5)
    bullish = price > ema9v > ema21v > ema50v and ema50v > ema200v and slope >= 0
    bearish = price < ema9v < ema21v < ema50v and ema50v < ema200v and slope <= 0
    return {
        "timeframe": label,
        "available": True,
        "direction": "LONG" if bullish else "SHORT" if bearish else "NONE",
        "price": round(price, 2),
        "ema9": round(ema9v, 2),
        "ema21": round(ema21v, 2),
        "ema50": round(ema50v, 2),
        "ema200": round(ema200v, 2),
        "ema21_slope": round(slope, 3),
    }


def pivot_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> dict:
    highs = []
    lows = []
    if len(df) < left + right + 1:
        return {"highs": highs, "lows": lows}
    for pos in range(left, len(df) - right):
        window = df.iloc[pos - left:pos + right + 1]
        row = df.iloc[pos]
        high = safe_float(row["high"])
        low = safe_float(row["low"])
        if high >= safe_float(window["high"].max()):
            highs.append({"pos": pos, "price": high, "time": str(df.index[pos])})
        if low <= safe_float(window["low"].min()):
            lows.append({"pos": pos, "price": low, "time": str(df.index[pos])})
    return {"highs": highs, "lows": lows}


def select_fib_impulse_leg(df: pd.DataFrame, direction: str, atr_value: float) -> dict | None:
    lookback = max(80, min(FIB_EMA_LOOKBACK_BARS, len(df)))
    recent = df.iloc[-lookback:].copy()
    pivots = pivot_points(recent)
    min_range = max(atr_value * FIB_EMA_MIN_IMPULSE_ATR, safe_float(recent["close"].iloc[-1]) * 0.001)

    if direction == "LONG":
        for high_pivot in reversed(pivots["highs"]):
            lows_before = [item for item in pivots["lows"] if item["pos"] < high_pivot["pos"]]
            if not lows_before:
                continue
            low_pivot = min(lows_before, key=lambda item: item["price"])
            impulse_range = high_pivot["price"] - low_pivot["price"]
            if impulse_range >= min_range:
                return {
                    "direction": direction,
                    "swing_low": round(low_pivot["price"], 2),
                    "swing_high": round(high_pivot["price"], 2),
                    "swing_low_time": low_pivot["time"],
                    "swing_high_time": high_pivot["time"],
                    "range": round(impulse_range, 2),
                    "impulse_atr": round(impulse_range / max(atr_value, 1e-9), 2),
                }
        high_pos = int(np.argmax(recent["high"].to_numpy()))
        if high_pos >= 8:
            low_pos = int(np.argmin(recent["low"].iloc[:high_pos].to_numpy()))
            low = safe_float(recent["low"].iloc[low_pos])
            high = safe_float(recent["high"].iloc[high_pos])
            if high - low >= min_range:
                return {
                    "direction": direction,
                    "swing_low": round(low, 2),
                    "swing_high": round(high, 2),
                    "swing_low_time": str(recent.index[low_pos]),
                    "swing_high_time": str(recent.index[high_pos]),
                    "range": round(high - low, 2),
                    "impulse_atr": round((high - low) / max(atr_value, 1e-9), 2),
                }

    if direction == "SHORT":
        for low_pivot in reversed(pivots["lows"]):
            highs_before = [item for item in pivots["highs"] if item["pos"] < low_pivot["pos"]]
            if not highs_before:
                continue
            high_pivot = max(highs_before, key=lambda item: item["price"])
            impulse_range = high_pivot["price"] - low_pivot["price"]
            if impulse_range >= min_range:
                return {
                    "direction": direction,
                    "swing_low": round(low_pivot["price"], 2),
                    "swing_high": round(high_pivot["price"], 2),
                    "swing_low_time": low_pivot["time"],
                    "swing_high_time": high_pivot["time"],
                    "range": round(impulse_range, 2),
                    "impulse_atr": round(impulse_range / max(atr_value, 1e-9), 2),
                }
        low_pos = int(np.argmin(recent["low"].to_numpy()))
        if low_pos >= 8:
            high_pos = int(np.argmax(recent["high"].iloc[:low_pos].to_numpy()))
            high = safe_float(recent["high"].iloc[high_pos])
            low = safe_float(recent["low"].iloc[low_pos])
            if high - low >= min_range:
                return {
                    "direction": direction,
                    "swing_low": round(low, 2),
                    "swing_high": round(high, 2),
                    "swing_low_time": str(recent.index[low_pos]),
                    "swing_high_time": str(recent.index[high_pos]),
                    "range": round(high - low, 2),
                    "impulse_atr": round((high - low) / max(atr_value, 1e-9), 2),
                }

    return None


def fib_pullback_trade_plan(price: float, atr_value: float, direction: str, leg: dict) -> dict:
    swing_low = safe_float(leg.get("swing_low"))
    swing_high = safe_float(leg.get("swing_high"))
    impulse_range = max(swing_high - swing_low, 1e-9)
    buffer = max(atr_value * FIB_EMA_STOP_BUFFER_ATR, price * 0.00012)

    if direction == "LONG":
        fib50 = swing_high - impulse_range * 0.500
        fib618 = swing_high - impulse_range * 0.618
        fib786 = swing_high - impulse_range * 0.786
        fib886 = swing_high - impulse_range * 0.886
        entry_low = fib786
        entry_high = fib50
        entry = price if entry_low <= price <= entry_high else (entry_low + entry_high) / 2
        stop = min(swing_low - buffer, fib886 - buffer * 0.25)
        risk = max(entry - stop, 1e-9)
        target = entry + risk * 2.0
        target_2 = swing_low + impulse_range * 2.618
        if target_2 <= target:
            target_2 = entry + risk * 3.0
        invalidates = "15M closes below the 0.786 retracement or the swing low breaks."
    else:
        fib50 = swing_low + impulse_range * 0.500
        fib618 = swing_low + impulse_range * 0.618
        fib786 = swing_low + impulse_range * 0.786
        fib886 = swing_low + impulse_range * 0.886
        entry_low = fib50
        entry_high = fib786
        entry = price if entry_low <= price <= entry_high else (entry_low + entry_high) / 2
        stop = max(swing_high + buffer, fib886 + buffer * 0.25)
        risk = max(stop - entry, 1e-9)
        target = entry - risk * 2.0
        target_2 = swing_high - impulse_range * 2.618
        if target_2 >= target:
            target_2 = entry - risk * 3.0
        invalidates = "15M closes above the 0.786 retracement or the swing high breaks."

    reward = abs(target - entry)
    reward_2 = abs(target_2 - entry)
    return {
        "entry": round(entry, 2),
        "entry_low": round(entry_low, 2),
        "entry_high": round(entry_high, 2),
        "fib50": round(fib50, 2),
        "fib618": round(fib618, 2),
        "fib786": round(fib786, 2),
        "fib886": round(fib886, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "target_2": round(target_2, 2),
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "reward_2": round(reward_2, 2),
        "rr": round(reward / max(risk, 1e-9), 2),
        "rr_tp2": round(reward_2 / max(risk, 1e-9), 2),
        "position_1pct_10k_oz": round(100 / max(risk, 1e-9), 4),
        "invalidates": invalidates,
    }


def fib_reversal_confirmation(df: pd.DataFrame, direction: str) -> dict:
    if len(df) < 10:
        return {"confirmed": False, "reason": "Not enough 15M candles."}
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    close = df["close"]
    e9 = ema(close, 9)
    e21 = ema(close, 21)
    idx = len(df) - 1
    ema9_slope = pct_change_now(e9, idx, 3)
    if direction == "LONG":
        candle_ok = safe_float(latest["close"]) > safe_float(latest["open"])
        reclaim_ok = safe_float(latest["close"]) >= safe_float(previous["close"])
        ema_ok = safe_float(e9.iloc[-1]) >= safe_float(e21.iloc[-1]) or ema9_slope > 0
        return {
            "confirmed": candle_ok and reclaim_ok and ema_ok,
            "reason": "Bullish 15M candle, close reclaim, and EMA9/21 trigger." if candle_ok and reclaim_ok and ema_ok else "Wait for bullish 15M close/reclaim inside the fib zone.",
            "ema9_slope": round(ema9_slope, 3),
        }
    candle_ok = safe_float(latest["close"]) < safe_float(latest["open"])
    reclaim_ok = safe_float(latest["close"]) <= safe_float(previous["close"])
    ema_ok = safe_float(e9.iloc[-1]) <= safe_float(e21.iloc[-1]) or ema9_slope < 0
    return {
        "confirmed": candle_ok and reclaim_ok and ema_ok,
        "reason": "Bearish 15M candle, close rejection, and EMA9/21 trigger." if candle_ok and reclaim_ok and ema_ok else "Wait for bearish 15M close/rejection inside the fib zone.",
        "ema9_slope": round(ema9_slope, 3),
    }


def compute_fibonacci_ema_pullback_signal(h1_df: pd.DataFrame, m15_df: pd.DataFrame, context: dict | None = None) -> dict:
    context = context or {}
    if h1_df is None or m15_df is None or len(h1_df) < 220 or len(m15_df) < 120:
        return {
            "signal": "WAIT",
            "quality": "NO_DATA",
            "direction": "NONE",
            "reason": "Need enough H1 and 15M candles to calculate H4 alignment and Fibonacci pullback levels.",
            "plan": None,
            "conditions": {},
            "alignments": [],
        }

    h4_df = resample_ohlcv(h1_df, "4h")
    alignments = [
        ema_alignment_snapshot(m15_df, "15M"),
        ema_alignment_snapshot(h1_df, "1H"),
        ema_alignment_snapshot(h4_df, "4H"),
    ]
    long_aligned = all(item.get("direction") == "LONG" for item in alignments)
    short_aligned = all(item.get("direction") == "SHORT" for item in alignments)
    direction = "LONG" if long_aligned else "SHORT" if short_aligned else "NONE"

    price = safe_float(m15_df["close"].iloc[-1])
    at = atr(m15_df, 14)
    ax, plus_di, minus_di = adx(m15_df, 14)
    atr_value = max(safe_float(at.iloc[-1]), price * 0.0008)
    atr_pct = atr_value / max(price, 1e-9) * 100
    adx_value = safe_float(ax.iloc[-1])
    plus_div = safe_float(plus_di.iloc[-1])
    minus_div = safe_float(minus_di.iloc[-1])
    volatility_ok = 0.08 <= atr_pct <= 1.20
    event_not_high = context.get("risk", "UNKNOWN") != "HIGH"

    if direction == "NONE":
        aligned_count = max(
            sum(item.get("direction") == "LONG" for item in alignments),
            sum(item.get("direction") == "SHORT" for item in alignments),
        )
        return {
            "signal": "WAIT",
            "quality": "NO_ALIGNMENT",
            "direction": "NONE",
            "confidence": int(clamp(aligned_count * 20, 5, 60)),
            "price": round(price, 2),
            "reason": "15M, 1H, and derived 4H EMA stacks are not aligned in one direction.",
            "plan": None,
            "conditions": {
                "m15_h1_h4_ema_alignment": False,
                "event_not_high": event_not_high,
                "atr_usable": volatility_ok,
            },
            "alignments": alignments,
            "inputs": {
                "m15_atr14": round(atr_value, 2),
                "atr_pct": round(atr_pct, 3),
                "m15_adx14": round(adx_value, 1),
                "m15_plus_di14": round(plus_div, 1),
                "m15_minus_di14": round(minus_div, 1),
            },
        }

    leg = select_fib_impulse_leg(m15_df, direction, atr_value)
    if not leg:
        return {
            "signal": f"{direction}_WAIT",
            "quality": "NO_IMPULSE",
            "direction": direction,
            "confidence": 45,
            "price": round(price, 2),
            "reason": f"{direction} EMA alignment exists, but no clean M15 impulse leg met the minimum ATR range.",
            "plan": None,
            "conditions": {
                "m15_h1_h4_ema_alignment": True,
                "m15_impulse_leg": False,
                "event_not_high": event_not_high,
                "atr_usable": volatility_ok,
            },
            "alignments": alignments,
            "inputs": {
                "m15_atr14": round(atr_value, 2),
                "atr_pct": round(atr_pct, 3),
                "m15_adx14": round(adx_value, 1),
            },
        }

    plan = fib_pullback_trade_plan(price, atr_value, direction, leg)
    tolerance = atr_value * FIB_EMA_ENTRY_TOLERANCE_ATR
    entry_low = safe_float(plan["entry_low"])
    entry_high = safe_float(plan["entry_high"])
    in_zone = entry_low <= price <= entry_high
    near_zone = entry_low - tolerance <= price <= entry_high + tolerance
    if price > entry_high:
        distance_to_zone = price - entry_high
        zone_state = "ABOVE_ZONE"
    elif price < entry_low:
        distance_to_zone = entry_low - price
        zone_state = "BELOW_ZONE"
    else:
        distance_to_zone = 0.0
        zone_state = "IN_ZONE"

    reversal = fib_reversal_confirmation(m15_df, direction)
    risk = safe_float(plan.get("risk"))
    rr_ok = safe_float(plan.get("rr")) >= 2.0 and risk > 0
    impulse_ok = safe_float(leg.get("impulse_atr")) >= FIB_EMA_MIN_IMPULSE_ATR
    di_ok = plus_div >= minus_div if direction == "LONG" else minus_div >= plus_div
    conditions = {
        "m15_h1_h4_ema_alignment": True,
        "m15_impulse_leg": impulse_ok,
        "price_inside_050_0786_zone": in_zone,
        "price_near_entry_zone": near_zone,
        "m15_reversal_confirmation": bool(reversal.get("confirmed")),
        "minimum_rr_2": rr_ok,
        "atr_usable": volatility_ok,
        "di_confirms_direction": di_ok,
        "event_not_high": event_not_high,
    }
    ready = all([
        conditions["m15_h1_h4_ema_alignment"],
        conditions["m15_impulse_leg"],
        conditions["price_inside_050_0786_zone"],
        conditions["m15_reversal_confirmation"],
        conditions["minimum_rr_2"],
        conditions["atr_usable"],
        conditions["event_not_high"],
    ])

    if not event_not_high:
        signal = "BLOCKED"
        quality = "EVENT_RISK"
        reason = "High-impact event risk is active; fib pullback entries are blocked until the event window clears."
    elif ready:
        signal = f"{direction}_ENTRY"
        quality = "CONFIRMED" if di_ok and adx_value >= 16 else "CAUTION"
        reason = (
            f"{direction} clean-room fib EMA entry: M15/H1/H4 aligned, price is inside "
            f"{plan['entry_low']}-{plan['entry_high']}, and reversal confirmed."
        )
    elif in_zone:
        signal = f"{direction}_WATCH"
        quality = "WAIT_CONFIRMATION"
        reason = f"Price is inside the fib entry zone; wait for reversal confirmation before entry. {reversal.get('reason')}"
    elif near_zone:
        signal = f"{direction}_WATCH"
        quality = "NEAR_ZONE"
        reason = f"Price is within {round(tolerance, 2)} of the fib entry zone; prepare but wait for a 15M trigger."
    else:
        signal = f"{direction}_PULLBACK"
        quality = "PENDING"
        if direction == "LONG" and zone_state == "ABOVE_ZONE":
            reason = f"Trend is long, but price is {distance_to_zone:.2f} above the 0.50 entry boundary. Wait for pullback."
        elif direction == "LONG":
            reason = f"Trend is long, but price is {distance_to_zone:.2f} below the 0.786 boundary. Wait for reclaim."
        elif zone_state == "BELOW_ZONE":
            reason = f"Trend is short, but price is {distance_to_zone:.2f} below the 0.50 entry boundary. Wait for bounce."
        else:
            reason = f"Trend is short, but price is {distance_to_zone:.2f} above the 0.786 boundary. Wait for rejection."

    confidence = int(clamp(
        25
        + 25
        + min(15, safe_float(leg.get("impulse_atr")) * 3)
        + (15 if in_zone else 8 if near_zone else 0)
        + (12 if reversal.get("confirmed") else 0)
        + (8 if di_ok else 0)
        + (5 if adx_value >= 18 else 0)
        - (20 if not event_not_high else 0)
        - (10 if not volatility_ok else 0),
        5,
        95,
    ))
    return {
        "signal": signal,
        "quality": quality,
        "direction": direction,
        "confidence": confidence,
        "price": round(price, 2),
        "reason": reason,
        "plan": plan,
        "conditions": conditions,
        "failed_conditions": [name for name, passed in conditions.items() if not passed],
        "alignments": alignments,
        "leg": leg,
        "zone_state": zone_state,
        "distance_to_zone": round(distance_to_zone, 2),
        "entry_tolerance": round(tolerance, 2),
        "reversal": reversal,
        "inputs": {
            "m15_atr14": round(atr_value, 2),
            "atr_pct": round(atr_pct, 3),
            "m15_adx14": round(adx_value, 1),
            "m15_plus_di14": round(plus_div, 1),
            "m15_minus_di14": round(minus_div, 1),
            "lookback_bars": min(FIB_EMA_LOOKBACK_BARS, len(m15_df)),
            "min_impulse_atr": FIB_EMA_MIN_IMPULSE_ATR,
        },
        "model": "clean_room_public_fib_ema_pullback",
        "source_note": "Approximates public Fibonacci/EMA claims; it is not copied proprietary EA code.",
    }


def fibonacci_ema_pullback_signal(context: dict | None = None, force: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    cached_at = FIB_EMA_CACHE.get("ts")
    if (
        not force
        and SIGNAL_CACHE_TTL_SECONDS > 0
        and FIB_EMA_CACHE.get("data") is not None
        and cached_at is not None
        and (now - cached_at).total_seconds() <= SIGNAL_CACHE_TTL_SECONDS
    ):
        return copy.deepcopy(FIB_EMA_CACHE["data"])

    with FIB_EMA_CACHE_LOCK:
        now = datetime.now(timezone.utc)
        cached_at = FIB_EMA_CACHE.get("ts")
        if (
            not force
            and SIGNAL_CACHE_TTL_SECONDS > 0
            and FIB_EMA_CACHE.get("data") is not None
            and cached_at is not None
            and (now - cached_at).total_seconds() <= SIGNAL_CACHE_TTL_SECONDS
        ):
            return copy.deepcopy(FIB_EMA_CACHE["data"])

        h1_df, h1_err = fetch_data("1h")
        m15_df, m15_err = fetch_data("15m")
        if h1_err or m15_err or h1_df is None or m15_df is None:
            result = {
                "signal": "WAIT",
                "quality": "OFFLINE",
                "direction": "NONE",
                "reason": h1_err or m15_err or "Missing H1/15M data.",
                "plan": None,
                "conditions": {},
                "alignments": [],
                "model": "clean_room_public_fib_ema_pullback",
            }
        else:
            result = compute_fibonacci_ema_pullback_signal(h1_df, m15_df, context=context)
            result["data_source"] = {
                "h1": h1_df.attrs.get("source", MARKET_DATA_SOURCE),
                "m15": m15_df.attrs.get("source", MARKET_DATA_SOURCE),
            }
            result["timestamp"] = now.strftime("%Y-%m-%d %H:%M UTC")

        FIB_EMA_CACHE["ts"] = now
        FIB_EMA_CACHE["data"] = copy.deepcopy(result)
        return result


def ct_session_label(ts: datetime | pd.Timestamp) -> str:
    if isinstance(ts, pd.Timestamp):
        value = ts.to_pydatetime()
    else:
        value = ts
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    ct = value.astimezone(ZoneInfo("America/Chicago"))
    hour = ct.hour + ct.minute / 60
    if 2 <= hour < 5:
        return "LONDON_OPEN_CT"
    if 7 <= hour < 10:
        return "NY_OPEN_OVERLAP_CT"
    return "OFF_SESSION"


def ict_session_windows(allowed_sessions: set[str] | None = None) -> list[str]:
    selected = allowed_sessions or ICT_DEFAULT_SESSIONS
    windows = {
        "LONDON_OPEN_CT": "02:00-05:00",
        "NY_OPEN_OVERLAP_CT": "07:00-10:00",
    }
    return [windows[name] for name in sorted(selected) if name in windows]


def ict_session_allowed(ts: datetime | pd.Timestamp, allowed_sessions: set[str] | None = None) -> bool:
    return ct_session_label(ts) in (allowed_sessions or ICT_DEFAULT_SESSIONS)


def is_event_blackout_time(ts: datetime | pd.Timestamp, minutes: int = 15) -> bool:
    if isinstance(ts, pd.Timestamp):
        value = ts.to_pydatetime()
    else:
        value = ts
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    for event in globals().get("PAST_EVENTS", []) + globals().get("UPCOMING_EVENTS", []):
        try:
            event_time = parse_utc(event.get("time_utc", ""))
        except Exception:
            continue
        if abs((value - event_time).total_seconds()) <= minutes * 60:
            return True
    return False


def directional_bias_from_frame(df: pd.DataFrame, label: str) -> dict:
    if df is None or len(df) < 55:
        return {"timeframe": label, "direction": "NONE", "reason": "insufficient bars"}
    close = df["close"]
    e21 = ema(close, 21)
    e50 = ema(close, 50)
    idx = len(df) - 1
    price = safe_float(close.iloc[idx])
    ema21v = safe_float(e21.iloc[idx])
    ema50v = safe_float(e50.iloc[idx])
    slope = pct_change_now(e21, idx, 5)
    recent = df.iloc[-24:] if label == "4H" else df.iloc[-20:]
    prior = df.iloc[-48:-24] if label == "4H" and len(df) >= 48 else df.iloc[-40:-20] if len(df) >= 40 else pd.DataFrame()
    higher_high = not prior.empty and safe_float(recent["high"].max()) > safe_float(prior["high"].max())
    higher_low = not prior.empty and safe_float(recent["low"].min()) > safe_float(prior["low"].min())
    lower_high = not prior.empty and safe_float(recent["high"].max()) < safe_float(prior["high"].max())
    lower_low = not prior.empty and safe_float(recent["low"].min()) < safe_float(prior["low"].min())

    bullish = price > ema21v > ema50v and slope >= -0.02 and (higher_high or higher_low or prior.empty)
    bearish = price < ema21v < ema50v and slope <= 0.02 and (lower_low or lower_high or prior.empty)
    direction = "LONG" if bullish else "SHORT" if bearish else "NONE"
    return {
        "timeframe": label,
        "direction": direction,
        "price": round(price, 2),
        "ema21": round(ema21v, 2),
        "ema50": round(ema50v, 2),
        "ema21_slope": round(slope, 3),
        "structure": {
            "higher_high": higher_high,
            "higher_low": higher_low,
            "lower_high": lower_high,
            "lower_low": lower_low,
        },
        "reason": f"{label} {direction.lower()} bias from EMA21/50 and recent swing structure." if direction != "NONE" else f"{label} is mixed/ranging.",
    }


def ict_higher_timeframe_bias(h1_df: pd.DataFrame) -> dict:
    h4_df = resample_ohlcv(h1_df, "4h")
    daily_df = resample_ohlcv(h1_df, "1d")
    h4 = directional_bias_from_frame(h4_df, "4H")
    daily = directional_bias_from_frame(daily_df, "Daily")
    h4_direction = h4.get("direction")
    daily_direction = daily.get("direction")
    if h4_direction in ("LONG", "SHORT") and daily_direction in ("NONE", h4_direction):
        direction = h4_direction
        allowed = True
    else:
        direction = "NONE"
        allowed = False
    return {
        "direction": direction,
        "allowed": allowed,
        "h4": h4,
        "daily": daily,
        "reason": (
            f"H4 bias is {h4_direction}; Daily is {daily_direction}. Trade {h4_direction.lower()} only."
            if allowed else
            f"No directional filter: H4 is {h4_direction}, Daily is {daily_direction}."
        ),
    }


def ict_levels(df: pd.DataFrame, ts: datetime | pd.Timestamp | None = None, lookback: int = 96) -> dict:
    if df.empty:
        return {}
    work = df.copy()
    if work.index.tz is None:
        work.index = work.index.tz_localize("UTC")
    else:
        work.index = work.index.tz_convert("UTC")
    current_ts = pd.Timestamp(ts).tz_convert("UTC") if ts is not None else work.index[-1]
    today = current_ts.date()
    prior_days = [day for day in sorted(set(work.index.date)) if day < today]
    previous_day = prior_days[-1] if prior_days else None
    current = work[work.index.date == today]
    previous = work[work.index.date == previous_day] if previous_day else pd.DataFrame()
    asian = current[(current.index.hour >= 0) & (current.index.hour < 7)]
    recent = work[work.index <= current_ts].tail(lookback)
    return {
        "previous_day": {
            "high": round(safe_float(previous["high"].max()), 2) if not previous.empty else None,
            "low": round(safe_float(previous["low"].min()), 2) if not previous.empty else None,
        },
        "asian": {
            "high": round(safe_float(asian["high"].max()), 2) if not asian.empty else None,
            "low": round(safe_float(asian["low"].min()), 2) if not asian.empty else None,
        },
        "recent_range": {
            "high": round(safe_float(recent["high"].max()), 2) if not recent.empty else None,
            "low": round(safe_float(recent["low"].min()), 2) if not recent.empty else None,
        },
    }


def range_thirds(levels: dict) -> dict:
    high = safe_float((levels.get("recent_range") or {}).get("high"))
    low = safe_float((levels.get("recent_range") or {}).get("low"))
    if high <= low:
        return {"low": None, "high": None, "discount_max": None, "premium_min": None}
    width = high - low
    return {
        "low": round(low, 2),
        "high": round(high, 2),
        "discount_max": round(low + width / 3, 2),
        "premium_min": round(high - width / 3, 2),
    }


def zone_touches_range(zone: dict, low: float, high: float, tolerance: float = 0.0) -> bool:
    return low <= safe_float(zone.get("high")) + tolerance and high >= safe_float(zone.get("low")) - tolerance


def zone_in_premium_discount(zone: dict, direction: str, thirds: dict) -> bool:
    zone_mid = (safe_float(zone.get("low")) + safe_float(zone.get("high"))) / 2
    if direction == "LONG":
        discount_max = thirds.get("discount_max")
        return discount_max is not None and zone_mid <= safe_float(discount_max)
    if direction == "SHORT":
        premium_min = thirds.get("premium_min")
        return premium_min is not None and zone_mid >= safe_float(premium_min)
    return False


def detect_ict_ob_fvg_zones(df: pd.DataFrame, direction: str, atr_series: pd.Series, lookback: int = 120) -> list[dict]:
    if df is None or len(df) < 30 or direction not in ("LONG", "SHORT"):
        return []
    start = max(3, len(df) - lookback)
    zones = []
    current_price = safe_float(df["close"].iloc[-1])
    current_atr = max(safe_float(atr_series.iloc[-1]), current_price * 0.0008)
    for idx in range(start, len(df) - 2):
        row = df.iloc[idx]
        future = df.iloc[idx + 1:idx + 4]
        if future.empty:
            continue
        open_price = safe_float(row["open"])
        close = safe_float(row["close"])
        high = safe_float(row["high"])
        low = safe_float(row["low"])
        atr_value = max(safe_float(atr_series.iloc[idx], current_atr), 1e-9)
        impulse_up = (safe_float(future["close"].max()) - close) / atr_value
        impulse_down = (close - safe_float(future["close"].min())) / atr_value
        if direction == "LONG" and close < open_price and impulse_up >= ICT_MIN_IMPULSE_ATR:
            zones.append({
                "type": "ORDER_BLOCK",
                "direction": direction,
                "low": round(low, 2),
                "high": round(high, 2),
                "mid": round((low + high) / 2, 2),
                "formed_at": str(df.index[idx]),
                "score": int(clamp(55 + impulse_up * 8, 0, 100)),
                "summary": f"LONG order block {low:.2f}-{high:.2f} before {impulse_up:.1f} ATR displacement.",
            })
        if direction == "SHORT" and close > open_price and impulse_down >= ICT_MIN_IMPULSE_ATR:
            zones.append({
                "type": "ORDER_BLOCK",
                "direction": direction,
                "low": round(low, 2),
                "high": round(high, 2),
                "mid": round((low + high) / 2, 2),
                "formed_at": str(df.index[idx]),
                "score": int(clamp(55 + impulse_down * 8, 0, 100)),
                "summary": f"SHORT order block {low:.2f}-{high:.2f} before {impulse_down:.1f} ATR displacement.",
            })

    for idx in range(start + 2, len(df)):
        prev2 = df.iloc[idx - 2]
        row = df.iloc[idx]
        if direction == "LONG" and safe_float(row["low"]) > safe_float(prev2["high"]):
            low = safe_float(prev2["high"])
            high = safe_float(row["low"])
            zones.append({
                "type": "FAIR_VALUE_GAP",
                "direction": direction,
                "low": round(low, 2),
                "high": round(high, 2),
                "mid": round((low + high) / 2, 2),
                "formed_at": str(df.index[idx]),
                "score": int(clamp(50 + ((high - low) / current_atr) * 10, 0, 100)),
                "summary": f"LONG fair value gap {low:.2f}-{high:.2f}.",
            })
        if direction == "SHORT" and safe_float(row["high"]) < safe_float(prev2["low"]):
            low = safe_float(row["high"])
            high = safe_float(prev2["low"])
            zones.append({
                "type": "FAIR_VALUE_GAP",
                "direction": direction,
                "low": round(low, 2),
                "high": round(high, 2),
                "mid": round((low + high) / 2, 2),
                "formed_at": str(df.index[idx]),
                "score": int(clamp(50 + ((high - low) / current_atr) * 10, 0, 100)),
                "summary": f"SHORT fair value gap {low:.2f}-{high:.2f}.",
            })
    deduped = []
    seen = set()
    for zone in sorted(zones, key=lambda item: (item["score"], item["formed_at"]), reverse=True):
        key = (zone["type"], zone["direction"], zone["low"], zone["high"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(zone)
    return deduped[:12]


def select_ict_zone(h1_df: pd.DataFrame, m15_df: pd.DataFrame, direction: str, levels: dict) -> dict | None:
    if direction not in ("LONG", "SHORT"):
        return None
    thirds = range_thirds(levels)
    m15_atr = atr(m15_df, 14)
    h1_atr = atr(h1_df, 14)
    zones = []
    for tf, source, atr_series, lookback in (
        ("15M", m15_df, m15_atr, ICT_LOOKBACK_M15_BARS),
        ("1H", h1_df, h1_atr, 72),
    ):
        for zone in detect_ict_ob_fvg_zones(source, direction, atr_series, lookback=lookback):
            if not zone_in_premium_discount(zone, direction, thirds):
                continue
            distance = abs(safe_float(source["close"].iloc[-1]) - safe_float(zone.get("mid"))) / max(safe_float(atr_series.iloc[-1]), 1e-9)
            zones.append({
                **zone,
                "timeframe": tf,
                "distance_atr": round(distance, 2),
                "premium_discount": thirds,
            })
    if not zones:
        return None
    return sorted(zones, key=lambda item: (item["distance_atr"], -item["score"]))[0]


def recent_swing_levels(df: pd.DataFrame, left: int = 2, right: int = 2, lookback: int = 80) -> dict:
    recent = df.tail(lookback)
    pivots = pivot_points(recent, left=left, right=right)
    highs = pivots.get("highs") or []
    lows = pivots.get("lows") or []
    return {
        "high": highs[-1] if highs else None,
        "low": lows[-1] if lows else None,
        "highs": highs,
        "lows": lows,
    }


def detect_5m_bos_retracement(m5_df: pd.DataFrame, direction: str, setup_zone: dict, atr_value: float) -> dict:
    if m5_df is None or len(m5_df) < 60 or direction not in ("LONG", "SHORT") or not setup_zone:
        return {"confirmed": False, "reason": "Missing 5M data, direction, or setup zone."}
    scan = min(ICT_BOS_LOOKBACK_5M_BARS, len(m5_df) - 15)
    best = None
    for age in range(scan):
        idx = len(m5_df) - 1 - age
        if idx < 16:
            continue
        row = m5_df.iloc[idx]
        history = m5_df.iloc[idx - 14:idx - 1]
        buffer = max(atr_value * 0.05, safe_float(row["close"]) * 0.00005)
        if direction == "LONG":
            swing = safe_float(history["high"].max())
            broke = safe_float(row["close"]) > swing + buffer
        else:
            swing = safe_float(history["low"].min())
            broke = safe_float(row["close"]) < swing - buffer
        if not broke:
            continue
        after = m5_df.iloc[idx + 1:]
        if after.empty:
            continue
        retrace_mask = (
            (after["low"] <= safe_float(setup_zone["high"]) + atr_value * ICT_ZONE_TOUCH_TOLERANCE_ATR)
            & (after["high"] >= safe_float(setup_zone["low"]) - atr_value * ICT_ZONE_TOUCH_TOLERANCE_ATR)
        )
        retraced = bool(retrace_mask.any())
        latest = m5_df.iloc[-1]
        latest_touch = zone_touches_range(
            setup_zone,
            safe_float(latest["low"]),
            safe_float(latest["high"]),
            atr_value * ICT_ZONE_TOUCH_TOLERANCE_ATR,
        )
        if retraced and latest_touch:
            best = {
                "confirmed": True,
                "direction": direction,
                "bos_level": round(swing, 2),
                "bos_time": str(m5_df.index[idx]),
                "age_bars": len(m5_df) - 1 - idx,
                "entry_zone": setup_zone,
                "reason": f"5M broke structure at {swing:.2f}, then retraced into {setup_zone.get('type')} zone.",
            }
            break
        if best is None:
            best = {
                "confirmed": False,
                "direction": direction,
                "bos_level": round(swing, 2),
                "bos_time": str(m5_df.index[idx]),
                "age_bars": len(m5_df) - 1 - idx,
                "entry_zone": setup_zone,
                "reason": "5M BOS exists, but the latest candle is not retracing into the setup zone.",
            }
    return best or {"confirmed": False, "direction": direction, "reason": "No recent 5M break of structure in the bias direction."}


def select_ict_target(entry: float, stop: float, direction: str, levels: dict, m15_df: pd.DataFrame) -> dict | None:
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    recent = recent_swing_levels(m15_df, left=3, right=3, lookback=160)
    raw_candidates = []
    if direction == "LONG":
        for label, value in (
            ("asian_high", (levels.get("asian") or {}).get("high")),
            ("previous_day_high", (levels.get("previous_day") or {}).get("high")),
            ("recent_swing_high", (recent.get("high") or {}).get("price")),
        ):
            price = safe_float(value)
            if price > entry:
                raw_candidates.append((label, price))
        raw_candidates = sorted(raw_candidates, key=lambda item: item[1])
        for label, price in raw_candidates:
            rr = (price - entry) / risk
            if rr >= ICT_MIN_RR:
                return {"price": round(price, 2), "rr": round(rr, 2), "source": label}
    if direction == "SHORT":
        for label, value in (
            ("asian_low", (levels.get("asian") or {}).get("low")),
            ("previous_day_low", (levels.get("previous_day") or {}).get("low")),
            ("recent_swing_low", (recent.get("low") or {}).get("price")),
        ):
            price = safe_float(value)
            if price < entry:
                raw_candidates.append((label, price))
        raw_candidates = sorted(raw_candidates, key=lambda item: item[1], reverse=True)
        for label, price in raw_candidates:
            rr = (entry - price) / risk
            if rr >= ICT_MIN_RR:
                return {"price": round(price, 2), "rr": round(rr, 2), "source": label}
    return None


def ict_trade_plan(direction: str, zone: dict, m5_df: pd.DataFrame, m15_df: pd.DataFrame, levels: dict) -> dict:
    price = safe_float(m5_df["close"].iloc[-1])
    at5 = atr(m5_df, 14)
    atr_value = max(safe_float(at5.iloc[-1]), price * 0.0008)
    entry = min(max(price, safe_float(zone["low"])), safe_float(zone["high"]))
    buffer = max(atr_value * ICT_STOP_BUFFER_ATR, price * 0.00012)
    if direction == "LONG":
        stop = safe_float(zone["low"]) - buffer
    else:
        stop = safe_float(zone["high"]) + buffer
    target = select_ict_target(entry, stop, direction, levels, m15_df)
    if not target:
        return {
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": None,
            "risk": round(abs(entry - stop), 2),
            "reward": None,
            "rr": None,
            "position_1pct_10k_oz": None,
            "invalidates": "No logical liquidity target gives at least 1:2.",
        }
    reward = abs(target["price"] - entry)
    risk = abs(entry - stop)
    return {
        "entry": round(entry, 2),
        "entry_low": zone.get("low"),
        "entry_high": zone.get("high"),
        "stop": round(stop, 2),
        "target": target["price"],
        "target_source": target["source"],
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "rr": target["rr"],
        "position_1pct_10k_oz": round(100 / max(risk, 1e-9), 4),
        "invalidates": f"Setup invalidates beyond {zone.get('type')} {zone.get('low')}-{zone.get('high')}.",
    }


def compute_ict_bos_retracement_signal(
    h1_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    m5_df: pd.DataFrame,
    context: dict | None = None,
    allowed_sessions: set[str] | None = None,
    model_name: str = "h4_daily_premium_discount_5m_bos_retracement",
) -> dict:
    context = context or {}
    if h1_df is None or m15_df is None or m5_df is None or len(h1_df) < 220 or len(m15_df) < 120 or len(m5_df) < 120:
        return {
            "signal": "WAIT",
            "quality": "NO_DATA",
            "direction": "NONE",
            "reason": "Need H1, 15M, and 5M candles for H4/Daily bias, zone, and BOS trigger.",
            "trade": {},
            "checks": {},
            "model": model_name,
            "allowed_sessions": sorted(allowed_sessions or ICT_DEFAULT_SESSIONS),
            "session_windows_ct": ict_session_windows(allowed_sessions or ICT_DEFAULT_SESSIONS),
        }
    now_ts = m5_df.index[-1]
    bias = ict_higher_timeframe_bias(h1_df)
    direction = bias.get("direction")
    levels = ict_levels(m15_df, now_ts)
    session = ct_session_label(now_ts)
    news_blocked = context.get("risk") == "HIGH" or is_event_blackout_time(now_ts, 15)
    selected_sessions = allowed_sessions or ICT_DEFAULT_SESSIONS
    session_check = "session_london_ct" if selected_sessions == ICT_LONDON_ONLY_SESSIONS else "session_london_or_ny_ct"
    checks = {
        "h4_daily_bias": direction in ("LONG", "SHORT"),
        session_check: ict_session_allowed(now_ts, selected_sessions),
        "news_blackout_clear": not news_blocked,
    }
    if direction not in ("LONG", "SHORT"):
        return {
            "signal": "WAIT",
            "quality": "NO_BIAS",
            "direction": "NONE",
            "reason": bias.get("reason"),
            "bias": bias,
            "levels": levels,
            "trade": {},
            "checks": checks,
            "session": session,
            "model": model_name,
            "allowed_sessions": sorted(selected_sessions),
            "session_windows_ct": ict_session_windows(selected_sessions),
        }
    zone = select_ict_zone(h1_df, m15_df, direction, levels)
    checks["premium_discount_zone"] = bool(zone)
    if not zone:
        return {
            "signal": f"{direction}_WAIT",
            "quality": "NO_ZONE",
            "direction": direction,
            "reason": "Bias is set, but no unmitigated H1/15M order block or FVG is sitting in premium/discount.",
            "bias": bias,
            "levels": levels,
            "zone": None,
            "trade": {},
            "checks": checks,
            "session": session,
            "model": model_name,
            "allowed_sessions": sorted(selected_sessions),
            "session_windows_ct": ict_session_windows(selected_sessions),
        }
    at5 = atr(m5_df, 14)
    bos = detect_5m_bos_retracement(m5_df, direction, zone, max(safe_float(at5.iloc[-1]), 1e-9))
    checks["m5_bos_retracement"] = bool(bos.get("confirmed"))
    trade = ict_trade_plan(direction, zone, m5_df, m15_df, levels) if bos.get("confirmed") else {}
    checks["minimum_rr_2"] = bool(trade and trade.get("rr") is not None and safe_float(trade.get("rr")) >= ICT_MIN_RR)
    executable = all(checks.values())
    quality = "CONFIRMED" if executable else "BLOCKED" if news_blocked else "WAIT_TRIGGER"
    signal = f"{direction}_ENTRY" if executable else f"{direction}_WATCH"
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "signal": signal,
        "quality": quality,
        "direction": direction,
        "session": session,
        "reason": (
            f"{direction} A+ BOS retracement: H4/Daily bias, premium/discount zone, 5M BOS retrace, and >=1:{ICT_MIN_RR:.0f} target."
            if executable else
            f"Waiting: {', '.join(failed)}."
        ),
        "bias": bias,
        "levels": levels,
        "zone": zone,
        "bos": bos,
        "trade": trade,
        "checks": checks,
        "failed_checks": failed,
        "model": model_name,
        "allowed_sessions": sorted(selected_sessions),
        "session_windows_ct": ict_session_windows(selected_sessions),
        "source_note": "Clean-room implementation of the described discretionary plan; not proprietary EA code.",
    }


def ict_bos_retracement_signal_for_sessions(
    context: dict | None,
    force: bool,
    allowed_sessions: set[str],
    cache: dict,
    cache_lock: threading.Lock,
    model_name: str,
) -> dict:
    now = datetime.now(timezone.utc)
    cached_at = cache.get("ts")
    if (
        not force
        and SIGNAL_CACHE_TTL_SECONDS > 0
        and cache.get("data") is not None
        and cached_at is not None
        and (now - cached_at).total_seconds() <= SIGNAL_CACHE_TTL_SECONDS
    ):
        return copy.deepcopy(cache["data"])
    with cache_lock:
        now = datetime.now(timezone.utc)
        cached_at = cache.get("ts")
        if (
            not force
            and SIGNAL_CACHE_TTL_SECONDS > 0
            and cache.get("data") is not None
            and cached_at is not None
            and (now - cached_at).total_seconds() <= SIGNAL_CACHE_TTL_SECONDS
        ):
            return copy.deepcopy(cache["data"])
        h1_df, h1_err = fetch_data("1h")
        m15_df, m15_err = fetch_data("15m")
        m5_df, m5_err = fetch_data("5m")
        if h1_err or m15_err or m5_err or h1_df is None or m15_df is None or m5_df is None:
            result = {
                "signal": "WAIT",
                "quality": "OFFLINE",
                "direction": "NONE",
                "reason": h1_err or m15_err or m5_err or "Missing timeframe data.",
                "trade": {},
                "checks": {},
                "model": model_name,
                "allowed_sessions": sorted(allowed_sessions),
                "session_windows_ct": ict_session_windows(allowed_sessions),
            }
        else:
            result = compute_ict_bos_retracement_signal(
                h1_df,
                m15_df,
                m5_df,
                context=context,
                allowed_sessions=allowed_sessions,
                model_name=model_name,
            )
            result["data_source"] = {
                "h1": h1_df.attrs.get("source", MARKET_DATA_SOURCE),
                "m15": m15_df.attrs.get("source", MARKET_DATA_SOURCE),
                "m5": m5_df.attrs.get("source", MARKET_DATA_SOURCE),
            }
            result["timestamp"] = now.strftime("%Y-%m-%d %H:%M UTC")
        cache["ts"] = now
        cache["data"] = copy.deepcopy(result)
        return result


def ict_bos_retracement_signal(context: dict | None = None, force: bool = False) -> dict:
    return ict_bos_retracement_signal_for_sessions(
        context,
        force,
        ICT_DEFAULT_SESSIONS,
        ICT_BOS_CACHE,
        ICT_BOS_CACHE_LOCK,
        "h4_daily_premium_discount_5m_bos_retracement",
    )


def ict_london_bos_signal(context: dict | None = None, force: bool = False) -> dict:
    return ict_bos_retracement_signal_for_sessions(
        context,
        force,
        ICT_LONDON_ONLY_SESSIONS,
        ICT_LONDON_BOS_CACHE,
        ICT_LONDON_BOS_CACHE_LOCK,
        "h4_daily_london_only_premium_discount_5m_bos_retracement",
    )


def fetch_yfinance_5m_history(days: int = 60) -> tuple[pd.DataFrame | None, str | None]:
    try:
        import yfinance as yf
        prepare_yfinance_cache(yf)
        period = f"{int(clamp(days, 5, 60))}d"
        df = yf.download("GC=F", period=period, interval="5m", progress=False, auto_adjust=False)
        if df.empty:
            return None, "Yahoo returned no 5M GC=F data"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]).lower().replace(" ", "_") for col in df.columns]
        else:
            df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.index = pd.to_datetime(df.index, utc=True)
        df.attrs["source"] = "yfinance:GC=F:5m"
        return df, None
    except Exception as e:
        return None, f"yfinance 5M backtest error: {compact_error(e)}"


def ict_backtest_trade_outcome(
    m5_df: pd.DataFrame,
    start_idx: int,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    max_hold_bars: int = 96,
) -> dict:
    end_idx = min(len(m5_df) - 1, start_idx + max_hold_bars)
    for idx in range(start_idx + 1, end_idx + 1):
        high = safe_float(m5_df["high"].iloc[idx])
        low = safe_float(m5_df["low"].iloc[idx])
        if direction == "LONG":
            hit_sl = low <= stop
            hit_tp = high >= target
        else:
            hit_sl = high >= stop
            hit_tp = low <= target
        if hit_sl:
            return {"exit_idx": idx, "exit_price": stop, "result": "SL"}
        if hit_tp:
            return {"exit_idx": idx, "exit_price": target, "result": "TP"}
    return {
        "exit_idx": end_idx,
        "exit_price": safe_float(m5_df["close"].iloc[end_idx]),
        "result": "TIME",
    }


def ct_trade_day(ts: pd.Timestamp) -> str:
    return ts.to_pydatetime().astimezone(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")


def run_ict_bos_backtest(
    days: int = 60,
    max_checks: int = 5000,
    risk_per_trade_pct: float = 1.0,
    allowed_sessions: set[str] | None = None,
    mode: str = "ict_bos_retracement_backtest",
) -> dict:
    m5_df, err = fetch_yfinance_5m_history(days)
    selected_sessions = allowed_sessions or ICT_DEFAULT_SESSIONS
    if err or m5_df is None:
        return {"error": err or "No 5M data", "mode": mode}
    m15_df = resample_ohlcv(m5_df, "15min")
    h1_df = resample_ohlcv(m5_df, "1h")
    min_m5 = 160
    min_m15 = 140
    min_h1 = 220
    if len(m5_df) < min_m5 or len(m15_df) < min_m15 or len(h1_df) < min_h1:
        return {
            "error": "Not enough history for ICT BOS replay",
            "bars": {"m5": len(m5_df), "m15": len(m15_df), "h1": len(h1_df)},
            "mode": mode,
        }

    equity = 10_000.0
    risk_fraction = clamp(risk_per_trade_pct / 100, 0.001, 0.05)
    trades = []
    skipped = {
        "not_session": 0,
        "daily_guardrail": 0,
        "news_blackout": 0,
        "insufficient_history": 0,
        "no_executable_setup": 0,
        "no_trade_plan": 0,
    }
    checked = 0
    daily_state: dict[str, dict] = {}
    start_idx = max(min_m5, min_h1 * 12)
    idx = start_idx

    while idx < len(m5_df) - 98 and checked < max_checks:
        ts = m5_df.index[idx]
        checked += 1
        if not ict_session_allowed(ts, selected_sessions):
            skipped["not_session"] += 1
            idx += 1
            continue
        day = ct_trade_day(ts)
        state = daily_state.setdefault(day, {"trades": 0, "consecutive_losses": 0})
        if state["trades"] >= ICT_MAX_TRADES_PER_DAY or state["consecutive_losses"] >= ICT_MAX_CONSECUTIVE_LOSSES:
            skipped["daily_guardrail"] += 1
            idx += 1
            continue
        if is_event_blackout_time(ts, 15):
            skipped["news_blackout"] += 1
            idx += 1
            continue

        h1_history = history_until(h1_df, ts, min_h1)
        m15_history = history_until(m15_df, ts, min_m15)
        m5_history = m5_df.iloc[:idx + 1]
        if h1_history is None or m15_history is None or len(m5_history) < min_m5:
            skipped["insufficient_history"] += 1
            idx += 1
            continue

        signal = compute_ict_bos_retracement_signal(
            h1_history,
            m15_history,
            m5_history,
            context={"risk": "LOW"},
            allowed_sessions=selected_sessions,
            model_name=(
                "h4_daily_london_only_premium_discount_5m_bos_retracement"
                if selected_sessions == ICT_LONDON_ONLY_SESSIONS else
                "h4_daily_premium_discount_5m_bos_retracement"
            ),
        )
        trade = signal.get("trade") or {}
        if signal.get("quality") != "CONFIRMED" or signal.get("direction") not in ("LONG", "SHORT"):
            skipped["no_executable_setup"] += 1
            idx += 1
            continue
        if trade.get("stop") is None or trade.get("target") is None:
            skipped["no_trade_plan"] += 1
            idx += 1
            continue

        direction = signal["direction"]
        entry = safe_float(trade.get("entry"))
        stop = safe_float(trade.get("stop"))
        target = safe_float(trade.get("target"))
        risk = abs(entry - stop)
        if risk <= 0:
            skipped["no_trade_plan"] += 1
            idx += 1
            continue

        outcome = ict_backtest_trade_outcome(m5_df, idx, direction, entry, stop, target)
        exit_idx = outcome["exit_idx"]
        exit_price = safe_float(outcome["exit_price"])
        pnl_per_oz = exit_price - entry if direction == "LONG" else entry - exit_price
        r_multiple = pnl_per_oz / risk
        size_oz = (equity * risk_fraction) / risk
        pnl = pnl_per_oz * size_oz
        equity += pnl
        state["trades"] += 1
        state["consecutive_losses"] = state["consecutive_losses"] + 1 if pnl < 0 else 0

        trades.append({
            "entry_time": ts.strftime("%Y-%m-%d %H:%M"),
            "exit_time": m5_df.index[exit_idx].strftime("%Y-%m-%d %H:%M"),
            "ct_session": ct_session_label(ts),
            "ct_day": day,
            "direction": direction,
            "result": outcome["result"],
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "target_source": trade.get("target_source"),
            "exit": round(exit_price, 2),
            "r": round(r_multiple, 2),
            "pnl": round(pnl, 2),
            "equity": round(equity, 2),
            "zone_type": (signal.get("zone") or {}).get("type"),
            "zone_timeframe": (signal.get("zone") or {}).get("timeframe"),
            "zone_low": (signal.get("zone") or {}).get("low"),
            "zone_high": (signal.get("zone") or {}).get("high"),
        })
        idx = exit_idx + 1

    if not trades:
        return {
            "mode": mode,
            "data": {
                "symbol": "GC=F",
                "interval": "5m",
                "start": str(m5_df.index[0]),
                "end": str(m5_df.index[-1]),
                "bars": len(m5_df),
            },
            "sampling": {"checked_bars": checked, "skipped": skipped},
            "metrics": {"total_signals": 0},
            "trades": [],
            "note": "No confirmed BOS-retracement trades passed all filters.",
        }

    total_pnl = sum(item["pnl"] for item in trades)
    wins = [item for item in trades if item["pnl"] > 0]
    losses = [item for item in trades if item["pnl"] < 0]
    gross_win = sum(item["pnl"] for item in wins)
    gross_loss = abs(sum(item["pnl"] for item in losses))
    profit_factor = gross_win / gross_loss if gross_loss else None
    equity_values = [10_000.0] + [item["equity"] for item in trades]
    peak = equity_values[0]
    max_dd = 0.0
    for value in equity_values:
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1)
    session_counts: dict[str, dict] = {}
    for item in trades:
        stats = session_counts.setdefault(item["ct_session"], {"trades": 0, "net_r": 0.0, "wins": 0, "pnl": 0.0})
        stats["trades"] += 1
        stats["net_r"] += item["r"]
        stats["wins"] += 1 if item["pnl"] > 0 else 0
        stats["pnl"] += item["pnl"]
    by_session = {
        key: {
            "trades": value["trades"],
            "win_rate": round(value["wins"] / value["trades"] * 100, 2),
            "net_r": round(value["net_r"], 2),
            "avg_r": round(value["net_r"] / value["trades"], 2),
            "pnl": round(value["pnl"], 2),
        }
        for key, value in sorted(session_counts.items())
    }
    return {
        "mode": mode,
        "data": {
            "symbol": "GC=F",
            "interval": "5m",
            "start": str(m5_df.index[0]),
            "end": str(m5_df.index[-1]),
            "bars": len(m5_df),
        },
        "settings": {
            "days": int(clamp(days, 5, 60)),
            "risk_per_trade_pct": risk_per_trade_pct,
            "max_trades_per_day": ICT_MAX_TRADES_PER_DAY,
            "max_consecutive_losses": ICT_MAX_CONSECUTIVE_LOSSES,
            "min_rr": ICT_MIN_RR,
            "sessions": sorted(selected_sessions),
            "sessions_ct": ict_session_windows(selected_sessions),
        },
        "sampling": {"checked_bars": checked, "skipped": skipped},
        "metrics": {
            "total_signals": len(trades),
            "win_rate": round(len(wins) / len(trades) * 100, 2),
            "net_r": round(sum(item["r"] for item in trades), 2),
            "avg_r": round(sum(item["r"] for item in trades) / len(trades), 3),
            "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
            "net_pnl_usd": round(total_pnl, 2),
            "return_pct": round((equity / 10_000 - 1) * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "final_equity": round(equity, 2),
            "by_session": by_session,
        },
        "trades": trades[-200:],
        "disclaimer": "Research backtest only. Uses GC=F Yahoo candles, not broker XAUUSD bid/ask execution.",
    }


def mt5_import():
    try:
        import MetaTrader5 as mt5
        return mt5, None
    except Exception as e:
        return None, f"MetaTrader5 Python package is not available: {compact_error(e)}"


def mt5_connect(mt5) -> tuple[bool, str | None]:
    kwargs = {}
    if MT5_LOGIN:
        try:
            kwargs["login"] = int(MT5_LOGIN)
        except Exception:
            return False, "MT5_LOGIN must be numeric."
    if MT5_PASSWORD:
        kwargs["password"] = MT5_PASSWORD
    if MT5_SERVER:
        kwargs["server"] = MT5_SERVER
    try:
        initialized = mt5.initialize(MT5_TERMINAL_PATH, **kwargs) if MT5_TERMINAL_PATH else mt5.initialize(**kwargs)
    except Exception as e:
        return False, f"MT5 initialize error: {compact_error(e)}"
    if not initialized:
        return False, f"MT5 initialize failed: {mt5.last_error()}"
    return True, None


def mt5_account_payload(account) -> dict:
    if not account:
        return {}
    return {
        "login": getattr(account, "login", None),
        "server": getattr(account, "server", None),
        "name": getattr(account, "name", None),
        "currency": getattr(account, "currency", None),
        "balance": round(safe_float(getattr(account, "balance", 0)), 2),
        "equity": round(safe_float(getattr(account, "equity", 0)), 2),
        "margin_free": round(safe_float(getattr(account, "margin_free", 0)), 2),
        "trade_mode": getattr(account, "trade_mode", None),
        "leverage": getattr(account, "leverage", None),
    }


def mt5_symbol_payload(symbol_info) -> dict:
    if not symbol_info:
        return {}
    return {
        "symbol": getattr(symbol_info, "name", MT5_SYMBOL),
        "digits": getattr(symbol_info, "digits", None),
        "point": getattr(symbol_info, "point", None),
        "contract_size": getattr(symbol_info, "trade_contract_size", None),
        "volume_min": getattr(symbol_info, "volume_min", None),
        "volume_step": getattr(symbol_info, "volume_step", None),
        "volume_max": getattr(symbol_info, "volume_max", None),
        "spread": getattr(symbol_info, "spread", None),
        "trade_mode": getattr(symbol_info, "trade_mode", None),
    }


def mt5_runtime_status(include_account: bool = False) -> dict:
    status = {
        "autotrade_enabled": MT5_AUTOTRADE_ENABLED,
        "manual_execution_enabled": MT5_MANUAL_EXECUTION_ENABLED,
        "demo_mode": MT5_DEMO_MODE,
        "symbol": MT5_SYMBOL,
        "risk_per_trade_pct": MT5_RISK_PER_TRADE_PCT,
        "max_volume": MT5_MAX_VOLUME,
        "magic": MT5_MAGIC,
        "check_seconds": MT5_CHECK_SECONDS,
        "max_entry_drift_usd": MT5_MAX_ENTRY_DRIFT_USD,
        "session": "LONDON_OPEN_CT",
        "session_window_ct": ict_session_windows(ICT_LONDON_ONLY_SESSIONS),
        "module_available": False,
        "connected": False,
    }
    mt5, err = mt5_import()
    if err:
        return {**status, "error": err}
    status["module_available"] = True
    if not include_account:
        return status
    ok, connect_err = mt5_connect(mt5)
    if not ok:
        return {**status, "error": connect_err}
    account = mt5.account_info()
    symbol_info = mt5.symbol_info(MT5_SYMBOL)
    if symbol_info is None:
        mt5.symbol_select(MT5_SYMBOL, True)
        symbol_info = mt5.symbol_info(MT5_SYMBOL)
    return {
        **status,
        "connected": True,
        "account": mt5_account_payload(account),
        "symbol_info": mt5_symbol_payload(symbol_info),
        "last_error": mt5.last_error(),
    }


def mt5_london_order_key(signal: dict) -> str:
    trade = signal.get("trade") or {}
    bos = signal.get("bos") or {}
    return "|".join([
        str(signal.get("direction")),
        str(bos.get("bos_time")),
        str(trade.get("entry")),
        str(trade.get("stop")),
        str(trade.get("target")),
    ])


def mt5_london_signal_blockers(signal: dict) -> list[str]:
    trade = signal.get("trade") or {}
    blockers = []
    if signal.get("quality") != "CONFIRMED":
        blockers.append(f"London strategy is not confirmed: {signal.get('quality', 'WAIT')}.")
    if signal.get("session") != "LONDON_OPEN_CT":
        blockers.append(f"Current signal session is {signal.get('session', 'OFF_SESSION')}; automation is London-only.")
    if signal.get("direction") not in ("LONG", "SHORT"):
        blockers.append("No LONG/SHORT direction.")
    if trade.get("entry") is None or trade.get("stop") is None or trade.get("target") is None:
        blockers.append("Entry, stop, or target is missing.")
    if trade.get("rr") is None or safe_float(trade.get("rr")) < ICT_MIN_RR:
        blockers.append(f"Reward/risk is below 1:{ICT_MIN_RR:.0f}.")
    if signal.get("failed_checks"):
        blockers.append(f"Failed checks: {', '.join(signal.get('failed_checks') or [])}.")
    return blockers


def mt5_daily_guardrail(mt5, symbol: str) -> dict:
    now = datetime.now(timezone.utc)
    ct_now = now.astimezone(ZoneInfo("America/Chicago"))
    ct_start = ct_now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = ct_start.astimezone(timezone.utc)
    deals = mt5.history_deals_get(start, now)
    if deals is None:
        return {
            "allowed": False,
            "trades_today": None,
            "consecutive_losses": None,
            "reason": f"Could not read MT5 deal history for daily guardrails: {mt5.last_error()}",
        }
    filtered = [
        deal for deal in deals
        if getattr(deal, "symbol", "") == symbol and int(getattr(deal, "magic", 0) or 0) == MT5_MAGIC
    ]
    entry_in = getattr(mt5, "DEAL_ENTRY_IN", 0)
    entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)
    entries = [deal for deal in filtered if getattr(deal, "entry", None) == entry_in]
    closed = sorted(
        [deal for deal in filtered if getattr(deal, "entry", None) == entry_out],
        key=lambda deal: getattr(deal, "time", 0),
    )
    consecutive_losses = 0
    for deal in reversed(closed):
        profit = safe_float(getattr(deal, "profit", 0)) + safe_float(getattr(deal, "swap", 0)) + safe_float(getattr(deal, "commission", 0))
        if profit < 0:
            consecutive_losses += 1
        else:
            break
    allowed = len(entries) < ICT_MAX_TRADES_PER_DAY and consecutive_losses < ICT_MAX_CONSECUTIVE_LOSSES
    return {
        "allowed": allowed,
        "trades_today": len(entries),
        "consecutive_losses": consecutive_losses,
        "max_trades_per_day": ICT_MAX_TRADES_PER_DAY,
        "max_consecutive_losses": ICT_MAX_CONSECUTIVE_LOSSES,
        "reason": "Daily guardrails clear." if allowed else "Daily trade or loss limit reached.",
    }


def mt5_volume_for_risk(mt5, symbol: str, direction: str, entry: float, stop: float, balance: float, symbol_info) -> dict:
    risk_usd = balance * clamp(MT5_RISK_PER_TRADE_PCT / 100, 0.001, 0.05)
    order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL
    per_lot_loss = None
    try:
        calc = mt5.order_calc_profit(order_type, symbol, 1.0, entry, stop)
        if calc is not None and abs(safe_float(calc)) > 0:
            per_lot_loss = abs(safe_float(calc))
    except Exception:
        per_lot_loss = None
    if not per_lot_loss:
        contract_size = safe_float(getattr(symbol_info, "trade_contract_size", 100), 100)
        per_lot_loss = abs(entry - stop) * contract_size
    raw_volume = risk_usd / max(per_lot_loss, 1e-9)
    min_volume = safe_float(getattr(symbol_info, "volume_min", 0.01), 0.01)
    step = safe_float(getattr(symbol_info, "volume_step", 0.01), 0.01)
    broker_max = safe_float(getattr(symbol_info, "volume_max", MT5_MAX_VOLUME), MT5_MAX_VOLUME)
    capped_max = max(min_volume, min(broker_max, MT5_MAX_VOLUME))
    steps = int(max(raw_volume, 0) / max(step, 1e-9))
    volume = round(max(min_volume, min(capped_max, steps * step)), 6)
    return {
        "risk_usd": round(risk_usd, 2),
        "risk_per_trade_pct": MT5_RISK_PER_TRADE_PCT,
        "loss_per_lot": round(per_lot_loss, 2),
        "raw_volume": round(raw_volume, 6),
        "volume": volume,
        "volume_min": min_volume,
        "volume_step": step,
        "volume_max": capped_max,
    }


def mt5_execute_london_strategy(dry_run: bool = True, force_signal: bool = True, source: str = "manual") -> dict:
    with MT5_ORDER_LOCK:
        context = get_market_context(force=force_signal)
        signal = ict_london_bos_signal(context, force=force_signal)
        trade = signal.get("trade") or {}
        blockers = mt5_london_signal_blockers(signal)
        if source == "worker" and not MT5_AUTOTRADE_ENABLED:
            blockers.append("MT5_AUTOTRADE_ENABLED is false.")
        if source == "manual" and not dry_run and not MT5_MANUAL_EXECUTION_ENABLED:
            blockers.append("MT5_MANUAL_EXECUTION_ENABLED is false.")
        if blockers:
            return {
                "ok": False,
                "dry_run": dry_run,
                "source": source,
                "blocked": True,
                "blockers": blockers,
                "signal": signal,
                "mt5": mt5_runtime_status(include_account=False),
            }

        mt5, err = mt5_import()
        if err:
            return {"ok": False, "dry_run": dry_run, "source": source, "blocked": True, "blockers": [err], "signal": signal}
        ok, connect_err = mt5_connect(mt5)
        if not ok:
            return {"ok": False, "dry_run": dry_run, "source": source, "blocked": True, "blockers": [connect_err], "signal": signal}

        account = mt5.account_info()
        if not account:
            return {"ok": False, "dry_run": dry_run, "source": source, "blocked": True, "blockers": [f"MT5 account unavailable: {mt5.last_error()}"], "signal": signal}
        real_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2)
        if MT5_DEMO_MODE and getattr(account, "trade_mode", None) == real_mode:
            return {
                "ok": False,
                "dry_run": dry_run,
                "source": source,
                "blocked": True,
                "blockers": ["MT5_DEMO_MODE is true and the connected account is real. Demo account required."],
                "account": mt5_account_payload(account),
                "signal": signal,
            }

        if not mt5.symbol_select(MT5_SYMBOL, True):
            return {"ok": False, "dry_run": dry_run, "source": source, "blocked": True, "blockers": [f"Could not select {MT5_SYMBOL}: {mt5.last_error()}"], "signal": signal}
        symbol_info = mt5.symbol_info(MT5_SYMBOL)
        tick = mt5.symbol_info_tick(MT5_SYMBOL)
        if not symbol_info or not tick:
            return {"ok": False, "dry_run": dry_run, "source": source, "blocked": True, "blockers": [f"Missing MT5 symbol/tick for {MT5_SYMBOL}."], "signal": signal}

        direction = signal["direction"]
        order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL
        price = safe_float(getattr(tick, "ask", 0)) if direction == "LONG" else safe_float(getattr(tick, "bid", 0))
        entry = safe_float(trade.get("entry"))
        stop = safe_float(trade.get("stop"))
        target = safe_float(trade.get("target"))
        drift = abs(price - entry)
        execution_blockers = []
        if drift > MT5_MAX_ENTRY_DRIFT_USD:
            execution_blockers.append(f"Live MT5 price drift ${drift:.2f} is above ${MT5_MAX_ENTRY_DRIFT_USD:.2f}.")
        if direction == "LONG" and not (stop < price < target):
            execution_blockers.append("LONG stop/target are not valid around the live execution price.")
        if direction == "SHORT" and not (target < price < stop):
            execution_blockers.append("SHORT stop/target are not valid around the live execution price.")
        positions = mt5.positions_get(symbol=MT5_SYMBOL)
        if positions is None:
            execution_blockers.append(f"Could not read open positions: {mt5.last_error()}")
        else:
            matching = [
                pos for pos in positions
                if int(getattr(pos, "magic", 0) or 0) == MT5_MAGIC and getattr(pos, "symbol", "") == MT5_SYMBOL
            ]
            if matching:
                execution_blockers.append(f"Existing {MT5_SYMBOL} position with magic {MT5_MAGIC} is already open.")
        order_key = mt5_london_order_key(signal)
        if MT5_LAST_ORDER_KEY.get("key") == order_key:
            execution_blockers.append("This London signal was already sent in the current app session.")
        guardrail = mt5_daily_guardrail(mt5, MT5_SYMBOL)
        if not guardrail.get("allowed"):
            execution_blockers.append(guardrail.get("reason", "Daily guardrail blocked the order."))
        if execution_blockers:
            return {
                "ok": False,
                "dry_run": dry_run,
                "source": source,
                "blocked": True,
                "blockers": execution_blockers,
                "signal": signal,
                "account": mt5_account_payload(account),
                "symbol_info": mt5_symbol_payload(symbol_info),
                "guardrail": guardrail,
                "live_price": round(price, 2),
                "entry_drift": round(drift, 2),
            }

        sizing = mt5_volume_for_risk(mt5, MT5_SYMBOL, direction, price, stop, safe_float(getattr(account, "balance", 0)), symbol_info)
        digits = int(getattr(symbol_info, "digits", 2) or 2)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": MT5_SYMBOL,
            "volume": sizing["volume"],
            "type": order_type,
            "price": round(price, digits),
            "sl": round(stop, digits),
            "tp": round(target, digits),
            "deviation": MT5_DEVIATION_POINTS,
            "magic": MT5_MAGIC,
            "comment": MT5_ORDER_COMMENT[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": getattr(mt5, "ORDER_FILLING_IOC", 1),
        }
        payload = {
            "ok": True,
            "dry_run": dry_run,
            "source": source,
            "blocked": False,
            "signal": signal,
            "account": mt5_account_payload(account),
            "symbol_info": mt5_symbol_payload(symbol_info),
            "guardrail": guardrail,
            "live_price": round(price, 2),
            "entry_drift": round(drift, 2),
            "sizing": sizing,
            "order_request": request,
        }
        if dry_run:
            return payload

        result = mt5.order_send(request)
        if result is None:
            return {**payload, "ok": False, "blocked": True, "blockers": [f"MT5 order_send returned None: {mt5.last_error()}"]}
        retcode = getattr(result, "retcode", None)
        accepted = retcode in {getattr(mt5, "TRADE_RETCODE_DONE", 10009), getattr(mt5, "TRADE_RETCODE_PLACED", 10008)}
        result_payload = result._asdict() if hasattr(result, "_asdict") else {"retcode": retcode, "raw": str(result)}
        if accepted:
            MT5_LAST_ORDER_KEY.update({
                "key": order_key,
                "ts": utc_now_iso(),
                "ticket": result_payload.get("order") or result_payload.get("deal"),
            })
        return {
            **payload,
            "ok": accepted,
            "blocked": not accepted,
            "mt5_result": result_payload,
            "blockers": [] if accepted else [f"MT5 rejected order with retcode {retcode}."],
        }


async def mt5_london_worker(check_seconds: int):
    await asyncio.sleep(max(1, MT5_AUTO_STARTUP_DELAY_SECONDS))
    while True:
        try:
            result = await asyncio.to_thread(mt5_execute_london_strategy, False, False, "worker")
            signal = result.get("signal") or {}
            if result.get("ok"):
                log.info(f"[MT5] London demo order sent: {result.get('mt5_result')}")
            else:
                blockers = result.get("blockers") or []
                infrastructure_blocked = any(
                    "MT5" in item or "MetaTrader5" in item or "account" in item.lower()
                    for item in blockers
                )
                if signal.get("quality") == "CONFIRMED" or infrastructure_blocked:
                    log.info(f"[MT5] London automation blocked: {blockers}")
        except Exception as e:
            log.error(f"[MT5] London automation error: {e}")
        await asyncio.sleep(max(10, check_seconds))


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def parse_trade_time(value) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        return parse_utc(text.replace(" UTC", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
        except Exception:
            return None


def format_trade_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def signal_valid_until_text(opened_at) -> str:
    opened = parse_trade_time(opened_at) or datetime.now(timezone.utc)
    return format_trade_time(opened + timedelta(minutes=SIGNAL_VALID_MINUTES))


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
    zone = final.get("zone") or (final.get("stacked_setup") or {}).get("zone") or {}
    h1 = snapshot.get("h1", {})
    m15 = snapshot.get("m15", {})
    m5 = snapshot.get("m5", {})
    execution_timeframe = final.get("execution_timeframe") or "15M"
    execution_data = snapshot_execution_data(snapshot, execution_timeframe)
    opened_at = snapshot.get("timestamp") or utc_now_text()
    return {
        "id": trade_record_id(alert_key),
        "alert_key": alert_key,
        "source": source,
        "status": "OPEN",
        "result": "OPEN",
        "direction": final.get("direction"),
        "setup_type": final.get("setup_type") or (final.get("stacked_setup") or {}).get("setup_type"),
        "quality": final.get("quality"),
        "execution_timeframe": execution_timeframe,
        "opened_at": opened_at,
        "opened_bar": sweep.get("bar_time") or execution_data.get("bar_time"),
        "valid_until": signal_valid_until_text(opened_at),
        "invalidated_at": None,
        "invalidation_reason": None,
        "user_action": "PENDING",
        "user_action_at": None,
        "user_notes": None,
        "entry_low": trade.get("entry_low"),
        "entry_high": trade.get("entry_high"),
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
        "zone_low": zone.get("low"),
        "zone_high": zone.get("high"),
        "zone_score": zone.get("score"),
        "zone_summary": zone.get("summary"),
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
            upsert_signal_to_supabase(trade)
            return trade, False

    record = build_trade_record(snapshot, alert_key, source=source)
    history["trades"].append(record)
    save_trade_history(history)
    upsert_signal_to_supabase(record)
    return record, True


def london_signal_executable(signal: dict) -> bool:
    trade = signal.get("trade") or {}
    return (
        signal.get("quality") == "CONFIRMED"
        and signal.get("direction") in ("LONG", "SHORT")
        and signal.get("session") == "LONDON_OPEN_CT"
        and trade.get("entry") is not None
        and trade.get("stop") is not None
        and trade.get("target") is not None
        and all((signal.get("checks") or {}).values())
    )


def london_alert_key(signal: dict) -> str | None:
    if not london_signal_executable(signal):
        return None
    trade = signal.get("trade") or {}
    bos = signal.get("bos") or {}
    return "LONDON_BOS:" + "|".join([
        str(signal.get("direction")),
        str(bos.get("bos_time") or signal.get("timestamp")),
        str(trade.get("entry")),
        str(trade.get("stop")),
        str(trade.get("target")),
    ])


def build_london_trade_record(signal: dict, alert_key: str, source: str = "london_telegram_alert") -> dict:
    trade = signal.get("trade") or {}
    zone = signal.get("zone") or {}
    bos = signal.get("bos") or {}
    bias = signal.get("bias") or {}
    opened_at = signal.get("timestamp") or utc_now_text()
    return {
        "id": trade_record_id(alert_key),
        "alert_key": alert_key,
        "source": source,
        "status": "OPEN",
        "result": "OPEN",
        "direction": signal.get("direction"),
        "setup_type": "LONDON_BOS_RETRACEMENT",
        "quality": signal.get("quality"),
        "execution_timeframe": "5M",
        "opened_at": opened_at,
        "opened_bar": bos.get("bos_time"),
        "valid_until": signal_valid_until_text(opened_at),
        "invalidated_at": None,
        "invalidation_reason": None,
        "user_action": "PENDING",
        "user_action_at": None,
        "user_notes": None,
        "entry_low": trade.get("entry_low"),
        "entry_high": trade.get("entry_high"),
        "entry": trade.get("entry"),
        "stop": trade.get("stop"),
        "target": trade.get("target"),
        "target_2": None,
        "risk": trade.get("risk"),
        "reward": trade.get("reward"),
        "rr": trade.get("rr"),
        "size_1pct_10k_oz": trade.get("position_1pct_10k_oz"),
        "weighted_score": None,
        "confidence": None,
        "h1_score": None,
        "m15_score": None,
        "m5_score": None,
        "sweep_summary": bos.get("reason"),
        "zone_low": zone.get("low"),
        "zone_high": zone.get("high"),
        "zone_score": zone.get("score"),
        "zone_summary": zone.get("summary"),
        "reason": signal.get("reason") or (bias.get("reason") if isinstance(bias, dict) else None),
        "closed_at": None,
        "closed_bar": None,
        "closed_price": None,
        "observed_price": None,
        "observed_high": None,
        "observed_low": None,
        "r_multiple": None,
        "pnl_1pct_10k_usd": None,
    }


def record_london_trade_alert(signal: dict, source: str = "london_telegram_alert") -> tuple[dict | None, bool]:
    alert_key = london_alert_key(signal)
    if not alert_key:
        return None, False

    history = load_trade_history()
    for trade in history["trades"]:
        if trade.get("alert_key") == alert_key:
            upsert_signal_to_supabase(trade)
            return trade, False

    record = build_london_trade_record(signal, alert_key, source=source)
    history["trades"].append(record)
    save_trade_history(history)
    upsert_signal_to_supabase(record)
    return record, True


def close_trade_record(trade: dict, result: str, snapshot: dict, reason: str | None = None) -> dict:
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
    if result == "TP":
        closed_price = target
    elif result == "SL":
        closed_price = stop
    else:
        closed_price = price
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
    if reason:
        trade["invalidation_reason"] = reason
    if result in {"EXPIRED", "INVALIDATED"}:
        trade["invalidated_at"] = trade["closed_at"]
    upsert_signal_to_supabase(trade)
    insert_signal_outcome_to_supabase(trade)
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


def trade_expiration_reason(trade: dict, snapshot: dict) -> str | None:
    if trade.get("status") != "OPEN":
        return None
    valid_until = parse_trade_time(trade.get("valid_until"))
    if not valid_until:
        return None
    now = snapshot_time(snapshot)
    if now >= valid_until:
        return f"Signal expired at {format_trade_time(valid_until)} after {SIGNAL_VALID_MINUTES} minutes."
    return None


def trade_zone_invalidation_reason(trade: dict, snapshot: dict) -> str | None:
    if trade.get("status") != "OPEN":
        return None
    direction = trade.get("direction")
    zone_low = safe_float(trade.get("zone_low"))
    zone_high = safe_float(trade.get("zone_high"))
    if direction not in ("LONG", "SHORT") or zone_low <= 0 or zone_high <= 0:
        return None

    m15 = snapshot.get("m15", {})
    m5 = snapshot.get("m5", {})
    m15_close = safe_float(m15.get("price"))
    m5_price = safe_float(snapshot.get("price") or m5.get("price"))
    atr_value = max(
        safe_float(m15.get("indicators", {}).get("atr14")),
        safe_float(m5.get("indicators", {}).get("atr14")),
        0.01,
    )
    buffer = atr_value * SIGNAL_INVALIDATION_ATR_BUFFER

    if direction == "LONG":
        if m15_close and m15_close < zone_low - buffer:
            return f"15M close {m15_close:.2f} invalidated demand zone below {zone_low:.2f}."
        if m5_price and m5_price < zone_low - buffer:
            return f"5M price {m5_price:.2f} invalidated demand zone below {zone_low:.2f}."
    else:
        if m15_close and m15_close > zone_high + buffer:
            return f"15M close {m15_close:.2f} invalidated supply zone above {zone_high:.2f}."
        if m5_price and m5_price > zone_high + buffer:
            return f"5M price {m5_price:.2f} invalidated supply zone above {zone_high:.2f}."
    return None


def update_trade_results(snapshot: dict) -> list[dict]:
    history = load_trade_history()
    updated = []
    for trade in history["trades"]:
        result = trade_hit_result(trade, snapshot)
        if result:
            updated.append(close_trade_record(trade, result, snapshot))
            continue
        invalidation_reason = trade_zone_invalidation_reason(trade, snapshot)
        if invalidation_reason:
            updated.append(close_trade_record(trade, "INVALIDATED", snapshot, invalidation_reason))
            continue
        expiration_reason = trade_expiration_reason(trade, snapshot)
        if expiration_reason:
            updated.append(close_trade_record(trade, "EXPIRED", snapshot, expiration_reason))

    if updated:
        save_trade_history(history)
    return updated


def mark_trade_action(trade_id: str, action: str, notes: str | None = None) -> tuple[dict | None, str | None]:
    action = str(action or "").upper()
    if action not in {"TAKEN", "SKIPPED", "PENDING"}:
        return None, "Action must be TAKEN, SKIPPED, or PENDING."

    history = load_trade_history()
    for trade in history["trades"]:
        if trade.get("id") == trade_id or trade.get("alert_key") == trade_id:
            trade["user_action"] = action
            trade["user_action_at"] = utc_now_iso()
            trade["user_notes"] = notes
            save_trade_history(history)
            upsert_signal_to_supabase(trade)
            return trade, None
    return None, "Trade not found."


def trade_history_summary(trades: list[dict]) -> dict:
    closed = [trade for trade in trades if trade.get("status") == "CLOSED"]
    wins = [trade for trade in closed if trade.get("result") == "TP"]
    losses = [trade for trade in closed if trade.get("result") == "SL"]
    expired = [trade for trade in closed if trade.get("result") == "EXPIRED"]
    invalidated = [trade for trade in closed if trade.get("result") == "INVALIDATED"]
    taken = [trade for trade in trades if trade.get("user_action") == "TAKEN"]
    skipped = [trade for trade in trades if trade.get("user_action") == "SKIPPED"]
    pending = [trade for trade in trades if (trade.get("user_action") or "PENDING") == "PENDING"]
    pnl = sum(safe_float(trade.get("pnl_1pct_10k_usd")) for trade in closed)
    r_total = sum(safe_float(trade.get("r_multiple")) for trade in closed)
    return {
        "total": len(trades),
        "open": len([trade for trade in trades if trade.get("status") == "OPEN"]),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "expired": len(expired),
        "invalidated": len(invalidated),
        "taken": len(taken),
        "skipped": len(skipped),
        "pending": len(pending),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "net_r": round(r_total, 2),
        "pnl_1pct_10k_usd": round(pnl, 2),
    }


def trade_time_value(trade: dict) -> datetime:
    for key in ("closed_at", "opened_at", "generated_at"):
        value = trade.get(key)
        if not value:
            continue
        try:
            return parse_utc(str(value).replace(" UTC", "+00:00"))
        except Exception:
            try:
                return datetime.strptime(str(value), "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return datetime.min.replace(tzinfo=timezone.utc)


def supabase_signal_to_trade(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "alert_key": row.get("alert_key"),
        "source": row.get("source"),
        "status": row.get("status"),
        "result": row.get("result"),
        "direction": row.get("direction"),
        "setup_type": row.get("setup_type"),
        "quality": row.get("quality"),
        "execution_timeframe": row.get("execution_timeframe"),
        "opened_at": row.get("generated_at"),
        "opened_bar": row.get("opened_bar"),
        "valid_until": row.get("valid_until"),
        "invalidated_at": row.get("invalidated_at"),
        "invalidation_reason": row.get("invalidation_reason"),
        "user_action": row.get("user_action"),
        "user_action_at": row.get("user_action_at"),
        "user_notes": row.get("user_notes"),
        "entry_low": row.get("entry_low"),
        "entry_high": row.get("entry_high"),
        "entry": row.get("entry"),
        "stop": row.get("stop_loss"),
        "target": row.get("take_profit_1"),
        "target_2": row.get("take_profit_2"),
        "risk": row.get("risk"),
        "reward": row.get("reward"),
        "rr": row.get("risk_reward"),
        "weighted_score": row.get("weighted_score"),
        "confidence": row.get("confidence"),
        "h1_score": row.get("h1_score"),
        "m15_score": row.get("m15_score"),
        "m5_score": row.get("m5_score"),
        "sweep_summary": row.get("sweep_summary"),
        "zone_low": row.get("zone_low"),
        "zone_high": row.get("zone_high"),
        "zone_score": row.get("zone_score"),
        "reason": row.get("reason"),
        "closed_at": row.get("closed_at"),
        "closed_bar": row.get("closed_bar"),
        "closed_price": row.get("closed_price"),
        "observed_price": row.get("observed_price"),
        "observed_high": row.get("observed_high"),
        "observed_low": row.get("observed_low"),
        "r_multiple": row.get("r_multiple"),
        "pnl_1pct_10k_usd": row.get("pnl_1pct_10k_usd"),
    }


def group_performance(trades: list[dict], key: str) -> dict:
    groups = {}
    for trade in trades:
        label = str(trade.get(key) or "UNKNOWN")
        groups.setdefault(label, []).append(trade)
    return {label: performance_metrics(items, include_breakdowns=False) for label, items in groups.items()}


def max_losing_streak(closed: list[dict]) -> int:
    streak = 0
    max_streak = 0
    for trade in closed:
        if safe_float(trade.get("r_multiple")) < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def equity_stats(closed: list[dict]) -> dict:
    equity = []
    running = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in closed:
        running += safe_float(trade.get("r_multiple"))
        peak = max(peak, running)
        max_drawdown = min(max_drawdown, running - peak)
        equity.append({
            "time": trade_time_value(trade).strftime("%Y-%m-%d %H:%M UTC"),
            "r": round(running, 2),
        })
    return {
        "equity_curve": equity[-200:],
        "max_drawdown_r": round(max_drawdown, 2),
    }


def performance_metrics(trades: list[dict], include_breakdowns: bool = True) -> dict:
    closed = sorted(
        [trade for trade in trades if trade.get("status") == "CLOSED" and trade.get("r_multiple") is not None],
        key=trade_time_value,
    )
    wins = [trade for trade in closed if safe_float(trade.get("r_multiple")) > 0]
    losses = [trade for trade in closed if safe_float(trade.get("r_multiple")) < 0]
    open_trades = [trade for trade in trades if trade.get("status") == "OPEN"]
    gross_win_r = sum(safe_float(trade.get("r_multiple")) for trade in wins)
    gross_loss_r = abs(sum(safe_float(trade.get("r_multiple")) for trade in losses))
    net_r = sum(safe_float(trade.get("r_multiple")) for trade in closed)
    monthly_counts = {}
    result_counts = {}
    for trade in trades:
        month = trade_time_value(trade).strftime("%Y-%m")
        monthly_counts[month] = monthly_counts.get(month, 0) + 1
        result = trade.get("result") or "UNKNOWN"
        result_counts[result] = result_counts.get(result, 0) + 1

    result = {
        "total_signals": len(trades),
        "open": len(open_trades),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "avg_r": round(net_r / len(closed), 2) if closed else 0,
        "net_r": round(net_r, 2),
        "gross_win_r": round(gross_win_r, 2),
        "gross_loss_r": round(gross_loss_r, 2),
        "profit_factor": round(gross_win_r / gross_loss_r, 2) if gross_loss_r else None,
        "max_losing_streak": max_losing_streak(closed),
        "monthly_signal_count": dict(sorted(monthly_counts.items())),
        "result_count": dict(sorted(result_counts.items())),
        **equity_stats(closed),
    }
    if include_breakdowns:
        result["breakdowns"] = {
            "timeframe": group_performance(trades, "execution_timeframe"),
            "direction": group_performance(trades, "direction"),
            "quality": group_performance(trades, "quality"),
        }
    return result


def resolve_analytics_trades(source: str = "auto", limit: int = 500) -> tuple[list[dict], str]:
    supabase_rows = fetch_supabase_signals(limit) if source in ("auto", "supabase") else []
    if source == "supabase":
        trades = [supabase_signal_to_trade(row) for row in supabase_rows]
        resolved_source = "supabase"
    elif source == "auto" and supabase_rows:
        trades = [supabase_signal_to_trade(row) for row in supabase_rows]
        resolved_source = "supabase"
    else:
        trades = load_trade_history()["trades"][-max(1, min(limit, 2000)):]
        resolved_source = "local_json"
    return trades, resolved_source


def analytics_payload(source: str = "auto", limit: int = 500) -> dict:
    trades, resolved_source = resolve_analytics_trades(source, limit)
    return {
        "source": resolved_source,
        "supabase_configured": supabase_configured(),
        "metrics": performance_metrics(trades),
        "sample_size": len(trades),
    }


def closed_trades(trades: list[dict]) -> list[dict]:
    return sorted(
        [trade for trade in trades if trade.get("status") == "CLOSED" and trade.get("r_multiple") is not None],
        key=trade_time_value,
    )


def trades_since(trades: list[dict], days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return [trade for trade in trades if trade_time_value(trade) >= cutoff]


def gate_result(
    name: str,
    passed: bool,
    severity: str,
    metric,
    threshold,
    detail: str,
) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "metric": metric,
        "threshold": threshold,
        "detail": detail,
    }


def score_distribution(trades: list[dict]) -> dict:
    confidence = {}
    direction = {}
    result = {}
    h1_scores = []
    m15_scores = []
    m5_scores = []
    for trade in trades:
        confidence[confidence_tier(trade.get("confidence"))] = confidence.get(confidence_tier(trade.get("confidence")), 0) + 1
        direction[trade.get("direction") or "UNKNOWN"] = direction.get(trade.get("direction") or "UNKNOWN", 0) + 1
        result[trade.get("result") or "UNKNOWN"] = result.get(trade.get("result") or "UNKNOWN", 0) + 1
        for key, target in (("h1_score", h1_scores), ("m15_score", m15_scores), ("m5_score", m5_scores)):
            value = trade.get(key)
            if value is not None:
                target.append(safe_float(value))

    def stats(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "avg": None, "min": None, "max": None}
        return {
            "count": len(values),
            "avg": round(sum(values) / len(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    return {
        "confidence_tier": dict(sorted(confidence.items())),
        "direction": dict(sorted(direction.items())),
        "result": dict(sorted(result.items())),
        "scores": {
            "h1": stats(h1_scores),
            "m15": stats(m15_scores),
            "m5": stats(m5_scores),
        },
    }


def stale_open_trades(trades: list[dict]) -> list[dict]:
    stale = []
    now = datetime.now(timezone.utc)
    for trade in trades:
        if trade.get("status") != "OPEN":
            continue
        valid_until = parse_trade_time(trade.get("valid_until"))
        if valid_until and valid_until < now:
            stale.append({
                "id": trade.get("id"),
                "direction": trade.get("direction"),
                "valid_until": format_trade_time(valid_until),
            })
    return stale


def current_model_snapshot() -> dict:
    try:
        snapshot = get_trade_snapshot(force_context=False)
        final = snapshot.get("final", {})
        hard_filters = snapshot.get("hard_filters", {})
        return {
            "ok": True,
            "timestamp": snapshot.get("timestamp"),
            "price": snapshot.get("price"),
            "decision": {
                "direction": final.get("direction"),
                "quality": final.get("quality"),
                "confidence": final.get("confidence"),
                "reason": final.get("reason"),
            },
            "scores": {
                "h1": snapshot.get("h1", {}).get("technical_score"),
                "m15": snapshot.get("m15", {}).get("technical_score"),
                "m5": snapshot.get("m5", {}).get("technical_score"),
            },
            "hard_filters": {
                "allowed": hard_filters.get("allowed"),
                "blocked_reasons": hard_filters.get("blocked_reasons"),
            },
            "data_sources": {
                "h1": snapshot.get("h1", {}).get("data_source"),
                "m15": snapshot.get("m15", {}).get("data_source"),
                "m5": snapshot.get("m5", {}).get("data_source"),
            },
        }
    except Exception as e:
        return {
            "ok": False,
            "error": compact_error(e),
        }


def model_health_payload(
    source: str = "auto",
    limit: int = 1000,
    recent_days: int = 30,
    refresh_current: bool = False,
) -> dict:
    limit = max(1, min(limit, 5000))
    recent_days = int(clamp(recent_days, 7, 180))
    trades, resolved_source = resolve_analytics_trades(source, limit)
    all_metrics = performance_metrics(trades)
    recent_trades = trades_since(trades, recent_days)
    recent_metrics = performance_metrics(recent_trades)
    closed = closed_trades(trades)
    recent_closed = closed_trades(recent_trades)
    stale_open = stale_open_trades(trades)
    result_count = all_metrics.get("result_count", {})
    closed_count = all_metrics.get("closed", 0)
    inactive_count = result_count.get("EXPIRED", 0) + result_count.get("INVALIDATED", 0)
    inactive_ratio = round(inactive_count / closed_count * 100, 1) if closed_count else 0

    gates = [
        gate_result(
            "closed_sample_size",
            closed_count >= GO_LIVE_MIN_CLOSED_SIGNALS,
            "critical",
            closed_count,
            f">= {GO_LIVE_MIN_CLOSED_SIGNALS}",
            f"{closed_count} closed signals available; target is {GO_LIVE_MIN_CLOSED_SIGNALS}+ before live capital decisions.",
        ),
        gate_result(
            "win_rate",
            all_metrics.get("win_rate", 0) >= GO_LIVE_MIN_WIN_RATE,
            "critical",
            all_metrics.get("win_rate", 0),
            f">= {GO_LIVE_MIN_WIN_RATE}%",
            f"Win rate is {all_metrics.get('win_rate', 0)}%; go-live target is {GO_LIVE_MIN_WIN_RATE}%.",
        ),
        gate_result(
            "average_r",
            all_metrics.get("avg_r", 0) >= GO_LIVE_MIN_AVG_R,
            "critical",
            all_metrics.get("avg_r", 0),
            f">= {GO_LIVE_MIN_AVG_R}R",
            f"Average R is {all_metrics.get('avg_r', 0)}; go-live target is {GO_LIVE_MIN_AVG_R}R.",
        ),
        gate_result(
            "profit_factor",
            (all_metrics.get("profit_factor") or 0) >= GO_LIVE_MIN_PROFIT_FACTOR,
            "critical",
            all_metrics.get("profit_factor"),
            f">= {GO_LIVE_MIN_PROFIT_FACTOR}",
            f"Profit factor is {all_metrics.get('profit_factor')}; target is {GO_LIVE_MIN_PROFIT_FACTOR}+.",
        ),
        gate_result(
            "losing_streak",
            all_metrics.get("max_losing_streak", 0) <= GO_LIVE_MAX_LOSING_STREAK,
            "critical",
            all_metrics.get("max_losing_streak", 0),
            f"<= {GO_LIVE_MAX_LOSING_STREAK}",
            f"Max losing streak is {all_metrics.get('max_losing_streak', 0)}; cap is {GO_LIVE_MAX_LOSING_STREAK}.",
        ),
        gate_result(
            "recent_sample",
            len(recent_closed) >= 10,
            "warning",
            len(recent_closed),
            ">= 10",
            f"{len(recent_closed)} closed signals in the last {recent_days} days.",
        ),
        gate_result(
            "recent_average_r",
            len(recent_closed) < 10 or recent_metrics.get("avg_r", 0) >= 0,
            "warning",
            recent_metrics.get("avg_r", 0),
            ">= 0R",
            f"Recent average R over {recent_days} days is {recent_metrics.get('avg_r', 0)}.",
        ),
        gate_result(
            "stale_open_signals",
            len(stale_open) == 0,
            "warning",
            len(stale_open),
            "0",
            f"{len(stale_open)} open signals are past valid_until and should be refreshed/closed.",
        ),
        gate_result(
            "expired_or_invalidated_ratio",
            inactive_ratio <= 60,
            "warning",
            f"{inactive_ratio}%",
            "<= 60%",
            f"{inactive_ratio}% of closed signals are expired/invalidated rather than TP/SL outcomes.",
        ),
    ]

    critical_failures = [gate for gate in gates if gate["severity"] == "critical" and not gate["passed"]]
    warnings = [gate for gate in gates if gate["severity"] == "warning" and not gate["passed"]]
    status = "ready" if not critical_failures else "blocked"
    if status == "ready" and warnings:
        status = "watch"

    next_actions = [gate["detail"] for gate in critical_failures + warnings]
    if closed_count < GO_LIVE_MIN_CLOSED_SIGNALS:
        next_actions.append("Keep collecting signals on demo/manual review until the closed-signal sample reaches the configured threshold.")
    if stale_open:
        next_actions.append("Run /api/trades?refresh=true or wait for the trade checker to close stale signals.")

    return {
        "status": status,
        "source": resolved_source,
        "generated_at": utc_now_iso(),
        "thresholds": {
            "min_closed_signals": GO_LIVE_MIN_CLOSED_SIGNALS,
            "min_win_rate": GO_LIVE_MIN_WIN_RATE,
            "min_avg_r": GO_LIVE_MIN_AVG_R,
            "min_profit_factor": GO_LIVE_MIN_PROFIT_FACTOR,
            "max_losing_streak": GO_LIVE_MAX_LOSING_STREAK,
        },
        "sample": {
            "total_signals": len(trades),
            "closed_signals": len(closed),
            "recent_days": recent_days,
            "recent_closed_signals": len(recent_closed),
        },
        "gates": gates,
        "metrics": {
            "all": all_metrics,
            "recent": recent_metrics,
        },
        "distribution": score_distribution(trades),
        "stale_open_signals": stale_open[:20],
        "current": current_model_snapshot() if refresh_current else None,
        "next_actions": next_actions,
    }


def confidence_tier(value) -> str:
    score = safe_float(value)
    if 0 < score <= 1:
        score *= 100
    if score >= 75:
        return "HIGH"
    if score >= 60:
        return "MEDIUM"
    if score >= 50:
        return "LOW"
    return "BELOW_THRESHOLD"


def signal_session(row: dict) -> str:
    value = row.get("generated_at") or row.get("opened_at")
    if not value:
        return "UNKNOWN"
    try:
        return utc_session_label(parse_utc(str(value).replace(" UTC", "+00:00")))
    except Exception:
        return "UNKNOWN"


def merge_signal_outcome(row: dict, outcome: dict | None = None) -> dict:
    trade = supabase_signal_to_trade(row)
    if outcome:
        trade["status"] = "CLOSED"
        trade["result"] = outcome.get("outcome") or trade.get("result")
        trade["closed_at"] = outcome.get("outcome_at") or trade.get("closed_at")
        trade["closed_price"] = outcome.get("outcome_price") or trade.get("closed_price")
        trade["r_multiple"] = outcome.get("pnl_r") if outcome.get("pnl_r") is not None else trade.get("r_multiple")
        trade["pnl_1pct_10k_usd"] = (
            round(safe_float(trade.get("r_multiple")) * 100, 2)
            if trade.get("r_multiple") is not None else trade.get("pnl_1pct_10k_usd")
        )
    trade["confidence_tier"] = confidence_tier(row.get("confidence"))
    trade["session"] = signal_session(row)
    return trade


def feedback_group_stats(items: list[dict]) -> dict:
    closed = [item for item in items if item.get("status") == "CLOSED" and item.get("r_multiple") is not None]
    taken = [item for item in items if item.get("user_action") == "TAKEN"]
    skipped = [item for item in items if item.get("user_action") == "SKIPPED"]
    pending = [item for item in items if item.get("user_action") == "PENDING"]
    closed_taken = [item for item in taken if item.get("status") == "CLOSED" and item.get("r_multiple") is not None]
    closed_skipped = [item for item in skipped if item.get("status") == "CLOSED" and item.get("r_multiple") is not None]

    def avg_r(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return round(sum(safe_float(item.get("r_multiple")) for item in rows) / len(rows), 2)

    def win_rate(rows: list[dict]) -> float:
        if not rows:
            return 0
        wins = [item for item in rows if safe_float(item.get("r_multiple")) > 0]
        return round(len(wins) / len(rows) * 100, 1)

    return {
        "signals": len(items),
        "taken": len(taken),
        "skipped": len(skipped),
        "pending": len(pending),
        "closed": len(closed),
        "avg_r_all": avg_r(closed),
        "avg_r_taken": avg_r(closed_taken),
        "avg_r_skipped_would_have": avg_r(closed_skipped),
        "win_rate_all": win_rate(closed),
        "win_rate_taken": win_rate(closed_taken),
        "win_rate_skipped_would_have": win_rate(closed_skipped),
        "net_r_taken": round(sum(safe_float(item.get("r_multiple")) for item in closed_taken), 2),
        "net_r_skipped_would_have": round(sum(safe_float(item.get("r_multiple")) for item in closed_skipped), 2),
    }


def feedback_breakdown(items: list[dict], key: str) -> dict:
    groups = {}
    for item in items:
        groups.setdefault(str(item.get(key) or "UNKNOWN"), []).append(item)
    return {
        label: feedback_group_stats(rows)
        for label, rows in sorted(groups.items())
    }


def feedback_recommendations(items: list[dict]) -> list[dict]:
    recommendations = []
    closed_taken_count = len([
        item for item in items
        if item.get("user_action") == "TAKEN"
        and item.get("status") == "CLOSED"
        and item.get("r_multiple") is not None
    ])
    if closed_taken_count < 50:
        recommendations.append({
            "type": "sample_size",
            "priority": "HIGH",
            "message": f"Only {closed_taken_count} closed taken signals. Do not tune aggressively before 50-100 outcomes.",
        })

    for setup, stats in feedback_breakdown(items, "setup_type").items():
        if stats["taken"] >= 5 and stats["avg_r_taken"] is not None and stats["avg_r_taken"] < 0:
            recommendations.append({
                "type": "reduce_exposure",
                "priority": "MEDIUM",
                "setup_type": setup,
                "message": f"{setup} is negative on taken trades ({stats['avg_r_taken']}R avg). Reduce or require extra confirmation.",
            })
        if stats["skipped"] >= 5 and stats["avg_r_skipped_would_have"] is not None and stats["avg_r_skipped_would_have"] >= 0.4:
            recommendations.append({
                "type": "review_skips",
                "priority": "MEDIUM",
                "setup_type": setup,
                "message": f"Skipped {setup} signals would have averaged {stats['avg_r_skipped_would_have']}R. Review skip criteria.",
            })
    return recommendations[:10]


def feedback_analytics_payload(user_id: str, limit: int = 500) -> dict:
    limit = max(1, min(limit, 5000))
    signal_rows = fetch_supabase_signals(limit)
    action_rows = fetch_supabase_user_actions(user_id, limit)
    outcome_rows = fetch_supabase_outcomes(limit)
    action_by_signal = {row.get("signal_id"): row for row in action_rows}
    outcome_by_signal = {row.get("signal_id"): row for row in outcome_rows}

    items = []
    for row in signal_rows:
        signal_id = row.get("id")
        action = action_by_signal.get(signal_id)
        item = merge_signal_outcome(row, outcome_by_signal.get(signal_id))
        item["user_action"] = (action or {}).get("action") or "PENDING"
        item["user_notes"] = (action or {}).get("notes")
        item["user_action_at"] = (action or {}).get("acted_at")
        items.append(item)

    return {
        "source": "supabase",
        "sample_size": len(items),
        "user_id": user_id,
        "summary": feedback_group_stats(items),
        "breakdowns": {
            "setup_type": feedback_breakdown(items, "setup_type"),
            "confidence_tier": feedback_breakdown(items, "confidence_tier"),
            "direction": feedback_breakdown(items, "direction"),
            "timeframe": feedback_breakdown(items, "execution_timeframe"),
            "session": feedback_breakdown(items, "session"),
        },
        "recommendations": feedback_recommendations(items),
        "recent": items[:100],
    }


def normalize_df_timezone(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def backtest_default_hold_bars(interval: str) -> int:
    if interval == "5m":
        return 288
    if interval == "15m":
        return 96
    return 48


def backtest_min_bars(interval: str) -> int:
    if interval == "5m":
        return 800
    if interval == "15m":
        return 400
    return 220


def backtest_trade_outcome(
    df: pd.DataFrame,
    ts: pd.Timestamp,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    max_hold_bars: int,
) -> dict:
    future = df.loc[df.index > ts].iloc[:max_hold_bars]
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if future.empty or risk <= 0:
        return {
            "status": "OPEN",
            "result": "OPEN",
            "closed_at": None,
            "closed_price": None,
            "r_multiple": None,
        }

    for bar_time, row in future.iterrows():
        high = safe_float(row["high"])
        low = safe_float(row["low"])
        if direction == "LONG":
            hit_sl = low <= stop
            hit_tp = high >= target
        else:
            hit_sl = high >= stop
            hit_tp = low <= target

        if hit_sl and hit_tp:
            return {
                "status": "CLOSED",
                "result": "SL",
                "closed_at": bar_time.strftime("%Y-%m-%d %H:%M UTC"),
                "closed_price": round(stop, 2),
                "r_multiple": -1.0,
            }
        if hit_tp:
            return {
                "status": "CLOSED",
                "result": "TP",
                "closed_at": bar_time.strftime("%Y-%m-%d %H:%M UTC"),
                "closed_price": round(target, 2),
                "r_multiple": round(reward / risk, 2),
            }
        if hit_sl:
            return {
                "status": "CLOSED",
                "result": "SL",
                "closed_at": bar_time.strftime("%Y-%m-%d %H:%M UTC"),
                "closed_price": round(stop, 2),
                "r_multiple": -1.0,
            }

    last = future.iloc[-1]
    close = safe_float(last["close"])
    pnl = close - entry if direction == "LONG" else entry - close
    return {
        "status": "CLOSED",
        "result": "EXPIRED",
        "closed_at": future.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
        "closed_price": round(close, 2),
        "r_multiple": round(pnl / risk, 2),
    }


def backtest_filter_allows(
    ts: pd.Timestamp,
    signal_data: dict,
    interval: str,
    direction: str,
    last_direction_time: dict[str, pd.Timestamp],
) -> tuple[bool, list[str]]:
    reasons = []
    now = ts.to_pydatetime()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    if utc_session_label(now) == "OFF_SESSION":
        reasons.append("off_session")
    if is_friday_close_window(now):
        reasons.append("friday_close")
    if is_sunday_open_window(now):
        reasons.append("sunday_open")
    if is_rollover_window(now):
        reasons.append("daily_rollover")

    indicators = signal_data.get("indicators", {})
    adx_value = safe_float(indicators.get("adx14"))
    atr_pct = safe_float(indicators.get("atr_pct"))
    if atr_pct > 0.95 and adx_value < 18:
        reasons.append("high_vol_choppy")

    last_time = last_direction_time.get(direction)
    if last_time is not None:
        minutes_since = (ts - last_time).total_seconds() / 60
        if minutes_since < RECENT_SIGNAL_COOLDOWN_MINUTES:
            reasons.append("recent_same_direction")

    return not reasons, reasons


def backtest_signal_record(
    interval: str,
    signal_data: dict,
    outcome: dict,
    filter_reasons: list[str],
) -> dict:
    trade = signal_data.get("trade", {})
    entry = safe_float(trade.get("entry"), safe_float(signal_data.get("price")))
    stop = safe_float(trade.get("stop"), safe_float(signal_data.get("stop")))
    target = safe_float(trade.get("target"), safe_float(signal_data.get("target")))
    risk = abs(entry - stop)
    reward = abs(target - entry)
    direction = signal_data.get("signal")
    alert_key = f"BACKTEST:{interval.upper()}:{direction}:{signal_data.get('bar_time')}"
    return {
        "id": trade_record_id(alert_key),
        "alert_key": alert_key,
        "source": "backtest",
        "status": outcome.get("status"),
        "result": outcome.get("result"),
        "direction": direction,
        "quality": "BACKTEST",
        "execution_timeframe": timeframe_label(interval),
        "opened_at": signal_data.get("bar_time"),
        "opened_bar": signal_data.get("bar_time"),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "target_2": trade.get("target_2"),
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "rr": round(reward / risk, 2) if risk else None,
        "weighted_score": signal_data.get("technical_score"),
        "confidence": abs(int(safe_float(signal_data.get("technical_score")))),
        "h1_score": None,
        "m15_score": signal_data.get("technical_score") if interval == "15m" else None,
        "m5_score": signal_data.get("technical_score") if interval == "5m" else None,
        "sweep_summary": signal_data.get("liquidity_sweep", {}).get("summary"),
        "reason": f"Rule-only {timeframe_label(interval)} replay. Filters skipped: {', '.join(filter_reasons) if filter_reasons else 'none'}.",
        "closed_at": outcome.get("closed_at"),
        "closed_price": outcome.get("closed_price"),
        "r_multiple": outcome.get("r_multiple"),
        "pnl_1pct_10k_usd": round(safe_float(outcome.get("r_multiple")) * 100, 2) if outcome.get("r_multiple") is not None else None,
    }


def run_rule_backtest(
    interval: str = "15m",
    days: int = 30,
    max_checks: int = 1200,
    max_hold_bars: int | None = None,
    include_filters: bool = True,
) -> dict:
    interval = interval.lower()
    if interval not in {"5m", "15m", "1h"}:
        return {"error": "interval must be one of 5m, 15m, 1h"}
    days = int(clamp(days, 1, 60))
    max_checks = int(clamp(max_checks, 50, 3000))
    max_hold_bars = max_hold_bars or backtest_default_hold_bars(interval)

    df, err = fetch_data(interval)
    if err or df is None:
        return {"error": err or "No data"}
    df = normalize_df_timezone(df)
    end = df.index[-1]
    start = end - pd.Timedelta(days=days)
    check_index = df.loc[df.index >= start].index
    min_bars = backtest_min_bars(interval)
    check_index = [ts for ts in check_index if df.index.get_loc(ts) >= min_bars]
    if len(check_index) > max_checks:
        step = int(np.ceil(len(check_index) / max_checks))
        check_index = check_index[::step]
    else:
        step = 1

    trades = []
    seen = set()
    skipped = {}
    last_direction_time = {}
    for ts in check_index:
        loc = df.index.get_loc(ts)
        history = df.iloc[:loc + 1]
        try:
            signal_data = compute_signal(history, interval)
        except Exception as e:
            skipped["compute_error"] = skipped.get("compute_error", 0) + 1
            log.warning(f"Backtest compute error {interval} {ts}: {e}")
            continue

        direction = signal_data.get("signal")
        trade = signal_data.get("trade", {})
        if direction not in ("LONG", "SHORT"):
            continue
        if trade.get("stop") is None or trade.get("target") is None:
            skipped["no_trade_levels"] = skipped.get("no_trade_levels", 0) + 1
            continue

        trigger_key = (direction, signal_data.get("bar_time"))
        if trigger_key in seen:
            continue
        seen.add(trigger_key)

        filter_reasons = []
        if include_filters:
            allowed, filter_reasons = backtest_filter_allows(ts, signal_data, interval, direction, last_direction_time)
            if not allowed:
                for reason in filter_reasons:
                    skipped[reason] = skipped.get(reason, 0) + 1
                continue

        entry = safe_float(trade.get("entry"), safe_float(signal_data.get("price")))
        stop = safe_float(trade.get("stop"))
        target = safe_float(trade.get("target"))
        outcome = backtest_trade_outcome(df, ts, direction, entry, stop, target, max_hold_bars)
        record = backtest_signal_record(interval, signal_data, outcome, filter_reasons)
        trades.append(record)
        last_direction_time[direction] = ts

    metrics = performance_metrics(trades)
    return {
        "mode": "rule_only_replay",
        "interval": timeframe_label(interval),
        "range": {
            "start": start.strftime("%Y-%m-%d %H:%M UTC"),
            "end": end.strftime("%Y-%m-%d %H:%M UTC"),
            "days": days,
        },
        "sampling": {
            "checked_bars": len(check_index),
            "sample_step": step,
            "max_checks": max_checks,
            "max_hold_bars": max_hold_bars,
            "filters_enabled": include_filters,
            "note": "This is a single-timeframe baseline replay of current rules, not the final multi-timeframe walk-forward model.",
        },
        "skipped": skipped,
        "metrics": metrics,
        "trades": trades[-200:],
    }


def stack_backtest_required_bars(days: int, max_hold_bars: int) -> dict[str, int]:
    bars_per_day = {"1h": 24, "15m": 96, "5m": 288}
    return {
        interval: int(min_bars_for_interval(interval) + bars_per_day[interval] * days + max_hold_bars + bars_per_day[interval])
        for interval in ("1h", "15m", "5m")
    }


def df_range_summary(df: pd.DataFrame) -> dict:
    return {
        "bars": len(df),
        "start": df.index[0].strftime("%Y-%m-%d %H:%M UTC") if not df.empty else None,
        "end": df.index[-1].strftime("%Y-%m-%d %H:%M UTC") if not df.empty else None,
    }


def history_until(df: pd.DataFrame, ts: pd.Timestamp, min_bars: int) -> pd.DataFrame | None:
    pos = int(df.index.searchsorted(ts, side="right")) - 1
    if pos < min_bars:
        return None
    return df.iloc[:pos + 1]


def stack_backtest_signal_record(
    snapshot: dict,
    stacked: dict,
    outcome: dict,
    filter_reasons: list[str],
) -> dict:
    trade = stacked.get("trade", {})
    zone = stacked.get("zone") or {}
    sweep = stacked.get("sweep") or {}
    direction = stacked.get("direction")
    entry = safe_float(trade.get("entry"), safe_float(snapshot.get("price")))
    stop = safe_float(trade.get("stop"))
    target = safe_float(trade.get("target"))
    risk = abs(entry - stop)
    reward = abs(target - entry)
    opened_bar = sweep.get("bar_time") or snapshot.get("m5", {}).get("bar_time") or snapshot.get("timestamp")
    alert_key = f"BACKTEST_STACK:5M:{direction}:{opened_bar}"
    return {
        "id": trade_record_id(alert_key),
        "alert_key": alert_key,
        "source": "supabase_stack_backtest",
        "status": outcome.get("status"),
        "result": outcome.get("result"),
        "direction": direction,
        "setup_type": stacked.get("setup_type"),
        "quality": "BACKTEST_STACK",
        "execution_timeframe": "5M",
        "opened_at": snapshot.get("timestamp"),
        "opened_bar": opened_bar,
        "entry": round(entry, 2),
        "entry_low": trade.get("entry_low"),
        "entry_high": trade.get("entry_high"),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "target_2": trade.get("target_2"),
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "rr": round(reward / risk, 2) if risk else None,
        "size_1pct_10k_oz": trade.get("position_1pct_10k_oz"),
        "weighted_score": None,
        "confidence": stacked.get("confidence"),
        "h1_score": stacked.get("h1_score"),
        "m15_score": stacked.get("m15_score"),
        "m5_score": stacked.get("m5_score"),
        "sweep_summary": sweep.get("summary"),
        "zone_low": zone.get("low"),
        "zone_high": zone.get("high"),
        "zone_score": zone.get("score"),
        "zone_summary": zone.get("summary"),
        "reason": (
            f"Supabase full-stack replay. Filters skipped: "
            f"{', '.join(filter_reasons) if filter_reasons else 'none'}."
        ),
        "h1_bar": snapshot.get("h1", {}).get("bar_time"),
        "m15_bar": snapshot.get("m15", {}).get("bar_time"),
        "m5_bar": snapshot.get("m5", {}).get("bar_time"),
        "closed_at": outcome.get("closed_at"),
        "closed_bar": outcome.get("closed_at"),
        "closed_price": outcome.get("closed_price"),
        "observed_price": None,
        "observed_high": None,
        "observed_low": None,
        "r_multiple": outcome.get("r_multiple"),
        "pnl_1pct_10k_usd": round(safe_float(outcome.get("r_multiple")) * 100, 2) if outcome.get("r_multiple") is not None else None,
    }


def run_supabase_stack_backtest(
    days: int = 30,
    max_checks: int = 2000,
    max_hold_bars: int | None = None,
    include_filters: bool = True,
) -> dict:
    if not supabase_configured():
        return {"error": "Supabase not configured"}

    days = int(clamp(days, 1, 120))
    max_checks = int(clamp(max_checks, 50, 10000))
    max_hold_bars = max_hold_bars or backtest_default_hold_bars("5m")
    required_bars = stack_backtest_required_bars(days, max_hold_bars)

    datasets = {}
    errors = {}
    for interval in ("1h", "15m", "5m"):
        df, err = fetch_supabase_ohlcv_history(interval, required_bars[interval])
        if err or df is None:
            errors[timeframe_label(interval)] = err or "No Supabase OHLCV data"
            continue
        datasets[interval] = normalize_df_timezone(df)

    if errors:
        return {
            "error": "Supabase OHLCV history is not ready for stack backtest",
            "details": errors,
            "required_bars": {timeframe_label(k): v for k, v in required_bars.items()},
            "hint": "Run /api/data/ingest with OANDA enabled until H1, M15, and M5 have enough complete candles, then rerun this endpoint.",
        }

    h1_df = datasets["1h"]
    m15_df = datasets["15m"]
    m5_df = datasets["5m"]
    end = m5_df.index[-1]
    start = end - pd.Timedelta(days=days)
    check_index = list(m5_df.loc[m5_df.index >= start].index)
    check_index = [
        ts for ts in check_index
        if history_until(m5_df, ts, backtest_min_bars("5m")) is not None
        and history_until(m15_df, ts, backtest_min_bars("15m")) is not None
        and history_until(h1_df, ts, backtest_min_bars("1h")) is not None
    ]
    if len(check_index) > max_checks:
        step = int(np.ceil(len(check_index) / max_checks))
        check_index = check_index[::step]
    else:
        step = 1

    trades = []
    seen = set()
    skipped = {}
    last_direction_time = {}

    for ts in check_index:
        h1_history = history_until(h1_df, ts, backtest_min_bars("1h"))
        m15_history = history_until(m15_df, ts, backtest_min_bars("15m"))
        m5_history = history_until(m5_df, ts, backtest_min_bars("5m"))
        if h1_history is None or m15_history is None or m5_history is None:
            skipped["insufficient_warmup"] = skipped.get("insufficient_warmup", 0) + 1
            continue

        try:
            h1_signal = compute_signal(h1_history, "1h")
            m15_signal = compute_signal(m15_history, "15m")
            m5_signal = compute_signal(m5_history, "5m")
        except Exception as e:
            skipped["compute_error"] = skipped.get("compute_error", 0) + 1
            log.warning(f"Supabase stack backtest compute error {ts}: {e}")
            continue

        timestamp = ts.strftime("%Y-%m-%d %H:%M UTC")
        snapshot = {
            "h1": h1_signal,
            "m15": m15_signal,
            "m5": m5_signal,
            "price": m5_signal.get("price") or m15_signal.get("price") or h1_signal.get("price"),
            "timestamp": timestamp,
        }
        stacked = stacked_execution_setup(snapshot, {})
        direction = stacked.get("direction")
        trade = stacked.get("trade", {})
        if not stacked.get("executable") or direction not in ("LONG", "SHORT"):
            failed = stacked.get("failed_checks") or ["not_executable"]
            for reason in failed:
                skipped[reason] = skipped.get(reason, 0) + 1
            continue
        if trade.get("stop") is None or trade.get("target") is None:
            skipped["no_trade_levels"] = skipped.get("no_trade_levels", 0) + 1
            continue

        opened_bar = (stacked.get("sweep") or {}).get("bar_time") or m5_signal.get("bar_time")
        trigger_key = (direction, opened_bar)
        if trigger_key in seen:
            skipped["duplicate_trigger"] = skipped.get("duplicate_trigger", 0) + 1
            continue
        seen.add(trigger_key)

        filter_reasons = []
        if include_filters:
            allowed, filter_reasons = backtest_filter_allows(ts, m5_signal, "5m", direction, last_direction_time)
            if not allowed:
                for reason in filter_reasons:
                    skipped[reason] = skipped.get(reason, 0) + 1
                continue

        entry = safe_float(trade.get("entry"), safe_float(snapshot.get("price")))
        stop = safe_float(trade.get("stop"))
        target = safe_float(trade.get("target"))
        outcome = backtest_trade_outcome(m5_df, ts, direction, entry, stop, target, max_hold_bars)
        trades.append(stack_backtest_signal_record(snapshot, stacked, outcome, filter_reasons))
        last_direction_time[direction] = ts

    metrics = performance_metrics(trades)
    months = max(days / 30.4375, 1 / 30.4375)
    metrics["estimated_trades_per_month"] = round(len(trades) / months, 1)
    metrics["estimated_closed_trades_per_month"] = round(metrics.get("closed", 0) / months, 1)

    return {
        "mode": "supabase_full_stack_replay",
        "source": f"supabase:{SUPABASE_OHLCV_SOURCE}:{OANDA_INSTRUMENT}",
        "range": {
            "start": start.strftime("%Y-%m-%d %H:%M UTC"),
            "end": end.strftime("%Y-%m-%d %H:%M UTC"),
            "days": days,
        },
        "datasets": {
            "H1": df_range_summary(h1_df),
            "M15": df_range_summary(m15_df),
            "M5": df_range_summary(m5_df),
        },
        "sampling": {
            "checked_bars": len(check_index),
            "sample_step": step,
            "max_checks": max_checks,
            "max_hold_bars": max_hold_bars,
            "filters_enabled": include_filters,
            "note": "Replays the production full stack: 1H bias, 15M order-block zone, 5M liquidity sweep, and Phase 1 session/regime/cooldown filters.",
        },
        "skipped": dict(sorted(skipped.items())),
        "metrics": metrics,
        "trades": trades[-200:],
    }


def readiness_check(name: str, passed: bool, severity: str, detail: str, data: dict | None = None) -> dict:
    result = {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    }
    if data is not None:
        result["data"] = data
    return result


def latest_supabase_row(table: str, select: str, order: str) -> dict | None:
    rows = supabase_request(
        "GET",
        table,
        params={
            "select": select,
            "order": order,
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


def ohlcv_readiness(days: int) -> dict:
    max_hold_bars = backtest_default_hold_bars("5m")
    required_bars = stack_backtest_required_bars(days, max_hold_bars)
    results = {}
    for interval in ("1h", "15m", "5m"):
        granularity = interval_granularity(interval)
        required = required_bars[interval]
        rows, err = supabase_ohlcv_rows(interval, required)
        if err:
            results[timeframe_label(interval)] = {
                "ready": False,
                "required_bars": required,
                "available_bars_checked": 0,
                "error": err,
            }
            continue
        df, parse_err = ohlcv_rows_to_df(rows, f"supabase:{SUPABASE_OHLCV_SOURCE}:{granularity}:readiness")
        if parse_err or df is None:
            results[timeframe_label(interval)] = {
                "ready": False,
                "required_bars": required,
                "available_bars_checked": len(rows),
                "error": parse_err or "No parsed OHLCV rows",
            }
            continue
        results[timeframe_label(interval)] = {
            "ready": len(df) >= required,
            "required_bars": required,
            "available_bars_checked": len(df),
            "start": df.index[0].strftime("%Y-%m-%d %H:%M UTC"),
            "end": df.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
        }
    return results


def system_readiness_report(backtest_days: int = 30) -> dict:
    backtest_days = int(clamp(backtest_days, 1, 120))
    checks = [
        readiness_check(
            "admin_token",
            bool(ADMIN_TOKEN),
            "critical",
            "ADMIN_TOKEN is configured." if ADMIN_TOKEN else "ADMIN_TOKEN is missing; protected routes are unusable.",
        ),
        readiness_check(
            "supabase",
            supabase_configured(),
            "critical",
            "Supabase URL and service role key are configured." if supabase_configured() else "SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing.",
        ),
        readiness_check(
            "telegram",
            bool(TG_TOKEN and TG_CHAT_ID),
            "critical",
            "Telegram delivery is configured." if TG_TOKEN and TG_CHAT_ID else "Telegram bot token or chat id is missing.",
        ),
        readiness_check(
            "email_backup",
            email_configured(),
            "warning",
            "Email backup delivery is configured." if email_configured() else "Email backup is not configured; Telegram will be the only alert path.",
        ),
        readiness_check(
            "oanda_candles",
            oanda_configured(),
            "critical",
            "OANDA candle API is configured." if oanda_configured() else "OANDA_API_KEY is missing.",
        ),
        readiness_check(
            "oanda_pricing",
            bool(OANDA_API_KEY and OANDA_ACCOUNT_ID) or XAUUSD_SPREAD_CENTS != "",
            "critical",
            "Spread source is configured." if (OANDA_API_KEY and OANDA_ACCOUNT_ID) or XAUUSD_SPREAD_CENTS != "" else "OANDA_ACCOUNT_ID or XAUUSD_SPREAD_CENTS is required for broker-grade spread checks.",
        ),
        readiness_check(
            "hard_filters",
            HARD_FILTERS_ENABLED,
            "critical",
            "Hard filters are enabled." if HARD_FILTERS_ENABLED else "Hard filters are disabled.",
        ),
        readiness_check(
            "stripe",
            not PAYMENTS_ENABLED or bool(STRIPE_SECRET_KEY and STRIPE_PRICE_ID and STRIPE_WEBHOOK_SECRET),
            "warning" if PAYMENTS_ENABLED else "info",
            "Stripe checkout and webhook env vars are configured." if PAYMENTS_ENABLED and STRIPE_SECRET_KEY and STRIPE_PRICE_ID and STRIPE_WEBHOOK_SECRET else "Payments are disabled for private trading mode." if not PAYMENTS_ENABLED else "Stripe is enabled but checkout or webhook env vars are incomplete.",
        ),
        readiness_check(
            "posthog",
            not PRODUCT_ANALYTICS_ENABLED or bool(POSTHOG_API_KEY),
            "warning" if PRODUCT_ANALYTICS_ENABLED else "info",
            "PostHog analytics is configured." if PRODUCT_ANALYTICS_ENABLED and POSTHOG_API_KEY else "Product analytics are disabled for private trading mode." if not PRODUCT_ANALYTICS_ENABLED else "PostHog is enabled but POSTHOG_API_KEY is missing.",
        ),
    ]

    data = {
        "private_trading_mode": not PAYMENTS_ENABLED,
        "payments_enabled": PAYMENTS_ENABLED,
        "product_analytics_enabled": PRODUCT_ANALYTICS_ENABLED,
        "market_data_source": MARKET_DATA_SOURCE,
        "supabase_ohlcv_source": SUPABASE_OHLCV_SOURCE,
        "instrument": OANDA_INSTRUMENT,
        "backtest_days": backtest_days,
        "macro_features": {
            "enabled": MACRO_FEATURES_ENABLED,
            "ingest_seconds": MACRO_FEATURES_INGEST_SECONDS,
        },
        "signal_lifecycle": {
            "valid_minutes": SIGNAL_VALID_MINUTES,
            "invalidation_atr_buffer": SIGNAL_INVALIDATION_ATR_BUFFER,
        },
        "notifications": {
            "email_enabled": EMAIL_ENABLED,
            "email_backup_mode": EMAIL_BACKUP_MODE,
            "email_recipients": len(email_recipients()),
            "telegram_configured": bool(TG_TOKEN and TG_CHAT_ID),
        },
        "go_live_thresholds": {
            "min_closed_signals": GO_LIVE_MIN_CLOSED_SIGNALS,
            "min_win_rate": GO_LIVE_MIN_WIN_RATE,
            "min_avg_r": GO_LIVE_MIN_AVG_R,
            "min_profit_factor": GO_LIVE_MIN_PROFIT_FACTOR,
            "max_losing_streak": GO_LIVE_MAX_LOSING_STREAK,
        },
    }

    if supabase_configured():
        latest_signal = latest_supabase_row("signals", "id,generated_at,direction,setup_type,status", "generated_at.desc")
        latest_spread = latest_supabase_row("market_spreads", "time,bid,ask,spread_cents", "time.desc")
        latest_macro = latest_supabase_row("macro_features", "ts,status,macro_score,macro_bias,dxy,us10y,vix", "ts.desc")
        ohlcv = ohlcv_readiness(backtest_days)
        data["latest_signal"] = latest_signal
        data["latest_spread"] = latest_spread
        data["latest_macro_features"] = latest_macro
        data["ohlcv"] = ohlcv
        all_ohlcv_ready = bool(ohlcv) and all(item.get("ready") for item in ohlcv.values())
        checks.append(readiness_check(
            "supabase_ohlcv_backtest_coverage",
            all_ohlcv_ready,
            "critical",
            f"Supabase has enough H1/M15/M5 candles for a {backtest_days}-day stack backtest." if all_ohlcv_ready else f"Supabase does not yet have enough H1/M15/M5 candles for a {backtest_days}-day stack backtest.",
            ohlcv,
        ))
        checks.append(readiness_check(
            "spread_history",
            latest_spread is not None or XAUUSD_SPREAD_CENTS != "",
            "warning",
            "Recent spread history exists or manual spread is configured." if latest_spread or XAUUSD_SPREAD_CENTS != "" else "No market_spreads row found yet; call /api/data/spread after OANDA pricing is configured.",
        ))
        checks.append(readiness_check(
            "macro_features_history",
            not MACRO_FEATURES_ENABLED or latest_macro is not None,
            "warning" if MACRO_FEATURES_ENABLED else "info",
            "Macro feature snapshots are being stored in Supabase." if latest_macro else "No macro_features row found yet; run /api/macro/ingest after applying the latest Supabase schema." if MACRO_FEATURES_ENABLED else "Macro feature snapshots are disabled.",
        ))

    critical_failures = [check for check in checks if check["severity"] == "critical" and not check["passed"]]
    warnings = [check for check in checks if check["severity"] == "warning" and not check["passed"]]
    next_actions = []
    for check in critical_failures + warnings:
        next_actions.append(check["detail"])
    if supabase_configured() and any(check["name"] == "supabase_ohlcv_backtest_coverage" and not check["passed"] for check in checks):
        next_actions.append("Run /api/data/ingest?token=ADMIN_TOKEN&granularities=M1,M5,M15,H1&count=5000 until the OHLCV coverage check passes.")
    if supabase_configured() and any(check["name"] == "macro_features_history" and not check["passed"] for check in checks):
        next_actions.append("Run /api/macro/ingest with Authorization: Bearer ADMIN_TOKEN after rerunning supabase_schema.sql.")

    return {
        "status": "ready" if not critical_failures else "blocked",
        "critical_failures": len(critical_failures),
        "warnings": len(warnings),
        "checks": checks,
        "data": data,
        "next_actions": next_actions,
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
    data = get_both_signals(force=force_context)
    context = get_market_context(force=force_context)
    snapshot = {
        **data,
        "context": context,
        "final": final_decision(data, context),
    }
    snapshot["hard_filters"] = hard_filter_status(snapshot, snapshot["final"])
    snapshot["timeframe_trades"] = timeframe_trade_decisions(snapshot)
    snapshot["watch_signals"] = watch_signal_candidates(snapshot)
    snapshot["prop_firm"] = prop_firm_snapshot(snapshot)
    return snapshot


def get_dashboard_snapshot(force: bool = False) -> dict:
    snapshot = get_trade_snapshot(force_context=force)
    return {
        **snapshot,
        "event_study": get_event_study(),
        "trade_history": trade_history_payload(),
        "trend_grid_bot": trend_grid_bot_signal(snapshot, snapshot["context"]),
        "fib_ema_pullback": fibonacci_ema_pullback_signal(snapshot["context"], force=force),
        "ict_bos_retracement": ict_bos_retracement_signal(snapshot["context"], force=force),
        "ict_london_bos": ict_london_bos_signal(snapshot["context"], force=force),
    }


# ═══════════════════════════════════════════════════════════════════
#  NOTIFICATIONS — TELEGRAM PRIMARY, EMAIL BACKUP
# ═══════════════════════════════════════════════════════════════════
def email_recipients() -> list[str]:
    return [item.strip() for item in EMAIL_TO.split(",") if item.strip()]


def email_configured() -> bool:
    return bool(
        EMAIL_ENABLED
        and SMTP_HOST
        and SMTP_PORT
        and EMAIL_FROM
        and email_recipients()
    )


def strip_html(value: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()


def email_html_body(value: str) -> str:
    return value.replace("\n", "<br>\n")


def send_email_sync(subject: str, body_html: str) -> bool:
    if not email_configured():
        log.warning("Email not configured")
        return False
    try:
        message = EmailMessage()
        message["Subject"] = subject[:180]
        message["From"] = EMAIL_FROM
        message["To"] = ", ".join(email_recipients())
        message.set_content(strip_html(body_html))
        message.add_alternative(email_html_body(body_html), subtype="html")

        if SMTP_USE_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(), timeout=15) as server:
                if SMTP_USER or SMTP_PASSWORD:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                if SMTP_USE_TLS:
                    server.starttls(context=ssl.create_default_context())
                if SMTP_USER or SMTP_PASSWORD:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
        log.info("Email ✅")
        return True
    except Exception as e:
        log.error(f"Email error: {e}")
        return False


async def email_send(subject: str, body_html: str) -> bool:
    return await asyncio.to_thread(send_email_sync, subject, body_html)


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


async def notify_send(subject: str, body_html: str) -> dict:
    telegram_ok = await tg_send(body_html)
    email_ok = False
    mode = EMAIL_BACKUP_MODE if EMAIL_BACKUP_MODE in {"failover", "all", "off"} else "failover"
    if mode == "all" or (mode == "failover" and not telegram_ok):
        email_ok = await email_send(subject, body_html)
    return {
        "sent": telegram_ok or email_ok,
        "telegram": telegram_ok,
        "email": email_ok,
        "mode": mode,
    }


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
    final = snapshot.get("final") or {}
    if is_executable_trade(final):
        snapshots.append({**snapshot, "final": final})
        final_key = (
            final.get("execution_timeframe"),
            final.get("direction"),
            (final.get("sweep") or {}).get("bar_time"),
        )
    else:
        final_key = None
    for decision in trade_decisions:
        if is_executable_trade(decision):
            decision_key = (
                decision.get("execution_timeframe"),
                decision.get("direction"),
                (decision.get("sweep") or {}).get("bar_time"),
            )
            if final_key and decision_key == final_key:
                continue
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
    zone = final.get("zone") or (final.get("stacked_setup") or {}).get("zone") or {}
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
    entry_zone_line = ""
    if trade.get("entry_low") is not None and trade.get("entry_high") is not None:
        entry_zone_line = (
            f"📍 <b>Entry zone</b>   <code>{fmt_money(trade.get('entry_low'))}</code> - "
            f"<code>{fmt_money(trade.get('entry_high'))}</code>\n"
        )
    zone_line = ""
    if zone:
        zone_line = (
            f"15M zone: <code>{fmt_money(zone.get('low'))}</code> - <code>{fmt_money(zone.get('high'))}</code> "
            f"score <b>{html_text(zone.get('score'))}</b>\n"
        )
    valid_until = signal_valid_until_text(snapshot.get("timestamp") or utc_now_text())
    next_event = context.get("event_risk", {}).get("next_event") or {}
    prop_line = ""
    if PROP_MODE_ENABLED:
        prop = prop_firm_snapshot(snapshot)
        prop_candidate = prop.get("candidate", {})
        prop_sizing = prop_candidate.get("sizing", {})
        if prop_candidate.get("ready"):
            prop_line = (
                f"🏦 <b>Prop size</b>     <code>{fmt_trade_size(prop_sizing.get('size_oz'))}</code> "
                f"risk <code>{safe_float(prop_sizing.get('risk_pct')):.2f}%</code> / "
                f"<code>{fmt_money(prop_sizing.get('risk_usd'))}</code>\n"
            )
        else:
            blockers = prop_candidate.get("blockers") or ["Prop gates not clear."]
            prop_line = f"🏦 <b>Prop mode</b>     <code>WAIT</code> - {html_text(blockers[0])}\n"

    return (
        f"{side} <b>XAU/USD TRADE — {html_text(execution_timeframe)} {html_text(direction)}</b>\n"
        f"<b>{action}</b> · Timeframe: <b>{html_text(execution_timeframe)}</b> · "
        f"Quality: <b>{html_text(quality)}</b> · Confidence: <b>{final.get('confidence', 0)}%</b>\n\n"
        f"{entry_zone_line}"
        f"💰 <b>Entry</b>        <code>{fmt_money(trade.get('entry'))}</code>\n"
        f"🛑 <b>Stop</b>         <code>{fmt_money(trade.get('stop'))}</code>  "
        f"Risk <code>{fmt_money(trade.get('risk'))}</code>\n"
        f"🎯 <b>Target 1</b>     <code>{fmt_money(trade.get('target'))}</code>  "
        f"Reward <code>{fmt_money(trade.get('reward'))}</code>\n"
        f"{target_2_line}"
        f"⚖️ <b>RR</b>           <code>{rr_text}</code>\n"
        f"💼 <b>Size</b>         <code>{fmt_trade_size(trade.get('position_1pct_10k_oz'))}</code> "
        f"for 1% risk on $10k\n\n"
        f"{prop_line}\n"
        f"📊 <b>Model</b>\n"
        f"Weighted score: <code>{fmt_signed_score(final.get('total_score'))}</code> · "
        f"1H: <code>{fmt_signed_score(h1.get('technical_score'))}</code> · "
        f"15M: <code>{fmt_signed_score(m15.get('technical_score'))}</code> · "
        f"5M: <code>{fmt_signed_score(m5.get('technical_score'))}</code>\n"
        f"Risk: <b>{html_text(final.get('risk'))}</b> · Event: <b>{html_text(final.get('event_level'))}</b>\n\n"
        f"🧭 <b>Trigger</b>\n"
        f"{zone_line}"
        f"{html_text(sweep.get('summary'))}\n"
        f"{html_text(execution_timeframe)} sweep bar: <code>{html_text(sweep.get('bar_time') or execution_data.get('bar_time'))}</code>\n\n"
        f"📋 <b>Reason</b>\n"
        f"{html_text(final.get('reason'))}\n\n"
        f"⏱ <b>Valid until</b> <code>{html_text(valid_until)}</code>\n"
        f"🚫 <b>Invalidation</b>\n"
        f"{html_text(trade.get('invalidates'))}\n\n"
        f"🗓 <b>Next event</b> {html_text(next_event.get('name') or 'None')} "
        f"{html_text(next_event.get('eta') or '')}\n"
        f"🕐 <i>{html_text(snapshot.get('timestamp'))}</i>\n"
        f"<i>Research only, not financial advice.</i>"
    )


def msg_london_trade_alert(signal: dict) -> str:
    trade = signal.get("trade") or {}
    zone = signal.get("zone") or {}
    bos = signal.get("bos") or {}
    bias = signal.get("bias") or {}
    levels = signal.get("levels") or {}
    prev = levels.get("previous_day") or {}
    asian = levels.get("asian") or {}
    direction = signal.get("direction", "WAIT")
    side = "🟢" if direction == "LONG" else "🔴"
    action = "BUY" if direction == "LONG" else "SELL"
    rr = trade.get("rr")
    rr_text = f"1 : {safe_float(rr):.2f}" if rr is not None else "--"
    valid_until = signal_valid_until_text(signal.get("timestamp") or utc_now_text())
    return (
        f"{side} <b>XAU/USD LONDON TRADE — {html_text(direction)}</b>\n"
        f"<b>{action}</b> · Strategy: <b>H4/Daily BOS Retracement</b> · Session: <b>London CT 02:00-05:00</b>\n\n"
        f"💰 <b>Entry</b>        <code>{fmt_money(trade.get('entry'))}</code>\n"
        f"📍 <b>Entry zone</b>   <code>{fmt_money(trade.get('entry_low'))}</code> - <code>{fmt_money(trade.get('entry_high'))}</code>\n"
        f"🛑 <b>Stop</b>         <code>{fmt_money(trade.get('stop'))}</code>  Risk <code>{fmt_money(trade.get('risk'))}</code>\n"
        f"🎯 <b>Target</b>       <code>{fmt_money(trade.get('target'))}</code>  "
        f"Reward <code>{fmt_money(trade.get('reward'))}</code>\n"
        f"⚖️ <b>RR</b>           <code>{rr_text}</code>\n"
        f"💼 <b>Size</b>         <code>{fmt_trade_size(trade.get('position_1pct_10k_oz'))}</code> "
        f"for 1% risk on $10k\n\n"
        f"🧭 <b>Trigger</b>\n"
        f"H4/Daily bias: <b>{html_text(direction)}</b> — {html_text(bias.get('reason'))}\n"
        f"Zone: <b>{html_text(zone.get('timeframe'))} {html_text(zone.get('type'))}</b> "
        f"<code>{fmt_money(zone.get('low'))}</code> - <code>{fmt_money(zone.get('high'))}</code>\n"
        f"5M BOS: <code>{html_text(bos.get('bos_level'))}</code> at <code>{html_text(bos.get('bos_time'))}</code>\n"
        f"{html_text(bos.get('reason'))}\n\n"
        f"🏦 <b>Liquidity targets</b>\n"
        f"Previous day H/L: <code>{fmt_money(prev.get('high'))}</code> / <code>{fmt_money(prev.get('low'))}</code>\n"
        f"Asian H/L: <code>{fmt_money(asian.get('high'))}</code> / <code>{fmt_money(asian.get('low'))}</code>\n"
        f"Target source: <b>{html_text(trade.get('target_source'))}</b>\n\n"
        f"⏱ <b>Valid until</b> <code>{html_text(valid_until)}</code>\n"
        f"🚫 <b>Invalidation</b>\n"
        f"{html_text(trade.get('invalidates'))}\n\n"
        f"🕐 <i>{html_text(signal.get('timestamp'))}</i>\n"
        f"<i>Research only, not financial advice.</i>"
    )


def msg_trade_result(trade: dict) -> str:
    result = trade.get("result")
    direction = trade.get("direction")
    if result == "TP":
        side = "✅"
        label = "TAKE PROFIT HIT"
    elif result == "SL":
        side = "🛑"
        label = "STOP LOSS HIT"
    elif result == "INVALIDATED":
        side = "⚠️"
        label = "SIGNAL INVALIDATED"
    else:
        side = "⏱"
        label = "SIGNAL EXPIRED"
    pnl = safe_float(trade.get("pnl_1pct_10k_usd"))
    pnl_sign = "+" if pnl > 0 else ""
    execution_timeframe = trade.get("execution_timeframe") or "15M"
    reason_line = ""
    if trade.get("invalidation_reason"):
        reason_line = f"Reason: {html_text(trade.get('invalidation_reason'))}\n\n"
    return (
        f"{side} <b>XAU/USD RESULT — {label}</b>\n\n"
        f"Direction: <b>{html_text(execution_timeframe)} {html_text(direction)}</b>\n"
        f"Entry: <code>{fmt_money(trade.get('entry'))}</code>\n"
        f"Exit: <code>{fmt_money(trade.get('closed_price'))}</code>\n"
        f"Stop: <code>{fmt_money(trade.get('stop'))}</code> · "
        f"TP: <code>{fmt_money(trade.get('target'))}</code>\n\n"
        f"R multiple: <code>{safe_float(trade.get('r_multiple')):+.2f}R</code>\n"
        f"P/L on 1% risk / $10k: <code>{pnl_sign}${pnl:.2f}</code>\n\n"
        f"{reason_line}"
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


def msg_watch_alert(snapshot: dict, candidate: dict) -> str:
    tf = candidate.get("timeframe", "--")
    direction = candidate.get("direction", "WAIT")
    quality = candidate.get("quality", "WATCH")
    trade = candidate.get("trade") or {}
    sweep = candidate.get("liquidity_sweep") or {}
    zone = candidate.get("zone") or {}
    hard_filters = candidate.get("hard_filters") or {}
    conditions = candidate.get("conditions") or {}
    score = int(safe_float(candidate.get("technical_score")))
    side = "🟡"
    action = "WATCH LONG" if direction == "LONG" else "WATCH SHORT"

    entry = trade.get("entry") if trade.get("entry") is not None else candidate.get("price")
    stop = trade.get("stop") if trade.get("stop") is not None else candidate.get("stop")
    target = trade.get("target") if trade.get("target") is not None else candidate.get("target")
    failed_checks = candidate.get("failed_stack_checks") or []
    blocked_reasons = hard_filters.get("blocked_reasons") or []
    stack_line = ", ".join(failed_checks) if failed_checks else "full stack still pending"
    filter_line = "; ".join(blocked_reasons[:3]) if blocked_reasons else "hard filters currently clear"
    zone_line = ""
    if zone:
        zone_line = (
            f"15M zone: <code>{fmt_money(zone.get('low'))}</code> - <code>{fmt_money(zone.get('high'))}</code> "
            f"score <b>{html_text(zone.get('score'))}</b>\n"
        )

    labels = {
        "trend_aligned": "EMA regime aligned",
        "liquidity_sweep": "Liquidity sweep",
        "ema_reclaim": "EMA reclaim/rejection",
        "momentum_confirmed": "Momentum confirmed",
        "not_overextended": "Not RSI-overextended",
    }
    rows = "\n".join([
        ("✅" if conditions.get(key) else "❌") + f" {label}"
        for key, label in labels.items()
    ])

    return (
        f"{side} <b>XAU/USD WATCH — {html_text(tf)} {html_text(direction)}</b>\n"
        f"<b>{html_text(action)}</b> · Tier: <b>{html_text(quality)}</b> · "
        f"Score: <code>{score:+d}</code> · "
        f"Conditions: <b>{candidate.get('conditions_met', 0)}/{candidate.get('max_conditions', 7)}</b>\n\n"
        f"💰 Price / entry ref: <code>{fmt_money(entry)}</code>\n"
        f"🛑 Est stop: <code>{fmt_money(stop)}</code>\n"
        f"🎯 Est target: <code>{fmt_money(target)}</code>\n\n"
        f"{rows}\n\n"
        f"🧭 Trigger\n"
        f"{zone_line}"
        f"{html_text(sweep.get('summary') or candidate.get('reason'))}\n"
        f"Bar: <code>{html_text(candidate.get('bar_time'))}</code>\n\n"
        f"Stack missing: <code>{html_text(stack_line)}</code>\n"
        f"Filter note: <code>{html_text(filter_line)}</code>\n\n"
        f"<i>Watch only. No executable trade is recorded until the full stack confirms.</i>\n"
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
    watch_minutes = max(1, WATCH_CHECK_SECONDS // 60)
    london_minutes = max(1, LONDON_ALERT_CHECK_SECONDS // 60)
    watch_line = (
        f"Watch alerts are <b>enabled</b> every <b>{watch_minutes} minutes</b>.\n"
        if WATCH_ALERTS_ENABLED else
        "Watch alerts are <b>disabled</b>.\n"
    )
    london_line = (
        f"London BOS alerts are <b>enabled</b> every <b>{london_minutes} minutes</b>.\n"
        if LONDON_ALERTS_ENABLED else
        "London BOS alerts are <b>disabled</b>.\n"
    )
    return (
        "✅ <b>XAU/USD Intelligence Bot Online</b>\n\n"
        f"Scanning for executable trades every <b>{interval_minutes} minutes</b>.\n"
        f"{watch_line}"
        f"{london_line}"
        "Trade alerts are recorded only when the full <b>1H/15M/5M</b> execution stack confirms.\n\n"
        "<b>Model:</b> Technical 50% · News 20% · Macro 15% · Sentiment 15%\n"
        "<b>Strategy:</b> 1H bias · 15M order block · 5M liquidity sweep/reclaim · 1:2 RR\n"
        "<b>Data:</b> yfinance + GDELT/Fed RSS + scheduled macro event seeds"
    )


def london_signal_blockers(signal: dict) -> list[str]:
    if london_signal_executable(signal):
        return []
    blockers = list(signal.get("failed_checks") or [])
    if signal.get("quality") != "CONFIRMED":
        blockers.append(f"quality={signal.get('quality', 'WAIT')}")
    if signal.get("session") != "LONDON_OPEN_CT":
        blockers.append(f"session={signal.get('session', 'OFF_SESSION')}")
    trade = signal.get("trade") or {}
    for key in ("entry", "stop", "target"):
        if trade.get(key) is None:
            blockers.append(f"missing_{key}")
    return list(dict.fromkeys(blockers))


def current_london_signal(force: bool = False) -> dict:
    context = get_market_context(force=force)
    return ict_london_bos_signal(context, force=force)


async def send_london_telegram_signal(force: bool = True, source: str = "london_check_now") -> dict:
    signal = await asyncio.to_thread(current_london_signal, force)
    alert_key = london_alert_key(signal)
    if not alert_key:
        return {
            "sent": False,
            "recorded": False,
            "already_recorded": False,
            "reason": "London setup is not executable.",
            "blockers": london_signal_blockers(signal),
            "signal": signal,
        }

    existing_trade = await asyncio.to_thread(find_trade_record, alert_key)
    if existing_trade:
        return {
            "sent": False,
            "recorded": False,
            "already_recorded": True,
            "trade_record": existing_trade,
            "trade_alert_key": alert_key,
            "signal": signal,
        }

    notification = await notify_send(
        f"XAU/USD London trade — {signal.get('direction')}",
        msg_london_trade_alert(signal),
    )
    should_record = notification["sent"] or supabase_configured()
    record, recorded = await asyncio.to_thread(record_london_trade_alert, signal, source) if should_record else (None, False)
    return {
        "sent": notification["sent"],
        "telegram_sent": notification["telegram"],
        "email_sent": notification["email"],
        "recorded": recorded,
        "trade_record": record,
        "trade_alert_key": alert_key,
        "signal": signal,
    }


# ═══════════════════════════════════════════════════════════════════
#  BACKGROUND CHECKERS
# ═══════════════════════════════════════════════════════════════════
async def london_alert_checker(check_secs: int):
    log.info(f"[LONDON] started — every {check_secs}s")
    if LONDON_ALERT_STARTUP_DELAY_SECONDS > 0:
        await asyncio.sleep(LONDON_ALERT_STARTUP_DELAY_SECONDS)

    while True:
        try:
            result = await send_london_telegram_signal(force=True, source="london_telegram_alert")
            signal = result.get("signal") or {}
            if result.get("sent") or result.get("recorded"):
                log.info(f"[LONDON] alert sent {signal.get('direction')} {result.get('trade_alert_key')}")
            elif result.get("already_recorded"):
                log.info(f"[LONDON] duplicate suppressed {result.get('trade_alert_key')}")
            elif signal.get("quality") == "CONFIRMED":
                log.info(f"[LONDON] confirmed but blocked: {result.get('blockers')}")
        except Exception as e:
            log.error(f"[LONDON] error: {e}")

        await asyncio.sleep(max(10, check_secs))


async def trade_checker(check_secs: int):
    log.info(f"[TRADE] started — every {check_secs}s")
    last_alert_keys = set()
    if TRADE_STARTUP_DELAY_SECONDS > 0:
        await asyncio.sleep(TRADE_STARTUP_DELAY_SECONDS)

    while True:
        try:
            snapshot = await asyncio.to_thread(get_trade_snapshot)
            closed_trades = await asyncio.to_thread(update_trade_results, snapshot)
            for closed_trade in closed_trades:
                log.info(f"[TRADE] {closed_trade.get('result')} {closed_trade.get('direction')} {closed_trade.get('id')}")
                await notify_send(
                    f"XAU/USD result — {closed_trade.get('result')} {closed_trade.get('direction')}",
                    msg_trade_result(closed_trade),
                )

            trade_snapshots = await asyncio.to_thread(executable_trade_snapshots, snapshot)
            if trade_snapshots:
                for trade_snapshot in trade_snapshots:
                    final = trade_snapshot.get("final", {})
                    key = trade_alert_key(trade_snapshot)
                    if not key:
                        continue

                    direction = final.get("direction")
                    timeframe = final.get("execution_timeframe")
                    existing_trade = await asyncio.to_thread(find_trade_record, key)
                    if existing_trade:
                        last_alert_keys.add(key)
                        log.info(f"[TRADE] {timeframe} {direction} already recorded as {existing_trade.get('id')}")
                    elif key not in last_alert_keys:
                        log.info(f"[TRADE] 🚨 {timeframe} {direction} @ ${final.get('trade', {}).get('entry')}")
                        notification = await notify_send(
                            f"XAU/USD trade — {timeframe} {direction}",
                            msg_trade_alert(trade_snapshot),
                        )
                        sent = notification["sent"]
                        if sent or supabase_configured():
                            source = (
                                "telegram_alert" if notification["telegram"] else
                                "email_alert" if notification["email"] else
                                "signal_generated"
                            )
                            await asyncio.to_thread(record_trade_alert, trade_snapshot, source=source)
                            last_alert_keys.add(key)
                    else:
                        log.info(f"[TRADE] active {timeframe} {direction}; duplicate alert suppressed")
            else:
                final = snapshot.get("final", {})
                log.info(f"[TRADE] no executable timeframe trade: {final.get('direction')} {final.get('quality')}")

        except Exception as e:
            log.error(f"[TRADE] error: {e}")

        await asyncio.sleep(check_secs)


async def watch_checker(check_secs: int):
    log.info(f"[WATCH] started — every {check_secs}s")
    last_alerted_keys = {}
    last_bucket_times = {}
    if WATCH_STARTUP_DELAY_SECONDS > 0:
        await asyncio.sleep(WATCH_STARTUP_DELAY_SECONDS)

    while True:
        try:
            snapshot = await asyncio.to_thread(get_trade_snapshot)
            candidates = await asyncio.to_thread(watch_signal_candidates, snapshot)
            now = datetime.now(timezone.utc)
            stale_before = now - timedelta(hours=12)
            for key, sent_at in list(last_alerted_keys.items()):
                if sent_at < stale_before:
                    del last_alerted_keys[key]

            if not candidates:
                log.info("[WATCH] no forming setup")
            for candidate in candidates:
                key = watch_alert_key(candidate)
                if key in last_alerted_keys:
                    continue

                bucket = watch_alert_bucket(candidate)
                last_bucket_time = last_bucket_times.get(bucket)
                if last_bucket_time is not None:
                    minutes_since = (now - last_bucket_time).total_seconds() / 60
                    if minutes_since < WATCH_ALERT_COOLDOWN_MINUTES:
                        continue

                log.info(
                    f"[WATCH] {candidate.get('timeframe')} {candidate.get('direction')} "
                    f"{candidate.get('quality')} score={candidate.get('technical_score')}"
                )
                await notify_send(
                    f"XAU/USD watch — {candidate.get('timeframe')} {candidate.get('direction')}",
                    msg_watch_alert(snapshot, candidate),
                )
                last_alerted_keys[key] = now
                last_bucket_times[bucket] = now

        except Exception as e:
            log.error(f"[WATCH] error: {e}")

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


async def oanda_ingest_worker(check_secs: int):
    log.info(f"[OANDA] ingest worker started — every {check_secs}s")
    while True:
        try:
            result = await asyncio.to_thread(ingest_oanda_ohlcv, oanda_granularity_list(), count=500)
            if result.get("ok"):
                log.info(f"[OANDA] ingested {result.get('ingested')}")
            else:
                log.warning(f"[OANDA] ingest skipped: {result.get('error')}")
        except Exception as e:
            log.error(f"[OANDA] ingest error: {e}")
        await asyncio.sleep(check_secs)


async def macro_features_worker(check_secs: int):
    log.info(f"[MACRO] feature worker started — every {check_secs}s")
    while True:
        try:
            result = await asyncio.to_thread(insert_macro_features_to_supabase)
            if result.get("ok"):
                record = result.get("record", {})
                log.info(f"[MACRO] stored {record.get('macro_bias')} score={record.get('macro_score')}")
            else:
                log.warning(f"[MACRO] ingest skipped: {result.get('error')}")
        except Exception as e:
            log.error(f"[MACRO] ingest error: {e}")
        await asyncio.sleep(check_secs)


# ═══════════════════════════════════════════════════════════════════
#  APP LIFESPAN
# ═══════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    await notify_send("XAU/USD bot started", msg_startup())
    trade_task = asyncio.create_task(trade_checker(TRADE_CHECK_SECONDS))
    watch_task = asyncio.create_task(watch_checker(WATCH_CHECK_SECONDS)) if WATCH_ALERTS_ENABLED else None
    london_alert_task = asyncio.create_task(london_alert_checker(LONDON_ALERT_CHECK_SECONDS)) if LONDON_ALERTS_ENABLED else None
    mt5_task = asyncio.create_task(mt5_london_worker(MT5_CHECK_SECONDS)) if MT5_AUTOTRADE_ENABLED else None
    oanda_task = None
    macro_task = None
    if OANDA_INGEST_ENABLED:
        oanda_task = asyncio.create_task(oanda_ingest_worker(OANDA_INGEST_SECONDS))
    if MACRO_FEATURES_ENABLED and supabase_configured():
        macro_task = asyncio.create_task(macro_features_worker(MACRO_FEATURES_INGEST_SECONDS))
    log.info("Trade checker running")
    if watch_task:
        log.info("Watch checker running")
    if london_alert_task:
        log.info("London BOS alert checker running")
    if mt5_task:
        log.info("MT5 London demo automation running")
    yield
    trade_task.cancel()
    with suppress(asyncio.CancelledError):
        await trade_task
    if watch_task:
        watch_task.cancel()
        with suppress(asyncio.CancelledError):
            await watch_task
    if london_alert_task:
        london_alert_task.cancel()
        with suppress(asyncio.CancelledError):
            await london_alert_task
    if mt5_task:
        mt5_task.cancel()
        with suppress(asyncio.CancelledError):
            await mt5_task
    if oanda_task:
        oanda_task.cancel()
        with suppress(asyncio.CancelledError):
            await oanda_task
    if macro_task:
        macro_task.cancel()
        with suppress(asyncio.CancelledError):
            await macro_task

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

@app.get("/api/strategy/fib-ema-pullback")
def api_fib_ema_pullback(refresh: bool = False):
    context = get_market_context(force=refresh)
    return JSONResponse(fibonacci_ema_pullback_signal(context, force=refresh))

@app.get("/api/strategy/ict-bos-retracement")
def api_ict_bos_retracement(refresh: bool = False):
    context = get_market_context(force=refresh)
    return JSONResponse(ict_bos_retracement_signal(context, force=refresh))

@app.get("/api/strategy/ict-london-bos")
def api_ict_london_bos(refresh: bool = False):
    context = get_market_context(force=refresh)
    return JSONResponse(ict_london_bos_signal(context, force=refresh))

@app.get("/api/mt5/status")
def api_mt5_status(request: Request, token: str = "", include_account: bool = True):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    return JSONResponse(mt5_runtime_status(include_account=include_account))

@app.post("/api/mt5/london-execute")
def api_mt5_london_execute(
    request: Request,
    token: str = "",
    dry_run: bool = True,
    refresh: bool = True,
):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    result = mt5_execute_london_strategy(dry_run=dry_run, force_signal=refresh, source="manual")
    status_code = 200
    if not result.get("ok") and result.get("blocked") and not dry_run:
        status_code = 409
    if not result.get("ok") and any("MetaTrader5 Python package" in item for item in result.get("blockers", [])):
        status_code = 503
    return JSONResponse(result, status_code=status_code)

@app.get("/api/prop/trades")
def api_prop_trades(refresh: bool = False):
    snapshot = get_trade_snapshot(force_context=refresh)
    return JSONResponse({
        "timestamp": snapshot.get("timestamp"),
        "price": snapshot.get("price"),
        "final": snapshot.get("final"),
        "prop_firm": snapshot.get("prop_firm"),
        "watch_signals": snapshot.get("watch_signals", []),
    })

@app.get("/api/trend-grid-bot")
def api_trend_grid_bot(refresh: bool = False):
    data = get_both_signals(force=refresh)
    context = get_market_context(force=refresh)
    return JSONResponse(trend_grid_bot_signal(data, context))

@app.get("/api/signal")
def api_signal():
    return JSONResponse(get_signal("1h"))

@app.get("/api/integrations")
def api_integrations():
    return {
        "status": service_status(),
        "app_base_url": APP_BASE_URL,
    }

@app.get("/api/me")
async def api_me(request: Request):
    return await entitlement_for_request(request)

@app.get("/api/premium/signals")
async def api_premium_signals(request: Request, limit: int = 50):
    entitlement = await entitlement_for_request(request)
    if not entitlement["authenticated"]:
        return error_response("unauthorized", "Supabase user token required.", 401)
    if not entitlement["active"]:
        return error_response("subscription_required", "Active subscription required.", 402)
    rows = supabase_request(
        "GET",
        "signals",
        params={
            "select": "*",
            "order": "generated_at.desc",
            "limit": str(max(1, min(limit, 200))),
        },
    )
    await track_request_event(request, "premium_signals_viewed", {"limit": limit})
    return {"signals": rows or []}

@app.post("/api/signals/{signal_id}/action")
async def api_signal_action(signal_id: str, request: Request):
    if not supabase_configured():
        return error_response("supabase_missing", "Supabase is not configured.", 503)
    entitlement = await entitlement_for_request(request)
    if not entitlement["authenticated"]:
        return error_response("unauthorized", "Supabase user token required.", 401)
    if not entitlement["active"]:
        return error_response("subscription_required", "Active subscription required.", 402)
    try:
        body = await request.json()
    except Exception:
        body = {}
    action = str(body.get("action", "")).upper()
    if action not in {"TAKEN", "SKIPPED"}:
        return error_response("invalid_action", "Action must be TAKEN or SKIPPED.", 422)
    payload = {
        "user_id": entitlement["user"]["id"],
        "signal_id": signal_id,
        "action": action,
        "notes": body.get("notes"),
        "acted_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    result = supabase_request(
        "POST",
        "user_signal_actions",
        params={"on_conflict": "user_id,signal_id"},
        json_payload=payload,
        prefer="resolution=merge-duplicates,return=representation",
    )
    await track_request_event(request, "signal_action_recorded", {"signal_id": signal_id, "action": action})
    return {"action": result[0] if isinstance(result, list) and result else payload}

@app.post("/api/billing/checkout")
async def api_billing_checkout(request: Request):
    if not PAYMENTS_ENABLED:
        return error_response("payments_disabled", "Payments are disabled for private trading mode.", 503)
    if not stripe_configured() or not STRIPE_PRICE_ID:
        return error_response("stripe_missing", "Stripe secret key or price id is not configured.", 503)
    entitlement = await entitlement_for_request(request)
    if not entitlement["authenticated"]:
        return error_response("unauthorized", "Supabase user token required.", 401)
    try:
        body = await request.json()
    except Exception:
        body = {}
    user = entitlement["user"]
    success_url = body.get("success_url") or f"{APP_BASE_URL}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = body.get("cancel_url") or f"{APP_BASE_URL}/?checkout=cancelled"
    data = {
        "mode": "subscription",
        "success_url": success_url,
        "cancel_url": cancel_url,
        "client_reference_id": user["id"],
        "line_items[0][price]": STRIPE_PRICE_ID,
        "line_items[0][quantity]": "1",
        "metadata[supabase_user_id]": user["id"],
        "subscription_data[metadata][supabase_user_id]": user["id"],
    }
    if user.get("email"):
        data["customer_email"] = user["email"]
    session = stripe_request("POST", "checkout/sessions", data=data)
    if not session or not session.get("url"):
        return error_response("stripe_checkout_failed", "Could not create Stripe Checkout session.", 502)
    await track_request_event(request, "checkout_started", {"stripe_session_id": session.get("id")})
    return {"id": session.get("id"), "url": session["url"]}

@app.post("/api/billing/portal")
async def api_billing_portal(request: Request):
    if not PAYMENTS_ENABLED:
        return error_response("payments_disabled", "Payments are disabled for private trading mode.", 503)
    if not stripe_configured():
        return error_response("stripe_missing", "Stripe secret key is not configured.", 503)
    entitlement = await entitlement_for_request(request)
    if not entitlement["authenticated"]:
        return error_response("unauthorized", "Supabase user token required.", 401)
    subscription = entitlement.get("subscription") or {}
    customer_id = subscription.get("stripe_customer_id")
    if not customer_id:
        return error_response("no_stripe_customer", "No Stripe customer found for this user.", 404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    session = stripe_request(
        "POST",
        "billing_portal/sessions",
        data={
            "customer": customer_id,
            "return_url": body.get("return_url") or APP_BASE_URL,
        },
    )
    if not session or not session.get("url"):
        return error_response("stripe_portal_failed", "Could not create Stripe billing portal session.", 502)
    await track_request_event(request, "billing_portal_opened", {"stripe_customer_id": customer_id})
    return {"url": session["url"]}

@app.post("/api/webhooks/stripe")
async def api_stripe_webhook(request: Request):
    if not PAYMENTS_ENABLED:
        return error_response("payments_disabled", "Payments are disabled for private trading mode.", 503)
    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    if not stripe_verify_signature(payload, signature):
        return error_response("invalid_signature", "Invalid Stripe webhook signature.", 400)
    try:
        event = json.loads(payload.decode("utf-8"))
    except Exception:
        return error_response("invalid_payload", "Invalid Stripe webhook payload.", 400)

    if supabase_configured():
        supabase_request(
            "POST",
            "stripe_events",
            params={"on_conflict": "id"},
            json_payload={
                "id": event.get("id"),
                "type": event.get("type"),
                "payload": event,
                "created_at": utc_now_iso(),
            },
            prefer="resolution=ignore-duplicates,return=minimal",
        )

    event_type = event.get("type")
    data_object = event.get("data", {}).get("object", {})
    user_id = (
        data_object.get("client_reference_id")
        or (data_object.get("metadata") or {}).get("supabase_user_id")
    )

    if event_type == "checkout.session.completed":
        subscription_id = data_object.get("subscription")
        subscription = stripe_request("GET", f"subscriptions/{subscription_id}") if subscription_id else None
        if subscription:
            upsert_subscription_from_stripe(subscription, user_id=user_id)
            posthog_capture(user_id or data_object.get("customer") or "stripe", "subscription_checkout_completed", {
                "stripe_customer_id": data_object.get("customer"),
                "stripe_subscription_id": subscription_id,
            })
    elif event_type in {"customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"}:
        subscription_id = data_object.get("id")
        existing = get_subscription_by_stripe_id(subscription_id) or get_subscription_by_customer(data_object.get("customer"))
        resolved_user_id = user_id or (existing or {}).get("user_id")
        upsert_subscription_from_stripe(data_object, user_id=resolved_user_id)
        posthog_capture(resolved_user_id or data_object.get("customer") or "stripe", "subscription_status_changed", {
            "status": data_object.get("status"),
            "stripe_subscription_id": subscription_id,
            "event_type": event_type,
        })
    elif event_type in {"invoice.payment_succeeded", "invoice.payment_failed"}:
        existing = get_subscription_by_customer(data_object.get("customer"))
        posthog_capture((existing or {}).get("user_id") or data_object.get("customer") or "stripe", event_type.replace(".", "_"), {
            "stripe_customer_id": data_object.get("customer"),
            "stripe_subscription_id": data_object.get("subscription"),
        })

    return {"received": True, "type": event_type}

@app.get("/api/data/status")
def api_data_status():
    return {
        "oanda": {
            "configured": oanda_configured(),
            "pricing_configured": bool(OANDA_API_KEY and OANDA_ACCOUNT_ID),
            "environment": OANDA_ENV,
            "base_url": oanda_base_url(),
            "instrument": OANDA_INSTRUMENT,
            "granularities": oanda_granularity_list(),
            "ingest_enabled": OANDA_INGEST_ENABLED,
            "ingest_seconds": OANDA_INGEST_SECONDS,
        },
        "market_data": {
            "source": MARKET_DATA_SOURCE,
            "supabase_source": SUPABASE_OHLCV_SOURCE,
            "supabase_limit": SUPABASE_OHLCV_LIMIT,
            "fallback_order": ["supabase", "oanda", "yfinance"] if MARKET_DATA_SOURCE == "auto" else [MARKET_DATA_SOURCE],
        },
        "macro_features": {
            "enabled": MACRO_FEATURES_ENABLED,
            "ingest_seconds": MACRO_FEATURES_INGEST_SECONDS,
        },
        "global_catalysts": {
            "enabled": GLOBAL_CATALYST_ENABLED,
            "timespan": GLOBAL_CATALYST_TIMESPAN,
            "min_headlines": GLOBAL_CATALYST_MIN_HEADLINES,
            "legacy_trade_catalyst_env_supported": True,
        },
        "supabase": {
            "configured": supabase_configured(),
        },
    }

@app.get("/api/macro/features")
def api_macro_features(request: Request, token: str = "", limit: int = 50, refresh: bool = False):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    return {
        "storage": {
            "configured": supabase_configured(),
            "enabled": MACRO_FEATURES_ENABLED,
            "ingest_seconds": MACRO_FEATURES_INGEST_SECONDS,
        },
        "current": get_macro_context() if refresh else None,
        "rows": fetch_macro_features(limit),
    }

@app.post("/api/macro/ingest")
def api_macro_ingest(request: Request, token: str = ""):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    if not MACRO_FEATURES_ENABLED:
        return error_response("macro_features_disabled", "Macro feature storage is disabled.", 503)
    result = insert_macro_features_to_supabase()
    return JSONResponse(result, status_code=200 if result.get("ok") else 503)

@app.get("/api/admin/readiness")
def api_admin_readiness(request: Request, token: str = "", backtest_days: int = 30):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    return JSONResponse(system_readiness_report(backtest_days=backtest_days))

@app.get("/api/data/spread")
def api_data_spread(request: Request, token: str = ""):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    pricing = get_cached_spread()
    if not pricing:
        return error_response("pricing_unavailable", "OANDA pricing is not configured or unavailable.", 503)
    return pricing

@app.post("/api/data/ingest")
def api_data_ingest(request: Request, token: str = "", granularities: str = "", count: int = 500):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    selected = oanda_granularity_list(granularities) if granularities else oanda_granularity_list()
    result = ingest_oanda_ohlcv(selected, count=max(1, min(count, 5000)))
    return JSONResponse(result, status_code=200 if result.get("ok") else 503)

@app.get("/health")
@app.head("/health")
def health():
    return {
        "status":   "ok",
        "time":     datetime.now(timezone.utc).isoformat(),
        "mode":     "EMA regime + 1H/15M/5M liquidity sweep + live news/event sentiment",
        "data":     "yfinance GC=F + DXY/10Y/VIX + GDELT/Fed RSS + scheduled macro events",
        "telegram": "configured ✅" if TG_TOKEN and TG_CHAT_ID else "NOT configured ❌",
        "email":    "configured ✅" if email_configured() else "NOT configured ❌",
        "integrations": service_status(),
        "trade_check_seconds": TRADE_CHECK_SECONDS,
        "trade_startup_delay_seconds": TRADE_STARTUP_DELAY_SECONDS,
        "watch_alerts": {
            "enabled": WATCH_ALERTS_ENABLED,
            "check_seconds": WATCH_CHECK_SECONDS,
            "startup_delay_seconds": WATCH_STARTUP_DELAY_SECONDS,
            "cooldown_minutes": WATCH_ALERT_COOLDOWN_MINUTES,
            "min_conditions": WATCH_MIN_CONDITIONS,
        },
        "london_alerts": {
            "enabled": LONDON_ALERTS_ENABLED,
            "check_seconds": LONDON_ALERT_CHECK_SECONDS,
            "startup_delay_seconds": LONDON_ALERT_STARTUP_DELAY_SECONDS,
            "session_window_ct": ict_session_windows(ICT_LONDON_ONLY_SESSIONS),
            "telegram_configured": bool(TG_TOKEN and TG_CHAT_ID),
        },
        "market_data": {
            "source": MARKET_DATA_SOURCE,
            "fallback_order": ["supabase", "oanda", "yfinance"] if MARKET_DATA_SOURCE == "auto" else [MARKET_DATA_SOURCE],
            "supabase_source": SUPABASE_OHLCV_SOURCE,
            "signal_cache_ttl_seconds": SIGNAL_CACHE_TTL_SECONDS,
        },
        "oanda": {
            "environment": OANDA_ENV,
            "instrument": OANDA_INSTRUMENT,
            "ingest_enabled": OANDA_INGEST_ENABLED,
            "pricing_configured": bool(OANDA_API_KEY and OANDA_ACCOUNT_ID),
        },
        "hard_filters": {
            "enabled": HARD_FILTERS_ENABLED,
            "cooldown_minutes": RECENT_SIGNAL_COOLDOWN_MINUTES,
            "max_spread_cents": MAX_SPREAD_CENTS,
            "spread_configured": XAUUSD_SPREAD_CENTS != "",
        },
        "signal_tuning": {
            "sweep_scan_bars_5m": LIQUIDITY_SWEEP_SCAN_BARS_5M,
            "sweep_scan_bars_15m": LIQUIDITY_SWEEP_SCAN_BARS_15M,
            "sweep_scan_bars_1h": LIQUIDITY_SWEEP_SCAN_BARS_1H,
            "order_block_impulse_atr": ORDER_BLOCK_IMPULSE_ATR,
            "order_block_min_score": ORDER_BLOCK_MIN_SCORE,
            "order_block_tested_min_score": ORDER_BLOCK_TESTED_MIN_SCORE,
            "order_block_test_tolerance_atr": ORDER_BLOCK_TEST_TOLERANCE_ATR,
            "technical_signal_threshold_5m": TECHNICAL_SIGNAL_THRESHOLD_5M,
            "technical_signal_threshold_15m": TECHNICAL_SIGNAL_THRESHOLD_15M,
            "technical_signal_threshold_1h": TECHNICAL_SIGNAL_THRESHOLD_1H,
            "stack_h1_bias_min_score": STACK_H1_BIAS_MIN_SCORE,
        },
        "signal_lifecycle": {
            "valid_minutes": SIGNAL_VALID_MINUTES,
            "invalidation_atr_buffer": SIGNAL_INVALIDATION_ATR_BUFFER,
        },
        "prop_firm": prop_firm_rules(),
        "notifications": {
            "email_backup_mode": EMAIL_BACKUP_MODE,
            "email_recipients": len(email_recipients()),
        },
    }

@app.get("/api/test-notification")
async def test_notification(request: Request, token: str = ""):
    if not admin_authorized(request, token):
        return JSONResponse({"error": "admin token required"}, status_code=403)
    if not (TG_TOKEN and TG_CHAT_ID) and not email_configured():
        return JSONResponse({"error": "No notification channel configured"}, status_code=400)
    result = await notify_send(
        "Test — XAU/USD Intelligence Bot",
        "🧪 <b>Test — XAU/USD Intelligence Bot</b>\n\n"
        "✅ Notification channel working.\n\n"
        f"Trade scanner interval: {max(1, TRADE_CHECK_SECONDS // 60)} minutes\n"
        f"Watch scanner interval: {max(1, WATCH_CHECK_SECONDS // 60)} minutes\n"
        "Trade alerts record only after the full 1H/15M/5M stack confirms; watch alerts are heads-up only.\n\n"
        "<i>Test only.</i>"
    )
    return result

@app.get("/api/test-email")
async def test_email(request: Request, token: str = ""):
    if not admin_authorized(request, token):
        return JSONResponse({"error": "admin token required"}, status_code=403)
    if not email_configured():
        return JSONResponse({"error": "Email not configured"}, status_code=400)
    ok = await email_send(
        "Test — XAU/USD Email Backup",
        "🧪 <b>Test — XAU/USD Email Backup</b>\n\n"
        "Email backup delivery is working.\n\n"
        "<i>Test only.</i>",
    )
    return {"sent": ok}

@app.get("/api/check-now")
async def check_now(request: Request, token: str = ""):
    if not admin_authorized(request, token):
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

            notification = await notify_send(
                f"XAU/USD trade — {decision.get('execution_timeframe')} {decision.get('direction')}",
                msg_trade_alert(trade_snapshot),
            )
            ok = notification["sent"]
            should_record = ok or supabase_configured()
            record, recorded = record_trade_alert(trade_snapshot, source="check_now") if should_record else (None, False)
            notifications.append({
                "sent": ok,
                "telegram_sent": notification["telegram"],
                "email_sent": notification["email"],
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


@app.get("/api/london/check-now")
@app.post("/api/london/check-now")
async def api_london_check_now(request: Request, token: str = ""):
    if not admin_authorized(request, token):
        return JSONResponse({"error": "admin token required"}, status_code=403)
    result = await send_london_telegram_signal(force=True, source="london_check_now")
    return JSONResponse(result, status_code=200 if result.get("sent") or not result.get("trade_alert_key") else 202)


@app.get("/api/trades")
def api_trades(refresh: bool = False, limit: int = 100):
    if refresh:
        data = get_both_signals(force=True)
        update_trade_results(data)
    return JSONResponse(trade_history_payload(limit=max(1, min(limit, 500))))

@app.post("/api/trades/{trade_id}/action")
async def api_trade_action(trade_id: str, request: Request, token: str = ""):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    try:
        body = await request.json()
    except Exception:
        body = {}
    trade, err = mark_trade_action(
        trade_id,
        str(body.get("action", "")),
        notes=body.get("notes"),
    )
    if err:
        status_code = 404 if err == "Trade not found." else 422
        return error_response("trade_action_failed", err, status_code)
    return {"trade": trade}

@app.get("/api/analytics/performance")
def api_analytics_performance(source: str = "auto", limit: int = 500):
    if source not in {"auto", "supabase", "local"}:
        return error_response("invalid_source", "source must be auto, supabase, or local.", 422)
    resolved_source = "local" if source == "local" else source
    return JSONResponse(analytics_payload(source=resolved_source, limit=max(1, min(limit, 2000))))

@app.get("/api/model-health")
def api_model_health(
    request: Request,
    token: str = "",
    source: str = "auto",
    limit: int = 1000,
    recent_days: int = 30,
    refresh_current: bool = False,
):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    if source not in {"auto", "supabase", "local"}:
        return error_response("invalid_source", "source must be auto, supabase, or local.", 422)
    resolved_source = "local" if source == "local" else source
    return JSONResponse(model_health_payload(
        source=resolved_source,
        limit=max(1, min(limit, 5000)),
        recent_days=recent_days,
        refresh_current=refresh_current,
    ))

@app.get("/api/analytics/feedback")
async def api_analytics_feedback(request: Request, limit: int = 500):
    if not supabase_configured():
        return error_response("supabase_missing", "Supabase is not configured.", 503)
    entitlement = await entitlement_for_request(request)
    if not entitlement["authenticated"]:
        return error_response("unauthorized", "Supabase user token required.", 401)
    if not entitlement["active"]:
        return error_response("subscription_required", "Active subscription required.", 402)
    payload = feedback_analytics_payload(entitlement["user"]["id"], limit=max(1, min(limit, 5000)))
    await track_request_event(request, "feedback_analytics_viewed", {"limit": limit})
    return JSONResponse(payload)

@app.get("/api/backtest/rule-replay")
def api_backtest_rule_replay(
    request: Request,
    token: str = "",
    interval: str = "15m",
    days: int = 30,
    max_checks: int = 1200,
    max_hold_bars: int = 0,
    include_filters: bool = True,
):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    result = run_rule_backtest(
        interval=interval,
        days=days,
        max_checks=max_checks,
        max_hold_bars=max_hold_bars or None,
        include_filters=include_filters,
    )
    status_code = 400 if result.get("error") else 200
    return JSONResponse(result, status_code=status_code)


@app.get("/api/backtest/supabase-stack")
def api_backtest_supabase_stack(
    request: Request,
    token: str = "",
    days: int = 30,
    max_checks: int = 2000,
    max_hold_bars: int = 0,
    include_filters: bool = True,
):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    result = run_supabase_stack_backtest(
        days=days,
        max_checks=max_checks,
        max_hold_bars=max_hold_bars or None,
        include_filters=include_filters,
    )
    if not result.get("error"):
        return JSONResponse(result)
    status_code = 503 if "Supabase" in result.get("error", "") else 400
    return JSONResponse(result, status_code=status_code)


@app.get("/api/backtest/ict-bos-retracement")
def api_backtest_ict_bos_retracement(
    request: Request,
    token: str = "",
    days: int = 60,
    max_checks: int = 5000,
    risk_per_trade_pct: float = 1.0,
):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    result = run_ict_bos_backtest(
        days=max(5, min(days, 60)),
        max_checks=max(100, min(max_checks, 20000)),
        risk_per_trade_pct=max(0.1, min(risk_per_trade_pct, 5.0)),
    )
    return JSONResponse(result, status_code=503 if result.get("error") else 200)


@app.get("/api/backtest/ict-london-bos")
def api_backtest_ict_london_bos(
    request: Request,
    token: str = "",
    days: int = 60,
    max_checks: int = 5000,
    risk_per_trade_pct: float = 1.0,
):
    if not admin_authorized(request, token):
        return error_response("admin_token_required", "admin token required", 403)
    result = run_ict_bos_backtest(
        days=max(5, min(days, 60)),
        max_checks=max(100, min(max_checks, 20000)),
        risk_per_trade_pct=max(0.1, min(risk_per_trade_pct, 5.0)),
        allowed_sessions=ICT_LONDON_ONLY_SESSIONS,
        mode="ict_london_only_bos_retracement_backtest",
    )
    return JSONResponse(result, status_code=503 if result.get("error") else 200)


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
.technical-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
.panel{background:#141410;border:1px solid #2d2b22;border-radius:8px;padding:13px;box-shadow:0 1px 0 rgba(255,255,255,.03) inset}.panel.tight{padding:10px}
.data-alert{border:1px solid rgba(239,106,91,.45);background:rgba(239,106,91,.08);color:#f2b1a9;border-radius:7px;padding:9px;margin-top:10px;font-size:12px;line-height:1.4}
.source-strip{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}.source-strip span{border:1px solid #343126;background:#10100c;border-radius:6px;padding:5px 7px;font-size:11px;color:#c9c0ae}.source-strip b{color:#ece7dc}
.empty-chart{height:92px;display:grid;place-items:center;border:1px dashed #343126;border-radius:7px;margin-top:8px;color:#817b70;font-size:11px}
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
@media(max-width:920px){main{padding:11px}.top,.hero,.four,.three,.two,.technical-grid{grid-template-columns:1fr}.direction{font-size:42px}.kv{grid-template-columns:repeat(2,1fr)}.stamp{text-align:left}.barrow{grid-template-columns:76px 1fr 62px}}
</style>
</head>
<body>
<main>
  <div class="top">
    <div>
      <h1>AURUM Risk OS</h1>
      <div class="sub">XAU/USD intraday cockpit: live news search, scheduled event risk, public-web sentiment, 1H EMA regime, 15M/5M liquidity sweep execution.</div>
      <div class="tabs-nav"><a class="tablink active" href="/">Cockpit</a><a class="tablink" href="/london">London</a><a class="tablink" href="/trades">Journal</a><a class="tablink" href="/ops">Ops</a></div>
    </div>
    <div class="stamp"><div id="status">loading...</div><button onclick="loadDashboard(true)">Refresh Feed</button><a class="navlink" href="/london">London</a><a class="navlink" href="/trades">Journal</a><a class="navlink" href="/ops">Ops</a></div>
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

function pctValue(value){
  const num=Number(value);
  return Number.isFinite(num)?num.toFixed(1)+'%':'--';
}

function biasSummaryMetrics(source,fallbackItems=[]){
  const items=fallbackItems||[];
  const bullish=Number.isFinite(Number(source?.bullish_count))?Number(source.bullish_count):items.filter(item=>String(item.bias||item.gold_bias||'').toLowerCase().includes('bull')).length;
  const bearish=Number.isFinite(Number(source?.bearish_count))?Number(source.bearish_count):items.filter(item=>String(item.bias||item.gold_bias||'').toLowerCase().includes('bear')).length;
  const neutral=Number.isFinite(Number(source?.neutral_count))?Number(source.neutral_count):Math.max(0,items.length-bullish-bearish);
  const total=Math.max(1,bullish+bearish+neutral);
  const bullishPct=Number.isFinite(Number(source?.bullish_pct))?Number(source.bullish_pct):(bullish/total*100);
  const bearishPct=Number.isFinite(Number(source?.bearish_pct))?Number(source.bearish_pct):(bearish/total*100);
  const neutralPct=Number.isFinite(Number(source?.neutral_pct))?Number(source.neutral_pct):(neutral/total*100);
  return metrics([
    ['Bullish',pctValue(bullishPct),'long',`${bullish}/${total} items`],
    ['Bearish',pctValue(bearishPct),'short',`${bearish}/${total} items`],
    ['Neutral',pctValue(neutralPct),'wait',`${neutral}/${total} items`],
    ['Score',`${source?.score>0?'+':''}${source?.score??0}`,(source?.score??0)>0?'long':(source?.score??0)<0?'short':'wait',source?.source||'live feed'],
  ]);
}

function dataHealthCard(key,label,data){
  const d=data?.[key]||{}, chartBars=(d.chart?.price||[]).length;
  const ok=!d.error&&Number(d.data_bars||0)>0&&chartBars>0;
  return `<div class="metric">
    <span class="label">${label} data</span>
    <b class="${ok?'long':'short'}">${ok?'ONLINE':'NO DATA'}</b>
    <div class="tiny muted">${esc(d.data_source||'no source')} ${esc(d.data_granularity||'')} · ${d.data_bars??0} bars · chart ${chartBars}</div>
    <div class="tiny muted">Last bar ${esc(d.bar_time||d.timestamp||'--')}</div>
    ${d.error?`<div class="tiny short">${esc(d.error)}</div>`:''}
  </div>`;
}

function renderDataHealth(data){
  return `<section class="panel tight">
    <div class="split">
      <div><div class="label">Market data health</div><div class="tiny muted">Dashboard must have 1H, 15M, and 5M candles before trade logic is valid.</div></div>
      <div class="risk ${data?.m5?.error?'short':'long'}">${data?.m5?.error?'M5 DOWN':'M5 OK'}</div>
    </div>
    <div class="grid three" style="margin-top:10px">
      ${dataHealthCard('h1','1H',data)}
      ${dataHealthCard('m15','15M',data)}
      ${dataHealthCard('m5','5M',data)}
    </div>
  </section>`;
}

function renderPropFirm(data){
  const prop=data.prop_firm||{}, rules=prop.rules||{}, state=prop.state||{}, candidate=prop.candidate||{}, sizing=candidate.sizing||{}, pass=candidate.pass_math||{};
  const ready=!!candidate.ready;
  const blockers=(candidate.blockers||[]).map(item=>`<tr><td>${esc(item)}</td></tr>`).join('')||'<tr><td class="long">All prop gates clear for the current model signal.</td></tr>';
  const guards=(prop.guards||[]).map(item=>`<span class="pill">${esc(item)}</span>`).join('');
  return `<div class="section-title"><h2>Prop Firm Trade Generator</h2><span class="${ready?'long':'wait'}">${esc(candidate.status||'WAIT')}</span></div>
  <section class="panel">
    <div class="split">
      <div>
        <div class="label">${esc(rules.firm_name||'Prop account')}</div>
        <div class="tf-title ${ready?dirCls(candidate.direction):'wait'}">${esc(candidate.action||'NO_TRADE')} ${ready?esc(candidate.direction||''):''}</div>
        <div class="reason">${esc(candidate.reason||'Waiting for a full-stack, rule-compliant setup.')}</div>
      </div>
      <div class="risk ${ready?'long':'wait'}">${ready?'READY':'WAIT'}</div>
    </div>
    <div class="grid four" style="margin-top:12px">${metrics([
      ['Account',money(rules.account_size),'',`${rules.profit_target_pct??'--'}% target`],
      ['Progress',`${state.target_progress_pct??0}%`,state.target_progress_pct>=75?'long':state.target_progress_pct<0?'short':'wait',`${money(state.target_remaining_usd)} remaining`],
      ['Phase',state.phase||'--',state.phase==='PROTECT'?'wait':state.phase==='TARGET_REACHED'?'long':'',`${state.phase_risk_pct??0}% risk mode`],
      ['Today',`${state.trades_today??0}/${rules.max_trades_per_day??0}`,state.losses_today>0?'wait':'',`${state.losses_today??0} losses · ${money(state.daily_risk_left_usd)} risk left`],
    ])}</div>
    <div class="grid four" style="margin-top:10px">${metrics([
      ['Entry',money(candidate.entry),'',candidate.entry_low!=null?`${money(candidate.entry_low)}-${money(candidate.entry_high)}`:'market/zone'],
      ['Stop',money(candidate.stop),'short',`${money(sizing.stop_distance)} stop distance`],
      ['Target',money(candidate.target),'long',`RR ${candidate.rr??'--'}`],
      ['Size',`${sizing.size_oz??0} oz`,'wait',`${money(sizing.risk_usd)} risk · ${sizing.risk_pct??0}%`],
    ])}</div>
    <div class="grid four" style="margin-top:10px">${metrics([
      ['TP1 $',money(sizing.target_profit_usd),'long','estimated if TP1 hits'],
      ['TP2 $',money(sizing.target_2_profit_usd),'long','estimated if TP2 hits'],
      ['R to target',pass.r_to_target??'--','wait',`${pass.two_r_wins_to_target??'--'} TP1 wins needed`],
      ['Confidence',`${candidate.confidence??0}%`,candidate.confidence>=rules.min_confidence?'long':'wait',`min ${rules.min_confidence??'--'}%`],
    ])}</div>
    <div class="scoreline">${guards}</div>
    <table style="margin-top:10px"><thead><tr><th>Gate blockers</th></tr></thead><tbody>${blockers}</tbody></table>
    <div class="tiny muted" style="margin-top:8px">${esc(candidate.disclaimer||'Research only.')} ${esc(state.note||'')}</div>
  </section>`;
}

function renderWatchSetups(data){
  const items=data.watch_signals||[];
  const rows=items.map(item=>{
    const filters=item.hard_filters||{}, blocked=(filters.blocked_reasons||[]).slice(0,2).join('; ');
    const missing=(item.failed_stack_checks||[]).slice(0,4).join(', ')||'--';
    const quality=esc(item.quality||'WATCH');
    return `<tr>
      <td><b class="${dirCls(item.direction)}">${esc(item.timeframe||'--')} ${esc(item.direction||'--')}</b><div class="tiny muted">${quality}</div></td>
      <td class="${cls(item.technical_score||0)}">${signed(item.technical_score)}</td>
      <td>${item.conditions_met??0}/${item.max_conditions??7}</td>
      <td>${money(item.price)}</td>
      <td>${esc(missing)}</td>
      <td class="${filters.allowed?'long':'wait'}">${filters.allowed?'clear':esc(blocked||'watch')}</td>
    </tr>`;
  }).join('')||'<tr><td colspan="6" class="muted">No active watch setups.</td></tr>';
  return `<div class="section-title"><h2>Watch Setups</h2><span class="muted small">${items.length} active</span></div>
  <section class="panel">
    <table><thead><tr><th>Setup</th><th>Score</th><th>Conditions</th><th>Price</th><th>Stack missing</th><th>Filters</th></tr></thead><tbody>${rows}</tbody></table>
  </section>`;
}

function renderTf(key,data){
  const d=data?.[key]||{}, ind=d.indicators||{}, comp=d.components||{}, sweep=d.liquidity_sweep||{};
  const scoreValue=Number.isFinite(Number(d.technical_score))?Number(d.technical_score):null;
  const score=scoreValue??0;
  const hasChart=(d.chart?.price||[]).length>0;
  const id='chart-'+key;
  const label=key==='h1'?'1H EMA regime':key==='m15'?'15M execution tape':'5M micro execution tape';
  return `<section class="panel">
    <div class="tf-head">
      <div><div class="label">${label}</div><div class="tf-title ${dirCls(d.setup_direction)}">${esc(d.setup_direction||'NONE')}</div></div>
      <div class="tf-score ${scoreValue==null?'wait':cls(score)}">${scoreValue==null?'--':`${score>0?'+':''}${score}`}</div>
    </div>
    ${d.error?`<div class="data-alert">${esc(d.error)}</div>`:''}
    <div class="source-strip">
      <span>Source <b>${esc(d.data_source||'--')}</b></span>
      <span>Granularity <b>${esc(d.data_granularity||'--')}</b></span>
      <span>Bars <b>${d.data_bars??0}</b></span>
      <span>Last <b>${esc(d.bar_time||'--')}</b></span>
    </div>
    ${hasChart?`<canvas id="${id}"></canvas>`:`<div class="empty-chart">No ${label} chart data</div>`}
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

function renderFibBot(bot){
  const plan=bot.plan||{}, inputs=bot.inputs||{}, leg=bot.leg||{}, reversal=bot.reversal||{};
  const conditions=bot.conditions||{}, failed=bot.failed_conditions||[];
  const conditionRows=Object.entries(conditions).map(([key,value])=>`<span class="pill">${esc(key.replaceAll('_',' '))} <b class="${value?'long':'short'}">${value?'YES':'NO'}</b></span>`).join('');
  const alignmentRows=(bot.alignments||[]).map(row=>`<tr>
    <td>${esc(row.timeframe)}</td>
    <td class="${dirCls(row.direction)}">${esc(row.direction||'NONE')}</td>
    <td>${money(row.price)}</td>
    <td>${money(row.ema9)} / ${money(row.ema21)}</td>
    <td>${money(row.ema50)} / ${money(row.ema200)}</td>
    <td class="${cls(row.ema21_slope||0)}">${row.ema21_slope??'--'}</td>
  </tr>`).join('')||'<tr><td colspan="6" class="muted">No EMA alignment data.</td></tr>';
  const failedText=failed.length?failed.map(item=>esc(item.replaceAll('_',' '))).join(', '):'all setup checks passed';
  return `<section class="panel">
    <div class="split">
      <div>
        <div class="label">Fibonacci EMA pullback strategy</div>
        <div class="tf-title ${dirCls(bot.signal)}">${esc(bot.signal||'WAIT')}</div>
        <div class="reason">${esc(bot.reason||'')}</div>
      </div>
      <div class="risk ${dirCls(bot.direction)}">${esc(bot.quality||'--')}</div>
    </div>
    <div class="grid four" style="margin-top:12px">${metrics([
      ['Entry zone',plan.entry_low==null?'--':`${money(plan.entry_low)}-${money(plan.entry_high)}`,'',`state ${bot.zone_state||'--'}`],
      ['Preferred entry',money(plan.entry),'wait',`current ${money(bot.price)}`],
      ['Stop',money(plan.stop),'short',plan.invalidates||'no active plan'],
      ['TP1 / TP2',`${money(plan.target)} / ${money(plan.target_2)}`,'long',`RR ${plan.rr??'--'} / ${plan.rr_tp2??'--'}`],
    ])}</div>
    <div class="grid four" style="margin-top:10px">${metrics([
      ['Impulse',leg.range==null?'--':money(leg.range),'blue',`${leg.impulse_atr??'--'}x ATR`],
      ['Swing low',money(leg.swing_low),'',leg.swing_low_time||'--'],
      ['Swing high',money(leg.swing_high),'',leg.swing_high_time||'--'],
      ['Size',plan.position_1pct_10k_oz==null?'--':plan.position_1pct_10k_oz+' oz','wait','1% risk on $10k'],
    ])}</div>
    <div class="scoreline">
      <span class="pill">Fib 50 <b>${money(plan.fib50)}</b></span>
      <span class="pill">Fib 61.8 <b>${money(plan.fib618)}</b></span>
      <span class="pill">Fib 78.6 <b>${money(plan.fib786)}</b></span>
      <span class="pill">Distance <b>${money(bot.distance_to_zone)}</b></span>
      <span class="pill">Confidence <b>${bot.confidence??0}%</b></span>
    </div>
    <div class="scoreline">${conditionRows}</div>
    <table style="margin-top:10px"><thead><tr><th>TF</th><th>Bias</th><th>Price</th><th>EMA9/21</th><th>EMA50/200</th><th>EMA21 slope</th></tr></thead><tbody>${alignmentRows}</tbody></table>
    <div class="tiny muted" style="margin-top:8px">Trigger: ${esc(reversal.reason||'--')} Failed checks: ${failedText}. ${esc(bot.source_note||'')}</div>
    <div class="tiny muted">Inputs: M15 ATR ${money(inputs.m15_atr14)}, ADX ${inputs.m15_adx14??'--'}, ATR% ${inputs.atr_pct??'--'}, lookback ${inputs.lookback_bars??'--'} bars.</div>
  </section>`;
}

function renderIctBot(bot){
  const trade=bot.trade||{}, zone=bot.zone||{}, bias=bot.bias||{}, levels=bot.levels||{}, checks=bot.checks||{}, bos=bot.bos||{};
  const prev=levels.previous_day||{}, asian=levels.asian||{}, range=levels.recent_range||{};
  const checkRows=Object.entries(checks).map(([key,value])=>`<span class="pill">${esc(key.replaceAll('_',' '))} <b class="${value?'long':'short'}">${value?'YES':'NO'}</b></span>`).join('');
  const failed=(bot.failed_checks||[]).map(item=>esc(item.replaceAll('_',' '))).join(', ')||'all checks passed';
  return `<section class="panel">
    <div class="split">
      <div>
        <div class="label">H4/Daily premium-discount BOS strategy</div>
        <div class="tf-title ${dirCls(bot.signal)}">${esc(bot.signal||'WAIT')}</div>
        <div class="reason">${esc(bot.reason||'')}</div>
      </div>
      <div class="risk ${dirCls(bot.direction)}">${esc(bot.quality||'--')}</div>
    </div>
    <div class="grid four" style="margin-top:12px">${metrics([
      ['Bias',bot.direction||'NONE',dirCls(bot.direction),bias.reason||'H4/Daily filter'],
      ['Session',bot.session||'--',bot.session==='OFF_SESSION'?'wait':'long','Plano CT windows only'],
      ['Zone',zone.low==null?'--':`${money(zone.low)}-${money(zone.high)}`,'',`${zone.timeframe||'--'} ${zone.type||'--'}`],
      ['BOS',bos.bos_level==null?'--':money(bos.bos_level),'blue',bos.bos_time||bos.reason||'--'],
    ])}</div>
    <div class="grid four" style="margin-top:10px">${metrics([
      ['Entry',money(trade.entry),'',trade.entry_low!=null?`${money(trade.entry_low)}-${money(trade.entry_high)}`:'waiting'],
      ['Stop',money(trade.stop),'short',trade.invalidates||'structural stop'],
      ['Target',money(trade.target),'long',trade.target_source||'liquidity pool'],
      ['RR / Size',`${trade.rr??'--'} / ${trade.position_1pct_10k_oz??'--'} oz`,'wait','1% risk on $10k'],
    ])}</div>
    <div class="grid four" style="margin-top:10px">${metrics([
      ['Prev day H/L',`${money(prev.high)} / ${money(prev.low)}`,'','liquidity levels'],
      ['Asian H/L',`${money(asian.high)} / ${money(asian.low)}`,'','00:00-07:00 UTC'],
      ['Recent range',`${money(range.high)} / ${money(range.low)}`,'','premium/discount source'],
      ['Failed',failed,'wait','active blockers'],
    ])}</div>
    <div class="scoreline">${checkRows}</div>
    <div class="tiny muted" style="margin-top:8px">${esc(bot.source_note||'Clean-room strategy implementation.')} ${esc((zone||{}).summary||'')}</div>
  </section>`;
}

function render(data){
  const final=data.final||{}, context=data.context||{}, macro=context.macro||{}, news=context.news||{}, people=context.people||{}, event=context.event_risk||{};
  const next=event.next_event||{}, trade=final.trade||{}, sweep=final.sweep||{}, ws=final.weighted_scores||{}, weights=final.weights||{}, study=data.event_study||{}, gridBot=data.trend_grid_bot||{}, fibBot=data.fib_ema_pullback||{}, ictBot=data.ict_bos_retracement||{}, globalCatalysts=context.global_catalysts||news.global_catalysts||context.trade_catalyst||{};
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
  const newsSummary=biasSummaryMetrics(news,news.headlines||[]);
  const peopleSummary=biasSummaryMetrics(people,people.samples||[]);
  const catalystMetrics=metrics([
    ['Gold impact',`${globalCatalysts.bias||'MIXED'} ${globalCatalysts.score>0?'+':''}${globalCatalysts.score??0}`,biasCls(globalCatalysts.bias),globalCatalysts.assessment||globalCatalysts.reason||'No active catalyst'],
    ['USD impact',`${globalCatalysts.usd_bias||'MIXED'} ${globalCatalysts.usd_score>0?'+':''}${globalCatalysts.usd_score??0}`,biasCls(globalCatalysts.usd_bias),'inverse read used for gold pressure'],
    ['Impact risk',globalCatalysts.risk||'UNKNOWN',impactCls(globalCatalysts.risk),globalCatalysts.source||'global catalysts'],
    ['Headlines',globalCatalysts.headline_count??0,globalCatalysts.active?'wait':'muted',globalCatalysts.status||''],
  ]);
  const catalystRows=(globalCatalysts.headlines||[]).slice(0,8).map(h=>`<tr>
    <td><b>${esc(h.region||'Global')}</b><div class="tiny muted">${esc(h.catalyst||h.category||'macro catalyst')}</div></td>
    <td class="${h.gold_score>0?'long':h.gold_score<0?'short':'wait'}">${esc(h.gold_bias||h.bias||'neutral')} ${h.gold_score>0?'+':''}${h.gold_score??0}</td>
    <td class="${h.usd_score>0?'long':h.usd_score<0?'short':'wait'}">${esc(h.usd_bias||'neutral')} ${h.usd_score>0?'+':''}${h.usd_score??0}</td>
    <td class="${impactCls(h.impact)}">${esc(h.impact||'LOW')}</td>
    <td>${esc(h.title)}<div class="tiny muted">${esc(h.source||'news')} · ${esc(h.reason||'')}</div><div class="playbook tiny">${esc(h.playbook||'')}</div></td>
  </tr>`).join('')||'<tr><td colspan="5" class="muted">No active country or macro catalyst detected from the live tape.</td></tr>';
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
        <span class="pill">Global catalysts <b class="${biasCls(globalCatalysts.bias)}">${globalCatalysts.active?esc(globalCatalysts.bias||'MIXED'):'NONE'}</b></span>
        <span class="pill">Public sentiment <b class="${biasCls(people.bias)}">${esc(people.bias||'MIXED')}</b></span>
        <span class="pill">Bull/Bear <b>${people.bullish_count??0}/${people.bearish_count??0}</b></span>
      </div>
    </section>
  </div>

  ${renderDataHealth(data)}
  ${renderPropFirm(data)}
  ${renderWatchSetups(data)}

  <div class="section-title"><h2>Execution Playbook</h2><span class="muted small">${esc(sweep.summary||'Waiting for sweep')}</span></div>
  <div class="grid two">
    <section class="panel"><div class="grid four">${tradeMetrics}</div></section>
    <section class="panel"><div class="label">Next scheduled catalyst</div><div class="split"><div><b>${esc(next.name||'None')}</b><div class="small muted">${esc(next.period||'')} ${next.time_utc?'. '+esc(next.time_utc):''}</div><div class="playbook tiny">${esc(next.gold_playbook||'')}</div></div><div class="risk ${impactCls(event.level)}">${esc(next.eta||'')}</div></div></section>
  </div>

  <div class="section-title"><h2>Fibonacci EMA Pullback Entry</h2><span class="muted small">M15/H1/H4 EMA alignment + 0.50-0.786 pullback zone</span></div>
  ${renderFibBot(fibBot)}

  <div class="section-title"><h2>H4 Daily BOS Retracement Entry</h2><span class="muted small">premium/discount OB-FVG + 5M BOS retrace + daily guardrails</span></div>
  ${renderIctBot(ictBot)}

  <div class="section-title"><h2>EMA200 ADX ATR Grid Signal</h2><span class="muted small">trend-following, ATR-spaced, capped, non-martingale</span></div>
  ${renderGridBot(gridBot)}

  <div class="section-title"><h2>Technical State</h2><span class="muted small">1H regime + 15M zone + 5M micro sweep/reclaim</span></div>
  <div class="grid technical-grid">${renderTf('h1',data)}${renderTf('m15',data)}${renderTf('m5',data)}</div>

  <div class="section-title"><h2>Live News And Public Sentiment</h2><span class="muted small">${esc(context.timestamp||'')}</span></div>
  <section class="panel">
    <div class="label">Global country and macro catalyst impact</div>
    <div class="grid four" style="margin-top:10px">${catalystMetrics}</div>
    <table style="margin-top:10px"><thead><tr><th>Region</th><th>Gold</th><th>USD</th><th>Impact</th><th>Headline / playbook</th></tr></thead><tbody>${catalystRows}</tbody></table>
  </section>
  <div class="grid two">
    <section class="panel"><div class="label">Current market news impact</div><div class="grid four" style="margin-top:10px;margin-bottom:10px">${newsSummary}</div>${newsRows}</section>
    <section class="panel"><div class="label">People/public-web sentiment proxy</div><div class="grid four" style="margin-top:10px;margin-bottom:10px">${peopleSummary}</div>${peopleRows}</section>
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
  document.getElementById('status').textContent=force?'searching live feeds...':'loading cached state...';
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
loadDashboard(false);
setInterval(()=>loadDashboard(false),300000);
</script>
</body>
</html>"""

LONDON_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0b0b09">
<title>London XAU/USD BOS</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b0b09;color:#ece7dc;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:16px}.top{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:start;margin-bottom:12px}
h1{font-size:18px;margin:0;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.sub{font-size:12px;color:#9f9a8f;margin-top:4px}.stamp{font-size:12px;color:#9f9a8f;text-align:right}
button,a.btn,.tablink{border:1px solid #494534;background:#171712;color:#ece7dc;border-radius:6px;padding:8px 12px;cursor:pointer;font-weight:700;text-decoration:none;display:inline-block;margin-left:6px}button:hover,a.btn:hover,.tablink:hover{border-color:#b99036}button.danger{border-color:#73443d;color:#f0b4ac}button.mini{padding:5px 8px;margin:2px;font-size:11px}
input{border:1px solid #494534;background:#10100c;color:#ece7dc;border-radius:6px;padding:8px 9px;max-width:170px}.tabs-nav{display:flex;gap:8px;margin:12px 0}.tablink{margin-left:0}.tablink.active{background:#2a2518;border-color:#b99036;color:#f2c76b}
.panel{background:#141410;border:1px solid #2d2b22;border-radius:8px;padding:13px;box-shadow:0 1px 0 rgba(255,255,255,.03) inset}.grid{display:grid;gap:10px}.two{grid-template-columns:repeat(2,1fr)}.four{grid-template-columns:repeat(4,1fr)}
.metric{background:#10100c;border:1px solid #28251d;border-radius:7px;padding:10px}.metric span{display:block;font-size:10px;color:#9f9a8f;text-transform:uppercase;letter-spacing:.13em}.metric b{display:block;font-size:20px;margin-top:4px;overflow-wrap:anywhere}
.section-title{display:flex;justify-content:space-between;gap:10px;align-items:center;margin:16px 0 8px}.section-title h2{font-size:12px;margin:0;text-transform:uppercase;letter-spacing:.14em;color:#d8d0bf}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:10px}th,td{text-align:left;border-bottom:1px solid #2c2a21;padding:8px 6px;vertical-align:top}th{color:#9f9a8f;font-size:10px;text-transform:uppercase;letter-spacing:.1em}tr:last-child td{border-bottom:0}
.long{color:#4ecb71}.short{color:#ef6a5b}.wait{color:#d7a93f}.blue{color:#6bb7d7}.muted{color:#9f9a8f}.tiny{font-size:11px}.reason{font-size:13px;color:#c9c0ae;line-height:1.45}.tf-title{font-size:28px;font-weight:900}.risk{font-size:26px;font-weight:900}.pill{border:1px solid #393528;background:#1b1a14;border-radius:999px;padding:5px 8px;font-size:11px;display:inline-block;margin:3px}.actions{display:flex;gap:6px;flex-wrap:wrap}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
@media(max-width:900px){main{padding:11px}.top,.two,.four{grid-template-columns:1fr}.stamp{text-align:left}table{font-size:11px;display:block;overflow-x:auto;white-space:nowrap}}
</style>
</head>
<body>
<main>
  <div class="top">
    <div>
      <h1>London XAU/USD BOS</h1>
      <div class="sub">H4/Daily bias, premium-discount OB/FVG, 5M BOS retracement, London CT window only.</div>
      <div class="tabs-nav"><a class="tablink" href="/">Cockpit</a><a class="tablink active" href="/london">London</a><a class="tablink" href="/trades">Journal</a><a class="tablink" href="/ops">Ops</a></div>
    </div>
    <div class="stamp"><div id="status">loading...</div><input id="adminToken" type="password" placeholder="ADMIN_TOKEN"><button onclick="saveToken()">Save</button><button onclick="loadLondon(true)">Refresh</button></div>
  </div>
  <div class="panel">
    <div class="actions">
      <button onclick="loadLondon(true)">Signal</button>
      <button onclick="runTelegramCheck()">Telegram Check</button>
      <button onclick="runBacktest()">Backtest</button>
      <button onclick="runMt5(true)">MT5 Dry Run</button>
      <button class="danger" onclick="runMt5(false)">MT5 Demo Execute</button>
    </div>
    <div id="actionResult" class="tiny muted" style="margin-top:8px"></div>
  </div>
  <div id="app" style="margin-top:10px"><div class="panel">Loading London strategy...</div></div>
  <div class="section-title"><h2>Backtest</h2><span class="muted tiny">GC=F 5M replay</span></div>
  <div id="backtest" class="panel muted">No replay loaded.</div>
</main>
<script>
const money=v=>v==null?'--':'$'+Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const dirCls=d=>(d||'').includes('LONG')?'long':(d||'').includes('SHORT')?'short':'wait';
const statusCls=v=>v===true||v==='CONFIRMED'||v==='ONLINE'?'long':v===false||v==='BLOCKED'||v==='NO_DATA'?'short':'wait';
const flagText=v=>v===true?'ON':v===false?'OFF':'--';
const flagCls=v=>v===true?'long':v===false?'wait':'muted';

function saveToken(){
  const value=document.getElementById('adminToken').value.trim();
  if(value)localStorage.setItem('xau_admin_token',value);
  document.getElementById('status').textContent=value?'token saved':'token empty';
}

function token(){
  return localStorage.getItem('xau_admin_token')||document.getElementById('adminToken').value.trim()||'';
}

function authHeaders(extra={}){
  const adminToken=token();
  return adminToken?{...extra,'Authorization':`Bearer ${adminToken}`}:{...extra};
}

async function fetchJson(path, options={}){
  const response=await fetch(path,{...options,headers:{...authHeaders(options.headers||{})}});
  const text=await response.text();
  let data={};
  try{data=JSON.parse(text)}catch(_){data={raw:text}}
  if(!response.ok)throw new Error(data?.error?.message||data?.blockers?.join('; ')||data?.error||text.slice(0,220)||('HTTP '+response.status));
  return data;
}

function metric(label,value,klass='',hint=''){
  return `<div class="metric"><span>${esc(label)}</span><b class="${klass}">${value}</b><div class="tiny muted">${esc(hint)}</div></div>`;
}

function metrics(items){
  return `<div class="grid four">${items.map(item=>metric(...item)).join('')}</div>`;
}

function renderSignal(bot, mt5){
  const trade=bot.trade||{}, zone=bot.zone||{}, bias=bot.bias||{}, levels=bot.levels||{}, checks=bot.checks||{}, bos=bot.bos||{};
  const prev=levels.previous_day||{}, asian=levels.asian||{}, range=levels.recent_range||{};
  const checkRows=Object.entries(checks).map(([key,value])=>`<span class="pill">${esc(key.replaceAll('_',' '))} <b class="${value?'long':'short'}">${value?'YES':'NO'}</b></span>`).join('');
  const blockers=(bot.failed_checks||[]).map(item=>`<tr><td>${esc(item.replaceAll('_',' '))}</td></tr>`).join('')||'<tr><td class="long">All London setup checks passed.</td></tr>';
  const moduleState=mt5?.module_available?'ONLINE':'NO MODULE';
  const connected=mt5?.connected?'CONNECTED':'OFFLINE';
  document.getElementById('app').innerHTML=`
    <div class="grid two">
      <section class="panel">
        <div class="section-title"><h2>London Signal</h2><span class="${dirCls(bot.direction)}">${esc(bot.quality||'WAIT')}</span></div>
        <div class="tf-title ${dirCls(bot.signal)}">${esc(bot.signal||'WAIT')}</div>
        <div class="reason">${esc(bot.reason||'')}</div>
        ${metrics([
          ['Bias',bot.direction||'NONE',dirCls(bot.direction),bias.reason||'H4/Daily filter'],
          ['Session',bot.session||'--',bot.session==='LONDON_OPEN_CT'?'long':'wait',(bot.session_windows_ct||[]).join(', ')],
          ['Zone',zone.low==null?'--':`${money(zone.low)}-${money(zone.high)}`,'',`${zone.timeframe||'--'} ${zone.type||'--'}`],
          ['BOS',bos.bos_level==null?'--':money(bos.bos_level),'blue',bos.bos_time||bos.reason||'--'],
        ])}
        <div style="margin-top:10px">${metrics([
          ['Entry',money(trade.entry),'',trade.entry_low!=null?`${money(trade.entry_low)}-${money(trade.entry_high)}`:'waiting'],
          ['Stop',money(trade.stop),'short',trade.invalidates||'structural stop'],
          ['Target',money(trade.target),'long',trade.target_source||'liquidity pool'],
          ['RR / Size',`${trade.rr??'--'} / ${trade.position_1pct_10k_oz??'--'} oz`,'wait','1% risk on $10k'],
        ])}</div>
        <div style="margin-top:10px">${metrics([
          ['Prev day H/L',`${money(prev.high)} / ${money(prev.low)}`,'','liquidity'],
          ['Asian H/L',`${money(asian.high)} / ${money(asian.low)}`,'','session range'],
          ['Recent range',`${money(range.high)} / ${money(range.low)}`,'','premium/discount'],
          ['Model',bot.model||'--','wait','clean-room rules'],
        ])}</div>
        <div style="margin-top:8px">${checkRows}</div>
        <table><thead><tr><th>Active blockers</th></tr></thead><tbody>${blockers}</tbody></table>
      </section>
      <section class="panel">
        <div class="section-title"><h2>MT5 Demo Automation</h2><span class="${statusCls(mt5?.connected)}">${esc(connected)}</span></div>
        ${metrics([
          ['Module',moduleState,statusCls(mt5?.module_available),mt5?.error||'MetaTrader5 Python bridge'],
          ['Autotrade',flagText(mt5?.autotrade_enabled),flagCls(mt5?.autotrade_enabled),`${mt5?.check_seconds??'--'}s worker`],
          ['Manual execute',flagText(mt5?.manual_execution_enabled),flagCls(mt5?.manual_execution_enabled),'protected endpoint'],
          ['Demo mode',flagText(mt5?.demo_mode),mt5?.demo_mode===false?'short':flagCls(mt5?.demo_mode),'real account guard'],
        ])}
        <div style="margin-top:10px">${metrics([
          ['Symbol',mt5?.symbol||'--','','MT5 market watch symbol'],
          ['Risk/trade',`${mt5?.risk_per_trade_pct??'--'}%`,'wait',`max ${mt5?.max_volume??'--'} lots`],
          ['Account',mt5?.account?.login||'--','','trade mode '+(mt5?.account?.trade_mode??'--')],
          ['Equity',money(mt5?.account?.equity),'',mt5?.account?.server||''],
        ])}</div>
        <table><thead><tr><th>Setting</th><th>Value</th></tr></thead><tbody>
          <tr><td class="mono">MT5_SYMBOL</td><td>${esc(mt5?.symbol||'--')}</td></tr>
          <tr><td class="mono">MT5_MAGIC</td><td>${esc(mt5?.magic||'--')}</td></tr>
          <tr><td class="mono">MT5_MAX_ENTRY_DRIFT_USD</td><td>${money(mt5?.max_entry_drift_usd)}</td></tr>
          <tr><td class="mono">Session</td><td>${esc((mt5?.session_window_ct||[]).join(', ')||'02:00-05:00 CT')}</td></tr>
        </tbody></table>
      </section>
    </div>`;
}

function renderBacktest(data){
  const m=data.metrics||{}, s=data.settings||{}, rows=(data.trades||[]).slice(-12).reverse().map(t=>`<tr>
    <td>${esc(t.entry_time)}</td>
    <td class="${dirCls(t.direction)}">${esc(t.direction)}</td>
    <td class="${t.result==='TP'?'long':t.result==='SL'?'short':'wait'}">${esc(t.result)}</td>
    <td>${money(t.entry)}</td><td class="short">${money(t.stop)}</td><td class="long">${money(t.target)}</td>
    <td class="${Number(t.r)>=0?'long':'short'}">${Number(t.r).toFixed(2)}R</td>
    <td>${esc(t.zone_timeframe)} ${esc(t.zone_type)}</td>
  </tr>`).join('')||'<tr><td colspan="8" class="muted">No trades.</td></tr>';
  document.getElementById('backtest').innerHTML=`
    ${metrics([
      ['Trades',m.total_signals??0,'','London only'],
      ['Win rate',`${m.win_rate??'--'}%`,Number(m.win_rate)>=50?'long':'short','closed trades'],
      ['Net R',m.net_r??'--',Number(m.net_r)>=0?'long':'short',`avg ${m.avg_r??'--'}R`],
      ['Profit factor',m.profit_factor??'--',Number(m.profit_factor)>=1.5?'long':'short',`DD ${m.max_drawdown_pct??'--'}%`],
      ['Return',`${m.return_pct??'--'}%`,Number(m.return_pct)>=0?'long':'short',`final ${money(m.final_equity)}`],
      ['Sessions',(s.sessions_ct||[]).join(', '),'wait',`${s.days??'--'} days`],
      ['Risk',`${s.risk_per_trade_pct??'--'}%`,'wait',`max ${s.max_trades_per_day??'--'} trades/day`],
      ['Sample',data.sampling?.checked_bars??'--','','checked 5M bars'],
    ])}
    <table><thead><tr><th>Entry time</th><th>Side</th><th>Result</th><th>Entry</th><th>SL</th><th>TP</th><th>R</th><th>Zone</th></tr></thead><tbody>${rows}</tbody></table>
    <div class="tiny muted" style="margin-top:8px">${esc(data.disclaimer||'Research backtest only.')}</div>`;
}

async function loadLondon(force=false){
  document.getElementById('status').textContent=force?'refreshing...':'loading...';
  try{
    const signal=await fetchJson(`/api/strategy/ict-london-bos?refresh=${force?'1':'0'}&ts=${Date.now()}`);
    let mt5={};
    try{mt5=await fetchJson(`/api/mt5/status?include_account=true&ts=${Date.now()}`)}catch(error){mt5={module_available:false,connected:false,error:error.message}}
    renderSignal(signal,mt5);
    document.getElementById('status').textContent='updated '+new Date().toLocaleTimeString();
  }catch(error){
    document.getElementById('status').textContent='error';
    document.getElementById('app').innerHTML=`<div class="panel short">${esc(error.message)}</div>`;
  }
}

async function runBacktest(){
  document.getElementById('actionResult').textContent='running London replay...';
  try{
    const data=await fetchJson(`/api/backtest/ict-london-bos?days=60&max_checks=20000&risk_per_trade_pct=1&ts=${Date.now()}`);
    renderBacktest(data);
    document.getElementById('actionResult').textContent=`London replay: ${data.metrics?.total_signals??0} trades, ${data.metrics?.win_rate??'--'}% win rate, ${data.metrics?.net_r??'--'}R.`;
  }catch(error){
    document.getElementById('actionResult').textContent=error.message;
  }
}

async function runTelegramCheck(){
  document.getElementById('actionResult').textContent='checking London Telegram signal...';
  try{
    const data=await fetchJson(`/api/london/check-now?ts=${Date.now()}`,{method:'POST'});
    const blockers=(data.blockers||[]).join(', ');
    const status=data.sent?'sent':data.already_recorded?'already recorded':'not sent';
    document.getElementById('actionResult').textContent=`Telegram ${status} · ${data.signal?.signal||'WAIT'}${blockers?' · '+blockers:''}`;
    await loadLondon(false);
  }catch(error){
    document.getElementById('actionResult').textContent=error.message;
  }
}

async function runMt5(dryRun){
  document.getElementById('actionResult').textContent=dryRun?'checking MT5 dry run...':'sending demo order...';
  try{
    const data=await fetchJson(`/api/mt5/london-execute?dry_run=${dryRun?'true':'false'}&refresh=true&ts=${Date.now()}`,{method:'POST'});
    const size=data.sizing?.volume??'--';
    const result=data.mt5_result?.retcode?` retcode ${data.mt5_result.retcode}`:'';
    document.getElementById('actionResult').textContent=`${dryRun?'Dry run':'Demo execute'} ${data.ok?'OK':'blocked'} · ${data.signal?.signal||'WAIT'} · ${size} lots${result}`;
    await loadLondon(false);
  }catch(error){
    document.getElementById('actionResult').textContent=error.message;
  }
}

document.getElementById('adminToken').value=localStorage.getItem('xau_admin_token')||'';
loadLondon(false);
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
button,a.btn,.tablink{border:1px solid #494534;background:#171712;color:#ece7dc;border-radius:6px;padding:8px 12px;cursor:pointer;font-weight:700;text-decoration:none;display:inline-block;margin-left:6px}button:hover,a.btn:hover,.tablink:hover{border-color:#b99036}button.mini{padding:5px 8px;margin:2px;font-size:11px}input{border:1px solid #494534;background:#10100c;color:#ece7dc;border-radius:6px;padding:8px 9px;max-width:170px}
.tabs-nav{display:flex;gap:8px;margin:12px 0}.tablink{margin-left:0}.tablink.active{background:#2a2518;border-color:#b99036;color:#f2c76b}
.panel{background:#141410;border:1px solid #2d2b22;border-radius:8px;padding:13px;box-shadow:0 1px 0 rgba(255,255,255,.03) inset}.grid{display:grid;gap:10px}.four{grid-template-columns:repeat(4,1fr)}
.metric{background:#10100c;border:1px solid #28251d;border-radius:7px;padding:10px}.metric span{display:block;font-size:10px;color:#9f9a8f;text-transform:uppercase;letter-spacing:.13em}.metric b{display:block;font-size:22px;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:12px}th,td{text-align:left;border-bottom:1px solid #2c2a21;padding:8px 6px;vertical-align:top}th{color:#9f9a8f;font-size:10px;text-transform:uppercase;letter-spacing:.1em}tr:last-child td{border-bottom:0}
.long{color:#4ecb71}.short{color:#ef6a5b}.wait{color:#d7a93f}.muted{color:#9f9a8f}.tiny{font-size:11px}.pill{border:1px solid #393528;background:#1b1a14;border-radius:999px;padding:5px 8px;font-size:11px;display:inline-block}.actions{white-space:normal;min-width:120px}
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
      <div class="tabs-nav"><a class="tablink" href="/">Cockpit</a><a class="tablink" href="/london">London</a><a class="tablink active" href="/trades">Journal</a><a class="tablink" href="/ops">Ops</a></div>
    </div>
    <div class="stamp"><div id="status">loading...</div><input id="adminToken" type="password" placeholder="ADMIN_TOKEN"><button onclick="saveToken()">Save</button><button onclick="loadTrades(true)">Refresh</button><a class="btn" href="/">Cockpit</a><a class="btn" href="/london">London</a><a class="btn" href="/ops">Ops</a></div>
  </div>
  <div id="app" class="panel">Loading trade journal...</div>
</main>
<script>
const money=v=>v==null?'--':'$'+Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const signed=v=>v==null?'--':(Number(v)>0?'+':'')+Number(v).toFixed(2);
const cls=v=>v==='TP'||Number(v)>0?'long':v==='SL'||v==='INVALIDATED'||Number(v)<0?'short':'wait';
const dirCls=d=>(d||'').includes('LONG')?'long':(d||'').includes('SHORT')?'short':'wait';
const actionCls=a=>a==='TAKEN'?'long':a==='SKIPPED'?'short':'wait';

function metrics(summary){
  return `<div class="grid four">
    <div class="metric"><span>Total</span><b>${summary.total||0}</b></div>
    <div class="metric"><span>Open</span><b class="wait">${summary.open||0}</b></div>
    <div class="metric"><span>Win rate</span><b class="${summary.win_rate>=50?'long':'short'}">${summary.win_rate||0}%</b></div>
    <div class="metric"><span>Net R / $10k</span><b class="${cls(summary.net_r)}">${signed(summary.net_r)}R · ${money(summary.pnl_1pct_10k_usd)}</b></div>
    <div class="metric"><span>Taken</span><b class="long">${summary.taken||0}</b></div>
    <div class="metric"><span>Skipped</span><b class="short">${summary.skipped||0}</b></div>
    <div class="metric"><span>Pending</span><b class="wait">${summary.pending||0}</b></div>
    <div class="metric"><span>Invalid/Expired</span><b class="wait">${(summary.invalidated||0)+(summary.expired||0)}</b></div>
  </div>`;
}

function saveToken(){
  const value=document.getElementById('adminToken').value.trim();
  if(value)localStorage.setItem('xau_admin_token',value);
  document.getElementById('status').textContent=value?'token saved':'token empty';
}

function token(){
  return localStorage.getItem('xau_admin_token')||document.getElementById('adminToken')?.value.trim()||'';
}

function authHeaders(extra={}){
  const adminToken=token();
  return adminToken?{...extra,'Authorization':`Bearer ${adminToken}`}:{...extra};
}

async function markAction(id,action){
  const adminToken=token();
  if(!adminToken){document.getElementById('status').textContent='ADMIN_TOKEN required';return;}
  const notes=action==='SKIPPED'?prompt('Skip reason?', ''):'';
  const response=await fetch(`/api/trades/${encodeURIComponent(id)}/action`,{
    method:'POST',
    headers:authHeaders({'Content-Type':'application/json'}),
    body:JSON.stringify({action,notes})
  });
  if(!response.ok){
    const text=await response.text();
    throw new Error(text.slice(0,160));
  }
  await loadTrades(false);
}

function render(data){
  const summary=data.summary||{}, trades=data.trades||[];
  const rows=trades.map(t=>`<tr>
    <td><span class="pill ${cls(t.result)}">${esc(t.result||'OPEN')}</span><div class="tiny muted">${esc(t.id)}</div></td>
    <td><span class="pill ${actionCls(t.user_action||'PENDING')}">${esc(t.user_action||'PENDING')}</span><div class="tiny muted">${esc(t.user_notes||'')}</div></td>
    <td><b class="${dirCls(t.direction)}">${esc(t.direction)}</b><div class="tiny muted">${esc(t.quality||'')}</div></td>
    <td>${money(t.entry)}<div class="tiny muted">${esc(t.opened_at||'')}</div></td>
    <td class="short">${money(t.stop)}</td>
    <td class="long">${money(t.target)}</td>
    <td>${money(t.closed_price)}<div class="tiny muted">${esc(t.closed_at||'open')}</div></td>
    <td class="${cls(t.r_multiple)}">${t.r_multiple==null?'--':signed(t.r_multiple)+'R'}<div class="tiny">${money(t.pnl_1pct_10k_usd)}</div></td>
    <td>${esc(t.sweep_summary||'')}<div class="tiny muted">${esc(t.execution_timeframe||'15M')} ${esc(t.opened_bar||'')}</div></td>
    <td class="actions"><button class="mini" onclick="markAction('${esc(t.id)}','TAKEN').catch(e=>document.getElementById('status').textContent=e.message)">Taken</button><button class="mini" onclick="markAction('${esc(t.id)}','SKIPPED').catch(e=>document.getElementById('status').textContent=e.message)">Skip</button><button class="mini" onclick="markAction('${esc(t.id)}','PENDING').catch(e=>document.getElementById('status').textContent=e.message)">Reset</button></td>
  </tr>`).join('')||'<tr><td colspan="10" class="muted">No executed trade alerts recorded yet.</td></tr>';
  document.getElementById('app').innerHTML=`${metrics(summary)}
    <table><thead><tr><th>Result</th><th>Decision</th><th>Side</th><th>Entry</th><th>SL</th><th>TP</th><th>Exit</th><th>R / P&L</th><th>Trigger</th><th>Action</th></tr></thead><tbody>${rows}</tbody></table>
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
document.getElementById('adminToken').value=localStorage.getItem('xau_admin_token')||'';
loadTrades(false);
setInterval(()=>loadTrades(false),300000);
</script>
</body>
</html>"""

OPS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0b0b09">
<title>AURUM Ops</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b0b09;color:#ece7dc;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:16px}.top{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:start;margin-bottom:12px}
h1{font-size:18px;margin:0;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.sub{font-size:12px;color:#9f9a8f;margin-top:4px}.stamp{font-size:12px;color:#9f9a8f;text-align:right}
button,a.btn,.tablink{border:1px solid #494534;background:#171712;color:#ece7dc;border-radius:6px;padding:8px 12px;cursor:pointer;font-weight:700;text-decoration:none;display:inline-block;margin-left:6px}button:hover,a.btn:hover,.tablink:hover{border-color:#b99036}button.mini{padding:5px 8px;margin:2px;font-size:11px}
input,select{border:1px solid #494534;background:#10100c;color:#ece7dc;border-radius:6px;padding:8px 9px;max-width:190px}.tabs-nav{display:flex;gap:8px;margin:12px 0}.tablink{margin-left:0}.tablink.active{background:#2a2518;border-color:#b99036;color:#f2c76b}
.panel{background:#141410;border:1px solid #2d2b22;border-radius:8px;padding:13px;box-shadow:0 1px 0 rgba(255,255,255,.03) inset}.grid{display:grid;gap:10px}.two{grid-template-columns:repeat(2,1fr)}.four{grid-template-columns:repeat(4,1fr)}
.metric{background:#10100c;border:1px solid #28251d;border-radius:7px;padding:10px}.metric span{display:block;font-size:10px;color:#9f9a8f;text-transform:uppercase;letter-spacing:.13em}.metric b{display:block;font-size:22px;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:10px}th,td{text-align:left;border-bottom:1px solid #2c2a21;padding:8px 6px;vertical-align:top}th{color:#9f9a8f;font-size:10px;text-transform:uppercase;letter-spacing:.1em}tr:last-child td{border-bottom:0}
.long{color:#4ecb71}.short{color:#ef6a5b}.wait{color:#d7a93f}.muted{color:#9f9a8f}.tiny{font-size:11px}.section-title{display:flex;justify-content:space-between;align-items:center;margin:16px 0 8px}.section-title h2{font-size:12px;margin:0;text-transform:uppercase;letter-spacing:.14em}
.pill{border:1px solid #393528;background:#1b1a14;border-radius:999px;padding:5px 8px;font-size:11px;display:inline-block}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.actions{display:flex;gap:6px;flex-wrap:wrap}
@media(max-width:900px){main{padding:11px}.top,.two,.four{grid-template-columns:1fr}.stamp{text-align:left}table{font-size:11px;display:block;overflow-x:auto;white-space:nowrap}}
</style>
</head>
<body>
<main>
  <div class="top">
    <div>
      <h1>AURUM Ops</h1>
      <div class="sub">Readiness, model health, notification tests, and protected maintenance actions.</div>
      <div class="tabs-nav"><a class="tablink" href="/">Cockpit</a><a class="tablink" href="/london">London</a><a class="tablink" href="/trades">Journal</a><a class="tablink active" href="/ops">Ops</a></div>
    </div>
    <div class="stamp"><div id="status">loading...</div><input id="adminToken" type="password" placeholder="ADMIN_TOKEN"><button onclick="saveToken()">Save</button><button onclick="loadOps()">Refresh</button></div>
  </div>
  <div class="panel">
    <div class="actions">
      <select id="source"><option value="auto">auto</option><option value="local">local</option><option value="supabase">supabase</option></select>
      <input id="recentDays" type="number" min="7" max="180" value="30">
      <label class="pill"><input id="refreshCurrent" type="checkbox" style="max-width:14px;padding:0;margin-right:5px">current</label>
      <button onclick="loadOps()">Load Health</button>
      <button onclick="runAction('/api/test-notification')">Test Notify</button>
      <button onclick="runAction('/api/test-email')">Test Email</button>
      <button onclick="runAction('/api/check-now')">Check Now</button>
      <button onclick="runBacktest()">Stack Backtest</button>
      <button onclick="runMacroIngest()">Snapshot Macro</button>
    </div>
    <div class="tiny muted" id="actionResult" style="margin-top:8px"></div>
  </div>
  <div id="app" style="margin-top:10px"><div class="panel">Loading ops state...</div></div>
</main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const cls=v=>v==='ready'||v===true?'long':v==='watch'?'wait':v==='blocked'||v===false?'short':'muted';
const signed=v=>v==null?'--':(Number(v)>0?'+':'')+Number(v).toFixed(2);

function saveToken(){
  const value=document.getElementById('adminToken').value.trim();
  if(value)localStorage.setItem('xau_admin_token',value);
  document.getElementById('status').textContent=value?'token saved':'token empty';
}

function token(){
  return localStorage.getItem('xau_admin_token')||document.getElementById('adminToken').value.trim()||'';
}

function authHeaders(){
  const adminToken=token();
  return adminToken?{'Authorization':`Bearer ${adminToken}`}:{};
}

async function fetchJson(path, options={}){
  const adminToken=token();
  if(!adminToken)throw new Error('ADMIN_TOKEN required');
  const glue=path.includes('?')?'&':'?';
  const response=await fetch(`${path}${glue}ts=${Date.now()}`,{
    ...options,
    headers:{...authHeaders(),...(options.headers||{})},
  });
  const text=await response.text();
  let data={};
  try{data=JSON.parse(text)}catch(_){data={raw:text}}
  if(!response.ok)throw new Error(data?.error?.message||data?.error||text.slice(0,180)||('HTTP '+response.status));
  return data;
}

function metrics(items){
  return `<div class="grid four">${items.map(([label,value,klass,hint])=>`<div class="metric"><span>${esc(label)}</span><b class="${klass||''}">${esc(value)}</b><div class="tiny muted">${esc(hint||'')}</div></div>`).join('')}</div>`;
}

function checksTable(rows){
  return `<table><thead><tr><th>Gate</th><th>Status</th><th>Severity</th><th>Metric</th><th>Detail</th></tr></thead><tbody>${(rows||[]).map(row=>`<tr>
    <td class="mono">${esc(row.name)}</td>
    <td class="${cls(row.passed)}">${row.passed?'PASS':'FAIL'}</td>
    <td>${esc(row.severity||'')}</td>
    <td>${esc(row.metric??row.threshold??'')}</td>
    <td>${esc(row.detail||'')}</td>
  </tr>`).join('')||'<tr><td colspan="5" class="muted">No checks.</td></tr>'}</tbody></table>`;
}

function readinessTable(rows){
  return `<table><thead><tr><th>Check</th><th>Status</th><th>Severity</th><th>Detail</th></tr></thead><tbody>${(rows||[]).map(row=>`<tr>
    <td class="mono">${esc(row.name)}</td>
    <td class="${cls(row.passed)}">${row.passed?'PASS':'FAIL'}</td>
    <td>${esc(row.severity||'')}</td>
    <td>${esc(row.detail||'')}</td>
  </tr>`).join('')||'<tr><td colspan="4" class="muted">No checks.</td></tr>'}</tbody></table>`;
}

function render(readiness, health){
  const all=health.metrics?.all||{}, recent=health.metrics?.recent||{}, sample=health.sample||{}, current=health.current||{};
  const distribution=health.distribution||{}, distResult=distribution.result||{}, distConfidence=distribution.confidence_tier||{};
  const macro=readiness.data?.latest_macro_features||{}, macroCfg=readiness.data?.macro_features||{};
  document.getElementById('app').innerHTML=`
    <div class="grid two">
      <section class="panel">
        <div class="section-title"><h2>Readiness</h2><span class="${cls(readiness.status)}">${esc(readiness.status)}</span></div>
        ${metrics([
          ['Critical fails',readiness.critical_failures??0,readiness.critical_failures?'short':'long','launch blockers'],
          ['Warnings',readiness.warnings??0,readiness.warnings?'wait':'long','non-blocking'],
          ['Backtest days',readiness.data?.backtest_days??'--','','coverage window'],
          ['Email backup',readiness.data?.notifications?.email_recipients??0,'wait','recipients'],
        ])}
        ${readinessTable(readiness.checks)}
      </section>
      <section class="panel">
        <div class="section-title"><h2>Model Health</h2><span class="${cls(health.status)}">${esc(health.status)}</span></div>
        ${metrics([
          ['Closed',sample.closed_signals??0,sample.closed_signals>=health.thresholds?.min_closed_signals?'long':'wait','sample size'],
          ['Win rate',(all.win_rate??0)+'%',(all.win_rate??0)>=health.thresholds?.min_win_rate?'long':'short','all closed'],
          ['Avg R',signed(all.avg_r),(all.avg_r??0)>=health.thresholds?.min_avg_r?'long':'short','expectancy'],
          ['Profit factor',all.profit_factor??'--',(all.profit_factor??0)>=health.thresholds?.min_profit_factor?'long':'short','gross win/loss'],
        ])}
        ${checksTable(health.gates)}
      </section>
    </div>
    <div class="grid two" style="margin-top:10px">
      <section class="panel">
        <div class="section-title"><h2>Recent Window</h2><span>${esc(sample.recent_days)} days</span></div>
        ${metrics([
          ['Recent closed',sample.recent_closed_signals??0,'wait','closed signals'],
          ['Recent win rate',(recent.win_rate??0)+'%',recent.win_rate>=50?'long':'short','closed only'],
          ['Recent avg R',signed(recent.avg_r),recent.avg_r>=0?'long':'short','degradation check'],
          ['Max losing streak',all.max_losing_streak??0,all.max_losing_streak<=health.thresholds?.max_losing_streak?'long':'short','all history'],
        ])}
        <div class="section-title"><h2>Distribution</h2></div>
        <table><thead><tr><th>Bucket</th><th>Counts</th></tr></thead><tbody>
          <tr><td>Result</td><td>${esc(JSON.stringify(distResult))}</td></tr>
          <tr><td>Confidence</td><td>${esc(JSON.stringify(distConfidence))}</td></tr>
          <tr><td>Direction</td><td>${esc(JSON.stringify(distribution.direction||{}))}</td></tr>
        </tbody></table>
      </section>
      <section class="panel">
        <div class="section-title"><h2>Current Model</h2><span class="${cls(current.ok)}">${current.ok?'OK':'not loaded'}</span></div>
        ${current.ok?metrics([
          ['Decision',current.decision?.direction||'--',current.decision?.direction?.includes('LONG')?'long':current.decision?.direction?.includes('SHORT')?'short':'wait',current.decision?.quality||''],
          ['Price',current.price??'--','','snapshot'],
          ['H1/M15/M5',`${current.scores?.h1??'--'} / ${current.scores?.m15??'--'} / ${current.scores?.m5??'--'}`,'wait','technical scores'],
          ['Hard filters',current.hard_filters?.allowed?'allowed':'blocked',current.hard_filters?.allowed?'long':'short','current state'],
        ]):'<div class="muted">Use refresh_current=true by loading health with current snapshot enabled from API when needed.</div>'}
        <div class="section-title"><h2>Macro Features</h2><span class="${cls(!!macro.ts)}">${macro.ts?'stored':'not stored'}</span></div>
        ${metrics([
          ['Bias',macro.macro_bias||'--',macro.macro_bias==='BULLISH'?'long':macro.macro_bias==='BEARISH'?'short':'wait',macro.status||''],
          ['Score',macro.macro_score??'--',Number(macro.macro_score||0)>0?'long':Number(macro.macro_score||0)<0?'short':'wait','macro model'],
          ['DXY / 10Y / VIX',`${macro.dxy??'--'} / ${macro.us10y??'--'} / ${macro.vix??'--'}`,'','latest row'],
          ['Storage',macroCfg.enabled?'enabled':'disabled',macroCfg.enabled?'long':'muted',`${macroCfg.ingest_seconds??'--'}s cadence`],
        ])}
        <div class="section-title"><h2>Next Actions</h2></div>
        <table><tbody>${(health.next_actions||readiness.next_actions||[]).slice(0,12).map(item=>`<tr><td>${esc(item)}</td></tr>`).join('')||'<tr><td class="muted">No actions.</td></tr>'}</tbody></table>
      </section>
    </div>`;
}

async function loadOps(){
  document.getElementById('status').textContent='loading...';
  try{
    const days=Number(document.getElementById('recentDays').value||30);
    const source=document.getElementById('source').value;
    const current=document.getElementById('refreshCurrent').checked;
    const readiness=await fetchJson(`/api/admin/readiness?backtest_days=${days}`);
    const health=await fetchJson(`/api/model-health?source=${encodeURIComponent(source)}&limit=1000&recent_days=${days}&refresh_current=${current?'true':'false'}`);
    render(readiness,health);
    document.getElementById('status').textContent='updated '+new Date().toLocaleTimeString();
  }catch(error){
    document.getElementById('status').textContent='error';
    document.getElementById('app').innerHTML=`<div class="panel short">${esc(error.message)}</div>`;
  }
}

async function runAction(path){
  try{
    const data=await fetchJson(path);
    document.getElementById('actionResult').textContent=JSON.stringify(data).slice(0,360);
  }catch(error){
    document.getElementById('actionResult').textContent=error.message;
  }
}

async function runBacktest(){
  try{
    const data=await fetchJson('/api/backtest/supabase-stack?days=30&max_checks=1000&include_filters=true');
    document.getElementById('actionResult').textContent=`${data.mode||'backtest'} ${data.metrics?.total_signals??0} signals, avg R ${data.metrics?.avg_r??'--'}, trades/month ${data.metrics?.estimated_trades_per_month??'--'}`;
  }catch(error){
    document.getElementById('actionResult').textContent=error.message;
  }
}

async function runMacroIngest(){
  try{
    const data=await fetchJson('/api/macro/ingest',{method:'POST'});
    const record=data.record||{};
    document.getElementById('actionResult').textContent=`macro ${record.macro_bias||'stored'} score ${record.macro_score??'--'} at ${record.ts||'now'}`;
    await loadOps();
  }catch(error){
    document.getElementById('actionResult').textContent=error.message;
  }
}

document.getElementById('adminToken').value=localStorage.getItem('xau_admin_token')||'';
loadOps();
</script>
</body>
</html>"""

@app.get("/trades", response_class=HTMLResponse)
def trades_page():
    return HTMLResponse(TRADES_HTML)

@app.get("/london", response_class=HTMLResponse)
def london_page():
    return HTMLResponse(LONDON_HTML)

@app.get("/ops", response_class=HTMLResponse)
def ops_page():
    return HTMLResponse(OPS_HTML)

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
