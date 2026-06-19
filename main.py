#!/usr/bin/env python3
"""
APEX Bybit Futures Trading Bot — v4.2 "Pullback, Protected & Aware"
====================================================================
WHAT CHANGED FROM v4.1 (entry accuracy + liq safety + news + smarter learning):

  [E1] NEW ENTRY MODEL — coherent TREND + TIMING scoring.
       v4.1's 6 signals were almost ALL monotonic with price (EMA stack,
       MACD, VWAP, 1H EMA) so the score peaked exactly when a move was
       already mature → the bot bought the TOP. v4.2 splits the six signals:
         TREND (direction/permission): Trend4H, MACD4H, Trend1H
         TIMING (pullback into value): Pullback, Bounce, Confirm
       TIMING signals only fire on a retrace toward value, so a blow-off top
       scores low and can't reach the threshold. The bot now enters pullbacks.

  [E2] EXTENSION GUARD (hard gate) — never enter when price is more than
       MAX_EXTENSION_ATR away from the 1H EMA50. Backstop for [E1].

  [E3] SAFE DYNAMIC LEVERAGE — permanently fixes "SL below liquidation".
       Leverage is derived from the SL distance so liquidation always sits a
       safe buffer beyond the stop. Wide stop → lower leverage automatically.
       Verified again vs the exchange's real liqPrice after the fill.

  [E4] STRUCTURAL STOPS — SL below the recent swing low (long) / above the
       swing high (short), bounded by ATR. Tighter, more logical stops →
       better R:R on pullback entries.

  [E5] RUNNER-AWARE EXITS — v4.1 could close the whole position (incl. the
       TP1 runner) on a blind 0.8R grab, capping winners. v4.2:
         pre-TP1  → reversal / momentum / funding / time, plus a STALL exit
                    that needs a profit AND a reversal (no blind grab).
         post-TP1 → breakeven already protects it; only a STRONG reversal
                    (>=4/5) or a long flat hold closes the runner.

  [E6] HIGHER FREQUENCY — relaxed the volume filter (a 1.2x spike used to
       block the low-volume pullback candles the new model wants) to a dead-
       volume floor; SL min relaxed from flat 2% to an ATR floor; optional
       market entries (USE_MAKER_ENTRY=false) for more fills.

  [E7] NEWS AWARENESS (fail-safe) — pulls recent crypto news (free
       CryptoCompare/CoinDesk feed). Fresh breaking news (< NEWS_BLOCK_MINUTES)
       on a coin → skip the entry (avoids news whipsaw). Older news is just
       reported in Discord. Any failure is ignored; trading never blocks.

  [E8] SMARTER LEARNING — each trade now stores regime + extension; the
       weekly report adds exit-reason P&L, long-vs-short win rates, and the
       per-coin auto-threshold so you can see the bot adapting itself.

  [E9] RICHER DISCORD — entries report signals, regime, R:R, extension,
       leverage, liquidation price + safety margin, USDT risk, and news.

Preserved unchanged from v4.1: split-speed loop, partial TP + breakeven,
closed-PnL reconciliation, persistent/atomic storage, original-risk exits,
HTTP retry session, signed-GET fix, catch-all main loop, /status + /journal,
health heartbeat.

NOTE: entry signal NAMES changed (Trend4H_bull, …), so old signal weights in
learning_state.json become inert and new ones start at 1.0. Coin profiles and
trade history are KEPT. The bot relearns weights over the coming weeks.
"""

import os
import json
import math
import time
import hmac
import hashlib
import logging
import datetime
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import ta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY             = os.environ.get("BYBIT_API_KEY", "")
API_SECRET          = os.environ.get("BYBIT_API_SECRET", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

BASE_URL = "https://api.bybit.com"

# [C2] Persistent data dir — on Railway: add a Volume mounted at /data and
#      set env var DATA_DIR=/data
DATA_DIR = os.environ.get("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)

DYNAMIC_SCAN_TOP_N     = 50
DYNAMIC_VOLUME_MIN_USD = 50_000_000
STABLECOIN_BASES       = {"USDC","BUSD","TUSD","USDP","DAI","FDUSD","PYUSD"}

RISK_PER_TRADE        = 0.015          # fallback / lowest tier
MAX_LEVERAGE          = 10
MAX_OPEN_POSITIONS    = 3
DAILY_LOSS_LIMIT      = 0.05
WEEKLY_LOSS_LIMIT     = 0.10
CONFLUENCE_THRESHOLD  = 4

# [S1] Split-speed loop
FAST_INTERVAL_SECONDS = 300            # exits / TP1 / breakeven every 5 min
ENTRY_SCAN_MINUTES    = (0, 15, 30, 45)  # entry scans at candle closes
WATCHLIST_TTL_SECONDS = 3600           # refresh watchlist hourly

# [S3/E6] Entry execution
USE_MAKER_ENTRY       = os.environ.get("USE_MAKER_ENTRY", "true").lower() == "true"
ENTRY_FILL_TIMEOUT_S  = 90             # wait per attempt for PostOnly fill
ENTRY_MAX_ATTEMPTS    = 2              # re-quote once, then skip

# [S4] Partial take-profit
TP1_R                 = 1.5
TP1_CLOSE_FRACTION    = 0.5

LONG_ENABLED  = True
SHORT_ENABLED = True

# ---- [E1/E2/E4] Entry model ----
MAX_EXTENSION_ATR     = 1.2            # hard gate: max price distance from 1H EMA50
MIN_RR                = 1.5            # minimum reward:risk to place an order
SL_ATR_MULT           = 2.0           # ATR-based stop distance
TP_ATR_MULT           = 6.0           # raw ATR target (capped to structure)
SL_SWING_LOOKBACK     = 8             # 1H candles for swing low/high
SL_SWING_BUFFER_ATR   = 0.3           # stop sits this far beyond the swing
SL_MIN_DIST_PCT       = 0.010         # noise floor for the stop (was flat 0.02)

# ---- [E3] Liquidation safety ----
MAINT_MARGIN          = 0.005         # ~0.5% maintenance margin (conservative)
LIQ_BUFFER            = 0.02          # keep liq >= this fraction beyond the SL

# ---- [E5] Exit thresholds ----
PRE_TP1_REVERSAL      = 3             # full-position reversal exit
RUNNER_EXIT_THRESHOLD = 4             # runner exits only on STRONG reversal
RUNNER_TIME_EXIT_HRS  = 12            # close a flat runner to free a slot

# ---- [E6] Frequency ----
VOL_FLOOR_RATIO       = 0.6           # skip only near-dead candles (was 1.2 spike)

# ---- [E7] News awareness ----
NEWS_ENABLED          = os.environ.get("NEWS_ENABLED", "true").lower() == "true"
NEWS_API_URL          = "https://min-api.cryptocompare.com/data/v2/news/"
NEWS_TTL_SECONDS      = 900           # refresh news cache every 15 min
NEWS_RECENT_HOURS     = 2             # "breaking" window
NEWS_BLOCK_MINUTES    = int(os.environ.get("NEWS_BLOCK_MINUTES", "30"))  # skip entry if news younger

MAX_CORRELATION_POSITIONS = 2
BTC_CORRELATED = {
    "ETHUSDT","SOLUSDT","BNBUSDT","AVAXUSDT",
    "LINKUSDT","DOTUSDT","MATICUSDT","LTCUSDT",
    "ADAUSDT","ATOMUSDT","NEARUSDT","APTUSDT",
}

COIN_BLACKLIST = {
    "USDCUSDT","TUSDUSDT","BUSDUSDT","USDTUSDT",
    "FDUSDUSDT","LDOUSDT","STETHUSDT","WBTCUSDT",
    "SHIBUSDT","PEPEUSDT","FLOKIUSDT","BTTUSDT","EVAAUSDT",
}
MAX_SPREAD_PCT = 0.0015

QUICK_PROFIT_R    = 0.8               # STALL exit threshold (needs reversal too)
TIME_EXIT_HOURS   = 6
TIME_EXIT_MIN_R   = 0.3
REOPEN_COOLDOWN_HRS = float(os.environ.get("REOPEN_COOLDOWN_HRS", "2"))  # [X1] no re-fire same coin
SETTLE_MINUTES      = float(os.environ.get("SETTLE_MINUTES", "25"))      # [X3] weak exits gated until trade settles; strong reversal always acts

# [BO] Breakout momentum mode — independent asymmetric path (low win-rate, runner-driven).
# Fully self-contained: set BREAKOUT_ENABLED=false to disable and behave exactly like before.
BREAKOUT_ENABLED        = os.environ.get("BREAKOUT_ENABLED", "true").lower() == "true"
BREAKOUT_LOOKBACK       = int(os.environ.get("BREAKOUT_LOOKBACK", "20"))      # 1H bars defining the range high/low
BREAKOUT_VOL_MULT       = float(os.environ.get("BREAKOUT_VOL_MULT", "1.5"))   # breakout bar volume must exceed avg * this
BREAKOUT_RISK_PCT       = float(os.environ.get("BREAKOUT_RISK_PCT", "0.01"))  # 1% risk per breakout trade (half of pullback)
BREAKOUT_SL_BUFFER_ATR  = float(os.environ.get("BREAKOUT_SL_BUFFER_ATR", "0.5"))  # SL sits this far below the broken level
BREAKOUT_TRAIL_LOOKBACK = int(os.environ.get("BREAKOUT_TRAIL_LOOKBACK", "3"))     # swing low/high bars the trail rides under
BREAKOUT_TP_ATR         = float(os.environ.get("BREAKOUT_TP_ATR", "12"))      # far safety-net TP only; the trail does the real exit work
FUNDING_SPIKE_PCT = 0.0005
FUNDING_RATE_MAX  = 0.001
FUNDING_RATE_MIN  = -0.001

TRADE_JOURNAL_FILE  = os.path.join(DATA_DIR, "trade_journal.json")
LEARNING_STATE_FILE = os.path.join(DATA_DIR, "learning_state.json")

RECONCILE_LOOKBACK_DAYS = 7
STALE_TRADE_DAYS        = 3

_instrument_cache: dict = {}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(os.path.join(DATA_DIR, "bot.log"))],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP session with retries (stability)
# ---------------------------------------------------------------------------

SESSION = requests.Session()
_retry = Retry(total=3, backoff_factor=1.0,
               status_forcelist=[429, 500, 502, 503, 504],
               allowed_methods=frozenset(["GET", "POST"]))
SESSION.mount("https://", HTTPAdapter(max_retries=_retry))
SESSION.mount("http://", HTTPAdapter(max_retries=_retry))

# ---------------------------------------------------------------------------
# Learning state
# ---------------------------------------------------------------------------

DEFAULT_LEARNING_STATE = {
    "signal_weights": {
        "Buy":  {"Trend4H_bull":1.0,"MACD4H_bull":1.0,"Trend1H_bull":1.0,
                 "Pullback_bull":1.0,"Bounce_bull":1.0,"Confirm_bull":1.0},
        "Sell": {"Trend4H_bear":1.0,"MACD4H_bear":1.0,"Trend1H_bear":1.0,
                 "Pullback_bear":1.0,"Bounce_bear":1.0,"Confirm_bear":1.0},
    },
    "coin_profiles": {},
    "last_updated": None,
    "total_trades_analyzed": 0,
}

MIN_TRADES_TO_LEARN     = 20
MIN_TRADES_PER_SIGNAL   = 8
MIN_TRADES_PER_COIN     = 10
MIN_TRADES_FOR_HOURS    = 15
MAX_WEIGHT              = 1.5
MIN_WEIGHT              = 0.5
WEIGHT_MEAN_REVERSION   = 0.9


def _atomic_write_json(path: str, obj) -> None:
    """[C2] Write to temp file then atomically replace — no corrupt files."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def load_learning_state() -> dict:
    state = None
    if os.path.exists(LEARNING_STATE_FILE):
        try:
            with open(LEARNING_STATE_FILE) as f:
                state = json.load(f)
        except Exception as e:
            log.warning("Could not load learning state: %s", e)
    if state is None:
        state = json.loads(json.dumps(DEFAULT_LEARNING_STATE))
    # ensure structure + migrate to v4.2 signal names (new ones start at 1.0)
    state.setdefault("signal_weights", {})
    for side in ("Buy", "Sell"):
        if side not in state["signal_weights"]:
            state["signal_weights"][side] = {}
        for sig, w in DEFAULT_LEARNING_STATE["signal_weights"][side].items():
            state["signal_weights"][side].setdefault(sig, w)
    state.setdefault("coin_profiles", {})
    state.setdefault("total_trades_analyzed", 0)
    state.setdefault("last_updated", None)
    return state


def save_learning_state(state: dict) -> None:
    _atomic_write_json(LEARNING_STATE_FILE, state)


# ---------------------------------------------------------------------------
# Trade Journal
# ---------------------------------------------------------------------------

def load_journal() -> list:
    if os.path.exists(TRADE_JOURNAL_FILE):
        try:
            with open(TRADE_JOURNAL_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning("Journal load failed: %s", e)
    return []


def save_journal(journal: list) -> None:
    _atomic_write_json(TRADE_JOURNAL_FILE, journal)


def log_trade(entry: dict) -> None:
    j = load_journal()
    j.append(entry)
    save_journal(j)


def _ts_to_ms(iso_str: str) -> int:
    try:
        dt = datetime.datetime.fromisoformat(str(iso_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


def _find_open_trade(symbol: str, direction: str) -> tuple:
    """Return (journal, index) of most recent open trade, index=-1 if none."""
    journal = load_journal()
    for i in range(len(journal)-1, -1, -1):
        t = journal[i]
        if (t.get("symbol")==symbol and t.get("direction")==direction
                and t.get("outcome") is None):
            return journal, i
    return journal, -1


def update_trade_outcome(symbol: str, direction: str, pnl: float,
                         exit_reason: str = "bot_exit",
                         exit_price: Optional[float] = None,
                         closed_ms: Optional[int] = None,
                         add_tp1: bool = True) -> None:
    """add_tp1=True adds stored TP1 partial profit (bot exits);
    add_tp1=False means pnl is already the full total (reconciliation)."""
    journal, i = _find_open_trade(symbol, direction)
    if i < 0:
        log.warning("update_trade_outcome: no open journal entry for %s %s", symbol, direction)
        return
    trade = journal[i]
    total = pnl + (float(trade.get("tp1_realized") or 0) if add_tp1 else 0.0)
    closed_dt = (datetime.datetime.utcfromtimestamp(closed_ms/1000)
                 if closed_ms else datetime.datetime.utcnow())
    trade["outcome"]    = "win" if total > 0 else "loss"
    trade["pnl"]        = round(total, 4)
    trade["closed_at"]  = closed_dt.isoformat()
    trade["exit_reason"]= exit_reason
    if exit_price is not None:
        trade["exit_price"] = exit_price
    save_journal(journal)
    log.info("Outcome: %s %s pnl=%.4f -> %s (%s)",
             symbol, direction, total, trade["outcome"], exit_reason)


# ---------------------------------------------------------------------------
# [C1] Closed-PnL reconciliation
# ---------------------------------------------------------------------------

def get_closed_pnl_records(start_ms: int) -> list:
    records, cursor = [], ""
    for _ in range(5):
        params = {"category": "linear", "limit": "100", "startTime": str(start_ms)}
        if cursor:
            params["cursor"] = cursor
        data = _signed_get("/v5/position/closed-pnl", params)
        if not data or data.get("retCode") != 0:
            break
        result = data.get("result", {})
        records.extend(result.get("list", []))
        cursor = result.get("nextPageCursor", "")
        if not cursor:
            break
    return records


def reconcile_closed_trades() -> int:
    journal = load_journal()
    pending = [t for t in journal if t.get("outcome") is None and t.get("order_result")]
    if not pending:
        return 0

    open_now = get_open_positions()
    now_ms = int(time.time() * 1000)
    earliest = min((_ts_to_ms(t.get("timestamp","")) for t in pending), default=now_ms)
    lookback_floor = now_ms - RECONCILE_LOOKBACK_DAYS * 86_400_000
    start_ms = max(min(earliest - 60_000, now_ms), lookback_floor)

    records = get_closed_pnl_records(start_ms) or []
    def _rec_dir(r): return "Buy" if r.get("side") == "Sell" else "Sell"
    used_ids: set = set()
    fixed = 0

    for trade in pending:
        sym, d = trade.get("symbol",""), trade.get("direction","")
        if open_now.get(sym) == d:
            continue
        t_ms = _ts_to_ms(trade.get("timestamp",""))
        matches = sorted(
            [r for r in records
             if r.get("symbol")==sym and _rec_dir(r)==d
             and r.get("orderId") not in used_ids
             and int(r.get("updatedTime", 0)) >= t_ms - 60_000],
            key=lambda r: int(r.get("updatedTime", 0)),
        )
        if matches:
            for r in matches:
                used_ids.add(r.get("orderId"))
            try:
                pnl = sum(float(r.get("closedPnl", 0)) for r in matches)
                exit_px = float(matches[-1].get("avgExitPrice", 0)) or None
            except Exception:
                pnl, exit_px = 0.0, None
            update_trade_outcome(
                sym, d, pnl, exit_reason="exchange_close",
                exit_price=exit_px,
                closed_ms=int(matches[-1].get("updatedTime", now_ms)),
                add_tp1=False,
            )
            fixed += 1
            emoji = "✅" if pnl > 0 else "🛑"
            discord_notify(f"{emoji} **Reconciled {d} {sym}** | exchange close | "
                           f"PnL `{pnl:+.2f}` USDT (incl. fees)")
        else:
            if t_ms and now_ms - t_ms > STALE_TRADE_DAYS * 86_400_000:
                j2 = load_journal()
                for t2 in reversed(j2):
                    if (t2.get("symbol")==sym and t2.get("direction")==d
                            and t2.get("outcome") is None
                            and t2.get("timestamp")==trade.get("timestamp")):
                        t2["outcome"] = "unknown"
                        t2["exit_reason"] = "stale_unmatched"
                        save_journal(j2)
                        log.warning("Marked stale trade unknown: %s %s", sym, d)
                        break
    if fixed:
        log.info("Reconciled %d exchange-closed trades", fixed)
    return fixed


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def discord_notify(message: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        SESSION.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
    except Exception as e:
        log.warning("Discord failed: %s", e)


# ---------------------------------------------------------------------------
# [E7] News awareness — fail-safe, never blocks trading on error
# ---------------------------------------------------------------------------

_news_cache = {"ts": 0.0, "by_coin": {}, "count": 0}

def refresh_news_cache() -> None:
    """Fetch recent crypto news (free CryptoCompare/CoinDesk feed).
    Fail-open: any error is logged and ignored — trading continues."""
    if not NEWS_ENABLED:
        return
    if time.time() - _news_cache["ts"] < NEWS_TTL_SECONDS and _news_cache["count"]:
        return
    try:
        resp = SESSION.get(NEWS_API_URL, params={"lang": "EN"}, timeout=10)
        data = resp.json()
        items = data.get("Data", []) if isinstance(data, dict) else []
        by_coin: dict = {}
        now = time.time()
        for it in items:
            try:
                ts = float(it.get("published_on", 0) or 0)
            except Exception:
                ts = 0.0
            age_h = (now - ts) / 3600 if ts else 999.0
            cats = (it.get("categories", "") or "").upper()
            title = it.get("title", "") or ""
            for tag in cats.split("|"):
                tag = tag.strip()
                if not tag or not tag.isalnum() or len(tag) > 6:
                    continue
                sym = tag + "USDT"
                rec = by_coin.setdefault(sym, {"recent": 0, "title": "", "age_h": 999.0})
                if age_h <= NEWS_RECENT_HOURS:
                    rec["recent"] += 1
                if age_h < rec["age_h"]:
                    rec["age_h"] = age_h
                    rec["title"] = title
        _news_cache.update(ts=time.time(), by_coin=by_coin, count=len(items))
        log.info("News refreshed: %d items, %d coins tagged", len(items), len(by_coin))
    except Exception as e:
        log.warning("News refresh failed (ignored): %s", e)

def news_flag(symbol: str) -> dict:
    """{'breaking': bool, 'title': str, 'age_h': float} for a coin. Always safe."""
    try:
        rec = _news_cache.get("by_coin", {}).get(symbol)
        if not rec:
            return {"breaking": False, "title": "", "age_h": 999.0}
        return {"breaking": rec.get("recent", 0) > 0,
                "title": rec.get("title", ""),
                "age_h": float(rec.get("age_h", 999.0))}
    except Exception:
        return {"breaking": False, "title": "", "age_h": 999.0}


# ---------------------------------------------------------------------------
# Weekly Self-Learning Engine — [E8] richer adaptation + reporting
# ---------------------------------------------------------------------------

def run_weekly_learning() -> None:
    log.info("Running weekly self-learning ...")
    journal   = load_journal()
    state     = load_learning_state()
    completed = [t for t in journal if t.get("outcome") in ("win","loss")]

    if len(completed) < MIN_TRADES_TO_LEARN:
        discord_notify(f"📚 **Weekly Learning** | Only `{len(completed)}/{MIN_TRADES_TO_LEARN}` "
                       f"completed trades — skipping weight update.")
        return

    # anti-drift: mean-revert weights 10% toward 1.0 before updating
    for side in ("Buy","Sell"):
        for sig, w in state["signal_weights"][side].items():
            state["signal_weights"][side][sig] = round(1.0 + (w - 1.0) * WEIGHT_MEAN_REVERSION, 3)

    # 1. signal win rates (Buy/Sell separately; same-direction signals only)
    sig_stats: dict = {"Buy":{},"Sell":{}}
    for trade in completed:
        d = trade.get("direction","")
        if d not in ("Buy","Sell"): continue
        for sig in trade.get("signals",[]):
            if sig not in state["signal_weights"][d]:
                continue
            b = sig_stats[d].setdefault(sig, {"wins":0,"total":0})
            b["total"] += 1
            if trade.get("outcome") == "win": b["wins"] += 1

    for direction in ("Buy","Sell"):
        for sig, stats in sig_stats[direction].items():
            if stats["total"] < MIN_TRADES_PER_SIGNAL: continue
            wr = stats["wins"] / stats["total"]
            cur = state["signal_weights"][direction].get(sig, 1.0)
            if   wr >= 0.70: new_w = min(cur * 1.15, MAX_WEIGHT)
            elif wr >= 0.55: new_w = cur
            else:            new_w = max(cur * 0.85, MIN_WEIGHT)
            state["signal_weights"][direction][sig] = round(new_w, 3)

    # 2. coin profiling (>=10 trades; precompute best/worst hours + auto-threshold)
    coin_data: dict = {}
    for trade in completed:
        sym = trade.get("symbol","")
        if not sym: continue
        p = coin_data.setdefault(sym, {"wl":0,"tl":0,"ws":0,"ts":0,
                                       "win_hrs":[], "loss_hrs":[], "pnls":[]})
        d, o, h = trade.get("direction",""), trade.get("outcome"), trade.get("hour_utc")
        if d == "Buy":
            p["tl"] += 1
            if o == "win": p["wl"] += 1
        elif d == "Sell":
            p["ts"] += 1
            if o == "win": p["ws"] += 1
        if h is not None:
            (p["win_hrs"] if o=="win" else p["loss_hrs"]).append(h)
        p["pnls"].append(trade.get("pnl",0) or 0)

    profiles = state.setdefault("coin_profiles",{})
    for sym, data in coin_data.items():
        total = data["tl"] + data["ts"]
        if total < MIN_TRADES_PER_COIN: continue
        wr_l = data["wl"]/data["tl"] if data["tl"]>0 else None
        wr_s = data["ws"]/data["ts"] if data["ts"]>0 else None
        wr_a = (data["wl"]+data["ws"])/total
        score_bonus = -1 if wr_a>=0.70 else (0 if wr_a>=0.50 else (1 if wr_a>=0.35 else 2))
        best_hrs  = [h for h,_ in Counter(data["win_hrs"]).most_common(3)]  if data["win_hrs"]  else []
        worst_hrs = [h for h,_ in Counter(data["loss_hrs"]).most_common(3)] if data["loss_hrs"] else []
        profiles[sym] = {
            "win_rate_long":  round(wr_l,3) if wr_l is not None else None,
            "win_rate_short": round(wr_s,3) if wr_s is not None else None,
            "win_rate_all":   round(wr_a,3),
            "avg_pnl":        round(sum(data["pnls"])/total, 4),
            "score_bonus":    score_bonus,
            "best_hours_utc":  best_hrs,
            "worst_hours_utc": worst_hrs,
            "trades_count":   total,
        }

    state["last_updated"]          = datetime.datetime.utcnow().isoformat()
    state["total_trades_analyzed"] = len(completed)
    save_learning_state(state)

    # ---- report ----
    def _wr(sig, d):
        s = sig_stats[d].get(sig)
        return f"{s['wins']/s['total']*100:.0f}%" if s and s["total"]>=MIN_TRADES_PER_SIGNAL else "n/a"
    def _arr(w): return "↑" if w>1.05 else ("↓" if w<0.95 else "→")

    sell_lines = "\n".join(f"  `{s}`: {_wr(s,'Sell')} {_arr(w)} w={w}"
                           for s,w in state["signal_weights"]["Sell"].items())
    buy_lines  = "\n".join(f"  `{s}`: {_wr(s,'Buy')} {_arr(w)} w={w}"
                           for s,w in state["signal_weights"]["Buy"].items())
    top_coins = sorted(profiles.items(), key=lambda x:x[1].get("win_rate_all",0), reverse=True)[:5]
    coin_lines = "\n".join(
        f"  `{sym}`: {p['win_rate_all']*100:.0f}% ({p['trades_count']}t) "
        f"thr{p['score_bonus']:+d} avgPnL={p.get('avg_pnl',0):+.2f}"
        for sym,p in top_coins)

    # [E8] exit-reason P&L — which exits make money vs lose
    exit_stats: dict = {}
    for t in completed:
        er = t.get("exit_reason","?") or "?"
        e = exit_stats.setdefault(er, {"n":0,"w":0,"pnl":0.0})
        e["n"] += 1
        if t.get("outcome")=="win": e["w"] += 1
        e["pnl"] += (t.get("pnl") or 0)
    exit_lines = "\n".join(
        f"  `{er}`: {v['n']}t {v['w']/v['n']*100:.0f}%W pnl={v['pnl']:+.2f}"
        for er,v in sorted(exit_stats.items(), key=lambda x:x[1]['pnl'], reverse=True))

    # [E8] long vs short global win rate
    longs  = [t for t in completed if t.get("direction")=="Buy"]
    shorts = [t for t in completed if t.get("direction")=="Sell"]
    def _side_wr(lst):
        return f"{sum(1 for t in lst if t['outcome']=='win')/len(lst)*100:.0f}% ({len(lst)}t)" if lst else "n/a"

    overall = sum(1 for t in completed if t["outcome"]=="win")/len(completed)*100
    total_pnl = sum((t.get("pnl") or 0) for t in completed)
    discord_notify(
        f"📚 **WEEKLY LEARNING REPORT** | {datetime.date.today()}\n"
        f"Analyzed `{len(completed)}` | Win `{overall:.0f}%` | PnL `{total_pnl:+.2f}` USDT\n"
        f"LONG {_side_wr(longs)} | SHORT {_side_wr(shorts)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**SHORT signals:**\n{sell_lines}\n"
        f"**LONG signals:**\n{buy_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Exit reasons (P&L sorted):**\n{exit_lines if exit_lines else '  (none)'}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Coin profiles (top 5, thr=auto strictness):**\n"
        f"{coin_lines if coin_lines else '  (need >=10 trades/coin)'}"
    )
    log.info("Weekly learning done. %d trades analyzed.", len(completed))


# ---------------------------------------------------------------------------
# Dynamic watchlist
# ---------------------------------------------------------------------------

def get_dynamic_watchlist() -> list:
    try:
        resp = SESSION.get(f"{BASE_URL}/v5/market/tickers",
                           params={"category":"linear"}, timeout=10)
        data = resp.json()
        if data.get("retCode") != 0: return []
        def _tv(t):
            try: return float(t.get("turnover24h",0))
            except Exception: return 0.0
        top = sorted([t for t in data["result"]["list"] if t["symbol"].endswith("USDT")],
                     key=_tv, reverse=True)[:DYNAMIC_SCAN_TOP_N]
        qualified = []
        for t in top:
            sym = t["symbol"]
            if sym in COIN_BLACKLIST: continue
            if sym[:-4] in STABLECOIN_BASES: continue
            if _tv(t) < DYNAMIC_VOLUME_MIN_USD: continue
            qualified.append(sym)
        log.info("Watchlist: %d/%d qualified", len(qualified), len(top))
        return qualified
    except Exception as e:
        log.warning("get_dynamic_watchlist: %s", e)
        return []


# ---------------------------------------------------------------------------
# Bybit REST helpers
# ---------------------------------------------------------------------------

def _post_headers(body: str) -> dict:
    ts  = str(int(time.time()*1000))
    rw  = "5000"
    sig = hmac.new(API_SECRET.encode(), (ts+API_KEY+rw+body).encode(), hashlib.sha256).hexdigest()
    return {"X-BAPI-API-KEY":API_KEY,"X-BAPI-TIMESTAMP":ts,
            "X-BAPI-RECV-WINDOW":rw,"X-BAPI-SIGN":sig,"Content-Type":"application/json"}

def _signed_get(path: str, params: dict) -> Optional[dict]:
    ts  = str(int(time.time() * 1000))
    rw  = "5000"
    qs  = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = hmac.new(API_SECRET.encode(), (ts + API_KEY + rw + qs).encode(), hashlib.sha256).hexdigest()
    headers = {"X-BAPI-API-KEY":API_KEY,"X-BAPI-TIMESTAMP":ts,
               "X-BAPI-RECV-WINDOW":rw,"X-BAPI-SIGN":sig}
    try:
        resp = SESSION.get(f"{BASE_URL}{path}?{qs}", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error("GET %s: %s", path, e)
    return None

def get_account_balance() -> Optional[float]:
    data = _signed_get("/v5/account/wallet-balance", {"accountType":"UNIFIED","coin":"USDT"})
    if data and data.get("retCode")==0:
        for item in data["result"]["list"]:
            for coin in item.get("coin",[]):
                if coin["coin"]=="USDT": return float(coin["walletBalance"])
    return None

def get_open_positions() -> dict:
    data = _signed_get("/v5/position/list", {"category":"linear","settleCoin":"USDT"})
    if data and data.get("retCode")==0:
        return {item["symbol"]:item["side"] for item in data["result"].get("list",[])
                if float(item.get("size",0))!=0}
    return {}

def get_position_details() -> list:
    data = _signed_get("/v5/position/list", {"category":"linear","settleCoin":"USDT"})
    if data and data.get("retCode")==0:
        return [item for item in data["result"].get("list",[]) if float(item.get("size",0))!=0]
    return []

def get_position_entry_liq(symbol: str) -> tuple:
    """[E3] Real (avgPrice, liqPrice) for an open position. (0,0) if none."""
    for p in get_position_details():
        if p.get("symbol") == symbol:
            try:
                return float(p.get("avgPrice") or 0), float(p.get("liqPrice") or 0)
            except Exception:
                return 0.0, 0.0
    return 0.0, 0.0

def set_breakeven_sl(symbol: str, entry_price: float) -> bool:
    payload = {"category":"linear","symbol":symbol,"stopLoss":str(round(entry_price,4)),
               "slTriggerBy":"LastPrice","positionIdx":0}
    body = json.dumps(payload)
    try:
        resp = SESSION.post(f"{BASE_URL}/v5/position/trading-stop",
                            headers=_post_headers(body), data=body, timeout=10)
        return resp.json().get("retCode")==0
    except Exception as e:
        log.error("set_breakeven_sl %s: %s", symbol, e)
    return False

def get_funding_rate(symbol: str) -> Optional[float]:
    try:
        resp = SESSION.get(f"{BASE_URL}/v5/market/tickers",
                           params={"category":"linear","symbol":symbol}, timeout=10)
        data = resp.json()
        if data.get("retCode")==0 and data["result"]["list"]:
            return float(data["result"]["list"][0]["fundingRate"])
    except Exception as e:
        log.error("Funding %s: %s", symbol, e)
    return None

def get_spread_pct(symbol: str) -> Optional[float]:
    try:
        resp = SESSION.get(f"{BASE_URL}/v5/market/orderbook",
                           params={"category":"linear","symbol":symbol,"limit":1}, timeout=10)
        data = resp.json()
        if data.get("retCode")==0:
            bid = float(data["result"]["b"][0][0]); ask = float(data["result"]["a"][0][0])
            mid = (bid+ask)/2
            return (ask-bid)/mid if mid>0 else None
    except Exception:
        pass
    return None

def get_klines(symbol: str, interval: str, limit: int=300) -> Optional[pd.DataFrame]:
    try:
        resp = SESSION.get(f"{BASE_URL}/v5/market/kline",
                           params={"category":"linear","symbol":symbol,
                                   "interval":interval,"limit":limit}, timeout=10)
        data = resp.json()
        if data.get("retCode")!=0: return None
        df = pd.DataFrame(data["result"]["list"],
                          columns=["timestamp","open","high","low","close","volume","turnover"])
        df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df.sort_values("timestamp", inplace=True); df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        log.error("Klines %s %s: %s", symbol, interval, e)
    return None

def load_instrument_cache(symbols: list) -> None:
    url = f"{BASE_URL}/v5/market/instruments-info"
    for sym in symbols:
        if sym in _instrument_cache:
            continue
        try:
            resp = SESSION.get(url, params={"category":"linear","symbol":sym}, timeout=10)
            data = resp.json()
            if data.get("retCode")==0 and data["result"]["list"]:
                inst = data["result"]["list"][0]
                lot  = inst["lotSizeFilter"]
                tick = float(inst.get("priceFilter",{}).get("tickSize", 0) or 0)
                _instrument_cache[sym] = {"min_qty":float(lot["minOrderQty"]),
                                          "qty_step":float(lot["qtyStep"]),
                                          "tick_size":tick}
        except Exception as e:
            log.error("Instrument %s: %s", sym, e)

def snap_price(symbol: str, price: float) -> float:
    info = _instrument_cache.get(symbol)
    tick = (info or {}).get("tick_size") or 0
    if tick <= 0: return round(price, 4)
    snapped = round(round(price/tick)*tick, 10)
    dec = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
    return round(snapped, dec)

def snap_qty(symbol: str, qty: float) -> float:
    info = _instrument_cache.get(symbol)
    if not info: return round(qty, 3)
    step=info["qty_step"]; min_qty=info["min_qty"]
    snapped = math.floor(qty/step)*step
    dec = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    snapped = round(snapped, dec)
    return snapped if snapped>=min_qty else 0.0

def set_leverage(symbol: str, leverage: int) -> bool:
    payload = {"category":"linear","symbol":symbol,
               "buyLeverage":str(leverage),"sellLeverage":str(leverage)}
    body = json.dumps(payload)
    try:
        resp = SESSION.post(f"{BASE_URL}/v5/position/set-leverage",
                            headers=_post_headers(body), data=body, timeout=10)
        return resp.json().get("retCode") in (0,110043)
    except Exception as e:
        log.error("set_leverage: %s", e)
    return False

def place_order(symbol:str, side:str, qty:float, sl_price:float, tp_price:float) -> Optional[dict]:
    """Market IOC entry (used when USE_MAKER_ENTRY=false)."""
    payload = {"category":"linear","symbol":symbol,"side":side,"orderType":"Market","qty":str(qty),
               "stopLoss":str(round(sl_price,4)),"takeProfit":str(round(tp_price,4)),
               "timeInForce":"IOC","slTriggerBy":"LastPrice","tpTriggerBy":"LastPrice","positionIdx":0}
    body = json.dumps(payload)
    try:
        resp = SESSION.post(f"{BASE_URL}/v5/order/create",
                            headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode")==0:
            log.info("Market order placed: %s %s qty=%s",side,symbol,qty); return data["result"]
        log.error("Order failed %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("place_order: %s", e)
    return None

def close_position(symbol:str, side:str, qty:float) -> Optional[dict]:
    cs = "Sell" if side=="Buy" else "Buy"
    payload = {"category":"linear","symbol":symbol,"side":cs,"orderType":"Market","qty":str(qty),
               "timeInForce":"IOC","reduceOnly":True,"positionIdx":0}
    body = json.dumps(payload)
    try:
        resp = SESSION.post(f"{BASE_URL}/v5/order/create",
                            headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode")==0:
            log.info("Closed: %s %s qty=%s",symbol,cs,qty); return data["result"]
        log.error("close_position %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("close_position: %s", e)
    return None


# ---------------------------------------------------------------------------
# [E3] Leverage / liquidation safety
# ---------------------------------------------------------------------------

def liq_price_est(entry: float, leverage: int, direction: str, maint: float = MAINT_MARGIN) -> float:
    if leverage <= 0 or entry <= 0:
        return 0.0
    if direction == "Buy":
        return entry * (1 - 1.0/leverage + maint)
    return entry * (1 + 1.0/leverage - maint)

def safe_leverage_for_sl(entry: float, sl: float, max_lev: int = MAX_LEVERAGE,
                         maint: float = MAINT_MARGIN, buffer: float = LIQ_BUFFER) -> int:
    """Highest leverage (<=max) that keeps liq a safe buffer beyond the stop.
    Wide stop -> lower leverage automatically. Kills the SL-below-liq bug."""
    if entry <= 0:
        return 1
    sl_dist = abs(entry - sl) / entry
    denom = sl_dist + buffer + maint
    if denom <= 0:
        return max_lev
    return max(1, min(max_lev, int(1.0 / denom)))

def sl_inside_liq(entry: float, sl: float, direction: str, leverage: int,
                  min_gap: float = LIQ_BUFFER * 0.5) -> bool:
    liq = liq_price_est(entry, leverage, direction)
    if liq <= 0 or entry <= 0:
        return True
    if direction == "Buy":
        return (sl - liq) / entry >= min_gap
    return (liq - sl) / entry >= min_gap


# ---------------------------------------------------------------------------
# [S3] Maker entry helpers
# ---------------------------------------------------------------------------

def get_best_price(symbol: str, side: str) -> Optional[float]:
    try:
        resp = SESSION.get(f"{BASE_URL}/v5/market/orderbook",
                           params={"category":"linear","symbol":symbol,"limit":1}, timeout=10)
        data = resp.json()
        if data.get("retCode")==0:
            if side=="Buy":
                return float(data["result"]["b"][0][0])
            return float(data["result"]["a"][0][0])
    except Exception as e:
        log.error("get_best_price %s: %s", symbol, e)
    return None

def place_limit_postonly(symbol:str, side:str, qty:float, price:float,
                         sl_price:float, tp_price:float) -> Optional[dict]:
    payload = {"category":"linear","symbol":symbol,"side":side,
               "orderType":"Limit","qty":str(qty),"price":str(price),
               "stopLoss":str(sl_price),"takeProfit":str(tp_price),
               "timeInForce":"PostOnly",
               "slTriggerBy":"LastPrice","tpTriggerBy":"LastPrice","positionIdx":0}
    body = json.dumps(payload)
    try:
        resp = SESSION.post(f"{BASE_URL}/v5/order/create",
                            headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode")==0:
            log.info("PostOnly placed: %s %s qty=%s @%s",side,symbol,qty,price)
            return data["result"]
        log.error("PostOnly failed %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("place_limit_postonly: %s", e)
    return None

def get_order_status(symbol: str, order_id: str) -> Optional[str]:
    data = _signed_get("/v5/order/realtime",
                       {"category":"linear","orderId":order_id,"symbol":symbol})
    if data and data.get("retCode")==0:
        lst = data["result"].get("list",[])
        if lst: return lst[0].get("orderStatus")
    return None

def cancel_order(symbol: str, order_id: str) -> bool:
    payload = {"category":"linear","symbol":symbol,"orderId":order_id}
    body = json.dumps(payload)
    try:
        resp = SESSION.post(f"{BASE_URL}/v5/order/cancel",
                            headers=_post_headers(body), data=body, timeout=10)
        return resp.json().get("retCode")==0
    except Exception as e:
        log.error("cancel_order %s: %s", symbol, e)
    return False

def wait_for_fill(symbol: str, order_id: str, timeout_s: int) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = get_order_status(symbol, order_id)
        if st == "Filled": return "Filled"
        if st in ("Cancelled","Rejected","Deactivated"): return st
        time.sleep(8)
    return "Timeout"


# ---------------------------------------------------------------------------
# Regime detection — [C3] closed candles only
# ---------------------------------------------------------------------------

def _closed(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[:-1] if df is not None and len(df) > 1 else df

def _classify_ema(close: pd.Series) -> str:
    e9=ta.trend.EMAIndicator(close,window=9).ema_indicator().iloc[-1]
    e50=ta.trend.EMAIndicator(close,window=50).ema_indicator().iloc[-1]
    e200=ta.trend.EMAIndicator(close,window=200).ema_indicator().iloc[-1]
    if e9>e50>e200: return "BULL"
    if e9<e50<e200: return "BEAR"
    return "RANGE"

def _compute_regime(proxy: str) -> dict:
    df4h=get_klines(proxy,"240",limit=250); df1h=get_klines(proxy,"60",limit=250)
    fb={"regime_4h":"RANGE","regime_1h":"RANGE","allowed":"SKIP","threshold":5,"proxy":proxy}
    if df4h is None or len(df4h)<201: return fb
    df4h, df1h = _closed(df4h), _closed(df1h) if df1h is not None else None
    r4h=_classify_ema(df4h["close"])
    r1h=_classify_ema(df1h["close"]) if df1h is not None and len(df1h)>=50 else "RANGE"
    if r4h=="BULL":
        if r1h=="BULL":    allowed,thr="Buy",4
        elif r1h=="RANGE": allowed,thr="Buy",5
        else:              allowed,thr="SKIP",5
    elif r4h=="BEAR":
        if r1h=="BEAR":    allowed,thr="Sell",4
        elif r1h=="RANGE": allowed,thr="Sell",5
        else:              allowed,thr="SKIP",5
    elif r4h=="RANGE":
        if r1h=="BULL":    allowed,thr="Buy",5
        elif r1h=="BEAR":  allowed,thr="Sell",5
        else:              allowed,thr="SKIP",5
    log.info("REGIME [%s]: 4H=%s 1H=%s -> %s thr=%d",proxy,r4h,r1h,allowed,thr)
    return {"regime_4h":r4h,"regime_1h":r1h,"allowed":allowed,"threshold":thr,"proxy":proxy}

def get_market_regime() -> dict: return _compute_regime("BTCUSDT")
def get_symbol_regime(symbol: str) -> dict:
    return _compute_regime("XAUTUSDT" if "XAU" in symbol else "BTCUSDT")

def _symbol_regime_from_df(df4h_c: pd.DataFrame, df1h_c: pd.DataFrame) -> dict:
    """[R1] A symbol's OWN regime, computed from candles already loaded in the
    scan (no extra API call). The symbol decides its own direction; BTC is used
    only as a risk veto in scan_symbol. This unblocks good coin setups on days
    when BTC itself is sideways."""
    r4h = _classify_ema(df4h_c["close"])
    r1h = _classify_ema(df1h_c["close"]) if len(df1h_c) >= 50 else "RANGE"
    if r4h == "BULL":
        if r1h == "BULL":    allowed, thr = "Buy", 4.5
        elif r1h == "RANGE": allowed, thr = "Buy", 5
        else:                allowed, thr = "SKIP", 5
    elif r4h == "BEAR":
        if r1h == "BEAR":    allowed, thr = "Sell", 4.5
        elif r1h == "RANGE": allowed, thr = "Sell", 5
        else:                allowed, thr = "SKIP", 5
    else:  # 4H RANGE — only follow a clear 1H push
        if r1h == "BULL":    allowed, thr = "Buy", 5
        elif r1h == "BEAR":  allowed, thr = "Sell", 5
        else:                allowed, thr = "SKIP", 5
    return {"regime_4h": r4h, "regime_1h": r1h, "allowed": allowed, "threshold": thr}


# ---------------------------------------------------------------------------
# [E1] Entry scoring — coherent TREND + TIMING model
# ---------------------------------------------------------------------------

def _f(v, default=0.0) -> float:
    try:
        x = float(v)
        return default if math.isnan(x) else x
    except Exception:
        return default

def score_entry_signals(df4h_c: pd.DataFrame, df1h_c: pd.DataFrame,
                        atr: float, learning_state: Optional[dict]=None) -> tuple:
    """
    Six signals in two coherent groups:
      TREND  (direction/permission): Trend4H, MACD4H, Trend1H
      TIMING (pullback into value) : Pullback, Bounce, Confirm
    A blow-off top scores low on TIMING (RSI/StochRSI extreme, no reclaim) so
    it can't reach the threshold. Returns (score, direction, signals, meta).
    """
    wb = (learning_state or {}).get("signal_weights", {}).get("Buy", {})
    ws = (learning_state or {}).get("signal_weights", {}).get("Sell", {})

    c4 = df4h_c["close"]; c1 = df1h_c["close"]; o1 = df1h_c["open"]

    e50_4  = ta.trend.EMAIndicator(c4, window=50).ema_indicator()
    e200_4 = ta.trend.EMAIndicator(c4, window=200).ema_indicator()
    macd4  = ta.trend.MACD(c4, window_slow=26, window_fast=12, window_sign=9)
    m4_line, m4_sig = macd4.macd(), macd4.macd_signal()

    e9_1  = ta.trend.EMAIndicator(c1, window=9).ema_indicator()
    e50_1 = ta.trend.EMAIndicator(c1, window=50).ema_indicator()
    rsi1  = ta.momentum.RSIIndicator(c1, window=14).rsi()
    st1   = ta.momentum.StochRSIIndicator(c1, window=14, smooth1=3, smooth2=3)
    k1, d1 = st1.stochrsi_k(), st1.stochrsi_d()

    close4  = _f(c4.iloc[-1]);   ema50_4 = _f(e50_4.iloc[-1]);  ema200_4 = _f(e200_4.iloc[-1])
    m4      = _f(m4_line.iloc[-1]); ms4   = _f(m4_sig.iloc[-1])
    close1  = _f(c1.iloc[-1]);   open1   = _f(o1.iloc[-1])
    ema9_1  = _f(e9_1.iloc[-1]); ema50_1 = _f(e50_1.iloc[-1])
    rsi_now = _f(rsi1.iloc[-1], 50)
    k_now   = _f(k1.iloc[-1], 50);  d_now  = _f(d1.iloc[-1], 50)
    k_prev  = _f(k1.iloc[-2], 50);  d_prev = _f(d1.iloc[-2], 50)

    ext_long  = (close1 - ema50_1) / atr if atr > 0 else 99.0
    ext_short = (ema50_1 - close1) / atr if atr > 0 else 99.0

    # Recent extreme over the prior 5 closed candles (excludes the just-closed one).
    # Confirms a REAL retracement happened: a long pullback needs price to have come
    # off a recent high; a short bounce needs price to have come off a recent low.
    # Also blocks entering when the current close IS the local extreme (top/bottom).
    recent_high_c = _f(c1.iloc[-6:-1].max(), close1)
    recent_low_c  = _f(c1.iloc[-6:-1].min(), close1)
    pulled_back   = close1 < recent_high_c   # came off a recent high (long)
    bounced_up    = close1 > recent_low_c    # came off a recent low  (short)

    bs = ss = 0.0
    buy, sell = [], []

    # ---- LONG ----
    if close4 > ema50_4 > ema200_4:
        bs += wb.get("Trend4H_bull", 1.0); buy.append("Trend4H_bull")
    if m4 > ms4:
        bs += wb.get("MACD4H_bull", 1.0);  buy.append("MACD4H_bull")
    if ema9_1 > ema50_1:
        bs += wb.get("Trend1H_bull", 1.0); buy.append("Trend1H_bull")
    if 36.0 <= rsi_now <= 54.0 and pulled_back:
        bs += wb.get("Pullback_bull", 1.0); buy.append("Pullback_bull")
    if k_now > d_now and k_prev <= d_prev and k_now < 60:
        bs += wb.get("Bounce_bull", 1.0);   buy.append("Bounce_bull")
    if close1 > open1 and close1 > ema9_1:
        bs += wb.get("Confirm_bull", 1.0);  buy.append("Confirm_bull")

    # ---- SHORT ----
    if close4 < ema50_4 < ema200_4:
        ss += ws.get("Trend4H_bear", 1.0); sell.append("Trend4H_bear")
    if m4 < ms4:
        ss += ws.get("MACD4H_bear", 1.0);  sell.append("MACD4H_bear")
    if ema9_1 < ema50_1:
        ss += ws.get("Trend1H_bear", 1.0); sell.append("Trend1H_bear")
    if 46.0 <= rsi_now <= 64.0 and bounced_up:
        ss += ws.get("Pullback_bear", 1.0); sell.append("Pullback_bear")
    if k_now < d_now and k_prev >= d_prev and k_now > 40:
        ss += ws.get("Bounce_bear", 1.0);   sell.append("Bounce_bear")
    if close1 < open1 and close1 < ema9_1:
        ss += ws.get("Confirm_bear", 1.0);  sell.append("Confirm_bear")

    if bs >= ss:
        return bs, "Buy", buy, {"ext": ext_long, "rsi": rsi_now}
    return ss, "Sell", sell, {"ext": ext_short, "rsi": rsi_now}


# ---------------------------------------------------------------------------
# Reversal checker — [C3] closed candles
# ---------------------------------------------------------------------------

def check_reversal_signals(symbol:str, position_side:str, df:pd.DataFrame) -> tuple:
    df = _closed(df)
    if df is None or len(df)<3: return 0,[]
    c,h,l,o=df["close"],df["high"],df["low"],df["open"]
    e9=ta.trend.EMAIndicator(c,window=9).ema_indicator()
    rs=ta.momentum.RSIIndicator(c,window=14).rsi()
    st=ta.momentum.StochRSIIndicator(c,window=14,smooth1=3,smooth2=3)
    sk,sd=st.stochrsi_k(),st.stochrsi_d()
    hist=ta.trend.MACD(c,window_slow=26,window_fast=12,window_sign=9).macd_diff()
    cc,pc=c.iloc[-1],c.iloc[-2]; co,po=o.iloc[-1],o.iloc[-2]
    ch,cl=h.iloc[-1],l.iloc[-1]
    ce,pe=e9.iloc[-1],e9.iloc[-2]
    cr,pr=rs.iloc[-1],rs.iloc[-2]
    ck,pk=sk.iloc[-1],sk.iloc[-2]; cd,pd2=sd.iloc[-1],sd.iloc[-2]
    ch2,ph2=hist.iloc[-1],hist.iloc[-2]
    sigs=[]
    if position_side=="Sell":
        if pr<50 and cr>=50: sigs.append("RSI>50")
        if pk<20 and pk<=pd2 and ck>cd: sigs.append("StochRSI_cross_up")
        if ph2<0 and ch2>=0: sigs.append("MACD_hist_flip_up")
        if pc<pe and cc>=ce: sigs.append("price>EMA9")
        body=abs(cc-co); lw=min(cc,co)-cl; hw=ch-max(cc,co)
        if (body>0 and lw>=2*body and hw<=body) or (pc<po and cc>co and co<=pc and cc>=po):
            sigs.append("BullishPattern")
    else:
        if pr>50 and cr<=50: sigs.append("RSI<50")
        if pk>80 and pk>=pd2 and ck<cd: sigs.append("StochRSI_cross_down")
        if ph2>0 and ch2<=0: sigs.append("MACD_hist_flip_down")
        if pc>pe and cc<=ce: sigs.append("price<EMA9")
        body=abs(cc-co); lw=min(cc,co)-cl; hw=ch-max(cc,co)
        if (body>0 and hw>=2*body and lw<=body) or (pc>po and cc<co and co>=pc and cc<=po):
            sigs.append("BearishPattern")
    log.info("%s reversal (%s): %d/5 %s",symbol,position_side,len(sigs),sigs)
    return len(sigs),sigs


# ---------------------------------------------------------------------------
# 15M momentum — 3-candle confirmation, [C3] closed candles
# ---------------------------------------------------------------------------

def check_15m_momentum(symbol:str, position_side:str) -> bool:
    df=get_klines(symbol,"15",limit=60)
    if df is None or len(df)<21: return False
    df=_closed(df)
    c=df["close"]
    e9=ta.trend.EMAIndicator(c,window=9).ema_indicator()
    rsi=ta.momentum.RSIIndicator(c,window=14).rsi()
    if position_side=="Buy":
        ok=all(c.iloc[-i]<e9.iloc[-i] and rsi.iloc[-i]<50 for i in (1,2,3))
    else:
        ok=all(c.iloc[-i]>e9.iloc[-i] and rsi.iloc[-i]>50 for i in (1,2,3))
    if ok: log.info("%s 15M momentum confirmed 3/3 closed candles",symbol)
    return ok


# ---------------------------------------------------------------------------
# [S4] Partial take-profit + breakeven
# ---------------------------------------------------------------------------

def manage_tp1(positions: list) -> None:
    for pos in positions:
        sym  = pos.get("symbol",""); side = pos.get("side","")
        try:
            entry = float(pos.get("avgPrice") or 0)
            mark  = float(pos.get("markPrice") or 0)
            qty   = float(pos.get("size") or 0)
        except Exception:
            continue
        if entry==0 or mark==0 or qty==0: continue

        journal, i = _find_open_trade(sym, side)
        if i < 0: continue
        t = journal[i]
        if t.get("tp1_done"): continue
        try:
            e0 = float(t.get("entry") or entry)
            s0 = float(t.get("sl") or 0)
        except Exception:
            continue
        if s0 == 0: continue
        rd  = abs(e0 - s0)
        tp1 = (e0 + TP1_R*rd) if side=="Buy" else (e0 - TP1_R*rd)
        hit = (mark >= tp1) if side=="Buy" else (mark <= tp1)
        if not hit:
            log.info("%s TP1 %.2f%% away", sym, abs(tp1-mark)/mark*100)
            continue

        half = snap_qty(sym, qty*TP1_CLOSE_FRACTION)
        if half <= 0 or half >= qty:
            if set_breakeven_sl(sym, e0):
                journal[i]["tp1_done"] = True; save_journal(journal)
                discord_notify(f"🔒 **Breakeven** {sym} | TP1 `{mark:.4f}` | "
                               f"SL->`{e0:.4f}` (too small to split)")
            continue

        result = close_position(sym, side, half)
        if result:
            realized = (mark-e0)*half if side=="Buy" else (e0-mark)*half
            journal, i = _find_open_trade(sym, side)
            if i >= 0:
                journal[i]["tp1_done"]     = True
                journal[i]["tp1_realized"] = round(realized, 4)
                save_journal(journal)
            set_breakeven_sl(sym, e0)
            dlabel = "LONG" if side=="Buy" else "SHORT"
            discord_notify(f"🎯 **TP1 {dlabel} {sym}** | closed `{half}` @`{mark:.4f}` "
                           f"(`{realized:+.2f}` USDT est.) | SL->BE `{e0:.4f}` | "
                           f"runner `{round(qty-half,6)}`")


# ---------------------------------------------------------------------------
# [M8] Original risk lookup
# ---------------------------------------------------------------------------

def get_initial_risk_usdt(symbol: str, direction: str, qty: float) -> float:
    journal = load_journal()
    for t in reversed(journal):
        if (t.get("symbol")==symbol and t.get("direction")==direction
                and t.get("outcome") is None):
            e0, s0 = t.get("entry"), t.get("sl")
            if e0 and s0:
                return abs(float(e0)-float(s0)) * qty
    return 0.0


# ---------------------------------------------------------------------------
# [E5] Unified exit manager — runner-aware, no blind profit-grab
# ---------------------------------------------------------------------------

def manage_exits(position_details:list, regime:Optional[dict]=None) -> dict:
    closed={}; now_ms=int(time.time()*1000)
    for pos in position_details:
        sym=pos["symbol"]; side=pos["side"]
        qty=float(pos.get("size",0)); entry=float(pos.get("avgPrice",0) or 0)
        sl=float(pos.get("stopLoss",0) or 0); upnl=float(pos.get("unrealisedPnl",0) or 0)
        created_ms=int(pos.get("createdTime",now_ms) or now_ms)
        if entry==0 or qty==0: continue

        r_usdt = get_initial_risk_usdt(sym, side, qty)
        if r_usdt <= 0:
            rd = abs(entry-sl) if sl else 0.0
            r_usdt = rd*qty
        r_mult = (upnl / r_usdt) if r_usdt > 0 else 0.0

        journal, ji = _find_open_trade(sym, side)
        tp1_done = (ji >= 0 and bool(journal[ji].get("tp1_done")))

        # [X2] Holding time from OUR journal open-timestamp (Bybit createdTime
        # reads stale on re-opened symbols -> previously fired TIME EXIT in minutes).
        opened_ms = _ts_to_ms(journal[ji].get("timestamp","")) if ji >= 0 else 0
        if not opened_ms:
            opened_ms = created_ms
        hrs = (now_ms-opened_ms)/3_600_000 if opened_ms else 0.0
        age_min = hrs*60
        if opened_ms==0:
            log.info("%s SKIP exit checks (open time unknown)",sym); continue

        # [BO] Breakout positions ride a pure structural trail: ratchet the SL
        # under the most recent swing (no TP1, no reversal/settle/time logic) so
        # the rare winner can run as far as the trend allows. Bybit closes it
        # when the trailing SL is hit.
        if ji >= 0 and journal[ji].get("entry_mode") == "breakout":
            try:
                _raw = get_klines(sym, "60")
                if _raw is not None:
                    _df1h = _closed(_raw)
                    _atr = float(ta.volatility.AverageTrueRange(
                        _df1h["high"], _df1h["low"], _df1h["close"], window=14
                    ).average_true_range().iloc[-1])
                    _win = _df1h.iloc[-BREAKOUT_TRAIL_LOOKBACK:]
                    _px  = float(_df1h["close"].iloc[-1])
                    if side == "Buy":
                        _new = snap_price(sym, float(_win["low"].min()) - BREAKOUT_SL_BUFFER_ATR*_atr)
                        if _new > sl and _new < _px and set_breakeven_sl(sym, _new):
                            log.info("%s BO trail SL %.4f -> %.4f (R=%.2f)", sym, sl, _new, r_mult)
                    else:
                        _new = snap_price(sym, float(_win["high"].max()) + BREAKOUT_SL_BUFFER_ATR*_atr)
                        if (sl == 0 or _new < sl) and _new > _px and set_breakeven_sl(sym, _new):
                            log.info("%s BO trail SL %.4f -> %.4f (R=%.2f)", sym, sl, _new, r_mult)
            except Exception as e:
                log.error("%s BO trail error: %s", sym, e)
            continue

        pnl_str=f"+${upnl:.2f}" if upnl>=0 else f"-${abs(upnl):.2f}"
        dlabel="LONG" if side=="Buy" else "SHORT"
        reason=None; icon="⚡"; label="EXIT"

        # Evidence the entry thesis is breaking (closed-candle reversal + regime).
        rc=0; rf=[]
        df1h=get_klines(sym,"60",limit=100)
        if df1h is not None and len(df1h)>=4:
            rc,rf=check_reversal_signals(sym,side,df1h)
        rb=0
        if regime:
            al=regime.get("allowed","SKIP")
            if al=="SKIP" or al!=side: rb=1
        rev_list = rf + (["RegimeConflict"] if rb else [])
        rev_score  = rc+rb
        strong_rev = rev_score >= RUNNER_EXIT_THRESHOLD   # 4/5 = thesis clearly broken
        settled    = age_min >= SETTLE_MINUTES            # weak signals wait for settle

        if tp1_done:
            # RUNNER: breakeven protects it. Exit only on STRONG reversal or long-flat.
            if strong_rev:
                icon="🔄"; label="RUNNER EXIT"
                reason=f"strong reversal `{rev_score}/5` [{', '.join(rev_list)}]"
            elif hrs>RUNNER_TIME_EXIT_HRS and r_mult<0.5:
                icon="⏰"; label="RUNNER TIME"
                reason=f"runner flat `{hrs:.1f}h` (<0.5R) — freeing slot"
        else:
            # PRE-TP1. A STRONG reversal (4/5 on closed candles) means the thesis
            # is genuinely broken — honour it at ANY age, it is not entry noise.
            if strong_rev:
                reason=f"strong reversal `{rev_score}/5` [{', '.join(rev_list)}]"
            elif settled:
                # Weaker evidence only acts once the trade has had room to work.
                if rev_score >= PRE_TP1_REVERSAL:
                    reason=f"reversal `{rev_score}/5` [{', '.join(rev_list)}]"
                if reason is None and check_15m_momentum(sym,side):
                    icon="📉"; label="MOMENTUM EXIT"; reason="15M momentum flip (3-candle)"
                if reason is None and upnl>0:
                    fr=get_funding_rate(sym)
                    if fr is not None:
                        bad=(side=="Buy" and fr>=FUNDING_SPIKE_PCT) or (side=="Sell" and fr<=-FUNDING_SPIKE_PCT)
                        if bad:
                            icon="💸"; label="FUNDING EXIT"; reason=f"funding spike `{fr*100:.4f}%`"
                # STALL exit (replaces blind 0.8R grab): needs profit AND a reversal
                if reason is None and r_mult>=QUICK_PROFIT_R and rev_score>=2:
                    icon="🛟"; label="STALL EXIT"
                    reason=f"`{r_mult:.1f}R` + early reversal `{rev_score}/5` — locking profit"
                if reason is None and hrs>TIME_EXIT_HOURS and r_mult<TIME_EXIT_MIN_R:
                    icon="⏰"; label="TIME EXIT"; reason=f"held `{hrs:.1f}h` profit `{r_mult:.1f}R`<0.3R"
            else:
                log.info("%s settling (%.0fm/%.0fm) rev=%d/5 — only strong reversal acts",
                         sym, age_min, SETTLE_MINUTES, rev_score)

        if reason is None: continue
        log.warning("%s — %s [%s] | %s | uPnL=%s | %.1fR | held=%.1fh",
                    sym,label,dlabel,reason,pnl_str,r_mult,hrs)
        result=close_position(sym,side,qty)
        if result:
            update_trade_outcome(sym,side,upnl,exit_reason=label.lower().replace(" ","_"))
            discord_notify(f"{icon} **{label} {dlabel} {sym}**\n"
                           f"Reason: {reason}\n"
                           f"Achieved `{r_mult:+.1f}R` | uPnL `{pnl_str}` | Held `{hrs:.1f}h`")
            closed[sym]=True
        else:
            discord_notify(f"⚠️ **{sym}** — exit FAILED | {reason}")
    return closed


# ---------------------------------------------------------------------------
# Position sizing — [M5] threshold tiers, [M9] hour adj needs >=15 trades
# ---------------------------------------------------------------------------

def risk_pct_for_score(score: float) -> float:
    if score >= 6.0: return 0.025
    if score >= 5.0: return 0.020
    return 0.015

def calculate_position(balance:float, entry_price:float, sl_price:float,
                       leverage:int, score:float=4.0,
                       symbol:str="", learning_state:Optional[dict]=None) -> float:
    risk_pct = risk_pct_for_score(score)
    if learning_state and symbol:
        profile = learning_state.get("coin_profiles",{}).get(symbol,{})
        if profile.get("trades_count",0) >= MIN_TRADES_FOR_HOURS:
            now_h = datetime.datetime.utcnow().hour
            if now_h in profile.get("best_hours_utc",[]):
                risk_pct *= 1.10
            elif now_h in profile.get("worst_hours_utc",[]):
                risk_pct *= 0.80
    sl_dist=abs(entry_price-sl_price)
    if sl_dist==0: return 0.0
    qty_usdt=min((balance*risk_pct/sl_dist)*entry_price, balance*leverage)
    return round(qty_usdt/entry_price, 3)


# ---------------------------------------------------------------------------
# Loss guards
# ---------------------------------------------------------------------------

def daily_loss_exceeded(s,c): return s>0 and (s-c)/s>=DAILY_LOSS_LIMIT
def weekly_loss_exceeded(s,c): return s>0 and (s-c)/s>=WEEKLY_LOSS_LIMIT


# ---------------------------------------------------------------------------
# Daily summary — [M6] summarizes a specific date
# ---------------------------------------------------------------------------

def send_daily_summary(balance:float, for_date: Optional[datetime.date]=None) -> None:
    journal=load_journal()
    d=(for_date or datetime.date.today()).isoformat()
    day_t=[t for t in journal if str(t.get("timestamp","")).startswith(d)
           or str(t.get("closed_at","")).startswith(d)]
    comp=[t for t in day_t if t.get("outcome") in ("win","loss")]
    wins=sum(1 for t in comp if t["outcome"]=="win"); total=len(comp)
    wr=f"{wins/total*100:.0f}%" if total>0 else "n/a"
    pnl=sum((t.get("pnl") or 0) for t in comp)
    op=get_open_positions()
    discord_notify(
        f"📈 **DAILY SUMMARY** | {d}\n"
        f"• Trades closed: `{total}` | Win rate: `{wr}` ({wins}/{total})\n"
        f"• Realized PnL: `{'%+.2f'%pnl}` USDT (incl. fees where reconciled)\n"
        f"• Open: {', '.join(f'`{s}`' for s in op) if op else 'None'}\n"
        f"• Balance: `{balance:.2f}` USDT"
    )


# ---------------------------------------------------------------------------
# Order params builder — shared by maker & market paths
# ---------------------------------------------------------------------------

def build_order_params(symbol, direction, entry, atr, swing_low, swing_high,
                       pm, ph, pl, balance, score, leverage, learning_state):
    """Compute SL/TP/qty and run R:R + liq gates. Returns dict or None."""
    if direction == "Buy":
        sl = min(swing_low - SL_SWING_BUFFER_ATR*atr, entry - SL_ATR_MULT*atr)
        raw_tp = entry + TP_ATR_MULT*atr
        cands = [p for p in [pm, ph] if entry < p <= raw_tp*1.02]
        tp = min(cands) if cands else raw_tp
    else:
        sl = max(swing_high + SL_SWING_BUFFER_ATR*atr, entry + SL_ATR_MULT*atr)
        raw_tp = entry - TP_ATR_MULT*atr
        cands = [p for p in [pm, pl] if entry > p >= raw_tp*0.98]
        tp = max(cands) if cands else raw_tp
    sl = snap_price(symbol, sl); tp = snap_price(symbol, tp)

    risk = abs(entry - sl); reward = abs(tp - entry)
    if risk <= 0:
        log.info("%s SKIP zero risk", symbol); return None
    rr = reward / risk
    if rr < MIN_RR:
        log.info("%s SKIP R:R %.2f<%.1f", symbol, rr, MIN_RR); return None
    if abs(entry - sl)/entry < SL_MIN_DIST_PCT:
        log.info("%s SKIP SL too tight (%.2f%%)", symbol, abs(entry-sl)/entry*100); return None

    # [E3] liquidation safety — reduce leverage once, else skip
    if not sl_inside_liq(entry, sl, direction, leverage):
        leverage = safe_leverage_for_sl(entry, sl)
        set_leverage(symbol, leverage)
        if not sl_inside_liq(entry, sl, direction, leverage):
            log.info("%s SKIP SL beyond liq even at %dx", symbol, leverage); return None

    qty = snap_qty(symbol, calculate_position(balance, entry, sl, leverage,
                                              score, symbol, learning_state))
    if qty <= 0:
        log.info("%s SKIP qty 0", symbol); return None
    return {"sl": sl, "tp": tp, "qty": qty, "rr": rr, "leverage": leverage}


# ---------------------------------------------------------------------------
# Main scan — v4.2 entry pipeline
# ---------------------------------------------------------------------------

def scan_symbol(symbol:str, balance:float, regime:Optional[dict]=None,
                learning_state:Optional[dict]=None) -> Optional[dict]:
    log.info("Scanning %s ...",symbol)

    sp=get_spread_pct(symbol)
    if sp is not None and sp>MAX_SPREAD_PCT:
        log.info("%s SKIP spread %.4f%%",symbol,sp*100); return None

    df4h=get_klines(symbol,"240"); df1h=get_klines(symbol,"60")
    if df4h is None or df1h is None or len(df4h)<202 or len(df1h)<80:
        return None

    df4h_c=_closed(df4h); df1h_c=_closed(df1h)

    # [E6] Liquidity floor — only skip near-dead candles (lets low-vol pullbacks in)
    closed_vol=df4h_c["volume"].iloc[-1]
    avg_v=df4h_c["volume"].iloc[-21:-1].mean()
    if avg_v>0 and closed_vol/avg_v<VOL_FLOOR_RATIO:
        log.info("%s SKIP dead volume (%.2fx)",symbol,closed_vol/avg_v); return None

    # [S2] Real ATR — true range incl. gaps vs previous close
    try:
        atr=ta.volatility.AverageTrueRange(
            df1h_c["high"],df1h_c["low"],df1h_c["close"],window=14
        ).average_true_range().iloc[-1]
        atr=float(atr)
    except Exception:
        return None
    if not atr or atr<=0 or math.isnan(atr): return None

    # [E1] Entry model
    score,direction,details,meta=score_entry_signals(df4h_c,df1h_c,atr,learning_state)
    log.info("%s | score=%.2f dir=%s ext=%.2fATR rsi=%.0f %s",
             symbol,score,direction,meta.get("ext",99),meta.get("rsi",0),details)

    if direction=="Buy"  and not LONG_ENABLED:  return None
    if direction=="Sell" and not SHORT_ENABLED: return None

    # [R1] Regime gate — the SYMBOL decides direction; BTC is only a risk veto.
    sr = _symbol_regime_from_df(df4h_c, df1h_c)
    allowed = sr.get("allowed", "SKIP"); thr = float(sr.get("threshold", CONFLUENCE_THRESHOLD))

    # the coin itself must have a tradeable trend in the signal's direction
    if allowed == "SKIP" or allowed != direction:
        log.info("%s SKIP coin regime (%s/%s -> %s vs %s)",
                 symbol, sr.get("regime_4h"), sr.get("regime_1h"), allowed, direction)
        return None

    # [TWEAK2] block pullback when the coin's OWN 4H is RANGE (backtest PF 0.70 here).
    # Breakout entries use a separate path (scan_symbol_breakout) and are not affected.
    if sr.get("regime_4h") == "RANGE":
        log.info("%s SKIP pullback — coin 4H RANGE (no trend to pull back into)", symbol)
        return None

    # BTC veto (skip for gold, which is uncorrelated). regime = BTC market regime.
    btc4 = (regime or {}).get("regime_4h", "RANGE")
    if "XAU" not in symbol:
        if direction == "Buy" and btc4 == "BEAR":
            log.info("%s VETO long — BTC 4H BEAR", symbol); return None
        if direction == "Sell" and btc4 == "BULL":
            log.info("%s VETO short — BTC 4H BULL", symbol); return None
        # BTC undecided → require a slightly stronger coin setup
        if btc4 == "RANGE":
            thr += 0.5
        log.info("%s coin %s/%s | BTC 4H=%s | thr=%.1f",
                 symbol, sr.get("regime_4h"), sr.get("regime_1h"), btc4, thr)

    # per-coin auto-threshold from learning
    if learning_state:
        prof=learning_state.get("coin_profiles",{}).get(symbol,{})
        sb=prof.get("score_bonus",0); thr+=sb
        if sb!=0: log.info("%s coin auto-thr %+d -> %.1f",symbol,sb,thr)

    # [E2] Extension guard (hard) — never chase price far from value
    if meta.get("ext",99)>MAX_EXTENSION_ATR:
        log.info("%s SKIP over-extended %.2fATR > %.1f",symbol,meta["ext"],MAX_EXTENSION_ATR)
        return None

    if score<thr:
        log.info("%s SKIP score %.2f<%.1f",symbol,score,thr); return None

    # Correlation cap
    if symbol in BTC_CORRELATED:
        op=get_open_positions()
        if sum(1 for s in op if s in BTC_CORRELATED)>=MAX_CORRELATION_POSITIONS:
            log.info("%s SKIP correlation cap",symbol); return None

    # [E7] News gate — skip very fresh breaking news (avoid whipsaw)
    nf=news_flag(symbol)
    if nf["breaking"] and nf["age_h"]*60 < NEWS_BLOCK_MINUTES:
        log.info("%s SKIP fresh news %.0fmin: %s",symbol,nf["age_h"]*60,nf["title"][:60])
        discord_notify(f"📰 **{symbol} skipped** — fresh news {nf['age_h']*60:.0f}min ago, "
                       f"too risky to enter\n_{nf['title'][:140]}_")
        return None

    # [E4] Structure reference levels for SL / TP
    swing_low =df1h_c["low"].iloc[-SL_SWING_LOOKBACK:].min()
    swing_high=df1h_c["high"].iloc[-SL_SWING_LOOKBACK:].max()
    r4h=df4h_c.iloc[-20:]; ph=r4h["high"].max(); pl=r4h["low"].min(); pm=(ph+pl)/2

    # [E3] Pick safe leverage from a preliminary entry/SL
    prelim=get_best_price(symbol,direction)
    if prelim is None: return None
    prelim=snap_price(symbol,prelim)
    if direction=="Buy":
        prelim_sl=min(swing_low-SL_SWING_BUFFER_ATR*atr, prelim-SL_ATR_MULT*atr)
    else:
        prelim_sl=max(swing_high+SL_SWING_BUFFER_ATR*atr, prelim+SL_ATR_MULT*atr)
    leverage=safe_leverage_for_sl(prelim,prelim_sl)
    set_leverage(symbol,leverage)

    # ---- Entry execution ----
    result=None; entry=sl=tp=0.0; qty=0.0; rr=0.0
    if USE_MAKER_ENTRY:
        for attempt in range(ENTRY_MAX_ATTEMPTS):
            price=get_best_price(symbol,direction)
            if price is None: return None
            entry=snap_price(symbol,price)
            params=build_order_params(symbol,direction,entry,atr,swing_low,swing_high,
                                      pm,ph,pl,balance,score,leverage,learning_state)
            if params is None: return None
            sl,tp,qty,rr,leverage=params["sl"],params["tp"],params["qty"],params["rr"],params["leverage"]
            res=place_limit_postonly(symbol,direction,qty,entry,sl,tp)
            if not res: return None
            oid=res.get("orderId","")
            status=wait_for_fill(symbol,oid,ENTRY_FILL_TIMEOUT_S)
            if status=="Filled":
                result=res; break
            if status=="Timeout":
                cancel_order(symbol,oid)
            log.info("%s entry attempt %d/%d not filled (%s) — re-quoting",
                     symbol,attempt+1,ENTRY_MAX_ATTEMPTS,status)
        if not result:
            log.info("%s SKIP — maker not filled, not chasing",symbol); return None
    else:
        price=get_best_price(symbol,direction)
        if price is None: return None
        entry=snap_price(symbol,price)
        params=build_order_params(symbol,direction,entry,atr,swing_low,swing_high,
                                  pm,ph,pl,balance,score,leverage,learning_state)
        if params is None: return None
        sl,tp,qty,rr,leverage=params["sl"],params["tp"],params["qty"],params["rr"],params["leverage"]
        res=place_order(symbol,direction,qty,sl,tp)   # market IOC
        if not res: return None
        result=res

    # [E3] Post-fill: use REAL fill price + check liq
    real_entry, liq = get_position_entry_liq(symbol)
    if real_entry > 0:
        entry = real_entry                      # accurate R for exits
    liq_txt=""
    if liq>0:
        margin_pct = ((sl-liq) if direction=="Buy" else (liq-sl))/entry*100
        safe_icon="✅" if margin_pct>1.0 else "⚠️"
        liq_txt=f"\nLiq `{liq:.4f}` | SL sits `{margin_pct:+.1f}%` inside liq {safe_icon}"
        if margin_pct<=0:
            log.error("%s DANGER: SL beyond liq! margin=%.2f%%",symbol,margin_pct)

    risk_usdt=abs(entry-sl)*qty
    risk_pct_used=risk_pct_for_score(score)

    trade={
        "timestamp":   datetime.datetime.utcnow().isoformat(),
        "symbol":      symbol, "direction": direction,
        "entry_mode":  "pullback",
        "score":       round(score,3), "signals": details,
        "entry":       entry, "sl": sl, "tp": tp,
        "qty":         qty, "leverage": leverage, "rr": round(rr,2),
        "ext_atr":     round(meta.get("ext",0),2),
        "regime_4h":   sr.get("regime_4h"), "regime_1h": sr.get("regime_1h"),
        "hour_utc":    datetime.datetime.utcnow().hour,
        "outcome":     None, "pnl": None, "closed_at": None,
        "exit_reason": None,
        "tp1_done":    False, "tp1_realized": 0.0,
        "order_result":result,
    }
    log_trade(trade)

    # [E9] Rich Discord entry report
    news_txt=""
    if nf["title"] and nf["age_h"]<24:
        news_txt=f"\n📰 News {nf['age_h']:.0f}h ago: _{nf['title'][:110]}_"
    emoji="🟢" if direction=="Buy" else "🔴"
    dlabel="LONG" if direction=="Buy" else "SHORT"
    mode="maker pullback" if USE_MAKER_ENTRY else "market pullback"
    sl_pct=(sl-entry)/entry*100; tp_pct=(tp-entry)/entry*100
    discord_notify(
        f"{emoji} **{dlabel} {symbol}** ({mode})\n"
        f"Score `{score:.1f}/6` [{', '.join(details)}]\n"
        f"Coin regime 4H/1H: `{sr.get('regime_4h','?')}/{sr.get('regime_1h','?')}` | "
        f"BTC 4H: `{(regime or {}).get('regime_4h','?')}`\n"
        f"Entry `{entry}` | SL `{sl}` (`{sl_pct:+.1f}%`) | TP `{tp}` (`{tp_pct:+.1f}%`)\n"
        f"R:R `{rr:.2f}` | Ext `{meta.get('ext',0):.1f}` ATR | Lev `{leverage}x`"
        f"{liq_txt}\n"
        f"Qty `{qty}` | Risk `{risk_usdt:.2f}` USDT (`{risk_pct_used*100:.1f}%`)"
        f"{news_txt}"
    )
    return trade


# ---------------------------------------------------------------------------
# [BO] Breakout momentum module — fully independent of the pullback pipeline.
#   Pullback enters on a retrace to value (low ext). Breakout enters when price
#   PUSHES THROUGH a 20-bar range high/low on above-average volume — it leads
#   the trend instead of waiting for EMAs to stack. Deliberately skips the
#   extension guard (a breakout IS extended by nature), but is held in check by
#   tighter rules: volume confirmation, a closed bar beyond the level, BTC veto,
#   half risk (1%), a structural SL just inside the broken level, and a pure
#   trailing exit (no fixed TP) so the rare winner can run far enough to pay for
#   the many small losers. entry_mode="breakout" keeps its journal separate.
# ---------------------------------------------------------------------------

def detect_breakout(df4h_c: pd.DataFrame, df1h_c: pd.DataFrame):
    """Return (direction, signals, ref_level) on a confirmed 1H range break, else None."""
    if len(df1h_c) < BREAKOUT_LOOKBACK + 2 or len(df4h_c) < 51:
        return None
    window     = df1h_c.iloc[-(BREAKOUT_LOOKBACK + 1):-1]          # prior N bars, excl. just-closed
    range_high = float(window["high"].max())
    range_low  = float(window["low"].min())
    vol_avg    = float(window["volume"].mean())
    last_close = float(df1h_c["close"].iloc[-1])
    last_vol   = float(df1h_c["volume"].iloc[-1])
    if vol_avg <= 0:
        return None
    vol_x  = last_vol / vol_avg
    vol_ok = vol_x >= BREAKOUT_VOL_MULT

    # 4H direction filter — don't break out straight into the higher-TF trend.
    close4   = float(df4h_c["close"].iloc[-1])
    ema50_4  = float(df4h_c["close"].ewm(span=50, adjust=False).mean().iloc[-1])

    if last_close > range_high and vol_ok and close4 >= ema50_4:
        return "Buy", [f"BO_long", f"Vol{vol_x:.1f}x", f"Range{BREAKOUT_LOOKBACK}"], range_high
    if last_close < range_low and vol_ok and close4 <= ema50_4:
        return "Sell", [f"BO_short", f"Vol{vol_x:.1f}x", f"Range{BREAKOUT_LOOKBACK}"], range_low
    return None


def scan_symbol_breakout(symbol: str, balance: float, regime: Optional[dict] = None,
                         learning_state: Optional[dict] = None) -> Optional[dict]:
    if not BREAKOUT_ENABLED:
        return None

    sp = get_spread_pct(symbol)
    if sp is not None and sp > MAX_SPREAD_PCT:
        return None

    df4h = get_klines(symbol, "240"); df1h = get_klines(symbol, "60")
    if df4h is None or df1h is None or len(df4h) < 51 or len(df1h) < BREAKOUT_LOOKBACK + 2:
        return None
    df4h_c = _closed(df4h); df1h_c = _closed(df1h)

    bo = detect_breakout(df4h_c, df1h_c)
    if bo is None:
        return None
    direction, signals, ref_level = bo

    if direction == "Buy"  and not LONG_ENABLED:  return None
    if direction == "Sell" and not SHORT_ENABLED: return None

    # ATR for SL buffer / TP net
    try:
        atr = float(ta.volatility.AverageTrueRange(
            df1h_c["high"], df1h_c["low"], df1h_c["close"], window=14
        ).average_true_range().iloc[-1])
    except Exception:
        return None
    if not atr or atr <= 0 or math.isnan(atr):
        return None

    # BTC veto — never break out against the market leader (gold exempt).
    btc4 = (regime or {}).get("regime_4h", "RANGE")
    if "XAU" not in symbol:
        if direction == "Buy"  and btc4 == "BEAR":
            log.info("%s BO VETO long — BTC 4H BEAR", symbol); return None
        if direction == "Sell" and btc4 == "BULL":
            log.info("%s BO VETO short — BTC 4H BULL", symbol); return None

    # News gate (reuse) — avoid breaking out into a fresh-headline whipsaw.
    nf = news_flag(symbol)
    if nf["breaking"] and nf["age_h"] * 60 < NEWS_BLOCK_MINUTES:
        log.info("%s BO SKIP fresh news %.0fmin", symbol, nf["age_h"] * 60); return None

    price = get_best_price(symbol, direction)
    if price is None: return None
    entry = snap_price(symbol, price)

    # Structural SL just inside the level that was broken.
    if direction == "Buy":
        sl = snap_price(symbol, ref_level - BREAKOUT_SL_BUFFER_ATR * atr)
        tp = snap_price(symbol, entry + BREAKOUT_TP_ATR * atr)
        if sl >= entry: log.info("%s BO SKIP SL>=entry", symbol); return None
    else:
        sl = snap_price(symbol, ref_level + BREAKOUT_SL_BUFFER_ATR * atr)
        tp = snap_price(symbol, entry - BREAKOUT_TP_ATR * atr)
        if sl <= entry: log.info("%s BO SKIP SL<=entry", symbol); return None

    sl_dist = abs(entry - sl)
    if sl_dist <= 0: return None
    if sl_dist / entry < SL_MIN_DIST_PCT:
        log.info("%s BO SKIP SL too tight (%.2f%%)", symbol, sl_dist/entry*100); return None

    leverage = safe_leverage_for_sl(entry, sl)
    set_leverage(symbol, leverage)
    if not sl_inside_liq(entry, sl, direction, leverage):
        leverage = safe_leverage_for_sl(entry, sl); set_leverage(symbol, leverage)
        if not sl_inside_liq(entry, sl, direction, leverage):
            log.info("%s BO SKIP SL beyond liq", symbol); return None

    # Risk-based sizing at the (smaller) breakout risk budget.
    qty_usdt = min((balance * BREAKOUT_RISK_PCT / sl_dist) * entry, balance * leverage)
    qty = snap_qty(symbol, qty_usdt / entry)
    if qty <= 0:
        log.info("%s BO SKIP qty 0", symbol); return None

    log.info("%s BREAKOUT %s entry=%.4f sl=%.4f ref=%.4f %s",
             symbol, direction, entry, sl, ref_level, signals)

    res = place_order(symbol, direction, qty, sl, tp)   # market IOC — momentum entry
    if not res:
        return None

    real_entry, liq = get_position_entry_liq(symbol)
    if real_entry > 0:
        entry = real_entry
    risk_usdt = abs(entry - sl) * qty

    trade = {
        "timestamp":   datetime.datetime.utcnow().isoformat(),
        "symbol":      symbol, "direction": direction,
        "entry_mode":  "breakout",
        "score":       float(len(signals)), "signals": signals,
        "entry":       entry, "sl": sl, "tp": tp,
        "qty":         qty, "leverage": leverage, "rr": 0.0,
        "ext_atr":     0.0,
        "regime_4h":   "BO", "regime_1h": "BO",
        "hour_utc":    datetime.datetime.utcnow().hour,
        "outcome":     None, "pnl": None, "closed_at": None,
        "exit_reason": None,
        "tp1_done":    False, "tp1_realized": 0.0,
        "order_result": res,
    }
    log_trade(trade)

    emoji  = "🟢" if direction == "Buy" else "🔴"
    dlabel = "LONG" if direction == "Buy" else "SHORT"
    sl_pct = (sl - entry) / entry * 100
    discord_notify(
        f"{emoji} **BREAKOUT {dlabel} {symbol}** ⚡\n"
        f"Broke `{ref_level}` on {signals[1]} | risk `{BREAKOUT_RISK_PCT*100:.0f}%`\n"
        f"Entry `{entry}` | SL `{sl}` (`{sl_pct:+.1f}%`) | trailing exit\n"
        f"Qty `{qty}` | Risk `{risk_usdt:.2f}` USDT | {leverage}x"
    )
    return trade


# ---------------------------------------------------------------------------
# HTTP keep-alive + status
# ---------------------------------------------------------------------------

HTTP_PORT      = int(os.environ.get("PORT", 5000))
STATUS_TOKEN   = os.environ.get("STATUS_TOKEN", "")
BOT_STARTED_AT = datetime.datetime.utcnow().isoformat()

class _PingHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        body = json.dumps(obj, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self, query: str) -> bool:
        if not STATUS_TOKEN:
            return True
        from urllib.parse import parse_qs
        return parse_qs(query).get("token", [""])[0] == STATUS_TOKEN

    def do_GET(self):
        from urllib.parse import urlparse
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/status":
                if not self._authorized(parsed.query):
                    return self._send_json({"error": "unauthorized"}, 401)
                journal   = load_journal()
                completed = [t for t in journal if t.get("outcome") in ("win","loss")]
                wins      = sum(1 for t in completed if t["outcome"]=="win")
                open_trades = [
                    {k: t.get(k) for k in ("symbol","direction","entry","sl","tp",
                                           "qty","score","signals","rr","leverage",
                                           "ext_atr","regime_4h","regime_1h",
                                           "timestamp","tp1_done")}
                    for t in journal
                    if t.get("outcome") is None and t.get("order_result")
                ]
                ls = load_learning_state()
                self._send_json({
                    "bot": "APEX v4.2", "status": "running",
                    "started_at": BOT_STARTED_AT,
                    "time_utc": datetime.datetime.utcnow().isoformat(),
                    "maker_entry": USE_MAKER_ENTRY,
                    "news_enabled": NEWS_ENABLED,
                    "balance_usdt": get_account_balance(),
                    "open_positions": get_open_positions(),
                    "open_trades_journal": open_trades,
                    "completed_trades": len(completed),
                    "wins": wins,
                    "win_rate": round(wins/len(completed),3) if completed else None,
                    "total_pnl_usdt": round(sum((t.get("pnl") or 0) for t in completed),4),
                    "last_5_closed": [
                        {k: t.get(k) for k in ("symbol","direction","outcome","pnl",
                                               "exit_reason","closed_at")}
                        for t in completed[-5:]
                    ],
                    "learning_last_updated": ls.get("last_updated"),
                    "signal_weights": ls.get("signal_weights"),
                })
            elif parsed.path == "/journal":
                if not self._authorized(parsed.query):
                    return self._send_json({"error": "unauthorized"}, 401)
                self._send_json(load_journal()[-20:])
            else:
                body = b"APEX Bot Running v4.2"
                self.send_response(200)
                self.send_header("Content-Type","text/plain")
                self.send_header("Content-Length",str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except Exception as e:
            try: self._send_json({"error": str(e)}, 500)
            except Exception: pass

    def log_message(self,*a): pass

def _start_http_server():
    import socket
    for i in range(5):
        port=HTTP_PORT+i
        try:
            srv=HTTPServer(("0.0.0.0",port),_PingHandler)
            srv.socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            log.info("HTTP keep-alive on port %d",port); srv.serve_forever(); return
        except OSError as e:
            log.warning("Port %d in use (%s)",port,e.strerror)
    log.error("HTTP keep-alive disabled")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    log.info("="*60)
    log.info("APEX Bot v4.2 — Pullback, Protected & Aware ACTIVE")
    log.info("Data dir: %s | Long: %s | Short: %s | Maker: %s | News: %s",
             DATA_DIR, LONG_ENABLED, SHORT_ENABLED, USE_MAKER_ENTRY, NEWS_ENABLED)
    log.info("="*60)
    discord_notify(
        f"🚀 **APEX Bot v4.2 started** — Pullback entries + safe leverage + news\n"
        f"Exits every 5 min | Entries every 15 min (candle close)\n"
        f"Entry: {'MAKER' if USE_MAKER_ENTRY else 'MARKET'} | News: {'✅' if NEWS_ENABLED else '❌'} | "
        f"Long: {'✅' if LONG_ENABLED else '❌'} Short: {'✅' if SHORT_ENABLED else '❌'}\n"
        f"Data dir: `{DATA_DIR}`"
    )
    threading.Thread(target=_start_http_server, daemon=True).start()
    if not API_KEY or not API_SECRET:
        log.error("API keys not set"); return
    starting_balance=get_account_balance()
    if starting_balance is None:
        log.error("Balance fetch failed")
        discord_notify("💀 **APEX Bot CRASHED** | Balance fetch failed — API key issue!")
        return
    log.info("Starting balance: %.2f USDT", starting_balance)
    if DATA_DIR == ".":
        discord_notify("⚠️ **Warning**: DATA_DIR not set — journal/learning data "
                       "will be LOST on redeploy. Attach a Railway Volume and set "
                       "`DATA_DIR=/data`.")

    day_start=datetime.date.today()
    week_start=day_start-datetime.timedelta(days=day_start.weekday())
    week_bal=starting_balance; weekly_pause=None
    last_hc=datetime.datetime.utcnow()
    last_summary_date=datetime.date.today()
    weekly_done=False
    last_entry_slot=None
    _wl_cache={"ts":0.0,"wl":[]}

    def _sleep_to_next_tick():
        s = FAST_INTERVAL_SECONDS - (time.time() % FAST_INTERVAL_SECONDS) + 10
        time.sleep(s)

    while True:
        try:
            now=datetime.datetime.utcnow(); today=datetime.date.today()
            ls=load_learning_state()

            # [C1] Reconcile exchange-closed trades EVERY tick, before anything else
            try:
                reconcile_closed_trades()
            except Exception as e:
                log.error("Reconcile failed: %s", e)

            if weekly_pause and now<weekly_pause:
                log.warning("Weekly pause — %.1fh left",(weekly_pause-now).total_seconds()/3600)
                _sleep_to_next_tick(); continue

            if today!=last_summary_date:
                b=get_account_balance()
                if b: send_daily_summary(b, for_date=last_summary_date)
                last_summary_date=today

            if today!=day_start:
                day_start=today; starting_balance=get_account_balance() or starting_balance
                log.info("New day — %.2f USDT",starting_balance)

            cw=today-datetime.timedelta(days=today.weekday())
            if cw!=week_start:
                week_start=cw; week_bal=get_account_balance() or starting_balance
                weekly_done=False
                log.info("New week — %.2f USDT",week_bal)

            if now.weekday()==6 and not weekly_done:
                run_weekly_learning(); weekly_done=True

            if (now-last_hc).total_seconds()>=21_600:
                b=get_account_balance() or 0; op=get_open_positions()
                t_count=len([t for t in load_journal() if t.get("outcome") in ("win","loss")])
                discord_notify(f"💚 **APEX Bot alive** | Positions:`{len(op)}/{MAX_OPEN_POSITIONS}` | "
                               f"Balance:`{b:.2f}` USDT | Learned:`{t_count}` trades")
                last_hc=now

            bal=get_account_balance()
            if bal is None: _sleep_to_next_tick(); continue

            if daily_loss_exceeded(starting_balance,bal):
                lp=(starting_balance-bal)/starting_balance*100
                log.warning("Daily loss limit hit (-%.1f%%) — paused",lp)
                _sleep_to_next_tick(); continue

            if weekly_loss_exceeded(week_bal,bal):
                weekly_pause=now+datetime.timedelta(hours=48)
                lp=(week_bal-bal)/week_bal*100
                discord_notify(f"🛑 **Weekly loss limit** | `{week_bal:.2f}`->`{bal:.2f}` "
                               f"(`-{lp:.1f}%`) | Pausing 48h")
                _sleep_to_next_tick(); continue

            # ---- FAST path (every 5 min): protect & manage open positions ----
            op=get_open_positions(); pc=len(op)
            if pc>0:
                pd4=get_position_details()
                if pd4:
                    regime_for_exit=get_market_regime()
                    manage_tp1(pd4)                      # [S4]
                    pd4_fresh=get_position_details()     # refresh after partial close
                    if pd4_fresh:
                        closed=manage_exits(pd4_fresh,regime_for_exit)
                        if closed: op=get_open_positions(); pc=len(op)
                log.info("[fast] Positions:%d/%d %s | Balance:%.2f",
                         pc,MAX_OPEN_POSITIONS,list(op.keys()),bal)

            # ---- ENTRY path (every 15 min, aligned to candle close) ----
            slot=(now.hour, (now.minute//15)*15)
            is_entry_tick = now.minute % 15 < 5 and slot != last_entry_slot
            if is_entry_tick and pc<MAX_OPEN_POSITIONS:
                last_entry_slot=slot
                regime=get_market_regime()
                refresh_news_cache()                     # [E7] one fetch per scan, cached

                if time.time()-_wl_cache["ts"]>WATCHLIST_TTL_SECONDS or not _wl_cache["wl"]:
                    wl=get_dynamic_watchlist()
                    if wl:
                        _wl_cache.update(ts=time.time(), wl=wl)
                        load_instrument_cache(wl)
                wl=_wl_cache["wl"]

                # [X1] Re-open cooldown: don't re-fire a coin we just closed.
                _now_ms=int(time.time()*1000); _cd_ms=REOPEN_COOLDOWN_HRS*3_600_000
                _last_close={}
                for _t in load_journal():
                    if _t.get("outcome") in ("win","loss") and _t.get("closed_at"):
                        _ms=_ts_to_ms(_t.get("closed_at",""))
                        _s=_t.get("symbol","")
                        if _ms>_last_close.get(_s,0): _last_close[_s]=_ms

                for sym in wl:
                    if len(op)>=MAX_OPEN_POSITIONS: break
                    if sym in op: continue
                    if _now_ms-_last_close.get(sym,0) < _cd_ms:
                        log.info("%s SKIP cooldown (closed <%.0fh ago)",sym,REOPEN_COOLDOWN_HRS); continue
                    fr=get_funding_rate(sym)
                    if fr is None: continue
                    if not (FUNDING_RATE_MIN<=fr<=FUNDING_RATE_MAX): continue
                    try:
                        trade=scan_symbol(sym,bal,regime,ls)
                        if not (trade and trade.get("order_result")) and BREAKOUT_ENABLED:
                            trade=scan_symbol_breakout(sym,bal,regime,ls)
                        if trade and trade.get("order_result"): op=get_open_positions()
                    except Exception as e:
                        log.error("Error scanning %s: %s",sym,e)
                    time.sleep(1)

                rl=f"{regime.get('regime_4h','?')}/{regime.get('regime_1h','?')}->{regime.get('allowed','?')}"
                tc=len([t for t in load_journal() if t.get("outcome") in ("win","loss")])
                log.info("Entry scan complete [%02d:%02d] | Regime:%s",slot[0],slot[1],rl)
                if slot[1]==0:
                    discord_notify(f"📊 **Hourly scan** | Regime:`{rl}` | "
                                   f"Positions:`{len(op)}/{MAX_OPEN_POSITIONS}` | "
                                   f"Balance:`{bal:.2f}` USDT | Learned:`{tc}`")
            elif is_entry_tick:
                last_entry_slot=slot
                log.info("Entry slot %02d:%02d — max positions, skip scan",slot[0],slot[1])

            _sleep_to_next_tick()

        except Exception as e:
            log.exception("Main loop error: %s", e)
            discord_notify(f"⚠️ **Loop error (recovered)**: `{e}` — retrying in 5 min")
            time.sleep(300)


if __name__ == "__main__":
    main()
