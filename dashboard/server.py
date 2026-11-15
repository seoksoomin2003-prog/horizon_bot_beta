#!/usr/bin/env python3
"""Paper trading dashboard, separate from the bot.

Standard library only, and it imports nothing from the bot except paper/settings.py, which is standard library too:
the bot's config module loads .env, and this process must never have keys in memory. It listens on the loopback
address only and makes no outbound connection.

The one write route starts or stops paper trading through run_mac.sh, or the direction predictor through
run_predictor.sh (which places no orders). It can never start live trading: run_mac.sh needs --live and a typed phrase
for that, and this server passes neither. A control request must come from this page
(same origin and a per-start token), so another website open in the browser cannot send one.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import settings as paper_settings  # noqa: E402  standard library only
from paper import funding  # noqa: E402  standard library only

HOST = "127.0.0.1"
DEFAULT_PORT = 8791
PAGE = Path(__file__).with_name("page.html")
TAIL_BYTES = 256 * 1024
STALE_AFTER_SECONDS = 300
MAX_FILLS = 100
MAX_LOG_LINES = 30
ACTIVITY = re.compile(r"\[(ENTRY APPROVED|ENTRY ABORTED[^\]]*|EXIT ENGINE|TIMEOUT|STARTUP|ORDER_IN_FLIGHT|MIN_ORDER|SKIPPED: PAPER_RULES|REJECTED: SLIPPAGE|SLIPPAGE)\]|COOLDOWN ACTIVATED")
MAX_BODY = 4096
CONTROL_TIMEOUT = 60
LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - \S+ - ([A-Z]+) - (.*)$")
SECRET_LIKE = re.compile(r"[A-Za-z0-9_\-+/=]{32,}")
BAR_MS = 4 * 3_600_000
PREDICTOR_STALE_MS = BAR_MS + 20 * 60_000
ROUND_TRIP_COST = 0.002     # the research cost: 0.10% per side
MAX_PREDICTION_ROWS = 60
MAX_LATEST_ROWS = 15
SIZE_PERIODS = {2: "8시간", 6: "24시간", 18: "3일", 42: "7일"}      # research/size_model.PERIODS, in 4h bars
SIZE_SHARE = 0.1            # the share of the coins on each end that is scored, as in the research: 3 of 30


def mask(text: str) -> str:
    """The bot's logs should hold no keys. This keeps a mistake there from reaching the page."""
    return SECRET_LIKE.sub("***", text)


def tail_lines(path: Path) -> list[str]:
    try:
        with open(path, "rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - TAIL_BYTES))
            data = f.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()


def log_entries(path: Path, pattern: Optional[re.Pattern] = None) -> list[dict]:
    out = []
    for line in tail_lines(path):
        m = LOG_LINE.match(line)
        if m and (pattern is None or pattern.search(m.group(3))):
            out.append({"time": m.group(1), "level": m.group(2), "text": mask(m.group(3))[:300]})
    return out


def read_ledger(path: Path) -> list[dict]:
    fills = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    fills.append(json.loads(line))
                except ValueError:  # a line still being written
                    continue
    except OSError:
        pass
    return fills


def state_dirs(root: Path) -> list[Path]:
    return [p for p in root.glob("runtime/paper_*") if p.is_dir()]


def find_state_dir(root: Path) -> Optional[Path]:
    found = sorted(root.glob("runtime/paper_*/paper_broker.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0].parent if found else None


def bot_status(root: Path, state_mtime: Optional[float], now: float) -> dict:
    running = False
    try:
        pid = int((root / ".run" / "bot.pid").read_text().strip())
        os.kill(pid, 0)  # signal 0 only checks that the process exists
        running = True
    except (OSError, ValueError):
        pass
    age = None if state_mtime is None else max(0.0, now - state_mtime)
    return {"running": running, "age_seconds": age, "stale": bool(running and age is not None and age > STALE_AFTER_SECONDS)}


def fill_view(t: dict) -> dict:
    fee = float(t.get("fee", {}).get("cost", 0.0))
    realized = float(t.get("realized", 0.0) or 0.0)
    return {"time_ms": t.get("timestamp"), "symbol": t.get("symbol"), "side": t.get("side"), "price": t.get("price"),
            "amount": t.get("amount"), "reason": t.get("reason"), "fee": fee, "realized": realized,
            "is_close": bool(t.get("realized")), "cash_after": t.get("cash_after")}


def build_summary(root: Path, cache: Optional[dict] = None, now: Optional[float] = None) -> dict:
    now = time.time() if now is None else now
    cache = {} if cache is None else cache
    state_dir = find_state_dir(root)
    state, mtime = None, None
    if state_dir is not None:
        path = state_dir / "paper_broker.json"
        try:
            mtime = path.stat().st_mtime
            state = json.loads(path.read_text(encoding="utf-8"))
            cache["state"] = state
        except (OSError, ValueError):
            state = cache.get("state")
    result = {"now_ms": now * 1000, "has_data": state is not None, "bot": bot_status(root, mtime, now),
              "settings": paper_settings.load(root / "runtime"), "run_settings": (state or {}).get("settings"),
              "limits": {"max_leverage": paper_settings.MAX_LEVERAGE, "min_initial": paper_settings.MIN_INITIAL,
                         "fees": paper_settings.FEES},
              "past_runs": past_runs(root), "predictions": prediction_summary(root, now * 1000)}
    log_dir = root / "logs" / "paper"
    errors = log_entries(log_dir / "errors.log")
    result["error_count_recent"] = len(errors)
    result["errors"] = errors[-MAX_LOG_LINES:][::-1]
    result["activity"] = log_entries(log_dir / "trades.log", ACTIVITY)[-MAX_LOG_LINES:][::-1]
    if state is None:
        return result

    fills = read_ledger(state_dir / "paper_trades.jsonl")
    closes = [t for t in fills if t.get("realized")]
    wins = [t for t in closes if t["realized"] - t["fee"]["cost"] > 0]
    initial = float(state["initial"])
    cash = peak = initial
    drawdown = 0.0
    by_symbol: dict = defaultdict(lambda: {"pnl": 0.0, "closes": 0, "wins": 0})
    curve = []
    for t in fills:
        cash = t.get("cash_after", cash)
        peak = max(peak, cash)
        drawdown = min(drawdown, cash / peak - 1)
        curve.append({"time_ms": t["timestamp"], "cash": cash})
        row = by_symbol[t["symbol"]]
        row["pnl"] += t.get("realized", 0.0) - t["fee"]["cost"]
        if t.get("realized"):
            row["closes"] += 1
            row["wins"] += t["realized"] - t["fee"]["cost"] > 0

    orders = list(state.get("orders", {}).values())
    run_settings = state.get("settings") or {}
    marks = state.get("marks", {})
    positions = []
    unrealized_total = 0.0
    for symbol, p in state.get("positions", {}).items():
        mine = [o for o in orders if o.get("symbol") == symbol]
        stop = next((o.get("trigger") for o in mine if str(o.get("type", "")).startswith("STOP")), None)
        take = next((o.get("trigger") for o in mine if str(o.get("type", "")).startswith("TAKE_PROFIT")), None)
        mark = marks.get(symbol)
        unrealized = None if mark is None else (1 if p["side"] == "LONG" else -1) * (mark - p["entry"]) * p["amount"]
        unrealized_total += unrealized or 0.0
        lev = float(run_settings.get("leverage") or 0)
        liq = paper_settings.liquidation_price(p["side"], p["entry"], lev) if run_settings.get("market") == "futures" and lev > 0 else None
        positions.append({"symbol": symbol, "side": p["side"], "amount": p["amount"], "entry": p["entry"],
                          "opened_ms": p.get("opened_ms"), "stop": stop, "take_profit": take, "mark": mark,
                          "unrealized": unrealized, "liquidation": liq})
    result.update({
        "initial": initial, "cash": state["cash"], "return": state["cash"] / initial - 1, "realized": state["realized"],
        "equity": state["cash"] + unrealized_total, "marks_ms": state.get("marks_ms"),
        "funding": funding.round_trip(initial, state["cash"] + unrealized_total, result["predictions"].get("xrp_price")),
        "fees": state["fees"], "fills": len(fills), "closes": len(closes), "wins": len(wins),
        "win_rate": len(wins) / len(closes) if closes else None, "drawdown": drawdown, "positions": positions,
        "open_orders": len(orders), "curve": curve, "first_ms": fills[0]["timestamp"] if fills else None,
        "by_symbol": sorted(({"symbol": k, **v} for k, v in by_symbol.items()), key=lambda r: r["pnl"], reverse=True),
        "recent_fills": [fill_view(t) for t in fills[-MAX_FILLS:]][::-1],
    })
    return result


def past_runs(root: Path, limit: int = 20) -> list[dict]:
    """One row per archived paper run, newest first. Reads only the archive folder."""
    rows = []
    for broker in (root / "runtime" / "archive").glob("*/paper_*/paper_broker.json"):
        try:
            state = json.loads(broker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        fills = read_ledger(broker.parent / "paper_trades.jsonl")
        closes = [t for t in fills if t.get("realized")]
        initial = float(state.get("initial") or 0) or None
        rows.append({"name": broker.parent.parent.name, "settings": state.get("settings"), "initial": initial,
                     "cash": state.get("cash"), "return": state["cash"] / initial - 1 if initial else None,
                     "fills": len(fills), "closes": len(closes), "open_positions": len(state.get("positions", {})),
                     "first_ms": fills[0]["timestamp"] if fills else None, "last_ms": fills[-1]["timestamp"] if fills else None})
    rows.sort(key=lambda r: r["name"], reverse=True)
    return rows[:limit]


def pid_alive(path: Path) -> bool:
    try:
        os.kill(int(path.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def now_view(folder: Path) -> dict:
    """The predictor's newest "as of now" answer and whether a newer request is still waiting for one."""
    answer, asked = read_json(folder / "now.json"), read_json(folder / "now_request.json").get("requested_ms") or 0
    coins = answer.get("coins") if isinstance(answer.get("coins"), list) else []
    return {"pending": asked > (answer.get("requested_ms") or 0), "requested_ms": asked or None, "ok": answer.get("ok"),
            "made_ms": answer.get("made_ms"), "message": mask(str(answer.get("message", "")))[:200], "coins": coins}


def size_summary(folder: Path) -> list[dict]:
    """The live "how much" record (v14) for the large coins. Per period: what the coins with the highest and the lowest
    predicted sizes at each close were told and what they got, as returns over the median coin's. Only predictions
    by the newest model in the file count."""
    predicted = read_ledger(folder / "size_predictions.jsonl")
    answers = {(o.get("symbol"), o.get("close_ms"), o.get("horizon")): o.get("relative_return")
               for o in read_ledger(folder / "size_outcomes.jsonl")}
    newest = max((p.get("version", 0) for p in predicted), default=None)
    latest = max((p.get("close_ms", 0) for p in predicted), default=None)
    mean = lambda values: sum(values) / len(values) if values else None
    out = []
    for horizon, label in SIZE_PERIODS.items():
        top, bottom, closes, now_top, now_bottom = [], [], set(), [], []
        for p in predicted:
            if p.get("horizon") != horizon or p.get("version", 0) != newest:
                continue
            k = max(int(p["coins"] * SIZE_SHARE + 0.5), 1)
            side = top if p["rank"] <= k else bottom if p["rank"] > p["coins"] - k else None
            if side is None:
                continue
            if p["close_ms"] == latest:
                (now_top if side is top else now_bottom).append({"symbol": p["symbol"], "size": p["size"], "rank": p["rank"]})
            answer = answers.get((p["symbol"], p["close_ms"], horizon))
            if answer is not None:
                side.append((p["size"], answer))
                closes.add(p["close_ms"])
        top_actual, bottom_actual = mean([a for _, a in top]), mean([a for _, a in bottom])
        out.append({"horizon": horizon, "label": label, "closes": len(closes), "top": len(top),
                    "top_predicted": mean([s for s, _ in top]), "top_actual": top_actual, "top_above": sum(a > 0 for _, a in top),
                    "bottom": len(bottom), "bottom_predicted": mean([s for s, _ in bottom]), "bottom_actual": bottom_actual,
                    "spread": None if top_actual is None or bottom_actual is None else top_actual - bottom_actual,
                    "now_top": sorted(now_top, key=lambda c: c["rank"]), "now_bottom": sorted(now_bottom, key=lambda c: -c["rank"])})
    return [{"latest_ms": latest, **row} for row in out] if predicted else []


def shadow_summary(folder: Path) -> dict:
    """v15's shadow book (predictor/shadow.py): virtual money, no orders. Its path, costs, positions and the alphas'
    rank IC once their 7 days have passed."""
    book, ics = read_ledger(folder / "shadow_book.jsonl"), read_ledger(folder / "shadow_ic.jsonl")
    try:
        state = json.loads((folder / "shadow_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if not book:
        return {"rows": 0}
    equity = [1000.0] + [r["equity"] for r in book]
    peak, drawdown = equity[0], 0.0
    for e in equity:
        peak = max(peak, e)
        drawdown = max(drawdown, 1 - e / peak)
    days = (book[-1]["close_ms"] - book[0]["close_ms"]) / 86_400_000
    # the first rebalance builds the book from nothing; turnover counts the trading after it, once a day has passed
    turnover = sum(r["traded"] for r in book[1:]) / days if days >= 1 else None
    values = [r["ic"] for r in ics]
    spaced = values[::42]                       # closes 7 days apart: their horizons do not overlap
    sd = (sum((v - sum(spaced) / len(spaced)) ** 2 for v in spaced) / (len(spaced) - 1)) ** 0.5 if len(spaced) > 2 else 0.0
    weights = sorted((state.get("weights") or {}).items(), key=lambda kv: -kv[1])
    return {"rows": len(book), "first_ms": book[0]["close_ms"], "last_ms": book[-1]["close_ms"], "equity": book[-1]["equity"],
            "total": book[-1]["equity"] / 1000.0 - 1, "max_drawdown": drawdown,
            "cost": sum(r["cost"] for r in book), "funding": sum(r["funding"] for r in book),
            "turnover_per_day": turnover, "gross": book[-1]["gross"], "net": book[-1]["net"],
            "beta": book[-1]["beta"], "positions": book[-1]["positions"], "rebalanced": sum(bool(r["rebalanced"]) for r in book),
            "note": mask(str(book[-1].get("note", "")))[:120],
            "longs": [{"symbol": s, "weight": w} for s, w in weights if w > 0],
            "shorts": [{"symbol": s, "weight": w} for s, w in weights[::-1] if w < 0],
            "ic_count": len(values), "ic_mean": sum(values) / len(values) if values else None,
            "ic_t": (sum(spaced) / len(spaced)) / (sd / len(spaced) ** 0.5) if sd > 0 else None, "ic_spaced": len(spaced)}


def prediction_summary(root: Path, now_ms: float) -> dict:
    """The live direction predictor's record (predictor/live.py), scored against what happened next."""
    folder = root / "runtime" / "predictor"
    predicted = read_ledger(folder / "predictions.jsonl")
    outcomes = {(o.get("symbol"), o.get("close_ms"), o.get("horizon", 1), o.get("kind", "direction")): o
                for o in read_ledger(folder / "outcomes.jsonl")}
    try:
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = {}
    running = pid_alive(root / ".run" / "predictor.pid")
    updated = status.get("updated_ms")
    rows, scored, signals, confident = [], [], [], []
    tally = lambda: {"scored": 0, "hits": 0, "confident": 0, "confident_hits": 0, "top5": 0, "top5_hits": 0}
    eight, relative = tally(), {"4h": tally(), "8h": tally()}      # 8-hour predictions; "beats the median coin" predictions
    eight_p, relative_p, eight_flags = {}, {}, {}
    # The relative model changed with v11; only predictions by the newest model in the file are scored.
    newest_relative = max((p.get("version", 9) for p in predicted if p.get("kind") == "relative"), default=None)
    for p in predicted:
        horizon, kind = p.get("horizon", 1), p.get("kind", "direction")     # older rows are 4-hour direction ones
        o = outcomes.get((p.get("symbol"), p.get("close_ms"), horizon, kind))
        if horizon != 1 or kind != "direction":
            counts = eight if kind == "direction" else relative.get(f"{4 * horizon}h")
            if counts is None:
                continue
            if horizon == 2:
                (eight_p if kind == "direction" else relative_p)[(p.get("symbol"), p.get("close_ms"))] = p.get("p_up")
                if kind == "direction":
                    eight_flags[(p.get("symbol"), p.get("close_ms"))] = (bool(p.get("confident")), bool(p.get("top5")))
            answer = None if o is None else o.get("relative_return" if kind == "relative" else "log_return")
            if kind == "relative" and p.get("version", 9) != newest_relative:
                answer = None
            if answer is not None:
                hit = (p["p_up"] > 0.5) == (answer > 0)
                for key, counted in (("scored", True), ("confident", p.get("confident")), ("top5", p.get("top5"))):
                    if counted:
                        counts[key] += 1
                        counts["hits" if key == "scored" else f"{key}_hits"] += hit
            continue
        row = {"close_ms": p.get("close_ms"), "symbol": p.get("symbol"), "price": p.get("price"), "p_up": p.get("p_up"),
               "expected_gain": p.get("expected_gain"), "signal": bool(p.get("signal")), "confident": bool(p.get("confident")),
               "top5": bool(p.get("top5")), "next_price": None, "up": None, "hit": None}
        if o is not None:
            up = o["log_return"] > 0
            row.update(next_price=o.get("next_price"), up=up, hit=(p["p_up"] > 0.5) == up)
            scored.append((row, o["log_return"]))
            if row["signal"]:
                signals.append((row, o["log_return"]))
            if row["confident"]:
                confident.append((row, o["log_return"]))
        rows.append(row)
    latest_close = max((r["close_ms"] for r in rows), default=None)
    n = len(scored)
    net = [(1 if r["p_up"] > 0.5 else -1) * (math.exp(ret) - 1) - ROUND_TRIP_COST for r, ret in signals]
    confident_net = [(1 if r["p_up"] > 0.5 else -1) * (math.exp(ret) - 1) - ROUND_TRIP_COST for r, ret in confident]
    latest = sorted((r for r in rows if r["close_ms"] == latest_close), key=lambda r: -abs(r["p_up"] - 0.5))
    latest = [{**r, "p_up_8h": eight_p.get((r["symbol"], r["close_ms"])),
               "p_above_8h": relative_p.get((r["symbol"], r["close_ms"]))} for r in latest]
    # Every coin at the latest close, for looking one up: its rank is its place among the coins by confidence, 1 = surest.
    surest_8h = sorted((r for r in latest if r["p_up_8h"] is not None), key=lambda r: -abs(r["p_up_8h"] - 0.5))
    rank_8h = {r["symbol"]: i + 1 for i, r in enumerate(surest_8h)}
    coins = []
    for i, r in enumerate(latest):
        flags = eight_flags.get((r["symbol"], r["close_ms"]), (False, False))
        coins.append({"symbol": r["symbol"], "close_ms": r["close_ms"], "price": r["price"], "p_up": r["p_up"], "rank": i + 1,
                      "confident": r["confident"], "top5": r["top5"], "p_up_8h": r["p_up_8h"], "rank_8h": rank_8h.get(r["symbol"]),
                      "confident_8h": flags[0], "top5_8h": flags[1]})
    return {
        "running": running, "stale": bool(running and updated and now_ms - updated > PREDICTOR_STALE_MS),
        "updated_ms": updated, "ok": status.get("ok"), "message": mask(str(status.get("message", "")))[:200],
        "model_month": status.get("model_month"), "first_ms": rows[0]["close_ms"] if rows else None,
        "latest": latest[:MAX_LATEST_ROWS], "latest_count": len(latest),
        "coins": sorted(coins, key=lambda c: c["symbol"]),
        "now": now_view(folder),
        "xrp_price": next((r["price"] for r in latest if r["symbol"] == "XRPUSDT"), None),
        "collected": status.get("collected"), "eight": eight, "relative": relative, "relative_version": newest_relative,
        "size": size_summary(folder),
        "shadow": shadow_summary(folder),
        "confident": len(confident), "confident_hits": sum(r["hit"] for r, _ in confident),
        "confident_net": sum(confident_net) / len(confident_net) if confident_net else None,
        "scored": n, "hits": sum(r["hit"] for r, _ in scored), "ups": sum(r["up"] for r, _ in scored),
        "signals": len(signals), "signal_hits": sum(r["hit"] for r, _ in signals),
        "signal_net": sum(net) / len(net) if net else None,
        "recent": [r for r in rows if r["hit"] is not None][::-1][:MAX_PREDICTION_ROWS],
    }


def run_script(root: Path, *args: str, script: str = "run_mac.sh") -> tuple[bool, str]:
    """Runs run_mac.sh without any live-trading variable, so only paper mode can start."""
    env = {k: v for k, v in os.environ.items() if k not in ("TRADING_MODE", "LIVE_TRADING_CONFIRM")}
    try:
        done = subprocess.run([str(root / script), *args], cwd=root, env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=CONTROL_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    return done.returncode == 0, mask((done.stdout + done.stderr).strip())[-500:]


def archive_records(root: Path, now: Optional[float] = None) -> Optional[Path]:
    """Moves the current paper records aside, never deleting them."""
    sources = state_dirs(root) + [p for p in [root / "logs" / "paper"] if p.exists()]
    if not sources:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    dest = root / "runtime" / "archive" / f"paper_{stamp}"
    n = 1
    while dest.exists():
        n += 1
        dest = root / "runtime" / "archive" / f"paper_{stamp}_{n}"
    dest.mkdir(parents=True)
    for src in sources:
        shutil.move(str(src), str(dest / ("logs_paper" if src.name == "paper" else src.name)))
    return dest


def is_running(root: Path) -> bool:
    return bot_status(root, None, time.time())["running"]


def predictor_control(root: Path, action: str, runner=run_script) -> tuple[int, dict]:
    """Starts or stops the direction predictor, which places no orders."""
    running = pid_alive(root / ".run" / "predictor.pid")
    if action == "predictor_start":
        if running:
            return 200, {"ok": True, "message": "예측 기록기가 이미 켜져 있습니다."}
        ok, out = runner(root, "start", script="run_predictor.sh")
        return (200, {"ok": True, "message": "예측 기록기를 켰습니다. 처음에는 과거 시세를 받느라 1~2분 걸리고, 첫 예측은 다음 4시간봉 마감 때 나옵니다."}) \
            if ok else (500, {"ok": False, "message": f"켜지 못했습니다. {out}"})
    if action == "predictor_stop":
        if not running:
            return 200, {"ok": True, "message": "예측 기록기가 이미 꺼져 있습니다."}
        ok, out = runner(root, "stop", script="run_predictor.sh")
        return (200, {"ok": True, "message": "예측 기록기를 껐습니다. 기록은 남아 있습니다."}) if ok else \
               (500, {"ok": False, "message": f"끄지 못했습니다. {out}"})
    if action == "predict_now":
        if not running:
            return 409, {"ok": False, "message": "예측 기록이 꺼져 있습니다. 먼저 켜 주세요."}
        folder = root / "runtime" / "predictor"
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "now_request.tmp"
        tmp.write_text(json.dumps({"requested_ms": int(time.time() * 1000)}), encoding="utf-8")
        os.replace(tmp, folder / "now_request.json")
        return 200, {"ok": True, "message": "지금 시세로 예측을 요청했습니다. 코인 60개의 시세를 새로 받아 계산하느라 1~2분 걸립니다."}
    return 400, {"ok": False, "message": "알 수 없는 요청입니다."}


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def control(root: Path, action: str, raw_settings: Optional[dict] = None, runner=run_script) -> tuple[int, dict]:
    if action in ("predictor_start", "predictor_stop", "predict_now"):
        return predictor_control(root, action, runner)
    if action == "stop":
        if not is_running(root):
            return 200, {"ok": True, "message": "이미 중지되어 있습니다."}
        ok, out = runner(root, "stop")
        return (200, {"ok": True, "message": "가상 매매를 중지했습니다."}) if ok and not is_running(root) else \
               (500, {"ok": False, "message": f"중지하지 못했습니다. {out}"})
    if action == "resume":
        if is_running(root):
            return 200, {"ok": True, "message": "이미 실행 중입니다."}
        ok, out = runner(root, "start")
        return (200, {"ok": True, "message": "기존 기록을 이어서 가상 매매를 켰습니다."}) if ok else \
               (500, {"ok": False, "message": f"켜지 못했습니다. {out}"})
    if action == "start_new":
        try:
            clean = paper_settings.validate(raw_settings)
        except ValueError as e:
            return 400, {"ok": False, "message": str(e)}
        if is_running(root):
            ok, out = runner(root, "stop")
            if not ok or is_running(root):
                return 500, {"ok": False, "message": f"기존 가상 매매를 중지하지 못해 새로 시작하지 않았습니다. {out}"}
        archived = archive_records(root)
        paper_settings.save(root / "runtime", clean)
        ok, out = runner(root, "start")
        where = f" 이전 기록은 {archived.relative_to(root)}에 보관했습니다." if archived else ""
        if not ok:
            return 500, {"ok": False, "message": f"설정은 저장했지만 켜지 못했습니다. {out}{where}"}
        return 200, {"ok": True, "message": f"새 가상 매매를 시작했습니다.{where}", "settings": clean}
    return 400, {"ok": False, "message": "알 수 없는 요청입니다."}


class Handler(BaseHTTPRequestHandler):
    server_version = "dashboard"
    sys_version = ""
    root: Path
    cache: dict
    token: str
    control_lock: threading.Lock

    def _allowed_host(self) -> bool:
        # Blocks DNS rebinding: a web page cannot reach this server through a name that merely resolves to loopback.
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _send(self, status: int, body: bytes, content_type: str, csp: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", csp)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _plain(self, status: int, text: str) -> None:
        self._send(status, text.encode(), "text/plain; charset=utf-8", "default-src 'none'")

    def do_GET(self) -> None:
        if not self._allowed_host():
            return self._plain(403, "forbidden")
        path = self.path.split("?", 1)[0]
        if path == "/":
            nonce = secrets.token_urlsafe(16)
            csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; connect-src 'self'; "
                   "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            html = PAGE.read_text(encoding="utf-8").replace("__NONCE__", nonce).replace("__TOKEN__", self.token)
            return self._send(200, html.encode(), "text/html; charset=utf-8", csp)
        if path == "/api/summary":
            body = json.dumps(build_summary(self.root, self.cache), ensure_ascii=False).encode()
            return self._send(200, body, "application/json; charset=utf-8", "default-src 'none'")
        return self._plain(404, "not found")

    do_HEAD = do_GET

    def do_POST(self) -> None:
        if not self._allowed_host():
            return self._plain(403, "forbidden")
        if self.path != "/api/control":
            return self._plain(405, "not allowed")
        port = self.server.server_address[1]
        if self.headers.get("Origin") not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
            return self._plain(403, "forbidden")
        if not secrets.compare_digest(self.headers.get("X-Control-Token", ""), self.token):
            return self._plain(403, "forbidden")
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._plain(415, "json only")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._plain(400, "bad length")
        if not 0 < length <= MAX_BODY:
            return self._plain(400, "bad length")
        try:
            body = json.loads(self.rfile.read(length))
            action, raw = body["action"], body.get("settings")
        except (ValueError, KeyError, TypeError):
            return self._plain(400, "bad request")
        if not self.control_lock.acquire(blocking=False):
            status, result = 409, {"ok": False, "message": "다른 요청을 처리하는 중입니다. 잠시 뒤 다시 눌러 주세요."}
        else:
            try:
                status, result = control(self.root, str(action), raw)
            finally:
                self.control_lock.release()
        self._send(status, json.dumps(result, ensure_ascii=False).encode(), "application/json; charset=utf-8", "default-src 'none'")

    def _not_allowed(self) -> None:
        self._plain(405, "not allowed")

    do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _not_allowed

    def log_message(self, *args) -> None:
        pass


def build_server(root: Path, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"root": Path(root), "cache": {}, "token": secrets.token_urlsafe(32),
                                                "control_lock": threading.Lock()})
    server = ThreadingHTTPServer((HOST, port), handler)
    if not ipaddress.ip_address(server.server_address[0]).is_loopback:
        server.server_close()
        raise RuntimeError("dashboard must listen on loopback only")
    return server


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    try:
        server = build_server(root)
    except OSError as e:
        print(f"포트 {DEFAULT_PORT}을 열 수 없습니다: {e}", file=sys.stderr)
        return 1
    print(f"http://{HOST}:{DEFAULT_PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
