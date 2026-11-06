#!/usr/bin/env python3
"""Read-only paper trading dashboard, separate from the bot.

Standard library only, and it imports nothing from the bot: the bot's config module loads .env, and this process
must never have keys in memory. It reads three known locations under the project, has no write route, makes no
outbound connection, and listens on the loopback address only.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import secrets
import sys
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

HOST = "127.0.0.1"
DEFAULT_PORT = 8791
PAGE = Path(__file__).with_name("page.html")
TAIL_BYTES = 256 * 1024
STALE_AFTER_SECONDS = 300
MAX_FILLS = 100
MAX_LOG_LINES = 30
ACTIVITY = re.compile(r"\[(ENTRY APPROVED|ENTRY ABORTED[^\]]*|EXIT ENGINE|TIMEOUT|STARTUP|ORDER_IN_FLIGHT)\]|COOLDOWN ACTIVATED")
LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - \S+ - ([A-Z]+) - (.*)$")
SECRET_LIKE = re.compile(r"[A-Za-z0-9_\-+/=]{32,}")


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
    result = {"now_ms": now * 1000, "has_data": state is not None, "bot": bot_status(root, mtime, now)}
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
    positions = []
    for symbol, p in state.get("positions", {}).items():
        mine = [o for o in orders if o.get("symbol") == symbol]
        stop = next((o.get("trigger") for o in mine if str(o.get("type", "")).startswith("STOP")), None)
        take = next((o.get("trigger") for o in mine if str(o.get("type", "")).startswith("TAKE_PROFIT")), None)
        positions.append({"symbol": symbol, "side": p["side"], "amount": p["amount"], "entry": p["entry"],
                          "opened_ms": p.get("opened_ms"), "stop": stop, "take_profit": take})
    result.update({
        "initial": initial, "cash": state["cash"], "return": state["cash"] / initial - 1, "realized": state["realized"],
        "fees": state["fees"], "fills": len(fills), "closes": len(closes), "wins": len(wins),
        "win_rate": len(wins) / len(closes) if closes else None, "drawdown": drawdown, "positions": positions,
        "open_orders": len(orders), "curve": curve, "first_ms": fills[0]["timestamp"] if fills else None,
        "by_symbol": sorted(({"symbol": k, **v} for k, v in by_symbol.items()), key=lambda r: r["pnl"], reverse=True),
        "recent_fills": [fill_view(t) for t in fills[-MAX_FILLS:]][::-1],
    })
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "dashboard"
    sys_version = ""
    root: Path
    cache: dict

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
            html = PAGE.read_text(encoding="utf-8").replace("__NONCE__", nonce)
            return self._send(200, html.encode(), "text/html; charset=utf-8", csp)
        if path == "/api/summary":
            body = json.dumps(build_summary(self.root, self.cache), ensure_ascii=False).encode()
            return self._send(200, body, "application/json; charset=utf-8", "default-src 'none'")
        return self._plain(404, "not found")

    do_HEAD = do_GET

    def _read_only(self) -> None:
        self._plain(405, "read only")

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _read_only

    def log_message(self, *args) -> None:
        pass


def build_server(root: Path, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"root": Path(root), "cache": {}})
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
