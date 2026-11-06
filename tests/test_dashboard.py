"""The dashboard must stay local, read-only and away from keys."""
import http.client
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from dashboard.server import HOST, build_server, build_summary, mask

REPO = Path(__file__).resolve().parent.parent
FAKE_SECRET = "FAKESECRETVALUE_abcdefghijklmnopqrstuvwxyz0123456789"


def fill(i, symbol, cash, realized=0.0, fee=0.05):
    return {"id": f"T{i}", "symbol": symbol, "side": "buy", "price": 10.0, "amount": 1.0, "fee": {"cost": fee}, "timestamp": 1000 * i,
            "realized": realized, "reason": "stop" if realized else "market", "cash_after": cash}


@pytest.fixture
def root(tmp_path):
    state = tmp_path / "runtime" / "paper_binance"
    state.mkdir(parents=True)
    (state / "paper_broker.json").write_text(json.dumps({
        "initial": 1000.0, "cash": 1000.9, "realized": 1.0, "fees": 0.1,
        "positions": {"AAA/USDT": {"side": "LONG", "amount": 1.0, "entry": 10.0, "opened_ms": 5000}},
        "orders": {"A1": {"symbol": "AAA/USDT", "type": "STOP_MARKET", "trigger": 9.0},
                   "A2": {"symbol": "AAA/USDT", "type": "TAKE_PROFIT_MARKET", "trigger": 12.0}}}))
    (state / "paper_trades.jsonl").write_text(
        json.dumps(fill(1, "BBB/USDT", 999.95)) + "\n" + json.dumps(fill(2, "BBB/USDT", 1000.9, realized=1.0)) + "\n{partial")
    logs = tmp_path / "logs" / "paper"
    logs.mkdir(parents=True)
    (logs / "trades.log").write_text(
        "2026-11-06 20:17:31,442 - trades_logger - INFO - [ENTRY APPROVED] AAAUSDT LONG\n"
        "2026-11-06 20:17:32,000 - trades_logger - INFO - [REJECTED: RISK_ENGINE] noise\n")
    (logs / "errors.log").write_text(f"2026-11-06 20:18:00,000 - errors_logger - ERROR - failed with key {FAKE_SECRET}\n")
    (tmp_path / ".env").write_text(f"BINANCE_API_KEY={FAKE_SECRET}\n")
    return tmp_path


@pytest.fixture
def server(root):
    srv = build_server(root, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def request(server, method="GET", path="/", host=None):
    port = server.server_address[1]
    conn = http.client.HTTPConnection(HOST, port, timeout=5)
    conn.putrequest(method, path, skip_host=True)
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    conn.endheaders()
    res = conn.getresponse()
    body = res.read()
    conn.close()
    return res, body


def test_listens_on_loopback_only(server):
    assert server.server_address[0] == "127.0.0.1"


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_nothing_can_be_changed(server, root, method):
    before = (root / "runtime/paper_binance/paper_broker.json").read_bytes()
    assert request(server, method, "/api/summary")[0].status == 405
    assert (root / "runtime/paper_binance/paper_broker.json").read_bytes() == before


def test_foreign_host_is_refused(server):
    assert request(server, host="evil.example")[0].status == 403
    assert request(server, path="/api/summary", host="evil.example")[0].status == 403


@pytest.mark.parametrize("path", ["/.env", "/../.env", "/%2e%2e/.env", "/runtime/paper_binance/paper_broker.json", "/dashboard/server.py"])
def test_only_known_routes_exist(server, path):
    res, body = request(server, path=path)
    assert res.status == 404 and FAKE_SECRET.encode() not in body


def test_key_never_reaches_a_response(server):
    for path in ("/", "/api/summary"):
        assert FAKE_SECRET.encode() not in request(server, path=path)[1]


def test_page_has_no_outside_resources_and_no_controls(server):
    res, body = request(server)
    html = body.decode()
    assert "connect-src 'self'" in res.getheader("Content-Security-Policy")
    assert res.getheader("X-Content-Type-Options") == "nosniff"
    assert "__NONCE__" not in html
    for banned in ("<button", "<form", "<input", "innerHTML", "https://", "method:"):
        assert banned not in html.replace("http://www.w3.org/2000/svg", "")


def test_process_never_loads_the_bot_config_or_env():
    code = ("import sys, runpy; sys.path.insert(0, 'dashboard'); import server; "
            "assert not {'config', 'dotenv', 'ccxt', 'utils.persistence'} & set(sys.modules), 'bot module loaded'")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, check=True, timeout=30)


def test_summary_matches_the_terminal_report(root):
    from scripts.paper_report import summarize
    state_dir = root / "runtime/paper_binance"
    state = json.loads((state_dir / "paper_broker.json").read_text())
    trades = [json.loads(line) for line in (state_dir / "paper_trades.jsonl").read_text().splitlines() if not line.startswith("{partial")]
    expected = summarize(state, trades)
    got = build_summary(root)
    for key in ("cash", "return", "realized", "fees", "fills", "closes", "win_rate", "drawdown"):
        assert got[key] == pytest.approx(expected[key]), key
    assert {r["symbol"]: r["pnl"] for r in got["by_symbol"]} == pytest.approx(expected["by_symbol"])


def test_position_shows_its_protective_orders(root):
    p = build_summary(root)["positions"][0]
    assert (p["stop"], p["take_profit"]) == (9.0, 12.0)


def test_activity_is_filtered_and_errors_are_masked(root):
    s = build_summary(root)
    assert [e["text"] for e in s["activity"]] == ["[ENTRY APPROVED] AAAUSDT LONG"]
    assert FAKE_SECRET not in s["errors"][0]["text"] and "***" in s["errors"][0]["text"]
    assert mask("short") == "short"


def test_half_written_state_keeps_the_last_good_values(root):
    cache = {}
    assert build_summary(root, cache)["cash"] == 1000.9
    (root / "runtime/paper_binance/paper_broker.json").write_text('{"initial": 10')
    assert build_summary(root, cache)["cash"] == 1000.9


def test_no_records_yet_is_not_an_error(tmp_path):
    s = build_summary(tmp_path)
    assert s["has_data"] is False and s["bot"]["running"] is False


def test_bot_status_reads_pid_and_flags_stale_records(root):
    run = root / ".run"
    run.mkdir()
    (run / "bot.pid").write_text(str(os.getpid()))
    mtime = (root / "runtime/paper_binance/paper_broker.json").stat().st_mtime
    assert build_summary(root, now=mtime + 10)["bot"] == {"running": True, "age_seconds": pytest.approx(10), "stale": False}
    assert build_summary(root, now=mtime + 1000)["bot"]["stale"] is True
    (run / "bot.pid").write_text("99999999")
    assert build_summary(root)["bot"]["running"] is False
