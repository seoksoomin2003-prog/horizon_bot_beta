"""The dashboard must stay local and away from keys, and its only write route must start paper trading, never live."""
import http.client
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from dashboard.server import HOST, archive_records, build_server, build_summary, control, mask, run_script
from paper import settings as paper_settings

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


def request(server, method="GET", path="/", host=None, headers=None, body=None):
    port = server.server_address[1]
    conn = http.client.HTTPConnection(HOST, port, timeout=5)
    conn.putrequest(method, path, skip_host=True)
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    for k, v in (headers or {}).items():
        conn.putheader(k, v)
    if body is not None:
        conn.putheader("Content-Length", str(len(body)))
    conn.endheaders(body)
    res = conn.getresponse()
    body = res.read()
    conn.close()
    return res, body


def test_listens_on_loopback_only(server):
    assert server.server_address[0] == "127.0.0.1"


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_records_cannot_be_written_through_other_routes(server, root, method):
    before = (root / "runtime/paper_binance/paper_broker.json").read_bytes()
    assert request(server, method, "/api/summary")[0].status == 405
    assert (root / "runtime/paper_binance/paper_broker.json").read_bytes() == before


def control_headers(server, token=None, origin=None):
    port = server.server_address[1]
    return {"Origin": origin or f"http://127.0.0.1:{port}", "X-Control-Token": server.RequestHandlerClass.token if token is None else token,
            "Content-Type": "application/json"}


@pytest.mark.parametrize("token,origin", [("wrong", None), ("", None), (None, "http://evil.example"), (None, "null")])
def test_control_needs_this_pages_token_and_origin(server, root, token, origin):
    body = json.dumps({"action": "stop"}).encode()
    res, _ = request(server, "POST", "/api/control", headers=control_headers(server, token, origin), body=body)
    assert res.status == 403


def test_control_without_origin_header_is_refused(server):
    headers = control_headers(server)
    del headers["Origin"]
    assert request(server, "POST", "/api/control", headers=headers, body=b'{"action":"stop"}')[0].status == 403


def test_control_rejects_bad_settings_before_touching_anything(server, root):
    body = json.dumps({"action": "start_new", "settings": {"market": "futures", "leverage": 500, "initial": 50, "min_order": True}}).encode()
    res, out = request(server, "POST", "/api/control", headers=control_headers(server), body=body)
    assert res.status == 400 and "레버리지" in json.loads(out)["message"]
    assert (root / "runtime/paper_binance/paper_broker.json").exists()


def test_page_carries_the_control_token(server):
    html = request(server)[1].decode()
    assert server.RequestHandlerClass.token in html and "__TOKEN__" not in html


def fake_runner(calls, root, running_after=None):
    def run(r, *args):
        calls.append(args)
        pid = r / ".run" / "bot.pid"
        if args == ("start",):
            pid.parent.mkdir(exist_ok=True)
            pid.write_text(str(os.getpid()))
        elif args == ("stop",):
            pid.unlink(missing_ok=True)
        return True, ""
    return run


def test_start_new_keeps_old_records_saves_settings_and_starts_paper(root):
    calls = []
    status, out = control(root, "start_new", {"market": "spot", "leverage": 5, "initial": 50, "min_order": True}, runner=fake_runner(calls, root))
    assert status == 200 and out["ok"], out
    assert calls == [("start",)]
    archived = list((root / "runtime/archive").iterdir())
    assert len(archived) == 1 and (archived[0] / "paper_binance/paper_broker.json").exists() and (archived[0] / "logs_paper/trades.log").exists()
    assert not (root / "runtime/paper_binance").exists() and not (root / "logs/paper").exists()
    assert paper_settings.load(root / "runtime") == {"market": "spot", "leverage": 1, "initial": 50.0, "min_order": True}


def test_start_new_stops_a_running_bot_first(root):
    calls = []
    (root / ".run").mkdir()
    (root / ".run/bot.pid").write_text(str(os.getpid()))
    status, _ = control(root, "start_new", {"market": "futures", "leverage": 3, "initial": 50, "min_order": True}, runner=fake_runner(calls, root))
    assert status == 200 and calls == [("stop",), ("start",)]


def test_stop_and_resume(root):
    calls = []
    run = fake_runner(calls, root)
    assert control(root, "stop", runner=run)[1]["message"] == "이미 중지되어 있습니다."
    assert control(root, "resume", runner=run)[0] == 200
    assert control(root, "stop", runner=run)[0] == 200
    assert calls == [("start",), ("stop",)] and (root / "runtime/paper_binance").exists()


def test_unknown_action_is_refused(root):
    assert control(root, "start_live", runner=lambda *a: pytest.fail("must not run"))[0] == 400


def test_archive_never_overwrites(root):
    first = archive_records(root, now=0)
    (root / "runtime/paper_binance").mkdir()
    second = archive_records(root, now=0)
    assert first != second and first.exists() and second.exists()


def test_script_runs_without_live_variables(tmp_path, monkeypatch):
    script = tmp_path / "run_mac.sh"
    script.write_text('#!/bin/bash\necho "mode=${TRADING_MODE:-unset} confirm=${LIVE_TRADING_CONFIRM:-unset} args=$*"\n')
    script.chmod(0o755)
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "I_ACCEPT_REAL_LOSSES")
    ok, out = run_script(tmp_path, "start")
    assert ok and out == "mode=unset confirm=unset args=start"


def test_real_launcher_never_starts_live_without_the_flag():
    text = (REPO / "run_mac.sh").read_text()
    assert 'if [ "${2:-}" = "--live" ]' in text and "export TRADING_MODE=paper" in text


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


def test_page_has_no_outside_resources(server):
    res, body = request(server)
    html = body.decode()
    assert "connect-src 'self'" in res.getheader("Content-Security-Policy")
    assert "form-action 'none'" in res.getheader("Content-Security-Policy")
    assert res.getheader("X-Content-Type-Options") == "nosniff"
    assert "__NONCE__" not in html
    for banned in ("<form", "innerHTML", "https://", "--live"):
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


def test_positions_are_valued_at_the_last_recorded_price(root):
    path = root / "runtime/paper_binance/paper_broker.json"
    state = json.loads(path.read_text())
    state.update({"settings": {"market": "futures", "leverage": 10, "fee": 0.0005, "min_order": True},
                  "marks": {"AAA/USDT": 12.0}, "marks_ms": 7000})
    path.write_text(json.dumps(state))
    s = build_summary(root)
    p = s["positions"][0]
    assert p["mark"] == 12.0 and p["unrealized"] == pytest.approx(2.0)
    assert p["liquidation"] == pytest.approx(10.0 * (1 - 0.1 + 0.005))
    assert s["equity"] == pytest.approx(1000.9 + 2.0) and s["marks_ms"] == 7000


def test_spot_or_unknown_settings_show_no_liquidation_price(root):
    assert build_summary(root)["positions"][0]["liquidation"] is None
    assert build_summary(root)["positions"][0]["unrealized"] is None


def test_past_runs_list_archived_records(root):
    archive_records(root, now=0)
    runs = build_summary(root)["past_runs"]
    assert len(runs) == 1 and runs[0]["initial"] == 1000.0 and runs[0]["closes"] == 1 and runs[0]["open_positions"] == 1
