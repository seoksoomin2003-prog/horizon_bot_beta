"""The replay must not see the future, must match the bot's own calculations, and must stay away from the running bot."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from replay.market_data import BAR_MS, MINUTE_MS, WINDOW_BARS, closed_bars, regular_grid, window_at
from replay.report import analyze, compare_entries, positions_from_fills
from replay.runner import Frames, ReplayFeed
from replay.signals import load_signals, signal_rows

REPO = Path(__file__).resolve().parent.parent
T0 = 1_750_000_500_000 // BAR_MS * BAR_MS + 5 * MINUTE_MS     # deliberately not on a 15-minute boundary
RULES = {"AAAUSDT": {"amount_step": 0.01, "price_tick": 0.001, "min_amount": 0.01}}


def make_bars(minutes=(WINDOW_BARS + 8) * 15, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, minutes)))
    openp = np.r_[close[0], close[:-1]]
    high = np.maximum(openp, close) * 1.0004
    low = np.minimum(openp, close) * 0.9996
    return np.column_stack([T0 + np.arange(minutes) * MINUTE_MS, openp, high, low, close, rng.uniform(1, 9, minutes)])


def test_missing_minutes_become_flat_bars():
    bars = make_bars(10)
    grid = regular_grid(np.delete(bars, 4, axis=0))
    assert len(grid) == 10
    assert list(grid[4, 1:5]) == [bars[3, 4]] * 4 and grid[4, 5] == 0.0


def test_closed_bars_start_on_the_quarter_hour():
    closed, first = closed_bars(make_bars())
    assert first == 10 and closed[0, 0] % BAR_MS == 0


def test_forming_bar_matches_the_bar_it_becomes():
    bars = make_bars()
    closed, first = closed_bars(bars)
    bar = 305
    final_minute = first + (bar + 1) * 15 - 1
    assert window_at(bars, closed, first, final_minute)[-1] == pytest.approx(closed[bar])


@pytest.mark.parametrize("inside", [0, 7, 14])
def test_window_holds_only_the_past(inside):
    bars = make_bars()
    closed, first = closed_bars(bars)
    minute = first + 303 * 15 + inside
    changed = bars.copy()
    changed[minute + 1:, 1:] *= 3.0
    changed_closed, _ = closed_bars(changed)
    assert np.array_equal(window_at(bars, closed, first, minute), window_at(changed, changed_closed, first, minute))
    assert len(window_at(bars, closed, first, minute)) == WINDOW_BARS


def test_stored_signals_equal_the_bots_own_calculation():
    from signal_engine import SignalEngine
    bars = make_bars()
    closed, first = closed_bars(bars)
    start = first + 301 * 15 + 3
    rows = signal_rows(bars, start, start + 25)
    for i in (0, 11, 24):
        df = pd.DataFrame(window_at(bars, closed, first, start + i), columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        last = SignalEngine().compute_all_signals(df, "T").iloc[-1]
        assert rows["atr"][i] == pytest.approx(last["atr"])
        assert rows["score"][i] == last["score"]
        assert rows["bias"][i] == (1 if last["trend_bias"] == "LONG" else -1)
        assert rows["trigger"][i] == {"LONG": 1, "SHORT": -1}.get(last["trigger"], 0)


def test_fast_percentile_equals_the_real_rolling_rank():
    """Runs in a separate process, because the fast version replaces pandas' rolling apply for the whole process."""
    code = """
import numpy as np, pandas as pd
from pandas.core.window.rolling import Rolling
from signal_engine import SignalEngine
from replay.signals import fast_percentile_apply
from tests.test_replay import make_bars
real_apply = Rolling.apply
names = ["timestamp", "open", "high", "low", "close", "volume"]
for seed in range(6):
    b = make_bars(seed=seed)
    if seed == 5:                       # ties and flat stretches, the hard case for ranking
        b[:, 1:5] = np.round(b[:, 1:5], 0)
        b[1000:2000, 1:5] = 100.0
    df = pd.DataFrame(b[::15][:310], columns=names)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    Rolling.apply = real_apply
    slow = SignalEngine().compute_all_signals(df, "T")
    Rolling.apply = fast_percentile_apply
    fast = SignalEngine().compute_all_signals(df, "T")
    pd.testing.assert_frame_equal(slow, fast, check_exact=False, rtol=1e-12)
    assert (slow["trigger"].fillna("") == fast["trigger"].fillna("")).all()
print("same")
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120, env=dict(os.environ, PYTHONPATH=str(REPO)))
    assert out.returncode == 0 and "same" in out.stdout, out.stderr[-1500:]


def test_future_prices_cannot_change_a_stored_signal():
    bars = make_bars()
    closed, first = closed_bars(bars)
    start = first + 302 * 15
    changed = bars.copy()
    changed[start + 5:, 1:] *= 2.0
    a, b = signal_rows(bars, start, start + 5), signal_rows(changed, start, start + 5)
    for name in a:
        assert np.array_equal(a[name], b[name]), name


def test_missing_signals_are_an_error_not_silence(tmp_path):
    (tmp_path / "signals").mkdir()
    np.save(tmp_path / "AAAUSDT.npy", make_bars(100))
    columns = {c: np.zeros(10) for c in ("trigger", "regime", "bias", "score", "confidence", "atr")}
    np.savez(tmp_path / "signals" / f"AAAUSDT_{T0}.npz", **columns)
    with pytest.raises(ValueError, match="missing"):
        load_signals("AAAUSDT", T0, T0 + 20 * MINUTE_MS, tmp_path)
    assert len(load_signals("AAAUSDT", T0, T0 + 10 * MINUTE_MS, tmp_path)["trigger"]) == 10


def test_feed_never_returns_a_later_bar():
    bars = make_bars()
    feed = ReplayFeed({"AAAUSDT": bars}, RULES)
    now = T0 + 4600 * MINUTE_MS
    feed.set_minute(now)
    assert feed.fetch_ticker("AAA/USDT")["last"] == bars[4600, 4]
    assert feed.fetch_ohlcv("AAA/USDT", "1m", since=now - 5 * MINUTE_MS, limit=1500)[-1][0] == now
    assert feed.fetch_ohlcv("AAA/USDT", "15m", limit=300)[-1][0] <= now


def test_feed_rounds_like_the_exchange():
    import ccxt
    feed = ReplayFeed({"AAAUSDT": make_bars()}, RULES)
    assert float(feed.amount_to_precision("AAA/USDT", 2.419)) == pytest.approx(2.41)
    with pytest.raises(ccxt.InvalidOrder):
        feed.amount_to_precision("AAA/USDT", 0.008)


def test_frame_exposes_only_the_current_row_and_past_trends():
    n = 2000
    start = T0 // BAR_MS * BAR_MS
    s = {"trigger": np.zeros(n, np.int8), "regime": np.ones(n, np.int8), "bias": np.ones(n, np.int8),
         "score": np.zeros(n, np.int8), "confidence": np.zeros(n), "atr": np.arange(n, dtype=float)}
    s["bias"][50:] = -1
    frames = Frames({"AAAUSDT": s}, start)
    frames.minute_ms = start + 37 * MINUTE_MS
    df = frames.frame("AAAUSDT")
    assert df.iloc[-1]["atr"] == 37.0
    assert df.iloc[-1]["timestamp"] == pd.Timestamp(start + 30 * MINUTE_MS, unit="ms")
    assert list(df["trend_bias"]) == ["LONG"] * 3


def test_partial_closes_count_as_one_position():
    def fill(ms, side, amount, realized=0.0, fee=0.1):
        return {"timestamp": ms, "symbol": "A/USDT", "side": side, "amount": amount, "price": 10.0, "realized": realized, "fee": {"cost": fee}}
    fills = [fill(1, "buy", 10.0), fill(2, "sell", 3.0, 5.0), fill(3, "sell", 7.0, -1.0), fill(4, "sell", 2.0), fill(5, "buy", 2.0, 0.5)]
    done = positions_from_fills(fills)
    assert len(done) == 2
    assert done[0]["pnl"] == pytest.approx(5.0 - 1.0 - 0.3) and done[0]["side"] == "LONG"
    assert done[1]["side"] == "SHORT"
    summary = {"initial": 1000.0, "cash": 1004.0, "realized": 4.5, "fees": 0.5, "funding": 0.0, "start_ms": 0, "reached_ms": 86_400_000, "curve": []}
    assert analyze(summary, fills)["win_rate"] == 1.0


def test_compare_matches_each_entry_once():
    def entry(ms, side="buy"):
        return {"timestamp": ms, "symbol": "A/USDT", "side": side, "amount": 1.0, "price": 1.0, "fee": {"cost": 0.0}}
    c = compare_entries([entry(0)], [entry(60_000), entry(120_000, "sell")])
    assert c["matched"] == 1 and c["minutes_apart"] == [1.0] and len(c["replayed"]) == 1


@pytest.fixture
def scripted_run(tmp_path):
    """Falls, then rallies past the entry and collapses, with real files and a separate process, as the tool runs."""
    data = tmp_path / "data"
    (data / "signals").mkdir(parents=True)
    bars = make_bars(minutes=(WINDOW_BARS + 20) * 15)
    bars[:, 1:5] = 100.0
    bars[:, 2] = 100.1
    bars[:, 3] = 99.9
    first = int((-bars[0, 0] % BAR_MS) // MINUTE_MS)
    start = first + (WINDOW_BARS + 2) * 15
    bars[start + 30:, 1:5] = 90.0
    bars[start + 30, 1] = 100.0
    bars[start + 30, 2] = 100.0
    np.save(data / "AAAUSDT.npy", bars)
    np.save(data / "AAAUSDT.funding.npy", np.array([[bars[start + 5, 0], 0.001]]))
    (data / "markets.json").write_text(json.dumps(RULES))
    n = len(bars) - start
    trigger = np.zeros(n, np.int8)
    trigger[3] = 1      # one LONG signal
    np.savez(data / "signals" / f"AAAUSDT_{int(bars[start, 0])}.npz", trigger=trigger, regime=np.ones(n, np.int8), bias=np.ones(n, np.int8),
             score=np.ones(n, np.int8), confidence=np.full(n, 0.25), atr=np.full(n, 1.0))
    start_ms, end_ms = int(bars[start, 0]), int(bars[start + 90, 0])
    live = {p: (p.read_bytes() if p.exists() else None) for p in [REPO / "runtime/paper_binance/paper_broker.json"]}
    code = (
        "import sys; sys.argv=['replay','run','t','--data-dir',%r,'--symbols','AAAUSDT','--start','x','--end','x']\n"
        "import runpy, calendar, time\n"
        "sys.path.insert(0, %r); ns = runpy.run_path(%r, run_name='notmain')\n"
        "from pathlib import Path\n"
        "from replay.runner import prepare_run_dir\n"
        "run_dir = prepare_run_dir('t', Path(%r)); ns['isolate'](str(run_dir / 'runtime'))\n"
        "from replay.runner import run\n"
        "run('t', %d, %d, ['AAAUSDT'], data_dir=Path(%r))\n"
    ) % (str(data), str(REPO), str(REPO / "scripts/replay.py"), str(data), start_ms, end_ms, str(data))
    env = {k: v for k, v in os.environ.items() if not k.startswith("HORIZON_")}
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout[-1500:] + out.stderr[-1500:]
    run_dir = data / "runs" / "t"
    return {"summary": json.loads((run_dir / "summary.json").read_text()), "bars": bars, "start": start, "run_dir": run_dir, "live": live,
            "fills": [json.loads(x) for x in (run_dir / "paper_trades.jsonl").read_text().splitlines()]}


def test_real_pipeline_enters_after_the_signal_and_its_stop_closes(scripted_run):
    fills, bars, start = scripted_run["fills"], scripted_run["bars"], scripted_run["start"]
    assert len(fills) == 2, fills
    entry, stop = fills
    assert entry["side"] == "buy" and entry["symbol"] == "AAA/USDT:USDT"
    assert entry["timestamp"] == bars[start + 3, 0] + MINUTE_MS - 1     # at the end of the signal minute, not before
    assert entry["price"] == pytest.approx(100.0 * 1.0005)
    assert stop["reason"] == "stop" and stop["side"] == "sell"
    assert stop["timestamp"] // MINUTE_MS * MINUTE_MS == bars[start + 30, 0]
    assert stop["price"] == pytest.approx(98.0 * 0.9995)    # 2 ATR below the entry-minute price, with slippage
    assert entry["amount"] == pytest.approx(round(entry["amount"], 2))


def test_funding_is_charged_on_open_positions(scripted_run):
    s = scripted_run["summary"]
    assert s["funding_events"] == 1
    assert s["funding"] == pytest.approx(scripted_run["fills"][0]["amount"] * 100.0 * 0.001)
    fills = scripted_run["fills"]
    assert s["cash"] == pytest.approx(1000.0 + sum(f["realized"] - f["fee"]["cost"] for f in fills) - s["funding"])


def test_replay_writes_nothing_to_the_running_bots_records(scripted_run):
    for path, before in scripted_run["live"].items():
        assert (path.read_bytes() if path.exists() else None) == before
    assert (scripted_run["run_dir"] / "runtime" / "paper_binance").exists()
    assert (scripted_run["run_dir"] / "runtime" / "logs" / "trades.log").exists()


def test_replay_refuses_to_run_without_isolated_state(tmp_path):
    code = ("import os, sys; os.environ['HORIZON_RUNTIME_DIR'] = %r; sys.path.insert(0, %r)\n"
            "from pathlib import Path\nfrom replay.runner import run\n"
            "run('t', 0, 60000, [], data_dir=Path(%r))\n") % (str(REPO / "runtime"), str(REPO), str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert out.returncode != 0 and "not isolated" in out.stderr


def test_the_tool_blanks_keys_before_any_bot_import():
    source = (REPO / "scripts/replay.py").read_text()
    body = source[source.index("def main()"):]
    for bot_loading_import in ("from replay.signals import compute", "from replay.runner import run\n"):
        at = body.index(bot_loading_import)
        assert "isolate(" in body[body.rindex("elif args.command", 0, at):at]
    code = ("import runpy, sys, os; sys.argv=['r','report']; sys.path.insert(0, %r)\n"
            "os.environ['BINANCE_API_KEY'] = 'x'\n"
            "ns = runpy.run_path(%r, run_name='notmain'); ns['isolate']('/tmp/none')\n"
            "assert os.environ['BINANCE_API_KEY'] == '' and os.environ['TRADING_MODE'] == 'paper'\n") % (str(REPO), str(REPO / "scripts/replay.py"))
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
