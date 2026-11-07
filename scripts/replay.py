#!/usr/bin/env python3
"""Replays past prices through the bot's real pipeline. It cannot place orders or touch the running bot's records.

  python scripts/replay.py download --start 2025-11-06 --end 2026-11-06
  python scripts/replay.py signals  --start 2025-11-06 --end 2026-11-06
  python scripts/replay.py run NAME --start 2025-11-06 --end 2026-11-06 [--slip 5] [--fee 0.0005]
  python scripts/replay.py report NAME
  python scripts/replay.py compare NAME     compares a run with the live paper ledger
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BLANKED = ("BINANCE_API_KEY", "API_KEY", "BINANCE_SECRET", "BINANCE_API_SECRET", "SECRET_KEY", "BITGET_API_KEY", "BITGET_API_SECRET",
           "BITGET_PASSPHRASE", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "BINANCE_TELEGRAM_TOKEN", "BINANCE_TELEGRAM_CHAT_ID")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT", "AVAXUSDT", "DOGEUSDT", "BNBUSDT", "SUIUSDT", "ADAUSDT"]


def isolate(runtime_dir: str) -> None:
    """Must run before any bot import. Set values win over .env, so keys stay blank and paths stay separate."""
    os.environ.update({"TRADING_MODE": "paper", "HORIZON_RUNTIME_DIR": runtime_dir, "HORIZON_LOG_DIR": os.path.join(runtime_dir, "logs")})
    os.environ.update({k: "" for k in BLANKED})


def utc_ms(text: str) -> int:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return calendar.timegm(time.strptime(text, fmt)) * 1000
        except ValueError:
            continue
    raise SystemExit(f"날짜 형식이 틀립니다: {text}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["download", "signals", "run", "report", "compare"])
    parser.add_argument("name", nargs="?")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--slip", type=float, default=5.0)
    parser.add_argument("--fee", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    args = parser.parse_args()
    symbols = args.symbols.split(",")

    from replay.market_data import DATA_DIR, MINUTE_MS, WINDOW_BARS
    data_dir = args.data_dir or DATA_DIR

    if args.command == "download":
        from replay.market_data import download
        warmup = (WINDOW_BARS + 2) * 15 * MINUTE_MS
        download(symbols, utc_ms(args.start) - warmup, utc_ms(args.end), data_dir)
    elif args.command == "signals":
        isolate(tempfile.mkdtemp(prefix="horizon_replay_signals_"))
        from replay.signals import compute
        compute(symbols, utc_ms(args.start), utc_ms(args.end), args.workers, data_dir)
    elif args.command == "run":
        if not args.name:
            raise SystemExit("실행 이름이 필요합니다")
        from replay.runner import prepare_run_dir
        run_dir = prepare_run_dir(args.name, data_dir)
        isolate(str(run_dir / "runtime"))
        os.nice(10)
        from replay.runner import run
        run(args.name, utc_ms(args.start), utc_ms(args.end), symbols, fee=args.fee, slippage_bps=args.slip, data_dir=data_dir)
        return _report(args.name, data_dir)
    elif args.command == "report":
        return _report(args.name, data_dir)
    elif args.command == "compare":
        return _compare(args.name, data_dir)
    return 0


def _report(name: str, data_dir: Path) -> int:
    from replay.report import analyze, read_ledger, render
    run_dir = data_dir / "runs" / name
    summary = json.loads((run_dir / "summary.json").read_text())
    print(render(summary, analyze(summary, read_ledger(run_dir / "paper_trades.jsonl"))))
    return 0


def _compare(name: str, data_dir: Path) -> int:
    from replay.report import compare_entries, read_ledger
    run_dir = data_dir / "runs" / name
    summary = json.loads((run_dir / "summary.json").read_text())
    live = read_ledger(ROOT / "runtime" / "paper_binance" / "paper_trades.jsonl")
    live = [t for t in live if summary["start_ms"] <= t["timestamp"] < summary["end_ms"]]
    c = compare_entries(live, read_ledger(run_dir / "paper_trades.jsonl"))
    print(f"실제 진입 {len(c['real'])}건, 재생 진입 {len(c['replayed'])}건, 15분 안에 일치 {c['matched']}건")
    for side, rows in (("실제", c["real"]), ("재생", c["replayed"])):
        for r in rows:
            print(f"  {side} {time.strftime('%m-%d %H:%M', time.gmtime(r['opened_ms'] / 1000))} UTC {r['symbol']} {r['side']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
