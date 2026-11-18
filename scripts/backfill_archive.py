"""Backfills the exchange's public archive (data.binance.vision) for the 59 reference coins, then checks the data.

    .venv/bin/python scripts/backfill_archive.py            # download (resumes) and check
    .venv/bin/python scripts/backfill_archive.py --check    # check only

Adds, up to VAL_END_MS (the final-check period stays closed):
  replay_data/h4/flow_archive/{coin}.klines.npy   4h volume, trade count and taker-buy volume, from 2020-09
  replay_data/h4/flow_archive/{coin}.premium.npy  4h premium index (perpetual over index price), from 2020-09
  replay_data/h4/depth/{coin}.npy                 order-book notional within 1..5 % of the mid before each 4h close, from 2023

Only data quality is checked here (coverage, agreement with the bars and flow already stored). No signal or return is
computed, so this script looks at nothing a later preregistration would have to declare.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from research import h4_data
from research.h4_data import BAR_MS, VAL_END_MS
from scripts.direction_v6_dev import ORIGINAL

KINDS = ("klines4h", "premium4h", "depth")
REPORT = h4_data.DATA_DIR / "archive_backfill_report.json"


def reference() -> list[str]:
    return [s for s in ORIGINAL + h4_data.choose_v6_symbols() if s != "BTCDOMUSDT"]


def stored_bars(s: str) -> np.ndarray:
    return (h4_data.load_validation(s) if s in ORIGINAL else h4_data.load_v6(s))[0]


def check(coins: list[str]) -> dict:
    out = {"coins": {}, "made": time.strftime("%FT%TZ", time.gmtime())}
    for s in coins:
        flow, depth, bars = h4_data.load_flow_archive(s), h4_data.load_depth(s), stored_bars(s)
        c = {}
        # the archive's 4h closes against the bars the research already uses
        both = np.intersect1d(flow[:, 0], bars[:, 0])
        fa, ba = flow[np.searchsorted(flow[:, 0], both)], bars[np.searchsorted(bars[:, 0], both)]
        rel = np.abs(fa[:, 4] / ba[:, 4] - 1)
        c["flow_rows"], c["bars_rows"], c["shared_rows"] = len(flow), len(bars), len(both)
        c["close_mismatch_rows"] = int((rel > 1e-9).sum())
        c["volume_max_rel_diff"] = float(np.nanmax(np.abs(fa[:, 5] / ba[:, 5] - 1))) if len(both) else None
        c["flow_first"] = time.strftime("%F", time.gmtime(flow[0, 0] / 1000))
        c["premium_missing"] = int(np.isnan(flow[:, 10]).sum())
        c["taker_share_median"] = float(np.nanmedian(flow[:, 8] / flow[:, 5]))
        if s in ORIGINAL:                                    # against the flow fetched from the live API for v5
            api = h4_data.load_flow(s)
            shared = np.intersect1d(api[:, 0], flow[:, 0])
            a, f = api[np.searchsorted(api[:, 0], shared)], flow[np.searchsorted(flow[:, 0], shared)]
            c["api_taker_max_rel_diff"] = float(np.nanmax(np.abs(f[:, 8] / a[:, 2] - 1)))
            ok = np.isfinite(a[:, 3]) & np.isfinite(f[:, 10])
            c["api_premium_max_abs_diff"] = float(np.max(np.abs(f[ok, 10] - a[ok, 3]))) if ok.any() else None
        if len(depth):
            first = int(depth[0, 0])
            closes = np.arange(first, VAL_END_MS, BAR_MS)
            c["depth_first"] = time.strftime("%F", time.gmtime(first / 1000))
            c["depth_coverage"] = round(len(depth) / len(closes), 4)
            c["depth_snapshots_median"] = float(np.median(depth[:, 1]))
            notional = depth[:, 2:]
            bid, ask = notional[:, :5][:, ::-1], notional[:, 5:]        # nearest level first
            c["depth_monotonic_share"] = float(np.mean(np.all(np.diff(bid, axis=1) >= -1e-6, axis=1) & np.all(np.diff(ask, axis=1) >= -1e-6, axis=1)))
            c["depth_1pct_median_usd"] = float(np.nanmedian(bid[:, 0] + ask[:, 0]))
        out["coins"][s] = c
    return out


def main() -> None:
    os.nice(10)
    coins = reference()
    if "--check" not in sys.argv:
        h4_data.download_archive(coins, KINDS)
    report = check(coins)
    REPORT.write_text(json.dumps(report, indent=1))
    rows = report["coins"]
    print(f"{len(rows)} coins; close mismatches {sum(r['close_mismatch_rows'] for r in rows.values())}; "
          f"premium missing {sum(r['premium_missing'] for r in rows.values())}")
    for s, r in rows.items():
        print(s, {k: v for k, v in r.items() if k not in ("flow_rows", "bars_rows")})


if __name__ == "__main__":
    main()
