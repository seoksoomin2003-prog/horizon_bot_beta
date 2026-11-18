"""v16: the top-scored coin alone, with and without buy/sell timing rules, against random coins. Seen data, not evidence.

    .venv/bin/python scripts/topcoin_v16_dev.py --smoke    # random scores: checks the machinery, saves nothing
    .venv/bin/python scripts/topcoin_v16_dev.py            # the real scores, once

Rules and gates: docs/research/direction_preregistration_v16.md (fixed before this script first ran).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from research import alpha, h4_data, risk, topcoin
from research.h4_data import BAR_MS
from scripts.direction_v6_dev import END, SCORE_START, TRAIN_START
from scripts.quant_explore import build, load, market_panels, reference

NAMES = ["carry", "low_vol", "low_beta"]       # the v15 score
HORIZON = 42
COUNT = 30
RANDOM_RUNS = 200
YEARS = range(2022, 2027)
OUT = h4_data.DATA_DIR / "direction_v16_dev_result.json"


def make_panel(smoke: bool) -> topcoin.Panel:
    coins = reference()
    member_rows = alpha.members({s: load(s)[0] for s in coins}, TRAIN_START, END, COUNT)
    closes, funding = market_panels(coins, h4_data.WARMUP_START_MS, END)
    rets = risk.returns(closes)
    sig = risk.sigma(rets)
    bet = risk.beta(rets, risk.market_return(rets, alpha.membership(member_rows, closes.index.to_numpy(), coins)))
    data = alpha.add_scores(build(HORIZON, coins, member_rows, sig, bet), NAMES)
    pred = alpha.walk_forward(data, NAMES, HORIZON, TRAIN_START, SCORE_START, END)
    if smoke:          # random scores as persistent as real ones (AR(1) per coin, about a week's memory), same spread
        rng, sd = np.random.default_rng(16), pred["alpha"].std()
        pred = pred.sort_values(["symbol", "close_ms"])
        noise = []
        for _, g in pred.groupby("symbol", sort=False):
            x = np.zeros(len(g))
            for i in range(1, len(g)):
                x[i] = 0.98 * x[i - 1] + rng.normal(0, np.sqrt(1 - 0.98 ** 2))
            noise.append(x)
        pred["alpha"] = np.concatenate(noise) * sd + pred["alpha"].mean()
    grid = closes.index.to_numpy()
    bars = {s: load(s)[0] for s in coins}
    ohlc = {name: pd.DataFrame({s: pd.Series(b[:, k], index=b[:, 0].astype(np.int64) + BAR_MS).reindex(grid) for s, b in bars.items()},
                               index=grid) for k, name in ((1, "open"), (2, "high"), (3, "low"))}
    a = pred.pivot(index="close_ms", columns="symbol", values="alpha").reindex(index=grid, columns=coins)
    return topcoin.Panel(times=grid, symbols=coins, open=ohlc["open"].to_numpy(), high=ohlc["high"].to_numpy(),
                         low=ohlc["low"].to_numpy(), close=closes[coins].to_numpy(), sigma=sig[coins].to_numpy(),
                         funding=funding[coins].to_numpy(), alpha=a.to_numpy())


def main() -> None:
    os.nice(10)
    smoke = "--smoke" in sys.argv
    started = time.time()
    panel = make_panel(smoke)
    start = int(np.searchsorted(panel.times, SCORE_START))
    end = int(np.searchsorted(panel.times, END)) - 1
    print(f"panel {panel.close.shape}, closes {start} .. {end} ({time.time() - started:.0f}s)", flush=True)
    months = pd.to_datetime(panel.times, unit="ms").to_period("M").astype(str).to_numpy()
    out: dict = {"smoke": smoke, "start_ms": int(panel.times[start]), "end_ms": int(panel.times[end]), "rules": {}}
    for name, rules in (("A", topcoin.RULES_A), ("B", topcoin.RULES_B)):
        r: dict = {}
        for k in (1, 2):
            r[f"top_cost{k}"] = topcoin.summary(panel, topcoin.simulate(panel, topcoin.top_picker(panel), rules, start, end, cost=k), start, end)
        r["bottom"] = topcoin.summary(panel, topcoin.simulate(panel, topcoin.bottom_picker(panel), rules, start, end), start, end)
        runs = []
        for seed in range(RANDOM_RUNS):
            null = topcoin.permuted(panel, months, seed)
            runs.append(topcoin.summary(null, topcoin.simulate(null, topcoin.top_picker(null), rules, start, end), start, end))
        totals = np.array([x["total"] for x in runs])
        r["random"] = {"median": float(np.median(totals)), "p05": float(np.quantile(totals, 0.05)),
                       "p95": float(np.quantile(totals, 0.95)), "top_percentile": float((totals < r["top_cost1"]["total"]).mean()),
                       "by_year_median": {y: float(np.median([x["by_year"].get(y, np.nan) for x in runs])) for y in YEARS},
                       "drawdown_median": float(np.median([x["max_drawdown"] for x in runs])),
                       "trades_median": float(np.median([x["trades"] for x in runs]))}
        out["rules"][name] = r
        t1 = r["top_cost1"]
        print(f"[{name}] top: net {t1['total']:+.1%} (x2 {r['top_cost2']['total']:+.1%}), {t1['trades']} trades, "
              f"fill {t1['fill_rate'] or 0:.0%}, exits {t1['exits']}, trade mean {t1['trade_mean'] or 0:+.2%} "
              f"[{(t1['trade_ci'] or [0, 0])[0]:+.2%}, {(t1['trade_ci'] or [0, 0])[1]:+.2%}], win {t1['win_rate'] or 0:.0%}, "
              f"in market {t1['time_in_market']:.0%}, drawdown {t1['max_drawdown']:.0%}, years "
              + " ".join(f"{y}:{v:+.0%}" for y, v in t1["by_year"].items()), flush=True)
        print(f"[{name}] shuffled scores x{RANDOM_RUNS}: median {r['random']['median']:+.1%}, 5-95% [{r['random']['p05']:+.1%}, "
              f"{r['random']['p95']:+.1%}], top beats {r['random']['top_percentile']:.0%} of them; bottom {r['bottom']['total']:+.1%} "
              f"({time.time() - started:.0f}s)", flush=True)
    hold_btc = topcoin.hold(panel, start, end, "BTCUSDT")
    hold_all = topcoin.hold(panel, start, end)
    out["hold"] = {"btc": float(hold_btc[-1] - 1), "universe": float(hold_all[-1] - 1),
                   "btc_drawdown": float((1 - hold_btc / np.maximum.accumulate(hold_btc)).max()),
                   "universe_drawdown": float((1 - hold_all / np.maximum.accumulate(hold_all)).max())}
    print(f"hold, no costs: BTC {out['hold']['btc']:+.1%} (drawdown {out['hold']['btc_drawdown']:.0%}), "
          f"30 coins equal weight {out['hold']['universe']:+.1%} (drawdown {out['hold']['universe_drawdown']:.0%})")

    A, B = out["rules"]["A"], out["rules"]["B"]
    years_above = sum(A["top_cost1"]["by_year"].get(y, -np.inf) > A["random"]["by_year_median"][y] for y in YEARS)
    gates = {"G1": A["top_cost1"]["total"] > A["random"]["p95"],
             "G2": A["top_cost1"]["total"] > B["top_cost1"]["total"] and A["random"]["median"] > B["random"]["median"],
             "G3": A["top_cost2"]["total"] > 0,
             "G4": years_above >= 4}
    out["gates"], out["years_above_random"] = gates, years_above
    out["verdict"] = "TOP_COIN_RULES_ON_SEEN_DATA" if all(gates.values()) else "NO_TOP_COIN_EDGE"
    print(f"gates: G1 picking {gates['G1']}, G2 timing {gates['G2']}, G3 cost x2 {gates['G3']}, "
          f"G4 years above random {years_above}/5 {gates['G4']} -> {out['verdict']}")
    out["when"] = time.strftime("%FT%TZ", time.gmtime())
    if smoke:
        print("(smoke run on random scores: nothing saved, not a result)")
    else:
        OUT.write_text(json.dumps(out, indent=1, default=str))
        print(f"saved {OUT}")


if __name__ == "__main__":
    main()
