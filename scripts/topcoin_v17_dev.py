"""v17: the top coin by mu = beta * (market trend forecast) + alpha, long or flat (C) and long, short or flat (D).
Seen data, not evidence.

    .venv/bin/python scripts/topcoin_v17_dev.py --smoke    # random trend and scores: checks the machinery, saves nothing
    .venv/bin/python scripts/topcoin_v17_dev.py            # the real data, once

Rules and gates: docs/research/direction_preregistration_v17.md (fixed before this script first ran).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from research import alpha, h4_data, risk, timed, topcoin
from research.topcoin import Panel
from scripts.direction_v6_dev import END, SCORE_START, TRAIN_START
from scripts.topcoin_v16_dev import COUNT, HORIZON, NAMES, YEARS, make_panel
from scripts.quant_explore import load, market_panels, reference

RUNS = 200
OUT = h4_data.DATA_DIR / "direction_v17_dev_result.json"


def ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    e = rng.normal(0, np.sqrt(1 - phi ** 2), n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def market_inputs(panel: Panel, smoke: bool) -> tuple[np.ndarray, np.ndarray, dict, pd.Series]:
    """beta panel, m_hat on the grid, the monthly slopes, and the index return series."""
    coins = panel.symbols
    member_rows = alpha.members({s: load(s)[0] for s in coins}, TRAIN_START, END, COUNT)
    closes, _ = market_panels(coins, h4_data.WARMUP_START_MS, END)
    rets = risk.returns(closes)
    market = risk.market_return(rets, alpha.membership(member_rows, closes.index.to_numpy(), coins))
    bet = risk.beta(rets, market)
    z = None
    if smoke:          # a random trend as persistent as a 28 .. 84-day one (about a month's memory per 4h bar)
        z = pd.Series(ar1(len(market), 0.995, np.random.default_rng(17)), index=market.index)
    m_hat, slopes = timed.forecast(market, SCORE_START, END, z)
    return bet[coins].reindex(panel.times).to_numpy(), m_hat.reindex(panel.times).to_numpy(), slopes, market


def run(panel: Panel, MU: np.ndarray, rules: timed.TimedRules, start: int, end: int, cost: float = 1.0) -> dict:
    res = timed.simulate(panel, MU, rules, start, end, cost=cost)
    out = topcoin.summary(panel, res, start, end)
    closed = [tr for tr in res.trades if tr["why"] != "open"]
    out["short_share"] = float(np.mean([tr["side"] < 0 for tr in closed])) if closed else None
    out["symbols"] = pd.Series([tr["symbol"] for tr in res.trades]).value_counts().head(8).to_dict()
    return out


def main() -> None:
    os.nice(10)
    smoke = "--smoke" in sys.argv
    started = time.time()
    panel = make_panel(smoke)                          # v16's panel: alpha is the v15 score (random in the smoke run)
    beta, m_hat, slopes, market = market_inputs(panel, smoke)
    start = int(np.searchsorted(panel.times, SCORE_START))
    end = int(np.searchsorted(panel.times, END)) - 1
    months = pd.to_datetime(panel.times, unit="ms").to_period("M").astype(str).to_numpy()
    print(f"panel {panel.close.shape}, closes {start} .. {end} ({time.time() - started:.0f}s)", flush=True)

    # the market forecast's own quality (reported, not a gate)
    fwd = timed.forward(market).reindex(panel.times).to_numpy()
    span = slice(start, end + 1)
    ok = np.isfinite(m_hat[span]) & np.isfinite(fwd[span])
    every = np.zeros(end + 1 - start, bool)
    every[::timed.HORIZON] = True
    corr = lambda mask: float(np.corrcoef(m_hat[span][mask], fwd[span][mask])[0, 1]) if mask.sum() > 2 else None
    out: dict = {"smoke": smoke, "start_ms": int(panel.times[start]), "end_ms": int(panel.times[end]),
                 "market": {"corr_all": corr(ok), "corr_every_7d": corr(ok & every), "n_every_7d": int((ok & every).sum()),
                            "share_above_cost": float((m_hat[span][ok] > timed.ROUND_TRIP).mean()),
                            "share_below_minus_cost": float((m_hat[span][ok] < -timed.ROUND_TRIP).mean()),
                            "slopes": {pd.Timestamp(k, unit="ms").strftime("%Y-%m"): v for k, v in slopes.items()}},
                 "rules": {}}
    mk = out["market"]
    print(f"market forecast: corr {mk['corr_all']:+.3f} (every 7 days {mk['corr_every_7d']:+.3f}, n={mk['n_every_7d']}), "
          f"above cost {mk['share_above_cost']:.0%}, below -cost {mk['share_below_minus_cost']:.0%}, "
          f"slope first/last {list(slopes.values())[0]:+.3f}/{list(slopes.values())[-1]:+.3f}", flush=True)

    MU = timed.mu(panel.alpha, beta, m_hat)
    for name, rules in (("C", timed.RULES_C), ("D", timed.RULES_D)):
        r: dict = {f"cost{k}": run(panel, MU, rules, start, end, cost=k) for k in (1, 2)}
        totals, years = [], []
        for seed in range(RUNS):
            null = timed.simulate(panel, timed.mu(panel.alpha, beta, timed.shifted(m_hat, start, end, seed)), rules, start, end)
            s = topcoin.summary(panel, null, start, end)
            totals.append(s["total"])
            # a run liquidated earlier has no money left: its later years are 0/0, counted as no change
            years.append({y: (0.0 if np.isnan(v) else v) for y, v in s["by_year"].items()})
        totals = np.array(totals)
        r["market_null"] = {"median": float(np.median(totals)), "p025": float(np.quantile(totals, 0.025)),
                            "p975": float(np.quantile(totals, 0.975)), "percentile": float((totals < r["cost1"]["total"]).mean()),
                            "by_year_median": {y: float(np.median([x.get(y, np.nan) for x in years])) for y in YEARS}}
        picks = []
        for seed in range(RUNS):
            p = topcoin.permuted(panel, months, seed)
            picks.append(topcoin.summary(p, timed.simulate(p, timed.mu(p.alpha, beta, m_hat), rules, start, end), start, end)["total"])
        picks = np.array(picks)
        r["pick_null"] = {"median": float(np.median(picks)), "p05": float(np.quantile(picks, 0.05)),
                          "p95": float(np.quantile(picks, 0.95)), "percentile": float((picks < r["cost1"]["total"]).mean())}
        out["rules"][name] = r
        t1 = r["cost1"]
        print(f"[{name}] net {t1['total']:+.1%} (cost x2 {r['cost2']['total']:+.1%}), {t1['trades']} trades, exits {t1['exits']}, "
              f"short {t1['short_share'] or 0:.0%}, trade mean {t1['trade_mean'] or 0:+.2%} "
              f"[{(t1['trade_ci'] or [0, 0])[0]:+.2%}, {(t1['trade_ci'] or [0, 0])[1]:+.2%}], win {t1['win_rate'] or 0:.0%}, "
              f"in market {t1['time_in_market']:.0%}, drawdown {t1['max_drawdown']:.0%}, years "
              + " ".join(f"{y}:{v:+.0%}" for y, v in t1["by_year"].items()), flush=True)
        print(f"[{name}] market shifted x{RUNS}: median {r['market_null']['median']:+.1%}, 2.5-97.5% "
              f"[{r['market_null']['p025']:+.1%}, {r['market_null']['p975']:+.1%}], beats {r['market_null']['percentile']:.0%}; "
              f"scores shuffled: median {r['pick_null']['median']:+.1%}, beats {r['pick_null']['percentile']:.0%} "
              f"({time.time() - started:.0f}s)", flush=True)

    # reference points, not gates
    btc = panel.symbols.index("BTCUSDT")
    only_btc = np.full_like(MU, np.nan)
    only_btc[:, btc] = m_hat
    hold_btc = topcoin.hold(panel, start, end, "BTCUSDT")
    out["reference"] = {
        "alpha_only_C": run(panel, timed.mu(panel.alpha, beta, np.zeros_like(m_hat)), timed.RULES_C, start, end)["total"],
        "btc_timed": run(panel, only_btc, timed.RULES_C, start, end)["total"],
        "btc_hold": float(hold_btc[-1] - 1),
        "btc_hold_drawdown": float((1 - hold_btc / np.maximum.accumulate(hold_btc)).max()),
    }
    ref = out["reference"]
    print(f"reference: alpha only (C) {ref['alpha_only_C']:+.1%}, BTC timed by m_hat {ref['btc_timed']:+.1%}, "
          f"BTC hold {ref['btc_hold']:+.1%} (drawdown {ref['btc_hold_drawdown']:.0%}), cash 0%")

    out["gates"] = {}
    for name in ("C", "D"):
        r = out["rules"][name]
        mine = {y: (0.0 if np.isnan(v) else v) for y, v in r["cost1"]["by_year"].items()}
        above = sum(mine.get(y, -np.inf) > r["market_null"]["by_year_median"][y] for y in YEARS)
        g = {"G1": r["cost2"]["total"] > 0, "G2": r["cost1"]["max_drawdown"] < ref["btc_hold_drawdown"] / 2,
             "G3": r["cost1"]["total"] > r["market_null"]["p975"], "G4": above >= 4, "years_above": above}
        out["gates"][name] = g
        print(f"gates {name}: G1 cost x2 {g['G1']}, G2 drawdown {g['G2']}, G3 timing {g['G3']}, G4 years {above}/5 {g['G4']}")
    passed = [n for n in ("C", "D") if all(out["gates"][n][k] for k in ("G1", "G2", "G3", "G4"))]
    out["verdict"] = f"MARKET_TIMED_TOP_COIN_ON_SEEN_DATA ({', '.join(passed)})" if passed else "NO_MARKET_TIMING_EDGE"
    print(f"-> {out['verdict']}")
    out["when"] = time.strftime("%FT%TZ", time.gmtime())
    if smoke:
        print("(smoke run on random numbers: nothing saved, not a result)")
    else:
        OUT.write_text(json.dumps(out, indent=1, default=str))
        print(f"saved {OUT}")


if __name__ == "__main__":
    main()
