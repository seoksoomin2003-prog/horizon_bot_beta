"""What it costs to bring the money from Bithumb to Binance and back. Standard library only: the dashboard imports it.

The route: buy USDT with won on Bithumb, send it on the Tron network, trade on Binance, send it back on Tron, sell it
for won on Bithumb. The fees were read on CHECKED from the two exchanges' own fee pages. They change, so check again
before relying on them. The price gap between the two exchanges (the "kimchi premium") is not a fee and is left out.

These costs are paid once per trip of the money, not once per trade, so they are a hurdle on the whole account and
are never added to a single trade's slippage.
"""

from __future__ import annotations

CHECKED = "2026-11-12"
BITHUMB_FEE = 0.0025            # per trade in the won market; 0.0004 only while Bithumb's 30-day "lowest fee" is applied for
TO_BINANCE_USDT = 0.0           # Bithumb's USDT withdrawal fee on Tron
TO_BITHUMB_USDT = 2.0           # Binance's USDT withdrawal fee on Tron (it refuses withdrawals under 5 USDT)


def round_trip(capital: float, final: float | None = None, fee: float = BITHUMB_FEE) -> dict:
    """The trip's costs in USDT for `capital` arriving at Binance and `final` (default: the same) leaving it.

    `net` is what comes back to Bithumb minus what was spent there. `break_even` is the Binance balance at which
    the two are equal.
    """
    final = capital if final is None else final
    spent = (capital + TO_BINANCE_USDT) * (1 + fee)
    arriving = max(final - TO_BITHUMB_USDT, 0.0)
    back = arriving * (1 - fee)
    break_even = spent / (1 - fee) + TO_BITHUMB_USDT
    costs = {"buy_fee": (capital + TO_BINANCE_USDT) * fee, "to_binance": TO_BINANCE_USDT,
             "to_bithumb": min(TO_BITHUMB_USDT, max(final, 0.0)), "sell_fee": arriving * fee}
    return {**costs, "total": sum(costs.values()), "spent": spent, "back": back, "net": back - spent,
            "break_even": break_even, "break_even_return": break_even / capital - 1, "bithumb_fee": fee, "checked": CHECKED}
