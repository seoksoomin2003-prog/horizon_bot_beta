"""What it costs to bring the money from Bithumb to Binance and back. Standard library only: the dashboard imports it.

The route: buy XRP with won on Bithumb, send it to Binance, sell it there for USDT, trade; then buy XRP on Binance,
send it back and sell it for won on Bithumb. The fees were read on CHECKED from the two exchanges' own fee pages.
They change, so check again before relying on them. Left out because they are not fees: the price gap between the
two exchanges (the "kimchi premium") and XRP's own price move while a transfer is on its way.

These costs are paid once per trip of the money, not once per trade, so they are a hurdle on the whole account and
are never added to a single trade's slippage.
"""

from __future__ import annotations

CHECKED = "2026-11-12"
BITHUMB_FEE = 0.0025            # per trade in the won market; 0.0004 only while Bithumb's 30-day "lowest fee" is applied for
BINANCE_SPOT_FEE = 0.001        # XRP <-> USDT on Binance spot
TO_BINANCE_XRP = 0.4            # Bithumb's XRP withdrawal fee
TO_BITHUMB_XRP = 0.2            # Binance's XRP withdrawal fee
XRP_PRICE = 1.05                # USDT on CHECKED; used only when no fresher price is given


def round_trip(capital: float, final: float | None = None, xrp_price: float | None = None, fee: float = BITHUMB_FEE) -> dict:
    """The trip's costs in USDT for `capital` USDT arriving at Binance and `final` (default: the same) leaving it.

    `net` is what comes back to Bithumb minus what was spent there, which equals final - capital - total.
    `break_even` is the Binance balance at which nothing is lost.
    """
    final = capital if final is None else final
    price = xrp_price if xrp_price and xrp_price > 0 else XRP_PRICE
    keep = 1 - BINANCE_SPOT_FEE
    sent = capital / keep + TO_BINANCE_XRP * price          # XRP bought on Bithumb, in USDT
    arriving = max(final * keep - TO_BITHUMB_XRP * price, 0.0)  # XRP reaching Bithumb, in USDT
    spent, back = sent * (1 + fee), arriving * (1 - fee)
    break_even = (spent / (1 - fee) + TO_BITHUMB_XRP * price) / keep
    costs = {"buy_fee": sent * fee, "to_binance": TO_BINANCE_XRP * price,
             "binance_fee": capital / keep - capital + max(final, 0.0) * BINANCE_SPOT_FEE,
             "to_bithumb": min(TO_BITHUMB_XRP * price, max(final, 0.0) * keep), "sell_fee": arriving * fee}
    return {**costs, "total": sum(costs.values()), "spent": spent, "back": back, "net": back - spent,
            "break_even": break_even, "break_even_return": break_even / capital - 1, "bithumb_fee": fee,
            "xrp_price": price, "to_binance_xrp": TO_BINANCE_XRP, "to_bithumb_xrp": TO_BITHUMB_XRP, "checked": CHECKED}
