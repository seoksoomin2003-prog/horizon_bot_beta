"""Paper run settings chosen on the dashboard. Standard library only: the dashboard imports this without the bot.

The bot reads the file once at start, in paper mode only. Live trading never reads it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

FILE_NAME = "paper_settings.json"
MARKETS = ("futures", "spot")
# Binance taker fees without discounts. Spot has no leverage and no short selling.
FEES = {"futures": 0.0005, "spot": 0.001}
MAX_LEVERAGE = 20
MIN_INITIAL, MAX_INITIAL = 10.0, 1_000_000.0
DEFAULTS = {"market": "futures", "leverage": 3, "initial": 50.0, "min_order": True}
MAINTENANCE_RATE = 0.005    # isolated-margin approximation of Binance's lowest tier


def liquidation_price(side: str, entry: float, leverage: float) -> float:
    if side == "LONG":
        return entry * (1 - 1 / leverage + MAINTENANCE_RATE)
    return entry * (1 + 1 / leverage - MAINTENANCE_RATE)


def path(runtime_dir: os.PathLike | str) -> Path:
    return Path(runtime_dir) / FILE_NAME


def validate(raw: dict) -> dict:
    """Returns clean settings or raises ValueError with a message for the screen."""
    if not isinstance(raw, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")
    market = raw.get("market")
    if market not in MARKETS:
        raise ValueError("시장은 선물 또는 현물이어야 합니다.")
    try:
        leverage = int(raw.get("leverage"))
        initial = float(raw.get("initial"))
    except (TypeError, ValueError):
        raise ValueError("레버리지와 시작 자금은 숫자여야 합니다.") from None
    if isinstance(raw.get("leverage"), bool) or float(raw.get("leverage")) != leverage:
        raise ValueError("레버리지는 정수여야 합니다.")
    if market == "spot":
        leverage = 1
    if not 1 <= leverage <= MAX_LEVERAGE:
        raise ValueError(f"레버리지는 1~{MAX_LEVERAGE}배 사이여야 합니다.")
    if not (MIN_INITIAL <= initial <= MAX_INITIAL) or initial != initial:
        raise ValueError(f"시작 자금은 {MIN_INITIAL:g}~{MAX_INITIAL:,.0f} USDT 사이여야 합니다.")
    min_order = raw.get("min_order")
    if not isinstance(min_order, bool):
        raise ValueError("최소 주문 맞춤은 켜기 또는 끄기여야 합니다.")
    return {"market": market, "leverage": leverage, "initial": round(initial, 2), "min_order": min_order}


def load(runtime_dir: os.PathLike | str) -> dict | None:
    """The saved settings, or None when nothing valid was saved."""
    try:
        return validate(json.loads(path(runtime_dir).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def save(runtime_dir: os.PathLike | str, settings: dict) -> dict:
    clean = validate(settings)
    target = path(runtime_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)
    return clean
