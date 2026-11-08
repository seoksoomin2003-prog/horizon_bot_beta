"""Short positions are read with the right side, and a position with no stop gets emergency recovery."""
import pytest

from utils.exceptions import DataIntegrityError
from utils.exit_engine import ExitAction, ExitEngine
from utils.schemas import PositionSchema
from utils.validator import Validator


def test_short_side_is_read_from_ccxt_and_from_info():
    ccxt_short = {"symbol": "BTC/USDT:USDT", "side": "short", "contracts": 1.5, "entryPrice": 60000.0,
                  "unrealizedPnl": 100.0, "leverage": 10}
    assert Validator.position(ccxt_short).side == "SHORT"
    info_short = {"symbol": "ETH/USDT:USDT", "contracts": 5.0, "info": {"positionSide": "SHORT"}}
    assert Validator.position(info_short).side == "SHORT"


def test_position_without_side_is_rejected():
    with pytest.raises(DataIntegrityError):
        Validator.position({"symbol": "SOL/USDT:USDT", "contracts": 10.0})


def test_position_without_stop_gets_emergency_recovery():
    pos = PositionSchema(symbol="BTC/USDT:USDT", side="SHORT", amount=1.0, entry_price=65000.0, stop_loss=0.0)
    action, _, reason, _ = ExitEngine().compute_exit(pos=pos, current_price=64000.0, current_atr=500.0,
                                                     trend_bias="SHORT", regime="TREND")
    assert action == ExitAction.EMERGENCY_RECOVERY
    assert reason == "INVALID_STATE"
