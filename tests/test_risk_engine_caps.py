import unittest

from risk_engine import RiskEngine


class TestRiskEngineCaps(unittest.TestCase):
    def test_atr_compression_floor_and_leverage_cap_blocks(self):
        r = RiskEngine()

        # DOGE-like low price / low ATR regime that used to explode sizing.
        # equity=100, risk_per_trade=0.75% => risk_amount ~0.75
        # MIN_STOP_DISTANCE_PCT=0.2% => stop_distance >= 0.0002 on price=0.1
        # size ~ 3750, notional ~ 375 => 3.75x leverage => must reject by 3.0x cap.
        out = r.validate_trade(
            symbol="DOGEUSDT",
            current_equity=100.0,
            active_positions=[],
            atr=0.00001,
            confidence=4.0,
            price=0.10,
            side="LONG",
            stop_loss=0.09999,
            micro_score=1.0,
            used_margin=0.0,
        )
        self.assertFalse(out["approved"])
        self.assertTrue(("Leverage cap breach" in out["reason"]) or ("Notional cap breach" in out["reason"]))


if __name__ == "__main__":
    unittest.main()
