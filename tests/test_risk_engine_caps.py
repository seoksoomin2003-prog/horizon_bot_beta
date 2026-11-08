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
            price=0.10,
            side="LONG",
            stop_loss=0.09999,
            micro_score=4.0,  # undoes the fixed 0.25 risk scale, so the size reaches the cap
            used_margin=0.0,
        )
        self.assertFalse(out["approved"])
        self.assertTrue(("Leverage cap breach" in out["reason"]) or ("Notional cap breach" in out["reason"]))


    def test_conviction_score_does_not_change_size(self):
        from conviction_engine import ConvictionScore
        import config
        from risk_engine import RISK_SCALE
        sizes = []
        for composite in (0.0, 0.5, 1.0):
            score = ConvictionScore(symbol="BTCUSDT", side="LONG", composite_score=composite, signal_score=2, confidence=0.5)
            out = RiskEngine().approve_order(score, current_equity=1000.0, active_positions=[], atr=10.0,
                                             last_price=10000.0, stop_loss_price=9900.0, used_margin=0.0)
            self.assertTrue(out.approved, out.rejection_reason)
            sizes.append(out.amount)
        self.assertEqual(len(set(sizes)), 1)
        stop_distance = 10000.0 + max(10.0 * 0.15, 10000.0 * 0.0015) - 9900.0
        self.assertAlmostEqual(sizes[0], 1000.0 * config.RISK_PER_TRADE * RISK_SCALE / stop_distance, places=6)


if __name__ == "__main__":
    unittest.main()
