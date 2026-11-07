"""Turns a replay ledger into per-position results. Partial closes count as one position, not several wins."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def positions_from_fills(fills: list[dict]) -> list[dict]:
    """Entry to flat, per symbol, net of fees. Positions still open at the end are left out."""
    open_by_symbol: dict = {}
    done = []
    for t in sorted(fills, key=lambda x: x["timestamp"]):
        sym, fee = t["symbol"], t["fee"]["cost"]
        pos = open_by_symbol.get(sym)
        if pos is None:
            open_by_symbol[sym] = {"symbol": sym, "side": "LONG" if t["side"] == "buy" else "SHORT", "opened_ms": t["timestamp"],
                                   "entry": t["price"], "left": t["amount"], "pnl": -fee, "fees": fee, "fills": 1}
            continue
        pos["pnl"] += t.get("realized", 0.0) - fee
        pos["fees"] += fee
        pos["fills"] += 1
        pos["left"] -= t["amount"]
        if pos["left"] <= 1e-9:
            pos["closed_ms"] = t["timestamp"]
            pos["last_reason"] = t.get("reason")
            done.append(pos)
            del open_by_symbol[sym]
    return done


def mean_interval(pnls: list[float], rounds: int = 2000, seed: int = 7) -> tuple:
    """Bootstrap 95% interval of the average position result. Ignores clustering, so the true width is wider."""
    if len(pnls) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(pnls, k=len(pnls))) / len(pnls) for _ in range(rounds))
    return means[int(rounds * 0.025)], means[int(rounds * 0.975)]


def analyze(summary: dict, fills: list[dict]) -> dict:
    positions = positions_from_fills(fills)
    pnls = [p["pnl"] for p in positions]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    peak, worst = summary["initial"], 0.0
    for row in summary.get("curve", []):
        peak = max(peak, row[1])
        worst = min(worst, row[1] / peak - 1)
    days = (summary["reached_ms"] - summary["start_ms"]) / 86_400_000
    monthly: dict = {}
    for p in positions:
        key = _month(p["closed_ms"])
        monthly[key] = monthly.get(key, 0.0) + p["pnl"]
    by_symbol: dict = {}
    for p in positions:
        by_symbol[p["symbol"]] = by_symbol.get(p["symbol"], 0.0) + p["pnl"]
    low, high = mean_interval(pnls)
    return {
        "days": days, "positions": len(positions), "fills": len(fills), "per_day": len(positions) / days if days else 0.0,
        "final_cash": summary["cash"], "return": summary["cash"] / summary["initial"] - 1,
        "gross": summary["realized"], "fees": summary["fees"], "funding": summary["funding"], "max_drawdown": worst,
        "win_rate": len(wins) / len(pnls) if pnls else None, "avg_win": sum(wins) / len(wins) if wins else None,
        "avg_loss": sum(losses) / len(losses) if losses else None,
        "profit_factor": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None,
        "avg_pnl": sum(pnls) / len(pnls) if pnls else None, "avg_pnl_low": low, "avg_pnl_high": high,
        "monthly": monthly, "by_symbol": dict(sorted(by_symbol.items(), key=lambda kv: kv[1])),
    }


def _month(ms: float) -> str:
    import time
    return time.strftime("%Y-%m", time.gmtime(ms / 1000))


def _num(x, spec=".2f") -> str:
    return "없음" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def render(summary: dict, a: dict) -> str:
    lines = [
        f"재생 결과: {summary['name']}" + ("" if summary["done"] else " (진행 중)"),
        f"기간: {_month(summary['start_ms'])} ~ {_month(summary['reached_ms'])} ({a['days']:.0f}일), 비용 가정: 수수료 {summary['fee_rate']:.2%}, 슬리피지 {summary['slippage_bps']:.0f}bp",
        "",
        f"시작 {summary['initial']:,.2f} → 현금 {a['final_cash']:,.2f} USDT ({a['return']:+.2%}), 최대 낙폭 {a['max_drawdown']:.2%}",
        f"비용 전 손익 {a['gross']:+,.2f} / 수수료 -{a['fees']:,.2f} / 자금조달비 {-a['funding']:+,.2f}",
        f"포지션 {a['positions']}건(하루 {a['per_day']:.1f}건), 승률 {_num(a['win_rate'], '.1%')}, 손익비 {_num(a['profit_factor'])}",
        f"평균 이익 {_num(a['avg_win'])} / 평균 손실 {_num(a['avg_loss'])} / 포지션당 평균 {_num(a['avg_pnl'], '.3f')} USDT "
        f"(95% 구간 {_num(a['avg_pnl_low'], '.3f')} ~ {_num(a['avg_pnl_high'], '.3f')})",
        f"손실 3연속 후 진입이 막힌 시간: 전체의 {summary['cooldown_minutes'] / max(1, a['days'] * 1440):.1%}",
        f"미청산 포지션: {len(summary['open_positions'])}건, 봇 오류 기록: {sum(summary['errors'].values())}건",
        "",
        "월별 손익: " + ", ".join(f"{k} {v:+.1f}" for k, v in sorted(a["monthly"].items())),
        "코인별 손익: " + ", ".join(f"{k.split('/')[0]} {v:+.1f}" for k, v in a["by_symbol"].items()),
        "청산 사유(봇 판단): " + ", ".join(f"{k} {v}" for k, v in sorted(summary["exit_reasons"].items(), key=lambda kv: -kv[1])),
        "",
        "빠진 것: 호가 깊이, 15초 단위 판단(재생은 1분 단위), 주문 지연과 거부. 실제 결과는 대체로 더 나쁩니다.",
    ]
    return "\n".join(lines)


def compare_entries(actual: list[dict], replayed: list[dict], window_ms: int = 900_000) -> dict:
    """How well the replay reproduces real entries: same symbol and side, within the window, one-to-one."""
    def entries(fills):
        return [{"symbol": p["symbol"], "side": p["side"], "opened_ms": p["opened_ms"]} for p in _all_positions(fills)]

    real, sim = entries(actual), entries(replayed)
    unused = list(sim)
    matched = []
    for r in real:
        hit = next((s for s in unused if s["symbol"] == r["symbol"] and s["side"] == r["side"]
                    and abs(s["opened_ms"] - r["opened_ms"]) <= window_ms), None)
        if hit:
            unused.remove(hit)
            matched.append((r, hit))
    return {"real": real, "replayed": sim, "matched": len(matched),
            "minutes_apart": [abs(a["opened_ms"] - b["opened_ms"]) / 60_000 for a, b in matched]}


def _all_positions(fills: list[dict]) -> list[dict]:
    held: dict = {}
    out = []
    for t in sorted(fills, key=lambda x: x["timestamp"]):
        pos = held.get(t["symbol"])
        if pos is None:
            pos = {"symbol": t["symbol"], "side": "LONG" if t["side"] == "buy" else "SHORT", "opened_ms": t["timestamp"], "left": t["amount"]}
            held[t["symbol"]] = pos
            out.append(pos)
        else:
            pos["left"] -= t["amount"]
            if pos["left"] <= 1e-9:
                del held[t["symbol"]]
    return out
