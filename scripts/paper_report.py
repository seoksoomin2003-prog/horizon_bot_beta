#!/usr/bin/env python3
"""Korean summary of the paper account. Reads local files only; opens no network connection."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.persistence import get_absolute_path  # noqa: E402


def summarize(state: dict, trades: list[dict]) -> dict:
    closes = [t for t in trades if t.get("realized")]
    wins = [t for t in closes if t["realized"] - t["fee"]["cost"] > 0]
    cash = peak = state["initial"]
    drawdown = 0.0
    by_symbol: dict = defaultdict(float)
    for t in trades:
        cash = t.get("cash_after", cash)
        peak = max(peak, cash)
        drawdown = min(drawdown, cash / peak - 1)
        by_symbol[t["symbol"]] += t.get("realized", 0.0) - t["fee"]["cost"]
    return {"initial": state["initial"], "cash": state["cash"], "return": state["cash"] / state["initial"] - 1,
            "realized": state["realized"], "fees": state["fees"], "fills": len(trades), "closes": len(closes),
            "win_rate": len(wins) / len(closes) if closes else None, "drawdown": drawdown,
            "positions": state["positions"], "open_orders": len(state["orders"]), "by_symbol": dict(by_symbol),
            "first_ms": trades[0]["timestamp"] if trades else None, "last_ms": trades[-1]["timestamp"] if trades else None}


def render(s: dict) -> str:
    when = lambda ms: time.strftime("%Y-%m-%d %H:%M", time.localtime(ms / 1000)) if ms else "없음"
    lines = ["가상 매매 현황 (실제 돈이 아닙니다)", "",
             f"시작 자금: {s['initial']:,.2f} USDT",
             f"현금 잔고: {s['cash']:,.2f} USDT ({s['return']:+.2%}, 미실현 손익 제외)",
             f"실현 손익: {s['realized']:+,.2f} USDT / 수수료: {s['fees']:,.2f} USDT",
             f"체결 {s['fills']}건, 청산 {s['closes']}건, 승률 " + ("없음" if s["win_rate"] is None else f"{s['win_rate']:.0%}"),
             f"청산 기준 최대 낙폭: {s['drawdown']:.2%}", f"기간: {when(s['first_ms'])} ~ {when(s['last_ms'])}",
             f"대기 중인 손절·익절 주문: {s['open_orders']}건", ""]
    if s["positions"]:
        lines.append("보유 포지션:")
        lines += [f"  {k}: {p['side']} {p['amount']} @ {p['entry']}" for k, p in s["positions"].items()]
    else:
        lines.append("보유 포지션: 없음")
    if s["by_symbol"]:
        lines += ["", "코인별 손익 (수수료 차감):"]
        lines += [f"  {k}: {v:+,.2f} USDT" for k, v in sorted(s["by_symbol"].items(), key=lambda kv: kv[1])]
    lines += ["", "자금조달비용, 강제청산, 호가 깊이는 반영하지 않아 실제 결과는 더 나쁠 수 있습니다.",
              "표본이 적으면 이 수치로 전략의 수익성을 판단할 수 없습니다."]
    return "\n".join(lines)


def main() -> int:
    path = get_absolute_path("paper_broker.json")
    if not os.path.exists(path):
        print("아직 가상 매매 기록이 없습니다.")
        return 0
    with open(path) as f:
        state = json.load(f)
    ledger = os.path.join(os.path.dirname(path), "paper_trades.jsonl")
    trades = []
    if os.path.exists(ledger):
        with open(ledger) as f:
            trades = [json.loads(line) for line in f if line.strip()]
    print(render(summarize(state, trades)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
