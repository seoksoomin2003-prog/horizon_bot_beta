from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from logger import trades_logger, errors_logger
from utils.schemas import PositionSchema


class LifecycleState(str, Enum):
    OPEN = "OPEN"
    TP1_PENDING = "TP1_PENDING"
    TP1_PARTIAL_PENDING = "TP1_PARTIAL_PENDING"
    TP1_PARTIALLY_FILLED = "TP1_PARTIALLY_FILLED"
    TP1_THRESHOLD_REACHED = "TP1_THRESHOLD_REACHED"
    TP1_DONE = "TP1_DONE"
    TP2_PENDING = "TP2_PENDING"
    TP2_PARTIAL_PENDING = "TP2_PARTIAL_PENDING"
    TP2_PARTIALLY_FILLED = "TP2_PARTIALLY_FILLED"
    TP2_THRESHOLD_REACHED = "TP2_THRESHOLD_REACHED"
    TP2_DONE = "TP2_DONE"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class LifecycleValidation:
    ok: bool
    repaired: bool
    state_before: str
    state_after: str
    reason: str = ""


def derive_lifecycle_state(pos: PositionSchema) -> str:
    if not pos:
        return LifecycleState.CLOSED.value
    if (pos.amount or 0.0) <= 0 or str(getattr(pos, "status", "")).lower() == "closed":
        return LifecycleState.CLOSED.value

    # Get quantity details
    intended = getattr(pos, "intended_qty", 0.0) or 0.0
    filled = getattr(pos, "filled_qty", 0.0) or 0.0
    ratio = filled / intended if intended > 0.0 else 0.0

    tp1_pending = bool(getattr(pos, "tp1_pending", False))
    tp1_done = bool(getattr(pos, "tp1_done", False))
    tp2_pending = bool(getattr(pos, "tp2_pending", False))
    tp2_done = bool(getattr(pos, "tp2_done", False))

    if tp2_done:
        return LifecycleState.TP2_DONE.value

    if tp2_pending:
        if ratio >= 0.95:
            return LifecycleState.TP2_THRESHOLD_REACHED.value
        if ratio > 0.0:
            return LifecycleState.TP2_PARTIAL_PENDING.value
        return LifecycleState.TP2_PENDING.value

    if tp1_done:
        if intended > 0.0 and 0.0 < ratio < 0.95:
            return LifecycleState.TP2_PARTIALLY_FILLED.value
        return LifecycleState.TP1_DONE.value

    if tp1_pending:
        if ratio >= 0.95:
            return LifecycleState.TP1_THRESHOLD_REACHED.value
        if ratio > 0.0:
            return LifecycleState.TP1_PARTIAL_PENDING.value
        return LifecycleState.TP1_PENDING.value

    if intended > 0.0 and 0.0 < ratio < 0.95:
        return LifecycleState.TP1_PARTIALLY_FILLED.value

    if intended > 0.0 and ratio >= 0.95:
        return LifecycleState.TP1_THRESHOLD_REACHED.value

    # Fallback to binary flags for simulation/legacy support
    if bool(getattr(pos, "tp2_done", False)):
        return LifecycleState.TP2_DONE.value
    if bool(getattr(pos, "tp1_done", False)):
        return LifecycleState.TP1_DONE.value

    return LifecycleState.OPEN.value


def validate_position_lifecycle(
    pos: PositionSchema,
    *,
    allow_repair: bool = True,
    hard_halt_on_unrecoverable: bool = True,
) -> LifecycleValidation:
    """
    Enforces lifecycle invariants (single source of truth: tp1_done/tp2_done).

    Invalid states:
      - tp2_done=True while tp1_done=False
      - trailing_active=True before TP1
      - breakeven_active=True before TP1
      - CLOSED while trailing/breakeven active
    """
    before = derive_lifecycle_state(pos)
    repaired = False
    reason_parts = []

    if not pos:
        return LifecycleValidation(ok=True, repaired=False, state_before=before, state_after=before, reason="")

    # Repair 1: TP ordering invariant
    if pos.tp2_done and not pos.tp1_done:
        msg = f"[LIFECYCLE_INVALID] {pos.symbol} tp2_done=True while tp1_done=False"
        errors_logger.critical(msg)
        if allow_repair:
            pos.tp1_done = True
            repaired = True
            reason_parts.append("repaired_tp_order")
        else:
            if hard_halt_on_unrecoverable:
                raise RuntimeError(msg)
            return LifecycleValidation(ok=False, repaired=False, state_before=before, state_after=before, reason="tp_order_violation")

    # Repair 2: trailing/breakeven cannot be active before TP1
    if (pos.trailing_active or pos.breakeven_active) and not pos.tp1_done:
        msg = f"[LIFECYCLE_INVALID] {pos.symbol} trailing/breakeven active before TP1"
        errors_logger.critical(msg)
        if allow_repair:
            pos.trailing_active = False
            pos.breakeven_active = False
            repaired = True
            reason_parts.append("repaired_pre_tp1_flags")
        else:
            if hard_halt_on_unrecoverable:
                raise RuntimeError(msg)
            return LifecycleValidation(ok=False, repaired=False, state_before=before, state_after=before, reason="pre_tp1_flag_violation")

    # Repair 3: CLOSED implies trailing/breakeven inactive
    after_derived = derive_lifecycle_state(pos)
    if after_derived == LifecycleState.CLOSED.value and (pos.trailing_active or pos.breakeven_active):
        msg = f"[LIFECYCLE_INVALID] {pos.symbol} CLOSED with trailing/breakeven active"
        errors_logger.critical(msg)
        if allow_repair:
            pos.trailing_active = False
            pos.breakeven_active = False
            repaired = True
            reason_parts.append("repaired_closed_flags")
        else:
            if hard_halt_on_unrecoverable:
                raise RuntimeError(msg)
            return LifecycleValidation(ok=False, repaired=False, state_before=before, state_after=before, reason="closed_flag_violation")

    # Normalize lifecycle_state field to derived truth
    after = derive_lifecycle_state(pos)
    if getattr(pos, "lifecycle_state", None) != after:
        pos.lifecycle_state = after
        if after != before:
            repaired = True
            reason_parts.append("normalized_lifecycle_state")

    ok = True
    reason = ",".join(reason_parts)
    if repaired and reason:
        trades_logger.warning(f"[LIFECYCLE_REPAIRED] {pos.symbol} {before} -> {after} ({reason})")
    return LifecycleValidation(ok=ok, repaired=repaired, state_before=before, state_after=after, reason=reason)


def apply_tp_fill(pos: PositionSchema, tp_level: int) -> Tuple[bool, str]:
    """
    Centralized transition for TP completion.
    Returns: (changed, transition_string)
    """
    if not pos:
        return False, "noop"

    before = derive_lifecycle_state(pos)
    changed = False

    intended = getattr(pos, "intended_qty", 0.0) or 0.0
    filled = getattr(pos, "filled_qty", 0.0) or 0.0

    is_fully_filled = (intended > 0 and filled >= (intended * 0.95)) or (intended == 0.0)

    if tp_level == 1:
        if is_fully_filled:
            if not pos.tp1_done:
                pos.tp1_done = True
                changed = True
        else:
            if pos.tp1_done:
                pos.tp1_done = False
                changed = True
    elif tp_level == 2:
        if is_fully_filled:
            if not pos.tp1_done:
                pos.tp1_done = True
                changed = True
            if not pos.tp2_done:
                pos.tp2_done = True
                changed = True
        else:
            if pos.tp2_done:
                pos.tp2_done = False
                changed = True
    else:
        raise ValueError(f"Invalid tp_level: {tp_level}")

    validate_position_lifecycle(pos, allow_repair=True, hard_halt_on_unrecoverable=True)
    after = derive_lifecycle_state(pos)
    return changed, f"{before}->{after}"

