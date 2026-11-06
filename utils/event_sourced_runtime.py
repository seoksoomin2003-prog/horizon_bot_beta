import os
import json
import time
import uuid
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from logger import trades_logger, errors_logger
from utils.schemas import PositionSchema
from utils.persistence import get_absolute_path, atomic_write_json
from utils.degraded_manager import DegradedExchangeManager, ExchangeDegradationState

@dataclass(frozen=True)
class LifecycleEvent:
    sequence: int
    stream_version: int
    timestamp: float
    symbol: str
    event_type: str  # ENTRY_SUBMITTED, ENTRY_FILLED, SL_PLACED, TP_PLACED, TP1_TRIGGERED, TP1_FILLED, SL_MOVED, POSITION_CLOSED, RECONCILE_CORRECTION
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConcurrencyException(Exception):
    """Raised when expected stream_version does not match the current length."""
    pass

def project_position_state(symbol: str, events: List[LifecycleEvent]) -> Optional[PositionSchema]:
    """
    [PURE STATE PROJECTION ENGINE]
    Generates the canonical PositionSchema from a stream of historical immutable events.
    """
    pos: Optional[PositionSchema] = None
    
    for event in sorted(events, key=lambda x: (x.sequence, x.timestamp)):
        etype = event.event_type
        m = event.metadata
        
        if etype == "ENTRY_FILLED":
            pos = PositionSchema(
                symbol=symbol,
                side=m.get("side", "LONG").upper(),
                amount=float(m.get("amount", 0.0)),
                initial_amount=float(m.get("amount", 0.0)),
                entry_price=float(m.get("price", 0.0)),
                status="open",
                stop_loss=0.0,
                take_profit=0.0,
                tp1_done=False,
                tp2_done=False,
                tp1_pending=False,
                tp2_pending=False,
                trailing_active=False,
                breakeven_active=False,
                candles_held=0
            )
            
        elif etype == "ENTRY_PARTIALLY_FILLED":
            pos = PositionSchema(
                symbol=symbol,
                side=m.get("side", "LONG").upper(),
                amount=float(m.get("amount", 0.0)),
                initial_amount=float(m.get("amount", 0.0)),
                entry_price=float(m.get("price", 0.0)),
                status="open",
                stop_loss=0.0,
                take_profit=0.0,
                lifecycle_state="PARTIAL",
                tp1_done=False,
                tp2_done=False,
                tp1_pending=False,
                tp2_pending=False,
                trailing_active=False,
                breakeven_active=False,
                candles_held=0
            )
            setattr(pos, "entry_order_id", m.get("order_id"))
            
        elif pos is not None:
            if etype == "ENTRY_PARTIALLY_FILLED":
                pos.amount = float(m.get("amount", 0.0))
                pos.initial_amount = float(m.get("amount", 0.0))
                pos.lifecycle_state = "PARTIAL"
            if etype == "SL_PLACED":
                pos.stop_loss = float(m.get("price", 0.0))
                pos.sl_plan_id = m.get("order_id")
                
            elif etype == "TP_PLACED":
                pos.take_profit = float(m.get("price", 0.0))
                pos.tp_plan_id = m.get("order_id")
                
            elif etype == "TP1_TRIGGERED":
                pos.tp1_pending = True
                pos.intended_qty = float(m.get("amount", 0.0))
                
            elif etype == "TP1_FILLED":
                pos.tp1_done = True
                pos.tp1_pending = False
                pos.filled_qty = float(m.get("amount", 0.0))
                pos.remaining_qty = max(0.0, pos.intended_qty - pos.filled_qty)
                pos.cumulative_fill_ratio = pos.filled_qty / pos.intended_qty if pos.intended_qty > 0 else 0.0
                pos.amount = max(0.0, pos.initial_amount - pos.filled_qty)
                
            elif etype == "TP2_TRIGGERED":
                pos.tp2_pending = True
                pos.intended_qty = float(m.get("amount", 0.0))
                
            elif etype == "TP2_FILLED":
                pos.tp2_done = True
                pos.tp2_pending = False
                pos.filled_qty = float(m.get("amount", 0.0))
                pos.remaining_qty = max(0.0, pos.intended_qty - pos.filled_qty)
                pos.cumulative_fill_ratio = pos.filled_qty / pos.intended_qty if pos.intended_qty > 0 else 0.0
                pos.amount = max(0.0, pos.amount - pos.filled_qty)
                
            elif etype == "SL_MOVED":
                pos.stop_loss = float(m.get("price", 0.0))
                pos.sl_plan_id = m.get("order_id")
                if pos.tp1_done:
                    pos.trailing_active = True
                    
            elif etype == "POSITION_CLOSED":
                pos.amount = 0.0
                pos.status = "closed"
                pos.tp1_pending = False
                pos.tp2_pending = False
                
            elif etype == "RECONCILE_CORRECTION":
                for k, v in m.items():
                    if hasattr(pos, k):
                        setattr(pos, k, v)
                        
    if pos and pos.status == "closed":
        return None
    return pos

class SymbolRuntimeOwner:
    """
    [SYMBOL SEQUENCER / SINGLE WRITER]
    The absolute and exclusive runtime authority for a single symbol.
    Serializes and source-logs all events in a deterministic sequence.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self._lock = threading.RLock()
        self.event_log_path = get_absolute_path("event_journals", f"{symbol}_events.jsonl")
        self.events: List[LifecycleEvent] = []
        self._load_and_replay_events()
        
    def _load_and_replay_events(self):
        with self._lock:
            self.events = []
            if os.path.exists(self.event_log_path):
                try:
                    expected_seq = 1
                    with open(self.event_log_path, 'r') as f:
                        for line_idx, line in enumerate(f, 1):
                            if not line.strip(): continue
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError as je:
                                raise ValueError(f"Malformed JSON on line {line_idx}: {je}")
                                
                            seq = data.get("sequence")
                            stream_ver = data.get("stream_version", seq)
                            if seq is None:
                                raise ValueError(f"Missing sequence number on line {line_idx}")
                            if seq != expected_seq:
                                raise ValueError(f"Sequence gap detected on line {line_idx}. Expected {expected_seq}, got {seq}")
                                
                            event = LifecycleEvent(
                                sequence=seq,
                                stream_version=stream_ver,
                                timestamp=data["timestamp"],
                                symbol=self.symbol,
                                event_type=data["event_type"],
                                metadata=data["metadata"]
                            )
                            self.events.append(event)
                            expected_seq += 1
                except Exception as e:
                    errors_logger.critical(f"FATAL: Event Journal replay failed for {self.symbol}: {e}. HALTING STARTUP.")
                    import sys
                    sys.exit(1)
                    
    def submit_event(self, event_type: str, metadata: Dict[str, Any] = None, expected_version: int = None) -> LifecycleEvent:
        """
        Appends an event to the append-only event store deterministically under lock.
        Throws ConcurrencyException if expected_version does not match current stream_version.
        """
        with self._lock:
            # Active lease validation
            from utils.retry_scheduler import TaskLease, WorkerGenerationRegistry
            TaskLease.verify_active()
            WorkerGenerationRegistry.verify_active()
            
            current_version = len(self.events)
            if expected_version is not None and expected_version != current_version:
                raise ConcurrencyException(f"OCC Failure on {self.symbol}: Expected {expected_version}, got {current_version}")
                
            metadata = metadata or {}
            next_seq = current_version + 1
            event = LifecycleEvent(
                sequence=next_seq,
                stream_version=next_seq,
                timestamp=time.time(),
                symbol=self.symbol,
                event_type=event_type,
                metadata=metadata
            )
            
            # Tiered fsync Durability (Tier 1 vs Tier 2)
            is_critical = event_type in ("ENTRY_FILLED", "SL_PLACED", "POSITION_CLOSED", "SL_MOVED", "TP1_FILLED", "TP2_FILLED")
            
            # Append-only persistence write
            try:
                os.makedirs(os.path.dirname(self.event_log_path), exist_ok=True)
                entry = {
                    "sequence": event.sequence,
                    "stream_version": event.stream_version,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "metadata": event.metadata
                }
                with open(self.event_log_path, 'a') as f:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    if is_critical:
                        try:
                            os.fsync(f.fileno())
                        except OSError:
                            pass
            except Exception as e:
                errors_logger.critical(f"FATAL: Failed to write event to journal for {self.symbol}: {e}")
                raise e
                
            self.events.append(event)
            
            # Target 3: Compaction on Close
            if event_type == "POSITION_CLOSED":
                self.compact_log_on_close()
                
            return event

    def compact_log_on_close(self):
        """Archives current events to historical file and truncates the active log."""
        with self._lock:
            if self.events:
                archive_path = get_absolute_path("event_journals", f"{self.symbol}_events_archive.jsonl")
                try:
                    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
                    with open(archive_path, 'a') as af:
                        for event in self.events:
                            entry = {
                                "sequence": event.sequence,
                                "stream_version": event.stream_version,
                                "timestamp": event.timestamp,
                                "event_type": event.event_type,
                                "metadata": event.metadata
                            }
                            af.write(json.dumps(entry) + "\n")
                        af.flush()
                except Exception as e:
                    errors_logger.error(f"Failed to archive closed events for {self.symbol}: {e}")

            try:
                if os.path.exists(self.event_log_path):
                    with open(self.event_log_path, 'w') as f:
                        f.write("") # Truncate active journal
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except OSError:
                            pass
            except Exception as e:
                errors_logger.error(f"Failed to truncate active journal for {self.symbol}: {e}")
                
            self.events = []
            
    def get_state(self) -> Optional[PositionSchema]:
        """
        Projects and returns the active position schema dynamically.
        """
        with self._lock:
            return project_position_state(self.symbol, self.events)

class OwnershipRegistry:
    """
    [ACTIVE SYMBOL OWNERSHIP REGISTRY]
    Ensures exactly one SymbolRuntimeOwner exists per symbol.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OwnershipRegistry, cls).__new__(cls)
                cls._instance.owners: Dict[str, SymbolRuntimeOwner] = {}
            return cls._instance
            
    def get_owner(self, symbol: str) -> SymbolRuntimeOwner:
        with self._lock:
            import utils.state_manager as sm
            expected_log_path = os.path.join(os.path.dirname(sm.POSITIONS_FILE), "event_journals", f"{symbol}_events.jsonl")
            if symbol not in self.owners or self.owners[symbol].event_log_path != expected_log_path:
                self.owners[symbol] = SymbolRuntimeOwner(symbol)
            return self.owners[symbol]

class ProtectionOrchestrator:
    """
    [UNIFIED PROTECTION ORCHESTRATOR]
    The sole authority in the system allowed to place, migrate, or cancel Stop-Loss and Take-Profit orders.
    Watchdogs, exit staging loops, and reconcilers must route all requests through this orchestrator.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProtectionOrchestrator, cls).__new__(cls)
            return cls._instance
            
    def place_initial_protection(self, symbol: str, side: str, amount: float, stop_price: float, take_profit_price: float, execution_engine) -> Tuple[Optional[str], Optional[str]]:
        """
        Unified placement of initial SL/TP orders on entry.
        """
        trades_logger.info(f"[PROTECTION_ORCHESTRATOR] Placing initial SL/TP for {symbol}...")
        sl_order = execution_engine.place_stop_loss(symbol, side, amount, stop_price)
        sl_id = sl_order.get("id") if sl_order else None
        
        tp_id = None
        if take_profit_price > 0:
            tp_order = execution_engine.place_take_profit(symbol, side, amount, take_profit_price)
            tp_id = tp_order.get("id") if tp_order else None
            
        owner = OwnershipRegistry().get_owner(symbol)
        if sl_id:
            owner.submit_event("SL_PLACED", {"price": stop_price, "order_id": sl_id})
        if tp_id:
            owner.submit_event("TP_PLACED", {"price": take_profit_price, "order_id": tp_id})
            
        return sl_id, tp_id
        
    def migrate_stop_loss(self, symbol: str, pos: PositionSchema, new_stop_price: float, execution_engine) -> bool:
        """
        Unified trailing/breakeven stop loss migration path.
        """
        trades_logger.info(f"[PROTECTION_ORCHESTRATOR] Migrating SL for {symbol} to {new_stop_price}...")
        if execution_engine.atomic_stop_update(symbol, pos, new_stop_price):
            owner = OwnershipRegistry().get_owner(symbol)
            owner.submit_event("SL_MOVED", {"price": new_stop_price, "order_id": pos.sl_plan_id})
            return True
        return False
        
    def execute_emergency_close(self, symbol: str, pos: PositionSchema, execution_engine) -> bool:
        """
        Unified emergency full close / failsafe liquidation.
        """
        trades_logger.critical(f"[PROTECTION_ORCHESTRATOR] Triggering emergency close for {symbol}!")
        if execution_engine.close_full(symbol, pos):
            owner = OwnershipRegistry().get_owner(symbol)
            owner.submit_event("POSITION_CLOSED", {"reason": "EMERGENCY_LIQUIDATION"})
            return True
        return False
