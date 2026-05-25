import json
import os
import time
import threading
from typing import Dict, Any
from utils.persistence import atomic_write_json
import logging

class ExecutionJournal:
    LOG_PATH = "data/execution_journal.log"
    SNAPSHOT_PATH = "data/lifecycle_state.json"
    _lock = threading.RLock()

    @classmethod
    def _ensure_dirs(cls):
        os.makedirs(os.path.dirname(cls.LOG_PATH), exist_ok=True)

    @classmethod
    def log_event(cls, symbol: str, event: str, metadata: Dict[str, Any] = None):
        """Appends to the execution log and updates the snapshot. Fully serialized."""
        with cls._lock:
            cls._ensure_dirs()
            metadata = metadata or {}
            entry = {
                "timestamp": time.time(),
                "symbol": symbol,
                "event": event,
                "metadata": metadata
            }
            
            # 1. Append-only log
            try:
                with open(cls.LOG_PATH, 'a') as f:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass # Allow best-effort fsync if not supported on platform
            except Exception as e:
                logging.getLogger('errors_logger').critical(f"FATAL: Execution Journal append failed: {e}")
                raise e

            # 2. Update O(1) Snapshot
            cls._update_snapshot(symbol, event)

    @classmethod
    def _update_snapshot(cls, symbol: str, event: str):
        """Updates the snapshot cache based on the logged event."""
        with cls._lock:
            snapshot = cls.get_all_snapshots()
            
            if symbol not in snapshot:
                snapshot[symbol] = {
                    "tp1_done": False,
                    "tp2_done": False,
                    "tp1_pending": False,
                    "tp2_pending": False,
                    "dust_quarantined": False,
                    "last_event": ""
                }
                
            state = snapshot[symbol]
            state["last_event"] = event
            
            if event == "TP1_TRIGGERED":
                state["tp1_pending"] = True
            elif event == "TP1_EXECUTED":
                state["tp1_done"] = True
                state["tp1_pending"] = False
            elif event == "TP1_PARTIALLY_FILLED":
                state["tp1_pending"] = True
                state["tp1_done"] = False
            elif event == "TP2_TRIGGERED":
                state["tp2_pending"] = True
            elif event == "TP2_EXECUTED":
                state["tp2_done"] = True
                state["tp2_pending"] = False
            elif event == "TP2_PARTIALLY_FILLED":
                state["tp2_pending"] = True
                state["tp2_done"] = False
            elif event == "DUST_QUARANTINED":
                state["dust_quarantined"] = True
            elif event == "POSITION_CLOSED" or event == "EMERGENCY_CLOSE":
                # Reset lifecycle on full close
                snapshot.pop(symbol, None)

            try:
                atomic_write_json(cls.SNAPSHOT_PATH, snapshot)
            except Exception as e:
                logging.getLogger('errors_logger').critical(f"Failed to atomically write snapshot: {e}")
                raise e

    @classmethod
    def _replay_journal(cls, write_to_disk: bool = True) -> Dict[str, Dict[str, Any]]:
        """Deterministically rebuilds the snapshot from the append-only log."""
        with cls._lock:
            cls._ensure_dirs()
            snapshot = {}
            if not os.path.exists(cls.LOG_PATH):
                return snapshot
                
            logging.getLogger('trades_logger').warning("Snapshot missing or corrupt. Replaying ExecutionJournal from log...")
            try:
                with open(cls.LOG_PATH, 'r') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            entry = json.loads(line)
                            symbol = entry.get("symbol")
                            event = entry.get("event")
                            if not symbol or not event: continue
                            
                            if symbol not in snapshot:
                                snapshot[symbol] = {
                                    "tp1_done": False,
                                    "tp2_done": False,
                                    "tp1_pending": False,
                                    "tp2_pending": False,
                                    "dust_quarantined": False,
                                    "last_event": ""
                                }
                                
                            state = snapshot[symbol]
                            state["last_event"] = event
                            
                            if event == "TP1_TRIGGERED":
                                state["tp1_pending"] = True
                            elif event == "TP1_EXECUTED":
                                state["tp1_done"] = True
                                state["tp1_pending"] = False
                            elif event == "TP1_PARTIALLY_FILLED":
                                state["tp1_pending"] = True
                                state["tp1_done"] = False
                            elif event == "TP2_TRIGGERED":
                                state["tp2_pending"] = True
                            elif event == "TP2_EXECUTED":
                                state["tp2_done"] = True
                                state["tp2_pending"] = False
                            elif event == "TP2_PARTIALLY_FILLED":
                                state["tp2_pending"] = True
                                state["tp2_done"] = False
                            elif event == "DUST_QUARANTINED":
                                state["dust_quarantined"] = True
                            elif event == "POSITION_CLOSED" or event == "EMERGENCY_CLOSE":
                                snapshot.pop(symbol, None)
                        except json.JSONDecodeError:
                            continue # Skip malformed log lines safely
                            
                # Atomically write the rebuilt snapshot
                if write_to_disk:
                    atomic_write_json(cls.SNAPSHOT_PATH, snapshot)
                    logging.getLogger('trades_logger').info("ExecutionJournal replay completed successfully.")
                return snapshot
                
            except Exception as e:
                logging.getLogger('errors_logger').critical(f"Journal replay failed: {e}")
                return snapshot

    @classmethod
    def get_all_snapshots(cls) -> Dict[str, Dict[str, Any]]:
        """Retrieves all snapshot cached lifecycles, replaying journal on error/missing."""
        with cls._lock:
            cls._ensure_dirs()
            if not os.path.exists(cls.SNAPSHOT_PATH):
                return cls._replay_journal()
                
            try:
                with open(cls.SNAPSHOT_PATH, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        raise ValueError("Snapshot is not a dictionary")
                    # Ensure all values possess correct field structure
                    for symbol, state in data.items():
                        if not isinstance(state, dict):
                            raise ValueError(f"State for {symbol} is not a dict")
                        state.setdefault("tp1_done", False)
                        state.setdefault("tp2_done", False)
                        state.setdefault("tp1_pending", False)
                        state.setdefault("tp2_pending", False)
                        state.setdefault("dust_quarantined", False)
                        state.setdefault("last_event", "")
                    return data
            except (json.JSONDecodeError, ValueError) as e:
                logging.getLogger('errors_logger').warning(f"Snapshot JSON corruption detected: {e}. Initiating replay.")
                return cls._replay_journal()
            except Exception as e:
                logging.getLogger('errors_logger').warning(f"Snapshot read error: {e}. Initiating replay.")
                return cls._replay_journal()

    @classmethod
    def get_snapshot(cls, symbol: str) -> Dict[str, Any]:
        """Gets snapshot state for a single symbol with safe defaults."""
        with cls._lock:
            return cls.get_all_snapshots().get(symbol, {
                "tp1_done": False,
                "tp2_done": False,
                "tp1_pending": False,
                "tp2_pending": False,
                "dust_quarantined": False,
                "last_event": ""
            })

    @classmethod
    def verify_and_align_snapshot(cls):
        """Verifies that the snapshot cache matches the replayed journal. If not, journal wins."""
        with cls._lock:
            replayed = cls._replay_journal(write_to_disk=False)
            cached = {}
            if os.path.exists(cls.SNAPSHOT_PATH):
                try:
                    with open(cls.SNAPSHOT_PATH, 'r') as f:
                        cached = json.load(f)
                except Exception:
                    cached = {}
            
            # Deep comparison of state flags
            diverged = False
            if set(replayed.keys()) != set(cached.keys()):
                diverged = True
            else:
                for sym, r_state in replayed.items():
                    c_state = cached.get(sym, {})
                    for k in ["tp1_done", "tp2_done", "tp1_pending", "tp2_pending", "dust_quarantined"]:
                        if r_state.get(k) != c_state.get(k):
                            diverged = True
                            break
                    if diverged:
                        break
            
            if diverged:
                logging.getLogger('trades_logger').warning(
                    "[LIFECYCLE] Snapshot cache diverged from ExecutionJournal. Aligning snapshot cache to journal truth."
                )
                try:
                    atomic_write_json(cls.SNAPSHOT_PATH, replayed)
                except Exception as e:
                    logging.getLogger('errors_logger').critical(f"Failed to atomically write aligned snapshot: {e}")

