import os
import json
import threading
from enum import Enum
from logger import trades_logger, errors_logger
from utils.persistence import atomic_write_json
from utils.telegram_commands import send_message

class ExchangeDegradationState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    READONLY = "READONLY"
    PROTECTION_ONLY = "PROTECTION_ONLY"
    CRITICAL_PROTECTION_FAILURE = "CRITICAL_PROTECTION_FAILURE"

class DegradedExchangeManager:
    """
    [C-04] Degraded Exchange Mode Manager.
    Persists and manages exchange outage/degradation states, implementing strict failsafe logic.
    """
    _instance = None
    _lock = threading.Lock()
    STATE_FILE = "data/degradation_state.json"

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DegradedExchangeManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        with self._lock:
            if self._initialized:
                return
            self.state = ExchangeDegradationState.HEALTHY
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.failure_lock = threading.Lock()
            self._load_state()
            self._initialized = True

    def _load_state(self):
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, 'r') as f:
                    data = json.load(f)
                    saved_state = data.get("state")
                    if saved_state in ExchangeDegradationState.__members__:
                        self.state = ExchangeDegradationState(saved_state)
                        trades_logger.info(f"[DEGRADED_MANAGER] Loaded persisted degradation state: {self.state}")
            except Exception as e:
                errors_logger.error(f"Failed to load degradation state: {e}")

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            atomic_write_json(self.STATE_FILE, {"state": self.state.value})
        except Exception as e:
            errors_logger.error(f"Failed to persist degradation state: {e}")

    def get_state(self) -> ExchangeDegradationState:
        with self._lock:
            return self.state

    def set_state(self, new_state: ExchangeDegradationState, reason: str = ""):
        with self._lock:
            if self.state != new_state:
                old_state = self.state
                self.state = new_state
                self._save_state()
                msg = f"⚠️ [DEGRADED_MANAGER] Exchange state transitioned: {old_state} -> {new_state}. Reason: {reason}"
                errors_logger.critical(msg)
                send_message(f"🚨 <b>EXCHANGE STATUS CHANGE</b>\n{old_state.value} ➡️ {new_state.value}\nReason: {reason}")

    def report_failure(self, reason: str = ""):
        with self.failure_lock:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            # 3 consecutive failures transitions to DEGRADED
            if self.consecutive_failures >= 3 and self.get_state() == ExchangeDegradationState.HEALTHY:
                self.set_state(ExchangeDegradationState.DEGRADED, f"3 consecutive failures. Last: {reason}")
            elif self.consecutive_failures >= 6 and self.get_state() == ExchangeDegradationState.DEGRADED:
                self.set_state(ExchangeDegradationState.CRITICAL, f"6 consecutive failures. Exchange is likely down.")

    def report_success(self):
        with self.failure_lock:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            
            # 3 consecutive successes recovers the state
            if self.consecutive_successes >= 3 and self.get_state() != ExchangeDegradationState.HEALTHY:
                self.set_state(ExchangeDegradationState.HEALTHY, "Exchange recovered, 3 consecutive successes.")

    def is_entry_allowed(self) -> bool:
        """New entries are frozen when degraded/critical/protection-only."""
        current = self.get_state()
        return current == ExchangeDegradationState.HEALTHY
