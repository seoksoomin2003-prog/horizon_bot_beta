import threading
import queue
from typing import Dict, Any, Optional

class RuntimeCoordinator:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RuntimeCoordinator, cls).__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.state_lock = threading.RLock()
        self.state = {
            "is_running": True,
            "trading_enabled": True,
            "stop_mode": "pause_only",
            "system_state": "NORMAL"
        }
        self.command_queue = queue.Queue()

    def get_state(self, key: str, default: Any = None) -> Any:
        with self.state_lock:
            return self.state.get(key, default)

    def set_state(self, key: str, value: Any):
        with self.state_lock:
            self.state[key] = value

    def update_state(self, updates: Dict[str, Any]):
        with self.state_lock:
            self.state.update(updates)

    def is_running(self) -> bool:
        return self.get_state("is_running", False)

    def enqueue_command(self, cmd: str, args: Optional[Dict[str, Any]] = None):
        """Used by Telegram to send commands to the main runtime."""
        self.command_queue.put({"command": cmd, "args": args or {}})

    def drain_commands(self) -> list:
        """Called by the main loop to process pending commands safely on the main thread."""
        commands = []
        while not self.command_queue.empty():
            try:
                commands.append(self.command_queue.get_nowait())
            except queue.Empty:
                break
        return commands
