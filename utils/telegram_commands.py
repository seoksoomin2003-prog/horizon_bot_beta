import time
import threading
import queue
import httpx
from datetime import datetime
import config
from utils.runtime_coordinator import RuntimeCoordinator
from logger import errors_logger, trades_logger

MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds
MAX_BACKOFF = 300.0    # 5 minutes
RATE_LIMIT_DELAY = 1.1 # 1 second + buffer per message to same chat

def _telegram_base_url() -> str:
    return f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"

class TelegramMessage:
    def __init__(self, text: str, priority: int = 1):
        self.text = text
        self.priority = priority  # 0 = High (Emergency), 1 = Normal
        self.timestamp = time.time()
        self.retries = 0

    def __lt__(self, other):
        # PriorityQueue sorts by lowest first
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

class TelegramManager:
    def __init__(self):
        self.msg_queue = queue.PriorityQueue(maxsize=100)
        self.worker_thread = None
        self.listener_thread = None
        self.stop_event = threading.Event()
        self._last_send_time = 0
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        return bool(
            config.TELEGRAM_ENABLED
            and config.TELEGRAM_BOT_TOKEN
            and config.TELEGRAM_CHAT_ID
        )
        
    def start(self, context=None):
        if not self.is_active:
            trades_logger.warning("Telegram disabled. TELEGRAM SAFE MODE is active.")
            return

        # 1. Start Sender Thread (Consumer)
        self.worker_thread = threading.Thread(
            target=self._sender_loop, 
            daemon=True, 
            name="TelegramSenderTask"
        )
        self.worker_thread.start()

        # 2. Start Receiver Thread (Listener)
        self.listener_thread = threading.Thread(
            target=self._polling_loop, 
            args=(context,), 
            daemon=True,
            name="TelegramListenerTask"
        )
        self.listener_thread.start()
        
        trades_logger.info("Telegram Reliability System Initialized (Non-blocking).")

    def send(self, text: str, priority: int = 1):
        """
        Public non-blocking API to queue a message with backpressure and dropping policy.
        """
        if not self.is_active:
            return
            
        msg = TelegramMessage(text, priority)
        with self._lock:
            if self.msg_queue.full():
                if priority == 0:
                    # Emergency message: drop a normal priority message to make room!
                    items = []
                    while not self.msg_queue.empty():
                        try:
                            items.append(self.msg_queue.get_nowait())
                        except queue.Empty:
                            break
                            
                    # Sort items by priority desc (lowest priority 1 first) then timestamp desc (oldest first)
                    items.sort(key=lambda x: (-x.priority, x.timestamp))
                    
                    dropped = False
                    if items and items[0].priority > 0:
                        dropped_msg = items.pop(0)
                        errors_logger.warning(f"Telegram Queue Full! Dropped normal-priority message to accommodate emergency message: {dropped_msg.text[:50]}")
                        dropped = True
                        
                    # Re-queue remaining items non-blockingly
                    for item in items:
                        try:
                            self.msg_queue.put(item, block=False)
                        except queue.Full:
                            errors_logger.warning(f"Telegram queue full while re-queueing. Dropping message: {item.text[:50]}")
                        
                    if not dropped:
                        # If queue is somehow filled entirely with emergency messages, drop oldest
                        try:
                            oldest = self.msg_queue.get_nowait()
                            errors_logger.warning(f"Telegram Queue full of emergency messages. Dropped oldest emergency: {oldest.text[:50]}")
                        except queue.Empty:
                            pass
                else:
                    # Normal priority and queue is full -> Drop it!
                    errors_logger.warning(f"Telegram Queue Full! Dropped normal-priority message: {text[:50]}")
                    return
            
            try:
                self.msg_queue.put(msg, block=False)
            except queue.Full:
                errors_logger.error("Telegram Queue FULL! Dropping message.")

    def _sender_loop(self):
        """
        Background worker that processes the message queue with retries.
        """
        while not self.stop_event.is_set():
            try:
                # Wait for a message
                msg = self.msg_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            success = self._attempt_send(msg)
            
            if success:
                self.msg_queue.task_done()
            else:
                # Handle non-blocking retry logic using threading.Timer to achieve retry isolation
                if msg.retries < MAX_RETRIES:
                    msg.retries += 1
                    backoff = min(INITIAL_BACKOFF * (2 ** (msg.retries - 1)), MAX_BACKOFF)
                    errors_logger.warning(f"Retrying Telegram message in {backoff}s (Attempt {msg.retries}/{MAX_RETRIES})")
                    
                    # Re-queue after delay non-blockingly using a Timer!
                    def re_queue():
                        with self._lock:
                            try:
                                self.msg_queue.put(msg, block=False)
                            except queue.Full:
                                errors_logger.error("Telegram Queue full during retry re-queue. Dropping message.")
                    threading.Timer(backoff, re_queue).start()
                    self.msg_queue.task_done()
                else:
                    errors_logger.critical(f"PERMANENT FAILURE: Telegram message dropped after {MAX_RETRIES} retries. Contents: {msg.text[:50]}...")
                    # Dead-letter log
                    self._log_to_dead_letter(msg)
                    self.msg_queue.task_done()

    def _attempt_send(self, msg: TelegramMessage) -> bool:
        """
        Performs the actual HTTP request. Returns True if sent or terminal error.
        """
        # Respect Telegram's 1 msg/sec limit
        elapsed = time.time() - self._last_send_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)

        url = f"{_telegram_base_url()}/sendMessage"
        payload = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": msg.text,
            "parse_mode": "HTML"
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload)
                self._last_send_time = time.time()

                if response.status_code == 200:
                    return True
                
                # Handle Specific Error Codes
                if response.status_code == 429:
                    retry_after = response.json().get("parameters", {}).get("retry_after", 30)
                    errors_logger.warning(f"Telegram Rate Limit (429). Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    return False # Trigger retry

                if response.status_code in [400, 401, 403]:
                    errors_logger.critical(f"Terminal Telegram Error ({response.status_code}): {response.text}")
                    # Terminal errors should not be retried indefinitely
                    return False # We let retry logic handle it, but maybe we should flag self.is_active = False

                errors_logger.error(f"Telegram API Error ({response.status_code}): {response.text}")
                return False

        except Exception as e:
            errors_logger.error(f"Telegram Connectivity Exception: {e}")
            return False

    def _log_to_dead_letter(self, msg: TelegramMessage):
        try:
            from utils.persistence import get_absolute_path
            log_path = get_absolute_path("logs", "failed_notifications.log")
            with open(log_path, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] PRIO={msg.priority} MSG={msg.text}\n")
        except:
            pass

    def _polling_loop(self, context):
        """
        Long-polling listener for commands.
        """
        offset = None
        while not self.stop_event.is_set():
            try:
                url = f"{_telegram_base_url()}/getUpdates"
                params = {"timeout": 30, "offset": offset}
                
                with httpx.Client(timeout=40.0) as client:
                    response = client.get(url, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("ok"):
                            for update in data.get("result", []):
                                offset = update["update_id"] + 1
                                message = update.get("message", {})
                                text = message.get("text")
                                chat_id = message.get("chat", {}).get("id")
                                
                                if text and chat_id:
                                    self._handle_command(text, chat_id, context)
                    elif response.status_code == 401:
                        errors_logger.critical("Invalid Telegram Token. Listener Stopped.")
                        break
                    else:
                        time.sleep(10)
            except Exception as e:
                time.sleep(10)

    def _handle_command(self, text, chat_id, context):
        """
        Logic for parsing commands. Security checked.
        """
        # Security Check
        if str(chat_id) != str(config.TELEGRAM_CHAT_ID):
            return

        cmd = text.lower().strip()
        pos_manager = context.get('pos_manager')
        execution_engine = context.get('execution_engine')
        dh = context.get('dh')

        if cmd == "/ping":
            self.send("[OK] BOT ONLINE", priority=0)

        elif cmd == "/help":
            help_text = (
                "STATUS: <b>COMMANDS</b>\n"
                "/status - Check bot health & balance\n"
                "/positions - View open positions\n"
                "/start - Resume trading\n"
                "/stop - Pause trading\n"
                "/stop_all - Emergency close & Pause\n"
                "/ping - Connectivity test"
            )
            self.send(help_text)

        elif cmd == "/status":
            active_pos_count = len(pos_manager.active_positions) if pos_manager else 0
            is_running_str = "YES" if RuntimeCoordinator().get_state("is_running") else "NO (PAUSED)"
            status_msg = (
                f"BOT STATUS\n"
                f"Running: {is_running_str}\n"
                f"Positions: {active_pos_count}\n"
                f"Stop Mode: {RuntimeCoordinator().get_state("stop_mode")}"
            )
            self.send(status_msg)

        elif cmd == "/positions":
            if not pos_manager or not pos_manager.active_positions:
                self.send("No open positions.")
                return
            
            pos_details = "OPEN POSITIONS\n"
            for pos in pos_manager.active_positions:
                pos_details += f"{pos.symbol} {pos.side} {pos.amount} @ {pos.entry_price}\n"
            self.send(pos_details)

        elif cmd == "/stop":
            RuntimeCoordinator().enqueue_command("PAUSE")
            self.send("[STOP] Pause command queued. Awaiting main loop confirmation.", priority=0)

        elif cmd == "/stop_all":
            self.send("[WARN] EMERGENCY STOP QUEUED", priority=0)
            RuntimeCoordinator().enqueue_command("STOP_ALL")

        elif cmd == "/start":
            RuntimeCoordinator().enqueue_command("RESUME")
            self.send("[START] Resume command queued.", priority=0)

# Global Instance
_manager = TelegramManager()

def send_message(text: str, priority: int = 1):
    """
    Standard entry point (Global wrapper).
    Harden text to ASCII for Windows safety.
    """
    try:
        # Final safety check for Windows cp1252 compatibility
        text = str(text).encode("ascii", "ignore").decode()
    except:
        pass
    _manager.send(text, priority)

def start_telegram_listener(pos_manager=None, risk_engine=None, execution_engine=None, dh=None):
    """
    Initializes the reliability manager and starts threads.
    """
    context = {
        'pos_manager': pos_manager,
        'risk_engine': risk_engine,
        'execution_engine': execution_engine,
        'dh': dh
    }
    _manager.start(context)
