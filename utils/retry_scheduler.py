import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from logger import trades_logger, errors_logger

class BoundedRetryScheduler:
    """
    [C-03] Institutional Bounded Retry Scheduler.
    Replaces unbounded daemon threads with a thread-safe, bounded execution worker pool,
    preventing thread explosions and implementing backpressure strategies.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BoundedRetryScheduler, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_workers: int = 5, max_queue_size: int = 100):
        with self._lock:
            if self._initialized:
                return
            self.max_workers = max_workers
            self.max_queue_size = max_queue_size
            self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="BoundedRetryWorker")
            self.pending_tasks = 0
            self.task_lock = threading.Lock()
            self._initialized = True
            trades_logger.info(f"Initialized BoundedRetryScheduler with max_workers={max_workers}, max_queue_size={max_queue_size}")

    def schedule_retry(self, task_fn, name: str, retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
        """
        Schedules a retry task to run asynchronously in the worker pool.
        Implements strict backpressure: rejects task if pending queue size limit is reached.
        """
        with self.task_lock:
            if self.pending_tasks >= self.max_queue_size:
                errors_logger.critical(
                    f"[BACKPRESSURE] BoundedRetryScheduler queue full! Rejecting retry task '{name}'. "
                    f"Pending tasks: {self.pending_tasks}"
                )
                raise RuntimeError(f"Retry queue full. Task '{name}' rejected due to backpressure.")
            self.pending_tasks += 1

        def execution_wrapper():
            try:
                delay = initial_delay
                for attempt in range(1, retries + 1):
                    # Sleep before attempt (if not the first one, or wait for next execution)
                    if attempt > 1:
                        time.sleep(delay)
                        delay *= backoff_factor
                    
                    try:
                        trades_logger.info(f"[RETRY_SCHEDULER] Running retry task '{name}' (Attempt {attempt}/{retries})...")
                        success = task_fn()
                        if success:
                            trades_logger.info(f"[RETRY_SCHEDULER] Retry task '{name}' succeeded on attempt {attempt}.")
                            return True
                    except Exception as e:
                        errors_logger.warning(f"[RETRY_SCHEDULER] Attempt {attempt} for task '{name}' failed: {e}")
                
                errors_logger.critical(f"[RETRY_SCHEDULER] Task '{name}' FAILED after maximum of {retries} retry attempts.")
                return False
            finally:
                with self.task_lock:
                    self.pending_tasks -= 1

        # Submit task to ThreadPoolExecutor
        self.executor.submit(execution_wrapper)
        return True

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)
