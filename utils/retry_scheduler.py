import time
import logging
import threading
import uuid
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from logger import trades_logger, errors_logger

class TaskAbandonRegistry:
    _lock = threading.Lock()
    _abandoned_leases = set()

    @classmethod
    def abandon(cls, lease_id: str):
        with cls._lock:
            cls._abandoned_leases.add(lease_id)

    @classmethod
    def is_abandoned(cls, lease_id: str) -> bool:
        with cls._lock:
            return lease_id in cls._abandoned_leases

class TaskLease:
    _local = threading.local()

    @classmethod
    def set_lease(cls, lease_id: str):
        cls._local.lease_id = lease_id

    @classmethod
    def get_lease(cls) -> Optional[str]:
        return getattr(cls._local, "lease_id", None)

    @classmethod
    def verify_active(cls):
        lease_id = cls.get_lease()
        if lease_id:
            if TaskAbandonRegistry.is_abandoned(lease_id):
                raise RuntimeError(f"Task lease {lease_id} has been revoked/abandoned. Aborting execution.")
            TaskHeartbeat.record_heartbeat(lease_id)

class TaskHeartbeat:
    _lock = threading.Lock()
    _heartbeats = {}

    @classmethod
    def record_heartbeat(cls, lease_id: str):
        with cls._lock:
            cls._heartbeats[lease_id] = time.time()

    @classmethod
    def get_last_heartbeat(cls, lease_id: str) -> Optional[float]:
        with cls._lock:
            return cls._heartbeats.get(lease_id)

class WorkerGenerationRegistry:
    _lock = threading.Lock()
    _active_generations = {}  # (symbol, worker_type) -> generation_id
    _local = threading.local()

    @classmethod
    def set_active_generation(cls, symbol: str, worker_type: str, gen_id: str):
        with cls._lock:
            cls._active_generations[(symbol, worker_type)] = gen_id

    @classmethod
    def set_thread_generation(cls, symbol: str, worker_type: str, gen_id: str):
        cls._local.symbol = symbol
        cls._local.worker_type = worker_type
        cls._local.generation_id = gen_id

    @classmethod
    def verify_active(cls):
        symbol = getattr(cls._local, "symbol", None)
        worker_type = getattr(cls._local, "worker_type", None)
        gen_id = getattr(cls._local, "generation_id", None)
        if symbol and worker_type and gen_id:
            with cls._lock:
                active_id = cls._active_generations.get((symbol, worker_type))
                if active_id is not None and active_id != gen_id:
                    raise RuntimeError(f"Worker generation {gen_id} stale/revoked for {symbol} ({worker_type}). Aborting.")

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

    def schedule_retry(self, task_fn, name: str, retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0, failure_callback=None, timeout: float = 5.0):
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
            success = False
            try:
                delay = initial_delay
                for attempt in range(1, retries + 1):
                    # Sleep before attempt (if not the first one, or wait for next execution)
                    if attempt > 1:
                        time.sleep(delay)
                        delay *= backoff_factor
                    
                    try:
                        trades_logger.info(f"[RETRY_SCHEDULER] Running retry task '{name}' (Attempt {attempt}/{retries})...")
                        # Execute task_fn under strict timeout isolation using a daemon thread
                        import queue
                        res_queue = queue.Queue()
                        lease_id = uuid.uuid4().hex
                        
                        def thread_wrapper(l_id):
                            try:
                                TaskLease.set_lease(l_id)
                                TaskLease.verify_active()
                                r = task_fn()
                                TaskLease.verify_active()
                                res_queue.put((True, r))
                            except Exception as ex:
                                res_queue.put((False, ex))
                                
                        thread = threading.Thread(target=thread_wrapper, args=(lease_id,), name=f"Task-{name}-Attempt-{attempt}", daemon=True)
                        thread.start()
                        
                        try:
                            success_flag, val = res_queue.get(timeout=timeout)
                            if success_flag:
                                res = val
                            else:
                                raise val
                        except queue.Empty:
                            TaskAbandonRegistry.abandon(lease_id)
                            errors_logger.error(f"[RETRY_SCHEDULER] Task '{name}' attempt {attempt} timed out after {timeout}s.")
                            raise TimeoutError(f"Task execution timed out after {timeout}s")
                        
                        if res:
                            trades_logger.info(f"[RETRY_SCHEDULER] Retry task '{name}' succeeded on attempt {attempt}.")
                            success = True
                            return True
                    except Exception as e:
                        errors_logger.warning(f"[RETRY_SCHEDULER] Attempt {attempt} for task '{name}' failed: {e}")
                
                errors_logger.critical(f"[RETRY_SCHEDULER] Task '{name}' FAILED after maximum of {retries} retry attempts.")
                if failure_callback:
                    try:
                        failure_callback()
                    except Exception as cb_err:
                        errors_logger.critical(f"[RETRY_SCHEDULER] Failure callback for '{name}' failed: {cb_err}")
                return False
            finally:
                with self.task_lock:
                    self.pending_tasks -= 1

        # Submit task to ThreadPoolExecutor
        self.executor.submit(execution_wrapper)
        return True

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)

