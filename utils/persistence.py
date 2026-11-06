import json
import os
import tempfile
import sys

def get_absolute_path(*path_segments: str) -> str:
    """
    Returns a canonical absolute path under the centralized project root's 'data' directory.
    Prevents relative path split-brain and cwd dependencies.
    """
    if path_segments and os.path.isabs(path_segments[0]):
        return os.path.abspath(os.path.join(*path_segments))
        
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(utils_dir)
    
    # Isolate all runtime state and persistence files by exchange deployment
    import config
    exchange_name = getattr(config, "EXCHANGE", "binance").lower()
    # Paper state never shares a folder with live state. Tests redirect the whole tree.
    if getattr(config, "IS_PAPER", True):
        exchange_name = f"paper_{exchange_name}"
    base_data_dir = os.path.join(os.environ.get("HORIZON_RUNTIME_DIR") or os.path.join(base_dir, 'runtime'), exchange_name)
    
    cleaned_segments = []
    for seg in path_segments:
        if seg.startswith('runtime/'):
            seg = seg[8:]
        elif seg.startswith('runtime\\'):
            seg = seg[8:]
        elif seg == 'runtime':
            continue
        # Also clean legacy data paths if passed
        if seg.startswith('data/'):
            seg = seg[5:]
        elif seg == 'data':
            continue
        cleaned_segments.append(seg)
        
    abs_path = os.path.abspath(os.path.join(base_data_dir, *cleaned_segments))
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    return abs_path


def atomic_write_json(filepath: str, data: dict):
    """
    Institution-grade atomic JSON write with Best-Effort POSIX directory fsync.
    """
    dirname = os.path.dirname(filepath)
    os.makedirs(dirname, exist_ok=True)
    
    # 1. Write to temporary file in the same directory
    fd, temp_path = tempfile.mkstemp(dir=dirname, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno()) # Tier 1: fsync file contents
            
        # 2. Atomic replacement with defender-safe retries on Windows
        import time
        max_retries = 5
        delay = 0.05
        for attempt in range(max_retries):
            try:
                os.replace(temp_path, filepath)
                break
            except PermissionError as pe:
                if sys.platform == 'win32' and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise pe
        
        # 3. Best-Effort POSIX Directory fsync
        if sys.platform != 'win32':
            try:
                dir_fd = os.open(dirname, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception as e:
                import logging
                logging.getLogger('errors_logger').warning(f"Directory fsync failed for {dirname}: {e}")
                
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise e


class ProcessLock:
    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self.fd = None
        
    def acquire(self) -> bool:
        """
        Atomically acquires an exclusive lock on the lock file.
        Returns True if lock was successfully acquired, False otherwise.
        """
        import os
        import sys
        
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        
        try:
            # 1. Open/create the lock file in read/write mode without truncating.
            self.fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        except Exception:
            return False
            
        # 2. Try to acquire exclusive, non-blocking OS lock
        try:
            if sys.platform != 'win32':
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                # Windows locking requires locking a specific byte range. Lock first 1 byte.
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
        except (BlockingIOError, PermissionError, OSError):
            # Lock is held by another active process!
            # Let's read the PID of the active process if possible
            try:
                os.lseek(self.fd, 0, os.SEEK_SET)
                pid_bytes = os.read(self.fd, 100)
                active_pid = int(pid_bytes.decode().strip())
                import logging
                logging.getLogger('trades_logger').critical(f"Another active process detected (PID {active_pid}) holding the lock.")
            except Exception:
                pass
            os.close(self.fd)
            self.fd = None
            return False
            
        # 3. We hold the lock! Write our PID to the file.
        try:
            os.ftruncate(self.fd, 0)
            os.lseek(self.fd, 0, os.SEEK_SET)
            os.write(self.fd, str(os.getpid()).encode())
            # Sync to disk
            os.fsync(self.fd)
            return True
        except Exception:
            self.release()
            return False
            
    def release(self):
        """
        Releases the lock and deletes the file only if we own it.
        """
        import os
        import sys
        if self.fd is not None:
            own_lock = False
            try:
                # 1. Read PID to ensure ownership before deleting
                try:
                    os.lseek(self.fd, 0, os.SEEK_SET)
                    pid_bytes = os.read(self.fd, 100)
                    pid = int(pid_bytes.decode().strip())
                    if pid == os.getpid():
                        own_lock = True
                except Exception:
                    pass
                
                # 2. Release OS lock
                try:
                    if sys.platform != 'win32':
                        import fcntl
                        fcntl.flock(self.fd, fcntl.LOCK_UN)
                    else:
                        import msvcrt
                        os.lseek(self.fd, 0, os.SEEK_SET)
                        msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            finally:
                # 3. Close handle FIRST to allow deletion on Windows
                try:
                    os.close(self.fd)
                except Exception:
                    pass
                self.fd = None
                
                # 4. Now unlink the lock file safely
                if own_lock:
                    try:
                        os.unlink(self.lock_path)
                    except OSError:
                        pass

