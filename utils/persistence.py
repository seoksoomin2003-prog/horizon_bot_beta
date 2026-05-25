import json
import os
import tempfile
import sys

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
            
        # 2. Atomic replacement
        os.replace(temp_path, filepath)
        
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
