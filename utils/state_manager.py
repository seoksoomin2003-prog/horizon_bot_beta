import json
import os
import shutil
from logger import errors_logger

from utils.persistence import get_absolute_path

POSITIONS_FILE = get_absolute_path('positions.json')
BACKUP_FILE = get_absolute_path('positions_backup.json')

def save_positions(positions):
    """
    Save positions list to JSON using an atomic write.
    """
    try:
        # 1. Prepare data directory
        os.makedirs(os.path.dirname(POSITIONS_FILE), exist_ok=True)

        # 2. Institutional Atomic Write
        from utils.persistence import atomic_write_json
        atomic_write_json(POSITIONS_FILE, positions)
        
        # 3. Update Backup
        shutil.copy2(POSITIONS_FILE, BACKUP_FILE)
        
    except Exception as e:
        errors_logger.critical(f"FATAL: Failed to save positions: {e}. HALTING BOT.")
        import config
        from utils.runtime_coordinator import RuntimeCoordinator
        RuntimeCoordinator().set_state("is_running", False)
        raise e

def load_positions():
    """
    Load positions from JSON. Fallback to backup if main file is corrupted or missing.
    """
    for file_path in [POSITIONS_FILE, BACKUP_FILE]:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if verify_integrity(data):
                    return data
                else:
                    errors_logger.warning(f"Integrity check failed for {file_path}")
        except (json.JSONDecodeError, Exception) as e:
            errors_logger.error(f"Error loading positions from {file_path}: {e}")
            
    return []

def verify_integrity(positions):
    """
    Basic schema check for loaded positions list.
    """
    if not isinstance(positions, list):
        return False
    
    # We check for basic existence of symbol and side. 
    # Internal fields might vary between versions, but symbol/side are core.
    for pos in positions:
        if not isinstance(pos, dict):
            return False
        if 'symbol' not in pos or 'side' not in pos:
            return False
            
    return True
