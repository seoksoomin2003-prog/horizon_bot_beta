class HorizonError(Exception):
    """Base Exception for Horizon Bot."""
    pass

class DataIntegrityError(HorizonError):
    """Raised when data from API is missing, None, or logically invalid."""
    pass

class ExecutionFailure(HorizonError):
    """Raised when an order placement or cancellation fails terminaly."""
    pass

class ZeroDivisionGuard(HorizonError):
    """Raised when a mathematical operation's denominator is logically zero."""
    pass
