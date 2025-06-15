class WebIQError(Exception):
    """Base exception class for WebIQ-related errors."""
    pass

class AutomationError(WebIQError):
    """Raised when automation operations fail."""
    pass

class ConfigurationError(WebIQError):
    """Raised when configuration is invalid or missing."""
    pass

class GoalNotAchievableError(WebIQError):
    """Raised when the goal cannot be achieved on the current page/context."""
    pass

class GoalBlockedError(WebIQError):
    """Raised when progress toward the goal is blocked by an obstacle (e.g., CAPTCHA, missing info)."""
    pass

class GoalNotCompletedError(WebIQError):
    """Raised when the automation attempts to complete the goal but verification fails."""
    pass

if __name__ == '__main__':
    # Example usage (for testing purposes)
    try:
        raise GoalNotAchievableError("The signup form is not present on this page.")
    except GoalNotAchievableError as e:
        print(f"Caught expected exception: {e}")

    try:
        raise GoalBlockedError("CAPTCHA detected, cannot proceed without solving.")
    except GoalBlockedError as e:
        print(f"Caught expected exception: {e}")

    try:
        raise GoalNotCompletedError("Account creation failed. Success message not found.")
    except GoalNotCompletedError as e:
        print(f"Caught expected exception: {e}")
