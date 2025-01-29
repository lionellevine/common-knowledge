# constants.py
import enum

class AgentType(enum.Enum):
    """Enumerates the possible agent types."""
    CLI = "cli"
    RANDOM = "random"
    GPT = "gpt"
    API = "api"

# Prefixes or key phrases used for detecting actions
KILL_PREFIX = "Kill "
SEARCH_PREFIX = "Search "
GO_TO_PREFIX = "Go to "
UNLOCK_DOOR_ACTION = "Unlock the door to escape and win the game!"
ESCAPE_DOOR_ACTION = "The door is unlocked! Escape and win the game."
