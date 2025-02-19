"""
constants.py
-------------
This module defines global constants and enumerations that are used throughout the Hoodwinked game.
These constants include the various agent types for controlling players and key prefixes for identifying
specific game actions such as killing, searching, and moving.
"""

import enum

class AgentType(enum.Enum):
    """
    Enumerates the possible types of agents that can control a player.
    
    Attributes:
        CLI: Represents a player controlled via a Command-Line Interface (human input).
        RANDOM: Represents a player that makes random decisions.
        GPT: Represents a player whose actions are generated using a GPT-based model.
    """
    CLI = "cli"
    RANDOM = "random"
    GPT = "gpt"

# Prefixes used to identify specific actions within player commands.
# These are used in parsing and determining what type of action a player intends to perform.
KILL_PREFIX = "Kill "      # Indicates a kill action (e.g., "Kill John")
SEARCH_PREFIX = "Search "  # Indicates a search action (e.g., "Search the fridge")
GO_TO_PREFIX = "Go to "    # Indicates a movement action (e.g., "Go to the Kitchen")
