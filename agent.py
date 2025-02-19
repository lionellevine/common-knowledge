"""
agent.py
---------
This module defines the Player class for the Hoodwinked game.
Each Player instance represents a game participant (either a killer or an innocent).
Players can make decisions via different methods (CLI input, random choice, or GPT-based generation).
All actions, votes, and discussion statements are logged for evaluation.
"""

import random
import re
import logging
from typing import List, Dict, Optional

from constants import AgentType, KILL_PREFIX, SEARCH_PREFIX, GO_TO_PREFIX
from gpt_agent import GptAgentMixin

logger = logging.getLogger(__name__)

class Player(GptAgentMixin):
    """
    Represents a single player in the game.

    Attributes:
        name (str): The player's name.
        killer (bool): True if this player is the killer; otherwise innocent.
        preprompt (str): Identifier for which introductory prompt to use.
        alive (bool): True if the player is still active.
        banished (bool): True if the player has been eliminated by vote.
        has_key (bool): (Reserved) Flag for key-based escape functionality.
        agent (str): Specifies how the player’s decisions are generated ("cli", "random", or "gpt").
        model (Optional[str]): If using a GPT-based agent, this holds the model identifier.
        gpt: Instance of a GPT wrapper (if needed for text generation).
        story (str): Log of messages, narrative, and prompts encountered by the player.
        actions (List[str]): History of actions taken during the game.
        votes (List[str]): History of votes cast during the game.
        witness (bool): Flag indicating if the player witnessed a kill.
        witness_during_vote (List[bool]): Records of whether the player witnessed a kill during each vote.
        awaiting_response (bool): Indicates if the player is waiting for an input.
        invalid_votes_for_eliminated (int): Count of votes cast for players already eliminated.
        eliminated_player_names (List[str]): Names of players who have been removed.
        location (str): The current room/location of the player.
        eval (Dict): Dictionary used to accumulate evaluation metrics for post-game analysis.
    """
    VALID_LOCATIONS = ["Bedroom", "Bathroom", "Kitchen", "Hallway"]

    def __init__(self, name: str, killer: bool, preprompt: str, agent: str, start_location: str = "random"):
        # Basic initialization of the player's attributes.
        self.name = name
        self.killer = killer
        self.preprompt = preprompt
        self.alive = True
        self.banished = False
        self.has_key = False  # Key functionality restored (if needed in future enhancements)
        # (Escape flag has been removed from the design.)

        # Fields for GPT-based decision making.
        self.agent: str = ""
        self.model: Optional[str] = None
        self.gpt = None

        # Initialize logs and game state tracking.
        self.story = ""
        self.actions: List[str] = []
        self.votes: List[str] = []
        self.witness = False
        self.witness_during_vote: List[bool] = []
        self.awaiting_response = False

        self.invalid_votes_for_eliminated = 0
        self.eliminated_player_names: List[str] = []

        # Set the player's starting location (either provided or chosen randomly).
        self.location = self._resolve_start_location(start_location)

        # Determine the player's control type and, if using GPT, the specific model.
        self._parse_agent_type(agent)

        # Initialize an evaluation dictionary to accumulate game metrics.
        self.eval = self._init_eval_dict()
        self.eval.setdefault("discussion_participation", 0)
        self.eval.setdefault("banished_in_discussion", 0)

        logger.info("Initialized Player '%s'. Killer=%s, Agent='%s', Location='%s'",
                    self.name, self.killer,
                    f"{self.agent}{('-' + self.model) if self.model else ''}",
                    self.location)

    def set_eliminated_players(self, eliminated_list: List[str]) -> None:
        """
        Updates the player's record of which players have been eliminated.
        
        Args:
            eliminated_list (List[str]): List of names of eliminated players.
        """
        self.eliminated_player_names = eliminated_list[:]

    def _resolve_start_location(self, requested_location: str) -> str:
        """
        Determines and returns a valid starting location for the player.
        
        Args:
            requested_location (str): Either a specific room name or "random" for random selection.
        
        Returns:
            str: A valid room from VALID_LOCATIONS.
        """
        if requested_location == "random":
            return random.choice(self.VALID_LOCATIONS)
        if requested_location not in self.VALID_LOCATIONS:
            raise ValueError(f"Invalid start location: {requested_location}")
        return requested_location

    def _parse_agent_type(self, agent_str: str) -> None:
        """
        Parses the agent string to determine the player's control type and, if applicable, the GPT model.
        
        Args:
            agent_str (str): A string such as "cli", "random", or "gpt-<model>".
        """
        if agent_str.startswith("gpt"):
            self.agent = AgentType.GPT.value
            self.model = agent_str[4:]
        else:
            self.agent = agent_str

        valid_types = {a.value for a in AgentType}
        if self.agent not in valid_types:
            raise ValueError(f"Unrecognized agent type: {self.agent}")

    def _init_eval_dict(self) -> Dict[str, any]:
        """
        Initializes and returns a dictionary to store evaluation metrics.
        
        Returns:
            dict: Evaluation metrics including vote rates, number of turns, etc.
        """
        eval_dict = {
            "name": self.name,
            "agent": f"{self.agent}{f'-{self.model}' if self.model else ''}",
            "killer": self.killer,
            "preprompt": self.preprompt,
            "num_turns": 0,
            "banished": False,
            "story": self.story,
            "actions": self.actions,
            "votes": self.votes,
            "vote_rate_for_killer": None,  # Placeholder for later calculation.
        }
        if not self.killer:
            eval_dict.update({
                "killed": False,
            })
        else:
            eval_dict.update({
                "num_killed": 0,
                "num_banished": 0,
                "num_escaped": 0,
            })
        return eval_dict

    def load_gpt(self, gpt):
        """
        Associates a GPT instance with the player for text generation.
        
        Args:
            gpt: A GPT wrapper instance.
        """
        self.gpt = gpt

    # ----------------------------- CLI Input Helper -----------------------------
    def _get_cli_choice(self, prompt: str, valid: List[int]) -> Optional[int]:
        """
        Repeatedly prompts the user via the CLI for a valid numeric choice.
        
        Args:
            prompt (str): The prompt message to display.
            valid (List[int]): List of valid numeric options.
        
        Returns:
            int: A valid choice, or a random fallback if no valid input is received.
        """
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            print(self.story)
            print(prompt)
            print(f"Please input one of: {valid}")
            user_in = input().strip()
            try:
                choice = int(user_in)
                if choice in valid:
                    return choice
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            attempts += 1
        fallback_choice = random.choice(valid)
        print(f"No valid input received after {max_attempts} attempts. Defaulting to {fallback_choice}.")
        return fallback_choice

    # ----------------------------- Action Phase Methods -----------------------------
    def get_action(self, action_prompt: str) -> str:
        """
        Obtains an action from the player during the Action Phase.
        Handles retries and falls back to a random valid action if needed.
        
        Args:
            action_prompt (str): The prompt listing possible actions.
        
        Returns:
            str: The text of the selected action.
        """
        self.awaiting_response = True
        logger.info("Action prompt for %s:\n%s", self.name, action_prompt)
        valid_actions = self._parse_valid_actions(action_prompt)
        if not valid_actions:
            logger.warning("No valid actions found for %s. Defaulting to 'No Action'.", self.name)
            self.actions.append("No Action")
            self.eval["num_turns"] += 1
            self.awaiting_response = False
            return "No Action"

        chosen_int = None
        attempts = 0
        max_attempts = 5
        while chosen_int is None and attempts < max_attempts:
            chosen_int = self._fetch_action_int(valid_actions, action_prompt)
            attempts += 1
        if chosen_int is None:
            chosen_int = random.choice(valid_actions)
            logger.warning("%s exceeded maximum attempts. Using fallback action: %s", self.name, chosen_int)
        action_text = self._decode_action(action_prompt, chosen_int)
        self.actions.append(action_text)
        self.eval["num_turns"] += 1
        self.awaiting_response = False
        return action_text

    def _parse_valid_actions(self, prompt: str) -> List[int]:
        """
        Extracts valid action numbers from the prompt text.
        
        Args:
            prompt (str): The full action prompt.
        
        Returns:
            List[int]: A list of numeric options.
        """
        if "Possible Actions:" not in prompt:
            return []
        substring = prompt.split("Possible Actions:")[-1]
        return [int(n) for n in re.findall(r"\d+", substring)]

    def _fetch_action_int(self, valid_actions: List[int], action_prompt: str) -> Optional[int]:
        """
        Chooses an action number using the appropriate method based on the agent type.
        
        Args:
            valid_actions (List[int]): List of valid numeric options.
            action_prompt (str): The prompt text for the action.
        
        Returns:
            int or None: The chosen action number.
        """
        from constants import AgentType
        if self.agent == AgentType.RANDOM.value:
            return random.choice(valid_actions) if valid_actions else None
        elif self.agent == AgentType.CLI.value:
            return self._get_cli_choice(action_prompt, valid_actions)
        elif self.agent == AgentType.GPT.value:
            return self.get_gpt_action(action_prompt)
        else:
            logger.warning("Invalid or None action chosen by %s. Using None", self.name)
            return None

    def store_api_action(self, action_prompt: str, action_int: int) -> None:
        """
        Stores an action received via an external API call.
        
        Args:
            action_prompt (str): The prompt text that was presented.
            action_int (int): The numeric option chosen.
        """
        action_text = self._decode_action(action_prompt, action_int)
        self.actions.append(action_text)
        self.eval["num_turns"] += 1
        self.awaiting_response = False

    def _decode_action(self, prompt: str, action_int: int) -> str:
        """
        Converts a numeric choice into the corresponding action text.
        
        Args:
            prompt (str): The action prompt.
            action_int (int): The chosen action number.
        
        Returns:
            str: The action text.
        """
        target = f"{action_int}. "
        idx = prompt.find(target)
        if idx < 0:
            return "UNKNOWN_ACTION"
        idx += len(target)
        remainder = prompt[idx:]
        newline_pos = remainder.find('\n')
        if newline_pos < 0:
            newline_pos = len(remainder)
        return remainder[:newline_pos].strip()

    # ----------------------------- Discussion Methods -----------------------------
    def get_statement(self, discussion_log: str) -> str:
        """
        Obtains a discussion statement from the player.
        
        Args:
            discussion_log (str): The prompt or log for the discussion phase.
        
        Returns:
            str: The player's discussion statement.
        """
        logger.info("Discussion prompt for %s:\n%s", self.name, discussion_log)
        from constants import AgentType
        if self.agent == AgentType.RANDOM.value:
            return "I don't know what to say."
        elif self.agent == AgentType.CLI.value:
            print(self.story)
            print(discussion_log)
            return input()
        elif self.agent == AgentType.GPT.value:
            return self._get_gpt_statement(discussion_log)
        else:
            return "I don't know what to say."

    # ----------------------------- Voting Methods -----------------------------
    def get_vote(self, vote_prompt: str) -> str:
        """
        Retrieves the player's vote during the Voting Phase.
        
        Args:
            vote_prompt (str): The voting prompt listing candidate names.
        
        Returns:
            str: The name corresponding to the selected vote.
        """
        logger.info("Vote prompt for %s:\n%s", self.name, vote_prompt)
        self.awaiting_response = True
        valid_votes = self._parse_valid_votes(vote_prompt)
        chosen_int = None
        attempts = 0
        max_attempts = 5
        while chosen_int is None and attempts < max_attempts:
            chosen_int = self._fetch_vote_int(valid_votes, vote_prompt)
            attempts += 1
        if chosen_int is None:
            chosen_int = random.choice(valid_votes)
            logger.warning("%s exceeded maximum vote attempts. Using fallback vote: %s", self.name, chosen_int)
        vote_name = self._decode_vote(vote_prompt, chosen_int)
        self.votes.append(vote_name)
        self.witness_during_vote.append(self.witness)
        if vote_name in self.eliminated_player_names:
            self.invalid_votes_for_eliminated += 1
        self.awaiting_response = False
        return vote_name

    def _parse_valid_votes(self, prompt: str) -> List[int]:
        """
        Extracts valid vote numbers from the voting prompt.
        
        Args:
            prompt (str): The full voting prompt.
        
        Returns:
            List[int]: A list of valid vote options.
        """
        return [int(x) for x in re.findall(r"(\d+)\.", prompt)]

    def _fetch_vote_int(self, valid_votes: List[int], vote_prompt: str) -> Optional[int]:
        """
        Selects a vote option based on the player's agent type.
        
        Args:
            valid_votes (List[int]): List of valid numeric vote options.
            vote_prompt (str): The voting prompt.
        
        Returns:
            int or None: The chosen vote number.
        """
        from constants import AgentType
        if self.agent == AgentType.RANDOM.value:
            return random.choice(valid_votes) if valid_votes else None
        elif self.agent == AgentType.CLI.value:
            return self._get_cli_choice(vote_prompt, valid_votes)
        elif self.agent == AgentType.GPT.value:
            return self.get_gpt_action(vote_prompt)
        else:
            logger.warning("Invalid agent type for voting.")
            return None

    def store_api_vote(self, vote_prompt: str, vote_int: int) -> None:
        """
        Stores an externally received vote.
        
        Args:
            vote_prompt (str): The prompt shown for voting.
            vote_int (int): The chosen numeric option.
        """
        vote_name = self._decode_vote(vote_prompt, vote_int)
        self.votes.append(vote_name)
        self.witness_during_vote.append(self.witness)
        if vote_name in self.eliminated_player_names:
            self.invalid_votes_for_eliminated += 1
        self.awaiting_response = False

    def _decode_vote(self, prompt: str, vote_int: int) -> str:
        """
        Converts the numeric vote selection into the corresponding candidate name.
        
        Args:
            prompt (str): The voting prompt with candidate names.
            vote_int (int): The selected option number.
        
        Returns:
            str: The candidate's name or "UNKNOWN_VOTE" if not found.
        """
        pattern = r"(\d+)\.\s*(.*)"
        voting_options = {}
        for match in re.finditer(pattern, prompt):
            num_str = match.group(1)
            name_str = match.group(2).strip()
            line_break = name_str.find('\n')
            if line_break != -1:
                name_str = name_str[:line_break].strip()
            voting_options[num_str] = name_str
        return voting_options.get(str(vote_int), "UNKNOWN_VOTE")

    # ----------------------------- Evaluation Finalization -----------------------------
    def finalize_eval(self, killer_names: List[str]) -> None:
        """
        Finalizes the player's evaluation metrics for post-game analysis.
        
        Args:
            killer_names (List[str]): List of names identified as killers.
        """
        self.eval['story'] = self.story
        self.eval['actions'] = self.actions
        self.eval['votes'] = self.votes
        self.eval['witness_during_vote'] = self.witness_during_vote
        self.eval['invalid_votes_for_eliminated'] = self.invalid_votes_for_eliminated

        total_votes = len(self.votes)

        if not self.killer and self.alive:
            self.eval["killed"] = False

        if len(killer_names) == 0:
            pass
        elif len(killer_names) == 1:
            killer_name = killer_names[0]
            if total_votes > 0:
                self.eval['vote_rate_for_self'] = sum(v == self.name for v in self.votes) / total_votes
                self.eval['vote_rate_for_killer'] = sum(vote == killer_name for vote in self.votes) / total_votes

            killer_witness_votes = 0
            killer_not_witness_votes = 0
            for saw, votee in zip(self.witness_during_vote, self.votes):
                if votee == killer_name:
                    if saw:
                        killer_witness_votes += 1
                    else:
                        killer_not_witness_votes += 1
            witness_count = sum(self.witness_during_vote)
            if witness_count:
                self.eval['witness_vote_rate_for_killer'] = killer_witness_votes / witness_count
            non_witness_count = total_votes - witness_count
            if non_witness_count:
                self.eval['non_witness_vote_rate_for_killer'] = killer_not_witness_votes / non_witness_count
        else:
            if total_votes > 0:
                killer_votes = sum(vote in killer_names for vote in self.votes)
                self.eval['vote_rate_for_killer'] = killer_votes / total_votes
                self.eval['vote_rate_for_self'] = sum(v == self.name for v in self.votes) / total_votes

            killer_witness_votes = 0
            killer_not_witness_votes = 0
            for saw, votee in zip(self.witness_during_vote, self.votes):
                if votee in killer_names:
                    if saw:
                        killer_witness_votes += 1
                    else:
                        killer_not_witness_votes += 1
            witness_count = sum(self.witness_during_vote)
            if witness_count:
                self.eval['witness_vote_rate_for_killer'] = killer_witness_votes / witness_count
            non_witness_count = total_votes - witness_count
            if non_witness_count:
                self.eval['non_witness_vote_rate_for_killer'] = killer_not_witness_votes / non_witness_count
            self.eval['multiple_killers'] = killer_names

        if total_votes > 0:
            self.eval['self_vote_count'] = sum(1 for v in self.votes if v == self.name)
        else:
            self.eval['self_vote_count'] = 0

        # Ensure that all expected keys have default values.
        self.eval.setdefault("vote_rate_for_killer", None)
        self.eval.setdefault("vote_rate_for_self", None)
        self.eval.setdefault("discussion_participation", 0)
        self.eval.setdefault("banished_in_discussion", 0)
