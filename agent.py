import random
import re
import logging
from typing import List, Dict, Optional

from constants import AgentType, KILL_PREFIX, SEARCH_PREFIX, GO_TO_PREFIX
from gpt_agent import GptAgentMixin

logger = logging.getLogger(__name__)

class Player(GptAgentMixin):
    """
    Represents a single player in the Hoodwinked game.
    Each player may be a killer or innocent (multiple killers possible).
    The agent type determines how actions and statements are chosen.
    """

    VALID_LOCATIONS = ["Bedroom", "Bathroom", "Kitchen", "Hallway"]

    def __init__(
        self,
        name: str,
        killer: bool,
        preprompt: str,
        agent: str,
        start_location: str = "random"
    ):
        """
        :param name: Name of the player
        :param killer: Whether this player is a killer (multiple can be True)
        :param preprompt: Which rules or introduction text to use
        :param agent: Agent type (cli/random/gpt/api, e.g. "gpt-3.5")
        :param start_location: Where the player starts (or 'random')
        """
        self.name = name
        self.killer = killer
        self.preprompt = preprompt
        self.alive = True
        self.banished = False
        self.has_key = False
        self.escaped = False

        # GPT-related fields (if agent is GPT)
        self.agent: str = ""
        self.model: Optional[str] = None
        self.gpt = None  # Will be loaded by the Game if needed

        # Player logs and metadata
        self.story = ""
        self.actions: List[str] = []
        self.votes: List[str] = []
        self.witness = False
        self.witness_during_vote: List[bool] = []
        self.awaiting_response = False

        # NEW: track invalid votes for already-eliminated players
        self.invalid_votes_for_eliminated = 0
        self.eliminated_player_names: List[str] = []

        # Start location
        self.location = self._resolve_start_location(start_location)

        # Determine final agent type (and model if GPT)
        self._parse_agent_type(agent)

        # Initialize evaluation dictionary
        self.eval = self._init_eval_dict()

        logger.info(
            "Initialized Player '%s'. Killer=%s, Agent='%s', Location='%s'",
            self.name, self.killer,
            f"{self.agent}{('-' + self.model) if self.model else ''}",
            self.location
        )

    # --------------------------------------------------------------------------
    #                         ELIMINATED PLAYERS TRACKER
    # --------------------------------------------------------------------------
    def set_eliminated_players(self, eliminated_list: List[str]) -> None:
        """
        Allows the environment to update this player's knowledge of
        who is already eliminated (dead or banished).

        If the player votes for someone in this list,
        self.invalid_votes_for_eliminated increments by 1.
        """
        self.eliminated_player_names = eliminated_list[:]

    # --------------------------------------------------------------------------
    #                         INIT & HELPER FUNCTIONS
    # --------------------------------------------------------------------------
    def _resolve_start_location(self, requested_location: str) -> str:
        """ Determines the player's initial location. """
        if requested_location == "random":
            return random.choice(self.VALID_LOCATIONS)
        if requested_location not in self.VALID_LOCATIONS:
            raise ValueError(f"Invalid start location: {requested_location}")
        return requested_location

    def _parse_agent_type(self, agent_str: str) -> None:
        """
        Splits out 'gpt' from, e.g., 'gpt-3.5'. Otherwise sets self.agent directly.
        Validates recognized agent types.
        """
        if agent_str.startswith("gpt"):
            self.agent = AgentType.GPT.value
            # e.g. "gpt-3.5" => model="3.5"
            self.model = agent_str[4:]
        else:
            self.agent = agent_str

        valid_types = {a.value for a in AgentType}
        if self.agent not in valid_types:
            raise ValueError(f"Unrecognized agent type: {self.agent}")

    def _init_eval_dict(self) -> Dict[str, any]:
        """
        Creates the evaluation dict that tracks end-of-game statistics.
        """
        eval_dict = {
            "name": self.name,
            "agent": f"{self.agent}{f'-{self.model}' if self.model else ''}",
            "killer": self.killer,
            "preprompt": self.preprompt,  # <-- STORE PREPROMPT HERE
            "num_turns": 0,
            "banished": False,
            "story": self.story,
            "actions": self.actions,
            "votes": self.votes,
        }

        if not self.killer:
            eval_dict.update({
                "killed": False,
                "escaped": False,
            })
        else:
            eval_dict.update({
                "num_killed": 0,     # how many victims this killer has
                "num_banished": 0,   # how many innocents got banished while killer is alive
                "num_escaped": 0,    # how many innocents escaped
            })
        return eval_dict

    def load_gpt(self, gpt):
        """
        If needed, the Game class can assign a GPT connection
        to this player for generating text or probabilities.
        """
        self.gpt = gpt

    # --------------------------------------------------------------------------
    #                             GETTING ACTIONS
    # --------------------------------------------------------------------------
    def get_action(self, action_prompt: str) -> str:
        """
        Retrieves the next action from this player. Depending on agent type,
        it can be user CLI input, random choice, or GPT-based logic.
        Returns the text of the chosen action (e.g. "Search the closet").
        """
        self.awaiting_response = True
        valid_actions = self._parse_valid_actions(action_prompt)

        chosen_int = None
        while chosen_int is None:
            chosen_int = self._fetch_action_int(valid_actions, action_prompt)

        action_text = self._decode_action(action_prompt, chosen_int)
        self.actions.append(action_text)
        self.eval["num_turns"] += 1
        self.awaiting_response = False

        return action_text

    def _parse_valid_actions(self, prompt: str) -> List[int]:
        """
        Searches for digit patterns in the portion after 'Possible Actions:'.
        This allows more than 10 possible actions (multi-digit).
        """
        if "Possible Actions:" not in prompt:
            return []
        substring = prompt.split("Possible Actions:")[-1]
        return [int(n) for n in re.findall(r"\d+", substring)]

    def _fetch_action_int(self, valid_actions: List[int], action_prompt: str) -> Optional[int]:
        """
        Gets an action integer (e.g. 1,2,3...) from the appropriate method.
        Validates. Returns None if invalid to re-try.
        """
        if self.agent == AgentType.RANDOM.value:
            value = random.choice(valid_actions) if valid_actions else None
        elif self.agent == AgentType.CLI.value:
            value = self._get_cli_action(valid_actions, action_prompt)
        elif self.agent == AgentType.GPT.value:
            value = self.get_gpt_action(action_prompt)
        elif self.agent == AgentType.API.value:
            value = None
        else:
            logger.error("Invalid agent type for action.")
            value = None

        if value is None or value not in valid_actions:
            logger.warning("Invalid action chosen by %s: %s. Valid: %s",
                           self.name, value, valid_actions)
            return None
        return value

    def _get_cli_action(self, valid_actions: List[int], prompt: str) -> Optional[int]:
        """
        Prompts the user for input. Returns the chosen integer or None if invalid.
        """
        print(self.story)
        print(prompt)
        print(f"Please input one of the following valid inputs: {valid_actions}")

        user_in = input().strip()
        try:
            choice = int(user_in)
            return choice
        except ValueError:
            return None

    def store_api_action(self, action_prompt: str, action_int: int) -> None:
        """
        If an external system provides the action integer, it can be stored here directly.
        """
        action_text = self._decode_action(action_prompt, action_int)
        self.actions.append(action_text)
        self.eval["num_turns"] += 1
        self.awaiting_response = False

    def _decode_action(self, prompt: str, action_int: int) -> str:
        """
        Extracts the action text from an action prompt given an integer.
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

    # --------------------------------------------------------------------------
    #                          GETTING STATEMENTS
    # --------------------------------------------------------------------------
    def get_statement(self, discussion_log: str) -> str:
        """
        Returns a statement from the player during group discussion.
        Returns only raw text (no extra quotes).
        """
        if self.agent == AgentType.RANDOM.value:
            statement = self._default_random_statement()
        elif self.agent == AgentType.CLI.value:
            statement = self._get_cli_statement(discussion_log)
        elif self.agent == AgentType.GPT.value:
            statement = self._get_gpt_statement(discussion_log)
        else:
            statement = self._default_random_statement()

        return statement

    def _default_random_statement(self) -> str:
        return "I don't know who the killer is."

    def _get_cli_statement(self, discussion_log: str) -> str:
        print(self.story)
        print(discussion_log)
        return input()

    # --------------------------------------------------------------------------
    #                               GETTING VOTES
    # --------------------------------------------------------------------------
    def get_vote(self, vote_prompt: str) -> str:
        """
        Returns the name of the voted player, or "No Vote" if not found.
        Also checks if the vote is for an eliminated player.
        """
        vote_int: Optional[int] = None
        if self.agent == AgentType.RANDOM.value:
            vote_int = self._get_random_vote(vote_prompt)
        elif self.agent == AgentType.CLI.value:
            vote_int = self._get_cli_vote(vote_prompt)
        elif self.agent == AgentType.GPT.value:
            vote_int = self.get_gpt_action(vote_prompt)
        # AgentType.API => might be set externally or no vote
        if vote_int is None:
            return "No Vote"

        vote_name = self._decode_vote(vote_prompt, vote_int)
        self.votes.append(vote_name)
        self.witness_during_vote.append(self.witness)

        # Check if that voted name is already eliminated
        if vote_name in self.eliminated_player_names:
            self.invalid_votes_for_eliminated += 1

        return vote_name

    def _get_random_vote(self, prompt: str) -> Optional[int]:
        """
        Randomly picks a vote from possible numeric options.
        """
        ints = re.findall(r"\d+", prompt)
        return int(random.choice(ints)) if ints else None

    def _get_cli_vote(self, prompt: str) -> Optional[int]:
        """
        For user input in CLI mode.
        """
        print(self.story)
        print(prompt)
        user_in = input().strip()
        try:
            return int(user_in)
        except ValueError:
            return None

    def _decode_vote(self, prompt: str, vote_int: int) -> str:
        """
        E.g. lines "1. Bob\n2. Jim\n3. Mira" => {1:'Bob', 2:'Jim', 3:'Mira'}
        """
        option_nums = re.findall(r"\d+", prompt)
        voting_options = {}
        for num in option_nums:
            start_idx = prompt.find(num) + 3
            end_idx = prompt[start_idx:].find('\n') + start_idx
            if end_idx < start_idx:
                end_idx = len(prompt)
            voting_options[num] = prompt[start_idx:end_idx].strip()

        return voting_options.get(str(vote_int), "UNKNOWN_VOTE")

    # --------------------------------------------------------------------------
    #                            FINAL EVALUATION
    # --------------------------------------------------------------------------
    def finalize_eval(self, killer_name: str) -> None:
        """
        Computes final evaluation metrics at game end.
        We also store how many invalid votes were cast for eliminated players.
        """
        self.eval['story'] = self.story
        self.eval['actions'] = self.actions
        self.eval['votes'] = self.votes
        self.eval['witness_during_vote'] = self.witness_during_vote

        # NEW: record how many times this player voted for an eliminated player
        self.eval['invalid_votes_for_eliminated'] = self.invalid_votes_for_eliminated

        total_votes = len(self.eval['votes'])
        if total_votes > 0:
            # fraction of self-votes
            self.eval['vote_rate_for_self'] = sum(v == self.name for v in self.votes) / total_votes
            # fraction of votes for the designated "killer_name"
            self.eval['vote_rate_for_killer'] = sum(v == killer_name for v in self.votes) / total_votes

        # Evaluate witness vs. non-witness votes
        killer_witness_votes = 0
        killer_not_witness_votes = 0
        for saw, votee in zip(self.witness_during_vote, self.votes):
            if votee == killer_name:
                if saw:
                    killer_witness_votes += 1
                else:
                    killer_not_witness_votes += 1

        witness_votes_count = sum(self.witness_during_vote)
        if witness_votes_count:
            self.eval['witness_vote_rate_for_killer'] = killer_witness_votes / witness_votes_count
        non_witness_count = total_votes - witness_votes_count
        if non_witness_count:
            self.eval['non_witness_vote_rate_for_killer'] = killer_not_witness_votes / non_witness_count

        # Evaluate search duplication
        search_actions = [a for a in self.actions if a.startswith(SEARCH_PREFIX)]
        if not self.killer and search_actions:
            visited = set()
            duplicates = 0
            for act in search_actions:
                loc = act[len(SEARCH_PREFIX):]
                if loc in visited:
                    duplicates += 1
                visited.add(loc)
            self.eval['duplicate_search_rate'] = duplicates / len(search_actions)
 
