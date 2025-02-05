# agent.py
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

    def __init__(self, name: str, killer: bool, preprompt: str, agent: str, start_location: str = "random"):
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
        self.gpt = None

        # Player logs
        self.story = ""
        self.actions: List[str] = []
        self.votes: List[str] = []
        self.witness = False
        self.witness_during_vote: List[bool] = []
        self.awaiting_response = False

        self.invalid_votes_for_eliminated = 0
        self.eliminated_player_names: List[str] = []

        # Set starting location
        self.location = self._resolve_start_location(start_location)

        # Determine agent type and model if applicable
        self._parse_agent_type(agent)

        # Initialize evaluation dictionary
        self.eval = self._init_eval_dict()

        logger.info("Initialized Player '%s'. Killer=%s, Agent='%s', Location='%s'",
                    self.name, self.killer,
                    f"{self.agent}{('-' + self.model) if self.model else ''}",
                    self.location)

    def set_eliminated_players(self, eliminated_list: List[str]) -> None:
        self.eliminated_player_names = eliminated_list[:]

    def _resolve_start_location(self, requested_location: str) -> str:
        if requested_location == "random":
            return random.choice(self.VALID_LOCATIONS)
        if requested_location not in self.VALID_LOCATIONS:
            raise ValueError(f"Invalid start location: {requested_location}")
        return requested_location

    def _parse_agent_type(self, agent_str: str) -> None:
        if agent_str.startswith("gpt"):
            self.agent = AgentType.GPT.value
            self.model = agent_str[4:]
        else:
            self.agent = agent_str

        valid_types = {a.value for a in AgentType}
        if self.agent not in valid_types:
            raise ValueError(f"Unrecognized agent type: {self.agent}")

    def _init_eval_dict(self) -> Dict[str, any]:
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
        }
        if not self.killer:
            eval_dict.update({
                "killed": False,
                "escaped": False,
            })
        else:
            eval_dict.update({
                "num_killed": 0,
                "num_banished": 0,
                "num_escaped": 0,
            })
        return eval_dict

    def load_gpt(self, gpt):
        self.gpt = gpt

    # --------------------------------------------------------------------------
    # Helper for CLI input with max retries
    # --------------------------------------------------------------------------
    def _get_cli_choice(self, prompt: str, valid: List[int]) -> Optional[int]:
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

    # --------------------------------------------------------------------------
    # Actions
    # --------------------------------------------------------------------------
    def get_action(self, action_prompt: str) -> str:
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
        if "Possible Actions:" not in prompt:
            return []
        substring = prompt.split("Possible Actions:")[-1]
        return [int(n) for n in re.findall(r"\d+", substring)]

    def _fetch_action_int(self, valid_actions: List[int], action_prompt: str) -> Optional[int]:
        from constants import AgentType
        if self.agent == AgentType.RANDOM.value:
            return random.choice(valid_actions) if valid_actions else None
        elif self.agent == AgentType.CLI.value:
            return self._get_cli_choice(action_prompt, valid_actions)
        elif self.agent == AgentType.GPT.value:
            return self.get_gpt_action(action_prompt)
        elif self.agent == AgentType.API.value:
            return random.choice(valid_actions) if valid_actions else None

        logger.warning("Invalid or None action chosen by %s. Using None", self.name)
        return None

    def store_api_action(self, action_prompt: str, action_int: int) -> None:
        action_text = self._decode_action(action_prompt, action_int)
        self.actions.append(action_text)
        self.eval["num_turns"] += 1
        self.awaiting_response = False

    def _decode_action(self, prompt: str, action_int: int) -> str:
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
    # Statements
    # --------------------------------------------------------------------------
    def get_statement(self, discussion_log: str) -> str:
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

    # --------------------------------------------------------------------------
    # Votes
    # --------------------------------------------------------------------------
    def get_vote(self, vote_prompt: str) -> str:
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
        # Avoid self-voting: if the vote candidate is yourself, choose an alternative.
        if vote_name == self.name:
            candidates = [v for v in valid_votes if self._decode_vote(vote_prompt, v) != self.name]
            if candidates:
                chosen_int = random.choice(candidates)
                vote_name = self._decode_vote(vote_prompt, chosen_int)
                logger.info("Self–vote avoided for %s; new vote candidate: %s", self.name, vote_name)
        self.votes.append(vote_name)
        self.witness_during_vote.append(self.witness)
        if vote_name in self.eliminated_player_names:
            self.invalid_votes_for_eliminated += 1
        self.awaiting_response = False
        return vote_name

    def _parse_valid_votes(self, prompt: str) -> List[int]:
        return [int(x) for x in re.findall(r"(\d+)\.", prompt)]

    def _fetch_vote_int(self, valid_votes: List[int], vote_prompt: str) -> Optional[int]:
        from constants import AgentType
        if self.agent == AgentType.RANDOM.value:
            return random.choice(valid_votes) if valid_votes else None
        elif self.agent == AgentType.CLI.value:
            return self._get_cli_choice(vote_prompt, valid_votes)
        elif self.agent == AgentType.GPT.value:
            return self.get_gpt_action(vote_prompt)
        elif self.agent == AgentType.API.value:
            return random.choice(valid_votes) if valid_votes else None
        else:
            logger.warning("Invalid agent type for voting.")
            return None

    def store_api_vote(self, vote_prompt: str, vote_int: int) -> None:
        vote_name = self._decode_vote(vote_prompt, vote_int)
        self.votes.append(vote_name)
        self.witness_during_vote.append(self.witness)
        if vote_name in self.eliminated_player_names:
            self.invalid_votes_for_eliminated += 1
        self.awaiting_response = False

    def _decode_vote(self, prompt: str, vote_int: int) -> str:
        pattern = r"(\d+)\.\s*(.*)"
        import re
        voting_options = {}
        for match in re.finditer(pattern, prompt):
            num_str = match.group(1)
            name_str = match.group(2).strip()
            line_break = name_str.find('\n')
            if line_break != -1:
                name_str = name_str[:line_break].strip()
            voting_options[num_str] = name_str
        return voting_options.get(str(vote_int), "UNKNOWN_VOTE")

    # --------------------------------------------------------------------------
    # Finalize Evaluation
    # --------------------------------------------------------------------------
    def finalize_eval(self, killer_names: List[str]) -> None:
        from constants import SEARCH_PREFIX
        self.eval['story'] = self.story
        self.eval['actions'] = self.actions
        self.eval['votes'] = self.votes
        self.eval['witness_during_vote'] = self.witness_during_vote
        self.eval['invalid_votes_for_eliminated'] = self.invalid_votes_for_eliminated

        total_votes = len(self.eval['votes'])

        # For non–killer players still alive, force "killed" flag to False.
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

# End of agent.py
