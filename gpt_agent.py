# gpt_agent.py
import random
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class GptAgentMixin:
    """
    A mixin that provides GPT-based decision logic, such as
    selecting an action from a probability distribution or
    generating a statement. Requires `self.gpt` and `self.model`
    to be defined in the main class.
    """

    def get_gpt_action(self, prompt: str, argmax: bool = False) -> int:
        """
        Uses self.gpt to retrieve a probability distribution over
        multiple choice options, then picks either the highest-probability
        or a sampled distribution.
        """
        logger.debug("Running get_gpt_action with prompt:\n%s", prompt)
        action_dict = self._extract_list_items(prompt)
        if not hasattr(self, 'gpt') or not self.gpt:
            logger.warning("No gpt instance found; defaulting to random choice!")
            if action_dict:
                return int(random.choice(list(action_dict.keys())))
            else:
                return None

        option_probs = self.gpt.get_probs(self.story + prompt, action_dict, self.model)

        if not option_probs:
            logger.warning("No probabilities returned from GPT; picking random choice.")
            if action_dict:
                return int(random.choice(list(action_dict.keys())))
            return None

        if argmax:
            return int(max(option_probs, key=option_probs.get))

        # Otherwise sample from distribution
        rand_val = random.random()
        cumulative = 0.0
        for action, prob in option_probs.items():
            cumulative += prob
            if rand_val <= cumulative:
                return int(action)
        # Fallback
        return int(random.choice(list(option_probs.keys())))

    def _get_gpt_statement(self, prompt: str) -> str:
        """
        Generates a short statement using GPT.
        # FIX 2: Remove the double-quote stop token to avoid truncation.
        """
        logger.debug("Generating GPT statement for prompt:\n%s", prompt)
        if not hasattr(self, 'gpt') or not self.gpt:
            logger.warning("No gpt instance found; returning default statement.")
            return "I don't know who the killer is."

        response = self.gpt.generate(
            prompt=self.story + prompt,
            max_tokens=50,
            model=self.model,
            stop_tokens=[]  # Removed the stop token for quotes
        )
        return response

    def _extract_list_items(self, text: str) -> Dict[int, str]:
        """
        Parses lines of the form '1. Some text' into {1: 'Some text', ...}.
        """
        import re
        pattern = r'(\d+)\.\s+(.*)'
        results = {}
        for match in re.finditer(pattern, text):
            num = int(match.group(1))
            content = match.group(2).strip()
            results[num] = content
        return results
