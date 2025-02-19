"""
gpt_agent.py
-------------
This module defines a mixin class, GptAgentMixin, which adds GPT-based decision-making
capabilities to a player. It includes methods for generating an action (as a numeric choice)
or a discussion statement based on a given prompt. It also provides a helper method to extract
numbered list items from a prompt.
"""

import random
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class GptAgentMixin:
    """
    Mixin class that provides methods for generating actions and statements using GPT.
    
    The methods in this mixin assume that the host class has attributes:
      - self.gpt: an instance of a GPT wrapper with methods 'get_probs' and 'generate'
      - self.story: a string containing the player's narrative history
      - self.model: the GPT model identifier (if using a GPT agent)
    """
    
    def get_gpt_action(self, prompt: str, argmax: bool = False) -> int:
        """
        Generates an action based on a prompt by using GPT to compute probabilities over possible actions.
        
        The function performs the following steps:
          1. Extracts numbered options from the prompt.
          2. If no GPT instance is available, selects a random option.
          3. Otherwise, calls self.gpt.get_probs to obtain probabilities for each option.
          4. If probabilities are available and argmax is True, returns the option with the highest probability.
          5. If argmax is False, uses a random draw weighted by the computed probabilities.
          6. Falls back to a random option if no probabilities are produced.
        
        Args:
            prompt (str): The prompt containing numbered action options.
            argmax (bool): If True, select the option with the highest probability; otherwise, sample.
        
        Returns:
            int: The chosen action number.
        """
        logger.debug("Running get_gpt_action with prompt:\n%s", prompt)
        action_dict = self._extract_list_items(prompt)
        
        # Fallback: if no GPT instance is available, randomly choose an action.
        if not hasattr(self, 'gpt') or not self.gpt:
            logger.warning("No gpt instance found; defaulting to random choice!")
            if action_dict:
                return int(random.choice(list(action_dict.keys())))
            else:
                return None

        # Obtain probability estimates for each option from GPT.
        option_probs = self.gpt.get_probs(self.story + prompt, action_dict, self.model)
        if not option_probs:
            logger.warning("No probabilities from GPT; random choice fallback.")
            if action_dict:
                return int(random.choice(list(action_dict.keys())))
            return None

        # If argmax is specified, choose the option with the highest probability.
        if argmax:
            return int(max(option_probs, key=option_probs.get))

        # Otherwise, sample an option based on the probability distribution.
        rand_val = random.random()
        cumulative = 0.0
        for act, pr in option_probs.items():
            cumulative += pr
            if rand_val <= cumulative:
                return int(act)
        # Fallback to random selection if sampling fails.
        return int(random.choice(list(option_probs.keys())))

    def _get_gpt_statement(self, prompt: str) -> str:
        """
        Generates a discussion statement using GPT based on the provided prompt.
        
        If a GPT instance is not available, returns a default statement.
        
        Args:
            prompt (str): The discussion prompt.
        
        Returns:
            str: The generated statement.
        """
        logger.debug("Generating GPT statement for prompt:\n%s", prompt)
        if not hasattr(self, 'gpt') or not self.gpt:
            logger.warning("No gpt instance found; returning default statement.")
            return "I don't know what to say."
        response = self.gpt.generate(
            prompt=self.story + prompt,
            max_tokens=50,
            model=self.model,
            stop_tokens=[]
        )
        return response

    def _extract_list_items(self, text: str) -> Dict[int, str]:
        """
        Extracts numbered list items from the provided text.
        
        This helper function searches for patterns like "1. Option text" in the text
        and returns a dictionary mapping each option number to its corresponding text.
        
        Args:
            text (str): The text containing numbered list items.
        
        Returns:
            Dict[int, str]: A dictionary where the key is the option number and the value is the option text.
        """
        import re
        pattern = r'(\d+)\.\s+(.*)'
        results = {}
        for match in re.finditer(pattern, text):
            num = int(match.group(1))
            content = match.group(2).strip()
            results[num] = content
        return results
