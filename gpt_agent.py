# gpt_agent.py
import random
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class GptAgentMixin:
    """
    Provides GPT-based logic for action or statement generation.
    """
    def get_gpt_action(self, prompt: str, argmax: bool = False) -> int:
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
            logger.warning("No probabilities from GPT; random choice fallback.")
            if action_dict:
                return int(random.choice(list(action_dict.keys())))
            return None

        if argmax:
            return int(max(option_probs, key=option_probs.get))

        rand_val = random.random()
        cumulative = 0.0
        for act, pr in option_probs.items():
            cumulative += pr
            if rand_val <= cumulative:
                return int(act)
        return int(random.choice(list(option_probs.keys())))

    def _get_gpt_statement(self, prompt: str) -> str:
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
        import re
        pattern = r'(\d+)\.\s+(.*)'
        results = {}
        for match in re.finditer(pattern, text):
            num = int(match.group(1))
            content = match.group(2).strip()
            results[num] = content
        return results
