# gpt.py
import os
import time
import logging

from openai import OpenAI, APIError
from dotenv import load_dotenv
from transformers import GPT2Tokenizer
from transformers.utils import logging as transformers_logging

logger = logging.getLogger(__name__)

class GPT:
    def __init__(self, temperature: float = 1.0):
        """
        Creates a GPT wrapper to handle OpenAI API calls.
        """
        logger.info("Configuring GPT instance...")
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not provided in the .env file")

        self.client = OpenAI(api_key=api_key)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.temperature = temperature

        self.chat_models = {
            "3.5": "gpt-3.5-turbo",
            "4":   "gpt-4",
            "4o":  "gpt-4o",
            "4o-2024-11-20":    "gpt-4o-2024-11-20",
            "4o-2024-08-06":    "gpt-4o-2024-08-06",
            "4o-mini":          "gpt-4o-mini",
            "4o-mini-2024-07-18":"gpt-4o-mini-2024-07-18",
            "o1-mini":          "o1-mini",
            "o1-mini-2024-09-12":"o1-mini-2024-09-12",
        }

    def generate(self, prompt: str, max_tokens: int, model: str, stop_tokens=None) -> str:
        stop_tokens = stop_tokens or []
        logger.debug("Sending generate request to OpenAI with model=%s", model)

        if model not in self.chat_models:
            raise ValueError(f"Unrecognized or unsupported chat model: {model}")

        full_model_name = self.chat_models[model]

        try:
            logger.info(f"Using chat-based model: {full_model_name}")
            response = self.client.chat.completions.create(
                model=full_model_name,
                messages=[
                    {"role": "system", "content": "This is a fictional game played for fun."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                stop=stop_tokens,
            )
            text = response.choices[0].message.content.strip().replace('\n', ' ')
            if len(text) < 2:
                raise ValueError("GPT returned an empty message.")
            return text

        except APIError as e:
            logger.error("API error on generate: %s. Retrying in 30s...", e)
            time.sleep(30)
            return self.generate(prompt, max_tokens, model, stop_tokens)

    def get_probs(self, prompt: str, option_dict: dict, model: str,
                  max_tokens: int = 8, n: int = 1, max_iters: int = 5) -> dict:
        logger.debug("Sending get_probs request to OpenAI with model=%s", model)

        if model not in self.chat_models:
            raise ValueError(f"Unrecognized or unsupported chat model: {model}")

        full_model_name = self.chat_models[model]
        votes = {k: 0 for k in option_dict.keys()}
        iters = 0

        while sum(votes.values()) == 0 and iters < max_iters:
            try:
                response = self.client.chat.completions.create(
                    model=full_model_name,
                    messages=[
                        {"role": "system", "content": "This is a fictional game played for fun."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    n=n
                )
                for choice in response.choices:
                    text_out = choice.message.content
                    for num, act_text in option_dict.items():
                        if str(num) in text_out or act_text in text_out:
                            votes[num] += 1

            except APIError as e:
                logger.error("API error on get_probs: %s. Retrying in 30s...", e)
                time.sleep(30)
            iters += 1

        if sum(votes.values()) == 0:
            votes = {k: 1 for k in option_dict.keys()}

        total = sum(votes.values())
        return {k: v / total for k, v in votes.items()}
