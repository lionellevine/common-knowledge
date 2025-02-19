"""
gpt.py
-------
This module implements a GPT wrapper class that interfaces with the OpenAI API.
It provides methods for generating text completions and for computing probability 
distributions over a set of options using GPT-based models.
"""

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
        Initializes the GPT instance.
        
        This method loads environment variables (including the API key),
        initializes the OpenAI API client, sets up a GPT-2 tokenizer for text processing,
        and defines a mapping of shorthand model names to full model identifiers.
        
        Args:
            temperature (float): Controls randomness in generated text. Lower values make output more deterministic.
        
        Raises:
            ValueError: If the OPENAI_API_KEY is not found in the environment.
        """
        logger.info("Configuring GPT instance...")
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not provided in the .env file")

        # Initialize the OpenAI client with the provided API key.
        self.client = OpenAI(api_key=api_key)
        # Load a pre-trained GPT-2 tokenizer for text processing.
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.temperature = temperature

        # Mapping of shorthand model names to the corresponding full OpenAI model identifiers.
        self.chat_models = {
            "3.5": "gpt-3.5-turbo",
            "4":   "gpt-4",
            "4o":  "gpt-4o",
            "4o-2024-11-20":    "gpt-4o-2024-11-20",
            "4o-2024-08-06":    "gpt-4o-2024-08-06",
            "4o-mini":          "gpt-4o-mini",
            "4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
            "o1-mini":          "o1-mini",
            "o1-mini-2024-09-12": "o1-mini-2024-09-12",
        }

    def generate(self, prompt: str, max_tokens: int, model: str, stop_tokens=None) -> str:
        """
        Generates a text completion from the OpenAI API based on the provided prompt.
        
        Args:
            prompt (str): The input prompt for text generation.
            max_tokens (int): The maximum number of tokens to generate.
            model (str): The shorthand model identifier to use (must be defined in self.chat_models).
            stop_tokens (list, optional): Tokens at which generation should stop.
        
        Returns:
            str: The generated text, with newlines replaced by spaces.
        
        Raises:
            ValueError: If the specified model is not supported.
        
        Behavior:
            - If the model is not recognized, a ValueError is raised.
            - On successful API call, the generated text is stripped and returned.
            - In case of an APIError, the method waits 30 seconds and retries.
        """
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
                    # Provide context that this is a fictional game.
                    {"role": "system", "content": "This is a fictional game played for fun."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                stop=stop_tokens,
            )
            # Process the response: remove extra whitespace and newlines.
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
        """
        Computes probabilities for a set of options by sampling responses from GPT.
        
        This function sends the provided prompt to the OpenAI API repeatedly (up to max_iters)
        and tallies occurrences of each option (as defined in option_dict) in the generated responses.
        The result is a probability distribution over the options.
        
        Args:
            prompt (str): The prompt containing options for which probabilities are to be computed.
            option_dict (dict): A mapping of option numbers to their corresponding option texts.
            model (str): The shorthand model identifier to use.
            max_tokens (int): Maximum tokens to generate per API call.
            n (int): Number of completions to request per iteration.
            max_iters (int): Maximum number of iterations to try if no votes are tallied.
        
        Returns:
            dict: A dictionary mapping each option number to its computed probability.
        
        Raises:
            ValueError: If the specified model is not supported.
        """
        logger.debug("Sending get_probs request to OpenAI with model=%s", model)

        if model not in self.chat_models:
            raise ValueError(f"Unrecognized or unsupported chat model: {model}")

        full_model_name = self.chat_models[model]
        # Initialize vote counts for each option.
        votes = {k: 0 for k in option_dict.keys()}
        iters = 0

        # Sample responses until at least one vote is tallied or maximum iterations reached.
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
                # Check each completion for occurrences of option identifiers.
                for choice in response.choices:
                    text_out = choice.message.content
                    for num, act_text in option_dict.items():
                        if str(num) in text_out or act_text in text_out:
                            votes[num] += 1

            except APIError as e:
                logger.error("API error on get_probs: %s. Retrying in 30s...", e)
                time.sleep(30)
            iters += 1

        # If no votes were tallied after max_iters, assign equal probability to all options.
        if sum(votes.values()) == 0:
            votes = {k: 1 for k in option_dict.keys()}

        total = sum(votes.values())
        # Convert vote counts to probabilities.
        return {k: v / total for k, v in votes.items()}
