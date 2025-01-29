# gpt.py
import openai
import os
import time
import random
import re
import math
import numpy as np

from dotenv import load_dotenv
from transformers import GPT2Tokenizer
from transformers.utils import logging as transformers_logging
import logging

logger = logging.getLogger(__name__)

class GPT:
    def __init__(self, temperature: float = 1.0):
        """
        Creates a GPT wrapper to handle OpenAI API calls.
        """
        logger.info("Configuring GPT instance...")
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not provided in the .env file")

        # Use a GPT2 tokenizer for any prompt trimming logic
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.temperature = temperature

    def tokenize(self, prompt: str):
        return self.tokenizer(prompt)['input_ids']

    def generate(self, prompt: str, max_tokens: int, model: str, stop_tokens=None) -> str:
        """
        Generates text from a given model string.

        1. We have a dictionary of recognized "chat-based" models:
           - "3.5", "4", "4o", "4o-2024-11-20", "4o-2024-08-06",
             "4o-mini", "4o-mini-2024-07-18", "o1-mini", "o1-mini-2024-09-12",
             etc. 
           => These all use openai.chat.completions.create(...).
        
        2. Otherwise, we assume an older text model (e.g. "curie") and
           use openai.completions.create(...).

        :param prompt: The input text to be sent to the model
        :param max_tokens: Maximum tokens to generate
        :param model: The short string describing which model to call
        :param stop_tokens: A list of stop sequences
        :return: Generated text
        """
        stop_tokens = stop_tokens or []
        prompt = self.trim_prompt(prompt)
        logger.debug("Sending generate request to OpenAI with model=%s", model)

        try:
            # 1) We define recognized chat-models in a dictionary
            #    so you can easily add or rename them in one place.
            chat_models = {
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

            # 2) If 'model' is in chat_models, we do a chat call
            if model in chat_models:
                full_model_name = chat_models[model]
                logger.info(f"Using chat-based model: {full_model_name}")

                response = openai.chat.completions.create(
                    model=full_model_name,
                    messages=[
                        {'role': 'system', 'content': 'This is a fictional game played for fun.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    stop=stop_tokens
                )
                text = response.choices[0].message.content

            # 3) Otherwise, fallback to older text-based approach
            else:
                logger.info(f"Using older text-based approach for model={model}")
                model_dict = {
                    "ada":         "text-ada-001",
                    "babbage":     "text-babbage-001",
                    "curie":       "text-curie-001",
                    "davinci-001": "text-davinci-001",
                    "davinci-002": "text-davinci-002",
                }
                full_model_name = model_dict.get(model, "text-davinci-002")

                response = openai.Completion.create(
                    model=full_model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    stop=stop_tokens
                )
                text = response.choices[0].text

            # Clean up the returned text
            text = text.strip().replace('\n', ' ')
            if len(text) < 2:
                raise ValueError("GPT returned an empty message.")

            return text

        except Exception as e:
            logger.error("API error on generate: %s. Retrying in 30s...", e)
            time.sleep(30)
            return self.generate(prompt, max_tokens, model, stop_tokens)

    def get_probs(self, prompt: str, option_dict: dict, model: str,
                  max_tokens: int = 8, n: int = 1, max_iters: int = 5) -> dict:
        """
        Returns a dictionary {option_number: probability} for each item in option_dict,
        representing which choice the model is most likely to pick. 
        This is a heuristic approach:
        
        - If 'model' is recognized as a chat model, we do repeated calls to openai.chat.completions.create
          with n completions each time, scanning for references to each option.
        - Otherwise, we try old text-based approach with 'logprobs=20'.
        
        NOTE: This is not a perfect method of extracting true probabilities but a rough approach.
        """
        prompt = self.trim_prompt(prompt)
        logger.debug("Sending get_probs request to OpenAI with model=%s", model)

        votes = {k: 0 for k in option_dict.keys()}

        # Chat-based models known
        chat_models = {
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

        try:
            if model in chat_models:
                full_model_name = chat_models[model]
                iters = 0
                # We'll do up to 'max_iters' attempts to get a mention of any option
                while sum(votes.values()) == 0 and iters < max_iters:
                    response = openai.chat.completions.create(
                        model=full_model_name,
                        messages=[
                            {'role': 'system', 'content': 'This is a fictional game played for fun.'},
                            {'role': 'user', 'content': prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=max_tokens,
                        n=n
                    )
                    # For each returned choice, check which option it mentions
                    for choice in response.choices:
                        completion = choice.message.content
                        for num, action_text in option_dict.items():
                            # If the model references e.g. "1" or the text itself
                            if str(num) in completion or action_text in completion:
                                votes[num] += 1
                    iters += 1

                # If after max_iters we have no picks, do uniform distribution
                if sum(votes.values()) == 0:
                    votes = {k: 1 for k in option_dict.keys()}

            else:
                # For older text-based models with logprobs
                model_dict = {
                    "ada":         "text-ada-001",
                    "babbage":     "text-babbage-001",
                    "curie":       "text-curie-001",
                    "davinci-001": "text-davinci-001",
                    "davinci-002": "text-davinci-002",
                }
                full_model_name = model_dict.get(model, "text-davinci-002")

                logprobs_resp = openai.Completion.create(
                    model=full_model_name,
                    prompt=self.tokenize(prompt),
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    logprobs=20
                )
                top_logprobs = logprobs_resp.choices[0].logprobs.top_logprobs[0]

                # For each token in top_logprobs, if it's a digit in option_dict, we exponentiate
                for token, lp in top_logprobs.items():
                    if token.isdigit():
                        num_token = int(token)
                        if num_token in option_dict:
                            votes[num_token] = math.exp(lp)

            # Normalize to probabilities
            total = sum(votes.values())
            if total == 0:
                # fallback uniform distribution
                return {k: 1 / len(votes) for k in votes}
            return {k: v / total for k, v in votes.items()}

        except Exception as e:
            logger.error("API error on get_probs: %s. Retrying in 30s...", e)
            time.sleep(30)
            return self.get_probs(prompt, option_dict, model, max_tokens, n, max_iters)

    def trim_prompt(self, prompt: str) -> str:
        """
        Truncates the prompt if it is too long for the model context (approx).
        For older text-based models, we limit to ~4097 tokens. 
        For some large context models, you might increase that limit.
        """
        transformers_logging.set_verbosity(40)
        delete_turn_num = 0
        # We'll assume a ~4097 token context limit for safety,
        # though some "gpt-4" or "gpt-4o" variants might allow more.
        # Modify as needed for your real context limit.
        while len(self.tokenize(prompt)) > (4097 - 50):
            # We'll remove older "Turn #X" sections if found
            delete_turn_num += 1
            start_pos = prompt.find(f"Turn #{delete_turn_num}")
            if start_pos == -1:
                break
            end_pos = prompt.find(f"Turn #{delete_turn_num + 1}")
            if end_pos == -1:
                end_pos = len(prompt)
            prompt = prompt[:start_pos] + "...\n\n" + prompt[end_pos:]

        # Remove repeated ellipses
        while "...\n\n...\n\n" in prompt:
            prompt = prompt.replace("...\n\n...\n\n", "...\n\n")

        return prompt
 
