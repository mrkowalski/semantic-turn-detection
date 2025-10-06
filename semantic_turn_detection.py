import os
import time
from itertools import cycle
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
DEBUG = int(os.environ.get("DEBUG", "0")) == 1


class Config(BaseModel):
    model_id: str
    eou_tokens: list[str]
    default_eou_token: int
    top_k: int
    default_threshold: float


class Conversation(BaseModel):
    chat: list[str]
    eou: bool

    def chat_with_roles(self):
        user_assistant = cycle(["user", "assistant"])
        return [{"role": next(user_assistant), "content": text} for text in self.chat]


class Data(BaseModel):
    config: Config
    conversations: list[Conversation]


class EndOfTurnModel:
    def __init__(self, config: Config):
        self._config = config

        print(
            f"Loading model {self._config.model_id} for local CPU inference. This may take a while..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._config.model_id,
            truncation_side="left",  # Truncate from the left if messages exceed max length
            token=os.environ["HF_TOKEN"],
        )
        self.model = AutoModelForCausalLM.from_pretrained(self._config.model_id)
        self.model.to("cpu")  # Ensure model is on CPU
        self.model.eval()  # Set model to evaluation mode
        print("Model loaded successfully.")
        print("\n" + "=" * 70 + "\n")

    def _convert_messages_to_chatml(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """
        Converts a list of messages into a single string in ChatML format.
        The EOT token (<|im_end|>) is removed from the last utterance, as the model
        is expected to predict its presence.

        Args:
            messages (list[dict[str, Any]]): A list of message dictionaries,
                                             each with "role" and "content".

        Returns:
            str: A string representing the conversation in ChatML format.
        """
        if not messages:
            return ""

        # Apply the chat template to format messages (e.g., adding special tokens like <|im_start|>)
        tokenized_convo = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,  # Do not add an assistant prompt for generation
            add_special_tokens=False,  # Special tokens are handled by the template
            tokenize=False,
        )

        # Important: Remove the end-of-turn token from the very end of the latest utterance,
        # as we want the model to predict this token.
        last_eou_index = tokenized_convo.rfind(
            self._config.eou_tokens[self._config.default_eou_token]
        )
        if last_eou_index != -1:
            text = tokenized_convo[:last_eou_index]
            return text
        return tokenized_convo

    def get_next_token_logprobs(self, prompt_text: str) -> dict[str, float]:
        """
        Performs local inference to get log probabilities for the next token.

        Args:
            prompt_text (str): The formatted conversation text.

        Returns:
            dict[str, float]: Dictionary mapping tokens to their log probabilities.
        """
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).to("cpu")

        with torch.no_grad():
            outputs = self.model(**inputs)

        next_token_logits = outputs.logits[
            0, -1, :
        ]  # Batch size 1, last token position
        log_softmax_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        top_logprobs_vals, top_logprobs_ids = torch.topk(
            log_softmax_probs, self._config.top_k
        )

        top_logprobs_dict = {}
        for i in range(self._config.top_k):
            token_id = top_logprobs_ids[i].item()
            # Decode the token ID to its string representation
            token_str = self.tokenizer.decode([token_id])
            logprob_val = top_logprobs_vals[i].item()
            top_logprobs_dict[token_str] = logprob_val

        # Directly return the dictionary of token -> logprob
        return top_logprobs_dict

    def process_result(
        self,
        top_logprobs: dict[str, float],
        target_tokens: Sequence[str],
    ) -> tuple[float, str]:
        """
        Processes the model's output to find the maximum probability
        among specified target tokens (e.g., EOT markers, punctuation).

        Args:
            top_logprobs (dict[str, float]): Dictionary mapping tokens to their log probabilities.
            target_tokens (list[str], optional): A list of tokens to look for.

        Returns:
            tuple[float, str]: A tuple containing the maximum probability found for any
                              target token, and the token itself. Returns (0.0, "") if
                              no target tokens are found or if there's an error.
        """
        try:
            if DEBUG:
                token_probs = {
                    token: f"{np.exp(logprob):.4f}"
                    for token, logprob in top_logprobs.items()
                }
                print(f"Token probabilities: {token_probs}")

            # Check for target tokens (like <|im_end|>)
            max_prob = 0.0
            best_token = ""

            for token_str, logprob in top_logprobs.items():
                if token_str in target_tokens:
                    # Convert log probability back to probability (exp(logprob)).
                    prob = np.exp(logprob)
                    if prob > max_prob:
                        max_prob = prob
                        best_token = token_str

            # If we found a target token, show its probability
            if DEBUG and best_token:
                print(
                    f"Found target token: '{best_token}' with probability: {max_prob:.4f}"
                )

            return max_prob, best_token

        except (KeyError, TypeError) as e:
            print(f"Error processing result: {type(e).__name__} - {e}")
            return 0.0, ""

    def predict_eou_prob(self, messages: Sequence[Mapping[str, Any]]) -> float:
        """
        Predicts the probability that the current turn is complete.

        Args:
            messages (list[dict[str, Any]]): The list of messages.

        Returns:
            float: The probability (0.0 to 1.0) that the turn is complete.
        """

        # Convert messages to the ChatML string format required by the model.
        text_input = self._convert_messages_to_chatml(messages)

        if DEBUG:
            print(f"EOT Input: '...{text_input}'")

        # Get log probabilities for the next token
        top_logprobs = self.get_next_token_logprobs(text_input)

        # Process the result to extract the probability of an EOT-indicating token.
        eou_prob, _ = self.process_result(top_logprobs, self._config.eou_tokens)

        if DEBUG:
            print(f"EOT Probability: {eou_prob:.4f}")

        return eou_prob

    def predict_eou(self, messages: Sequence[Mapping[str, Any]]) -> tuple[bool, float]:
        """
        Predicts whether the current turn in the conversation is complete.

        Args:
            messages (list[dict[str, Any]]): The list of messages in the conversation.

        Returns:
            bool: True if the turn is predicted to be complete, False otherwise.
        """
        eou_prob = self.predict_eou_prob(messages)
        return eou_prob >= self._config.default_threshold, eou_prob
