import asyncio
import aiohttp
import numpy as np
from transformers import AutoTokenizer
from typing import Any


class EndOfTurnModel:
    HF_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

    MAX_HISTORY = 4     # Maximum number of messages to consider in history
    TIMEOUT_MS = 500    # Timeout for API requests in milliseconds
    DEFAULT_THRESHOLD = 0.03    # Default probability threshold for determining end of turn

    def __init__(self, hostname: str, threshold: float = DEFAULT_THRESHOLD):
        self.hostname = hostname
        self.threshold = threshold
        self.baseurl = f"{self.hostname}/v1/completions"

        # Load the tokenizer for converting messages to the model's input format
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.HF_MODEL_ID,
            truncation_side="left",  # Truncate from the left if messages exceed max length
        )


    def _convert_messages_to_chatml(self, messages: list[dict[str, Any]]) -> str:
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
            add_special_tokens=False,     # Special tokens are handled by the template
            tokenize=False,
        )

        # Important: Remove the end-of-turn token from the very end of the latest utterance,
        # as we want the model to predict this token.
        eot_token = "<|im_end|>"
        last_eot_index = tokenized_convo.rfind(eot_token)
        if last_eot_index != -1:
            text = tokenized_convo[:last_eot_index]
            return text
        return tokenized_convo


    async def fetch_completion(self, prompt_text: str) -> dict[str, Any]:
        """
        Fetches completions from the VLLM server to get log probabilities for the next token.

        Args:
            prompt_text (str): The formatted conversation text (prompt for the model).

        Returns:
            dict[str, Any]: The JSON response from the server.
        """
        payload = {
            "model": self.HF_MODEL_ID,
            "prompt": prompt_text,
            "max_tokens": 1,  # We only need to predict the very next token.
            "logprobs": 20,   # Request log probabilities for the top N likely next tokens.
                              # This helps find EOT-related tokens even if they aren't the single most likely.
            "skip_special_tokens": False, # We need to see special tokens like <|im_end|>
        }
        headers = {"Content-Type": "application/json"}

        client_timeout = aiohttp.ClientTimeout(total=self.TIMEOUT_MS / 1000)

        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(self.baseurl, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error from VLLM server: {response.status}, {error_text}")
                    raise RuntimeError(f"VLLM server returned status {response.status}")
                return await response.json()
    

    def process_result(
        self, result: dict[str, Any], target_tokens: list[str] = ["<|im_end|>"]
    ) -> tuple[float, str]:
        """
        Processes the VLLM server's result to find the maximum probability
        among specified target tokens (e.g., EOT markers, punctuation).

        Args:
            result (dict[str, Any]): The JSON response from the VLLM server.
            target_tokens (list[str], optional): A list of tokens to look for.
                                                 Defaults to common EOT indicators.

        Returns:
            tuple[float, str]: A tuple containing the maximum probability found for any
                               target token, and the token itself. Returns (0.0, "") if
                               no target tokens are found or if there's an error.
        """
        try:
            # Navigate the JSON response to find the log probabilities of the top predicted tokens.
            # It should look like: {"token1": logprob1, "token2": logprob2, ...}
            top_logprobs = result["choices"][0]["logprobs"]["top_logprobs"][0]
            print(f"Top logprobs: {top_logprobs}")
            
            max_prob = 0.0
            best_token = ""

            for token_str, logprob in top_logprobs.items():
                # The tokenizer might add leading spaces to tokens, so strip them.
                stripped_token = token_str.strip()
                if stripped_token in target_tokens:
                    # Convert log probability back to probability (exp(logprob)).
                    prob = np.exp(logprob)
                    if prob > max_prob:
                        max_prob = prob
                        best_token = stripped_token
            
            return max_prob, best_token

        except (KeyError, IndexError, TypeError) as e:
            print(f"Error processing VLLM result: {type(e).__name__} - {e}. Result: {result}")
            return 0.0, ""


    async def predict_eot_prob(self, messages: list[dict[str, Any]]) -> float:
        """
        Predicts the probability that the current turn is complete.

        Args:
            messages (list[dict[str, Any]]): The list of messages.

        Returns:
            float: The probability (0.0 to 1.0) that the turn is complete.
        """
        # Consider only the most recent messages, up to MAX_HISTORY.
        truncated_messages = messages[-self.MAX_HISTORY :]

        # Convert messages to the ChatML string format required by the model.
        text_input = self._convert_messages_to_chatml(truncated_messages)

        # Fetch model completion (which includes log probabilities for the next token).
        result = await self.fetch_completion(text_input)

        # Process the result to extract the probability of an EOT-indicating token.
        eot_prob, _ = self.process_result(result)

        print(f"EOT Input: '{text_input}'")
        print(f"Predicted EOT Probability: {eot_prob:.4f}")

        return eot_prob


    async def predict_eot(self, messages: list[dict[str, Any]]) -> bool:
        """
        Predicts whether the current turn in the conversation is complete.

        Args:
            messages (list[dict[str, Any]]): The list of messages in the conversation.

        Returns:
            bool: True if the turn is predicted to be complete, False otherwise.
        """
        try:
            eot_prob = await asyncio.wait_for(
                self.predict_eot_prob(messages),
                timeout=self.TIMEOUT_MS / 1000,
            )
            return eot_prob >= self.threshold
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            print(f"EOT prediction failed due to {type(e).__name__}: {str(e)}. Defaulting to True.")
            return True


async def main():
    # Replace with your VLLM server's actual hostname
    model = EndOfTurnModel(hostname="http://localhost:8000")

    # Example conversations
    conversation1 = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "How may I help you today?"},
        {"role": "user", "content": "Can I have two chicken McNuggets and"}
    ]
    is_eot1 = await model.predict_eot(conversation1)
    print(f"Conversation 1 - Is EOT? {is_eot1}") # Expected: False

    conversation2 = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "How may I help you today?"},
        {"role": "user", "content": "I have a problem with my card."}
    ]
    is_eot2 = await model.predict_eot(conversation2)
    print(f"Conversation 2 - Is EOT? {is_eot2}") # Expected: True


if __name__ == "__main__":
    asyncio.run(main())
