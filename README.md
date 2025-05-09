# Semantic Turn Detection

This script uses a Language Model to predict whether the current turn in a conversation is complete. It queries a VLLM server hosting the specified model to get the probability of an end-of-turn token.

## Prerequisites

1.  **Python Environment**: Ensure you have Python 3.7+ installed.
2.  **VLLM Server**: You need to host the language model (`HuggingFaceTB/SmolLM2-360M-Instruct`) on a VLLM server.
    *   Please refer to the [official VLLM documentation](https://docs.vllm.ai/) for instructions on how to set up and run a VLLM server with the required model.
3.  **Python Dependencies**: Install the necessary Python libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Setup

1.  **Update Hostname**:
    Open the `semantic-turn-detection.py` script.
    In the `main` function, locate the following line:
    ```python
    model = EndOfTurnModel(hostname="http://localhost:8000")
    ```
    Change `"http://localhost:8000"` to the actual address of your VLLM server.

## Running the Script

Once the VLLM server is running and the hostname in the script is updated, you can run the script directly:

```bash
python semantic-turn-detection.py
```

The script will output the end-of-turn prediction for the example conversations defined in the `main` function.

## How it Works

The `EndOfTurnModel` class handles the interaction with the VLLM server.
- It takes a list of messages representing the conversation history.
- It formats these messages into the ChatML format expected by the model.
- It sends a request to the VLLM server's completions endpoint, asking for the log probabilities of the next potential tokens.
- It then processes the server's response to find the probability of an end-of-turn token (e.g., `<|im_end|>`).
- If this probability is above a predefined threshold, the turn is considered complete.

Key parameters in `EndOfTurnModel`:
- `HF_MODEL_ID`: The Hugging Face model identifier.
- `MAX_HISTORY`: The maximum number of recent messages to consider for prediction.
- `TIMEOUT_MS`: Timeout for API requests to the VLLM server.
- `DEFAULT_THRESHOLD`: The probability threshold used to determine if a turn is complete.