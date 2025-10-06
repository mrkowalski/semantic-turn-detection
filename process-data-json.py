from semantic_turn_detection import *


def main():
    with open("data.json", "r") as f:
        data = Data.model_validate_json(f.read(), strict=False)
    model = EndOfTurnModel(data.config)
    start = time.perf_counter_ns()
    for i in range(len(data.conversations)):
        is_eou, prob = model.predict_eou(data.conversations[i].chat_with_roles())
        print(
            f'{i:03d} - Is EOU? {is_eou} - {prob:.4f} - Correct? {data.conversations[i].eou == is_eou} - "{data.conversations[i].chat[-1]}"'
        )
    end = time.perf_counter_ns()
    print(
        f"Inference took {(end - start) / len(data.conversations) / 1_000_000:.2f} ms per conversation."
    )


if __name__ == "__main__":
    main()
