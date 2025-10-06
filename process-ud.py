from semantic_turn_detection import *
import re

RX_SENTENCE = re.compile(r"^#\s+text\s+=\s+(.*)\s+(\w+)([.!?])$")


def main() -> None:
    with open("data.json", "r") as f:
        data = Data.model_validate_json(f.read(), strict=False)
    model = EndOfTurnModel(data.config)
    with open("../UD_Polish-PDB/pl_pdb-ud-train.conllu", "r") as f:
        start = time.perf_counter_ns()
        _max = 100
        _remaining = _max
        correct = 0
        convcount = 0
        for line in f:
            m = RX_SENTENCE.match(line.strip())
            if m is not None:
                convs = [
                    Conversation(
                        chat=[f"{m.group(1)} {m.group(2)}{m.group(3)}"], eou=True
                    ),
                    Conversation(chat=[f"{m.group(1)} {m.group(2)}"], eou=True),
                    Conversation(chat=[f"{m.group(1)}"], eou=False),
                ]
                for i in range(len(convs)):
                    is_eou, prob = model.predict_eou(convs[i].chat_with_roles())
                    print(
                        f'{i:03d} - Is EOU? {is_eou} - {prob:.4f} - Correct? {convs[i].eou == is_eou} - "{convs[i].chat[-1]}"'
                    )
                    if is_eou == convs[i].eou:
                        correct += 1
                convcount += 3
                _remaining -= 1
                if _remaining < 1:
                    break
        end = time.perf_counter_ns()
        print(
            f"Inference took {(end - start) / convcount / 1_000_000:.2f} ms per conversation."
            f"Correct: {correct} / {_max*3}"
        )


if __name__ == "__main__":
    main()
