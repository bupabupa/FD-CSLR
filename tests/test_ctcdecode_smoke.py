import torch
from ctcdecode import CTCBeamDecoder


def main():
    labels = ["_", "a", "b"]
    logits = torch.full((1, 4, 3), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 0] = 10.0
    logits[0, 2, 2] = 10.0
    logits[0, 3, 0] = 10.0
    probabilities = logits.softmax(dim=-1)
    decoder = CTCBeamDecoder(
        labels,
        beam_width=8,
        blank_id=0,
        num_processes=1,
        log_probs_input=False,
    )
    beams, _, _, lengths = decoder.decode(
        probabilities,
        torch.tensor([4], dtype=torch.int32),
    )
    decoded = [
        labels[index]
        for index in beams[0, 0, : lengths[0, 0]].tolist()
    ]
    assert decoded == ["a", "b"], decoded
    print("ctcdecode smoke passed:", decoded)


if __name__ == "__main__":
    main()
