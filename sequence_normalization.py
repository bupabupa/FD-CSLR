import subprocess
import tempfile
from pathlib import Path


def _write_ctm(path, sequences):
    with open(path, "w", encoding="utf-8") as destination:
        for sample_index, sequence in enumerate(sequences):
            sample_id = "s{:06d}".format(sample_index)
            for token_index, token in enumerate(sequence):
                destination.write(
                    "{} 1 {:.2f} {:.2f} {}\n".format(
                        sample_id,
                        token_index * 0.01,
                        (token_index + 1) * 0.01,
                        token,
                    )
                )


def _read_ctm(path, sample_count):
    sequences = [[] for _ in range(sample_count)]
    with open(path, encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if len(fields) < 5:
                continue
            if not fields[0].startswith("s"):
                raise ValueError("unexpected normalized CTM sample id")
            sample_index = int(fields[0][1:])
            if sample_index < 0 or sample_index >= sample_count:
                raise ValueError("normalized CTM sample id is out of range")
            sequences[sample_index].append(fields[4])
    return sequences


def normalize_token_sequences(sequences, preprocess_script):
    sequences = [
        [str(token) for token in sequence]
        for sequence in sequences
    ]
    preprocess_script = Path(preprocess_script).expanduser().resolve()
    if not preprocess_script.is_file():
        raise FileNotFoundError(
            "preprocess script does not exist: {}".format(preprocess_script)
        )
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        hypothesis = directory / "hypothesis.ctm"
        temporary = directory / "temporary.ctm"
        normalized = directory / "normalized.ctm"
        _write_ctm(hypothesis, sequences)
        subprocess.run(
            [
                "bash",
                str(preprocess_script),
                str(hypothesis),
                str(temporary),
                str(normalized),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        return _read_ctm(normalized, len(sequences))


def edit_distance(reference, hypothesis):
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    previous = list(range(columns))
    for row in range(1, rows):
        current = [row] + [0] * (columns - 1)
        for column in range(1, columns):
            substitution = (
                0
                if reference[row - 1] == hypothesis[column - 1]
                else 1
            )
            current[column] = min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + substitution,
            )
        previous = current
    return int(previous[-1])


def aggregate_wer(references, hypotheses):
    edits = 0
    reference_length = 0
    for reference, hypothesis in zip(references, hypotheses):
        edits += edit_distance(reference, hypothesis)
        reference_length += len(reference)
    return {
        "edits": edits,
        "reference_length": reference_length,
        "wer": (
            100.0 * edits / reference_length
            if reference_length
            else 0.0
        ),
    }


def normalized_wer(references, hypotheses, preprocess_script):
    normalized_references = normalize_token_sequences(
        references,
        preprocess_script,
    )
    normalized_hypotheses = normalize_token_sequences(
        hypotheses,
        preprocess_script,
    )
    return aggregate_wer(normalized_references, normalized_hypotheses)
