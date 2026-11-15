import os
from pathlib import Path
import shutil
import subprocess
import sys

from .python_wer_evaluation import wer_calculation


ROOT = Path(__file__).resolve().parents[2]
SCLITE = ROOT / "software" / "sclite"


def _run(command, capture_output=False):
    return subprocess.run(
        [str(value) for value in command],
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _sort_stm_by_document(source, destination):
    with open(source, encoding="utf-8") as input_file:
        lines = input_file.readlines()

    def sort_key(line):
        fields = line.split()
        document = fields[0] if fields else ""
        return document, line

    with open(destination, "w", encoding="utf-8") as output_file:
        for line in sorted(lines, key=sort_key):
            output_file.write(line.rstrip("\r\n") + "\n")


def _parse_sclite_wer(output):
    for line in output.splitlines():
        if "Error" not in line:
            continue
        return float(line.split("=")[1].split("%")[0].strip())
    raise RuntimeError("sclite output does not contain an Error summary")


def evaluate(
    prefix="./",
    mode="dev",
    evaluate_dir=None,
    evaluate_prefix=None,
    output_file=None,
    output_dir=None,
    python_evaluate=False,
    triplet=False,
):
    if evaluate_dir is None or evaluate_prefix is None or output_file is None:
        raise ValueError(
            "evaluate_dir, evaluate_prefix and output_file are required"
        )
    prefix = Path(prefix)
    evaluate_dir = Path(evaluate_dir)
    hypothesis = prefix / output_file
    temporary_ctm = prefix / "tmp.ctm"
    normalized_ctm = prefix / "tmp2.ctm"
    temporary_stm = prefix / "tmp.stm"
    merged_output = prefix / f"out.{output_file}"
    if not hypothesis.is_file():
        raise FileNotFoundError(f"hypothesis CTM does not exist: {hypothesis}")

    _run(
        [
            "bash",
            evaluate_dir / "preprocess.sh",
            hypothesis,
            temporary_ctm,
            normalized_ctm,
        ]
    )
    if not normalized_ctm.is_file():
        raise RuntimeError("preprocess.sh did not create the normalized CTM")
    ground_truth = evaluate_dir / f"{evaluate_prefix}-{mode}.stm"
    _sort_stm_by_document(ground_truth, temporary_stm)
    _run(
        [
            sys.executable,
            evaluate_dir / "mergectmstm.py",
            normalized_ctm,
            temporary_stm,
        ]
    )
    shutil.copyfile(normalized_ctm, merged_output)

    if python_evaluate:
        result = wer_calculation(ground_truth, merged_output)
        if triplet:
            wer_calculation(
                ground_truth,
                merged_output,
                Path(str(merged_output).replace(".ctm", "-conv.ctm")),
            )
        return result

    report_command = [
        SCLITE,
        "-h",
        merged_output,
        "ctm",
        "-r",
        temporary_stm,
        "stm",
        "-f",
        "0",
        "-o",
        "sgml",
        "sum",
        "rsum",
        "pra",
    ]
    if output_dir is not None:
        report_directory = prefix / output_dir
        report_directory.mkdir(parents=True, exist_ok=True)
        report_command.extend(["-O", str(report_directory) + os.sep])
    _run(report_command)

    detail = _run(
        [
            SCLITE,
            "-h",
            merged_output,
            "ctm",
            "-r",
            temporary_stm,
            "stm",
            "-f",
            "0",
            "-o",
            "dtl",
            "stdout",
        ],
        capture_output=True,
    )
    return _parse_sclite_wer(detail.stdout + "\n" + detail.stderr)
