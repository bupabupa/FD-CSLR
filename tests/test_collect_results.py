import json
import tempfile
from pathlib import Path

from scripts.collect_results import collect_results, markdown_report


def _write_training(run_dir, checkpoint_dev, final_dev=None, final_test=None):
    run_dir.mkdir(parents=True)
    logs = []
    for epoch in range(40):
        logs.append("Mean training loss: 1.0.")
        logs.append("Dev WER: 40.00%")
    run_dir.joinpath("log.txt").write_text("\n".join(logs), encoding="utf-8")
    run_dir.joinpath(
        "dev_{:.2f}_epoch39_model.pt".format(checkpoint_dev)
    ).write_bytes(b"checkpoint")
    if final_dev is not None:
        _write_prediction(
            run_dir / "final" / "predictions-dev.jsonl",
            reference=[1, 2, 3],
            hypothesis=[1, 2, 3][: 3 - final_dev],
        )
    if final_test is not None:
        _write_prediction(
            run_dir / "final" / "predictions-test.jsonl",
            reference=[4, 5, 6, 7],
            hypothesis=[4, 5, 6, 7][: 4 - final_test],
        )


def _write_prediction(path, reference, hypothesis):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        output.write(json.dumps({
            "sample_id": "sample-0",
            "signer": "Signer01",
            "reference_ids": reference,
            "hypothesis_ids": hypothesis,
            "reference_glosses": [str(value) for value in reference],
            "hypothesis_glosses": [str(value) for value in hypothesis],
        }))
        output.write("\n")


def _write_calibration(path, best_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"best_loss": best_loss}),
        encoding="utf-8",
    )


def test_collect_results_selects_by_dev_only():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_training(root / "baseline", 30.0, final_dev=0, final_test=3)
        _write_training(root / "candidate", 31.0, final_dev=1, final_test=0)
        result = collect_results([
            "baseline={}".format(root / "baseline"),
            "candidate={}".format(root / "candidate"),
        ])

    assert result["selected_by_dev"] == ["baseline"]
    assert result["selected_dev_wer"] == 0.0
    assert result["runs"]["candidate"]["final_test"]["wer"] == 0.0
    report = markdown_report(result)
    assert "Selected by Dev (FDGC NLL tie-break): baseline" in report
    assert "D_test is reported but never used for selection" in report


def test_collect_results_marks_incomplete_and_ties():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_training(root / "first", 30.0, final_dev=0)
        _write_training(root / "second", 30.0, final_dev=0)
        incomplete = root / "incomplete"
        incomplete.mkdir()
        incomplete.joinpath(
            "dev_20.00_epoch3_model.pt"
        ).write_bytes(b"checkpoint")
        incomplete.joinpath("log.txt").write_text(
            "Mean training loss: 1.0.\nDev WER: 20.00%\n",
            encoding="utf-8",
        )
        no_final = root / "no_final"
        _write_training(no_final, 10.0)
        result = collect_results([
            "first={}".format(root / "first"),
            "second={}".format(root / "second"),
            "incomplete={}".format(incomplete),
            "no_final={}".format(no_final),
        ])

    assert result["selected_by_dev"] == ["first", "second"]
    assert not result["runs"]["incomplete"]["selection_eligible"]
    assert not result["runs"]["no_final"]["selection_eligible"]


def test_explicit_prediction_dir_wins_over_newer_nested_result():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run_dir = root / "candidate"
        _write_training(run_dir, 30.0, final_dev=0)
        _write_prediction(
            run_dir / "fdgc_ctc" / "final" / "predictions-dev.jsonl",
            reference=[1, 2, 3],
            hypothesis=[1],
        )
        result = collect_results([
            "candidate={}@{}".format(run_dir, run_dir / "final")
        ])

    assert result["runs"]["candidate"]["final_dev"]["wer"] == 0.0


def test_fdgc_dev_tie_uses_explicit_calibration_nll():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first"
        second = root / "second"
        _write_training(first, 30.0, final_dev=0)
        _write_training(second, 30.0, final_dev=0)
        _write_calibration(first / "fdgc" / "fdgc_metrics.json", 2.0)
        _write_calibration(second / "fdgc" / "fdgc_metrics.json", 1.0)
        result = collect_results([
            "first={}@{}@{}".format(
                first,
                first / "final",
                first / "fdgc" / "fdgc_metrics.json",
            ),
            "second={}@{}@{}".format(
                second,
                second / "final",
                second / "fdgc" / "fdgc_metrics.json",
            ),
        ])

    assert result["dev_tied"] == ["first", "second"]
    assert result["selected_by_dev"] == ["second"]
    assert result["selected_dev_wer"] == 0.0


def test_non_fdgc_tie_remains_explicitly_tied():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_training(root / "first", 30.0, final_dev=0)
        _write_training(root / "second", 30.0, final_dev=0)
        result = collect_results([
            "first={}".format(root / "first"),
            "second={}".format(root / "second"),
        ])

    assert result["dev_tied"] == ["first", "second"]
    assert result["selected_by_dev"] == ["first", "second"]


if __name__ == "__main__":
    test_collect_results_selects_by_dev_only()
    test_collect_results_marks_incomplete_and_ties()
    test_explicit_prediction_dir_wins_over_newer_nested_result()
    test_fdgc_dev_tie_uses_explicit_calibration_nll()
    test_non_fdgc_tie_remains_explicitly_tied()
    print("collect results tests passed")
