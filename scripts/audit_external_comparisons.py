import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.external_comparison_audit import (  # noqa: E402
    load_and_validate,
    write_template,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--split-sha256", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--write-template", default=None)
    args = parser.parse_args()

    if args.write_template:
        path = write_template(args.write_template)
        print("template: {}".format(path.resolve()))
        return
    if not args.input or not args.split_sha256:
        raise ValueError("--input and --split-sha256 are required for audit")
    report = load_and_validate(args.input, args.split_sha256)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
