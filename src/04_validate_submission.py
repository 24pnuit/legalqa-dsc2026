"""Validate a LegalQA prediction against the exact question ID set."""

from __future__ import annotations

import argparse
from pathlib import Path

from submission_utils import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/raw/public-official.json"),
        help="Question JSON defining the required IDs.",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=Path("data/submission.json"),
        help="Prediction JSON to validate.",
    )
    args = parser.parse_args()
    try:
        count = validate_submission(args.questions, args.pred)
    except ValueError as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    print(f"VALIDATION PASSED: {count} IDs; answers are non-empty UTF-8 strings.")


if __name__ == "__main__":
    main()
