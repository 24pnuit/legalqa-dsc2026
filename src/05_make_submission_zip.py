"""Validate and package exactly one prediction as submission.json."""

from __future__ import annotations

import argparse
import zipfile
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
        "--input",
        dest="prediction",
        type=Path,
        required=True,
        help="Prediction JSON to validate and package.",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="output",
        type=Path,
        required=True,
        help="Output ZIP path.",
    )
    args = parser.parse_args()

    try:
        count = validate_submission(args.questions, args.prediction)
    except ValueError as exc:
        parser.exit(1, f"FAIL: {exc}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.prediction, arcname="submission.json")
    print(f"Wrote {args.output} ({count} validated predictions)")


if __name__ == "__main__":
    main()
