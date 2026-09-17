"""Shared validation for submission checking and packaging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} file does not exist: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} root must be a JSON object: {path}")
    return data


def validate_submission(questions_path: Path, prediction_path: Path) -> int:
    questions = read_json_object(questions_path, "questions")
    predictions = read_json_object(prediction_path, "prediction")

    expected_ids = set(questions)
    prediction_ids = set(predictions)
    missing = sorted(expected_ids - prediction_ids)
    extra = sorted(prediction_ids - expected_ids)
    errors = []
    if missing:
        errors.append(f"missing {len(missing)} IDs, e.g. {missing[:5]}")
    if extra:
        errors.append(f"found {len(extra)} extra IDs, e.g. {extra[:5]}")

    for qid, value in predictions.items():
        if not isinstance(value, dict):
            errors.append(f"{qid}: value is not an object")
            continue
        if set(value) != {"answer"}:
            errors.append(f"{qid}: value must contain exactly the 'answer' field")
            continue
        answer = value["answer"]
        if not isinstance(answer, str) or not answer.strip():
            errors.append(f"{qid}: answer must be a non-empty string")

    if errors:
        preview = "\n - ".join(errors[:20])
        suffix = f"\n - ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Submission validation failed:\n - {preview}{suffix}")
    return len(expected_ids)
