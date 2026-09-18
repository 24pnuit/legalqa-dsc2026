"""Audit the final LegalQA submission artifact (exit 0=PASS, 1=FAIL/BLOCKED)."""

from __future__ import annotations

import argparse
import json
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

DEFAULT_ZIP = Path("outputs/submission.zip")
DEFAULT_QUESTIONS = Path("data/raw/public-official.json")
EXPECTED_MEMBER = "submission.json"


class PairsObject(list):
    """JSON object represented as pairs so duplicate keys remain visible."""


def materialize(value: Any, path: str, duplicates: list[tuple[str, str]]) -> Any:
    if isinstance(value, PairsObject):
        result: dict[str, Any] = {}
        seen: set[str] = set()
        for key, child in value:
            if key in seen:
                duplicates.append((path, key))
            seen.add(key)
            result[key] = materialize(child, f"{path}.{key}", duplicates)
        return result
    if isinstance(value, list):
        return [materialize(child, f"{path}[{i}]", duplicates) for i, child in enumerate(value)]
    return value


def state(value: bool | None) -> str:
    return "BLOCKED" if value is None else ("PASS" if value else "FAIL")


def examples(values: list[str] | set[str]) -> list[str]:
    return sorted(values)[:10]


def audit(zip_path: Path, questions_path: Path | None) -> int:
    checks: dict[str, bool | None] = {
        "zip_filename": zip_path.name == "submission.zip",
        "zip_readable": None,
        "internal_filename": None,
        "utf8": None,
        "valid_json": None,
        "root_object": None,
        "qid_type": None,
        "value_object": None,
        "answer_exists": None,
        "answer_string": None,
        "round_trip": None,
    }
    zip_entries: int | None = None
    duplicate_qids: list[str] = []
    duplicate_answers: list[str] = []
    empty_qids: list[str] = []
    extra_field_qids: list[str] = []
    replacement_qids: list[str] = []
    control_warnings: list[str] = []
    errors: list[str] = []
    blockers: list[str] = []
    submission: Any = None

    if not zip_path.is_file():
        blockers.append(f"Missing submission artifact: {zip_path}")
    else:
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                members = archive.infolist()
                zip_entries = len(members)
                corrupt_member = archive.testzip()
                checks["zip_readable"] = corrupt_member is None
                if corrupt_member:
                    errors.append(f"Corrupt ZIP member: {corrupt_member}")

                exact_member = len(members) == 1 and members[0].filename == EXPECTED_MEMBER
                checks["internal_filename"] = exact_member
                if len(members) != 1:
                    errors.append(f"ZIP contains {len(members)} entries; expected exactly 1")
                if not exact_member:
                    errors.append(
                        f"ZIP entries must be exactly ['{EXPECTED_MEMBER}']; "
                        f"got {[member.filename for member in members][:10]}"
                    )

                if exact_member and checks["zip_readable"]:
                    raw = archive.read(EXPECTED_MEMBER)
                    try:
                        text = raw.decode("utf-8", errors="strict")
                        checks["utf8"] = True
                    except UnicodeDecodeError as exc:
                        text = None
                        checks["utf8"] = False
                        errors.append(f"submission.json is not strict UTF-8: {exc}")

                    if text is not None:
                        try:
                            preserved = json.loads(text, object_pairs_hook=PairsObject)
                            checks["valid_json"] = True
                        except json.JSONDecodeError as exc:
                            checks["valid_json"] = False
                            errors.append(f"submission.json is not valid JSON: {exc}")
                        else:
                            duplicates: list[tuple[str, str]] = []
                            submission = materialize(preserved, "$", duplicates)
                            duplicate_qids = [key for path, key in duplicates if path == "$"]
                            duplicate_answers = [
                                path.removeprefix("$.")
                                for path, key in duplicates
                                if path != "$" and key == "answer"
                            ]
                            other_duplicates = [
                                f"{path}.{key}"
                                for path, key in duplicates
                                if path != "$" and key != "answer"
                            ]
                            if duplicate_qids:
                                errors.append(f"Duplicate top-level question_id(s): {examples(duplicate_qids)}")
                            if duplicate_answers:
                                errors.append(f"Duplicate answer field(s): {examples(duplicate_answers)}")
                            if other_duplicates:
                                errors.append(f"Other duplicate field(s): {examples(other_duplicates)}")
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            checks["zip_readable"] = False
            errors.append(f"Cannot read ZIP: {exc}")

    if submission is not None:
        checks["root_object"] = isinstance(submission, dict) and bool(submission)
        if not isinstance(submission, dict):
            errors.append("JSON root is not an Object")
        elif not submission:
            errors.append("JSON root Object is empty")
        else:
            checks["qid_type"] = all(isinstance(qid, str) and bool(qid.strip()) for qid in submission)
            checks["value_object"] = True
            checks["answer_exists"] = True
            checks["answer_string"] = True
            schema_qids: list[str] = []

            for qid, value in submission.items():
                label = str(qid)
                if not isinstance(qid, str) or not qid.strip():
                    schema_qids.append(repr(qid))
                if not isinstance(value, dict):
                    checks["value_object"] = False
                    schema_qids.append(label)
                    continue
                if set(value) - {"answer"}:
                    extra_field_qids.append(label)
                if "answer" not in value:
                    checks["answer_exists"] = False
                    schema_qids.append(label)
                    continue
                answer = value["answer"]
                if not isinstance(answer, str):
                    checks["answer_string"] = False
                    schema_qids.append(label)
                    continue
                if not answer.strip():
                    empty_qids.append(label)
                if "\ufffd" in answer:
                    replacement_qids.append(label)
                if any(unicodedata.category(char) == "Cc" and char not in "\n\t" for char in answer):
                    control_warnings.append(label)

            if extra_field_qids:
                errors.append(f"Entries with fields other than exactly 'answer': {examples(extra_field_qids)}")
            if empty_qids:
                errors.append(f"Empty/whitespace answer(s): {examples(empty_qids)}")
            if replacement_qids:
                errors.append(f"Answer(s) containing replacement character U+FFFD: {examples(replacement_qids)}")
            if schema_qids:
                errors.append(f"Schema error example qid(s): {examples(schema_qids)}")

            try:
                encoded = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8")
                reparsed = json.loads(encoded.decode("utf-8"))
                checks["round_trip"] = reparsed == submission and len(reparsed) == len(submission)
            except (UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                checks["round_trip"] = False
                errors.append(f"Unicode round-trip failed: {exc}")
            if checks["round_trip"] is False and not any("round-trip" in item for item in errors):
                errors.append("Unicode round-trip changed IDs or answers")

    expected_ids: set[str] | None = None
    submission_ids = set(submission) if isinstance(submission, dict) else None
    missing_ids: set[str] | None = None
    extra_ids: set[str] | None = None

    if questions_path is None:
        blockers.append("Official question source was not provided")
    elif not questions_path.is_file():
        blockers.append(f"Missing official question source: {questions_path}")
    else:
        try:
            question_text = questions_path.read_text(encoding="utf-8", errors="strict")
            questions = json.loads(question_text)
            if not isinstance(questions, dict):
                errors.append(f"Official question source root is not an Object: {questions_path}")
            else:
                expected_ids = set(questions)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"Cannot parse official question source {questions_path}: {exc}")

    if expected_ids is not None and submission_ids is not None:
        missing_ids = expected_ids - submission_ids
        extra_ids = submission_ids - expected_ids
        if missing_ids:
            errors.append(f"Missing question_id(s): {examples(missing_ids)}")
        if extra_ids:
            errors.append(f"Unexpected question_id(s): {examples(extra_ids)}")

    def count(value: set[str] | list[str] | None) -> str:
        return "BLOCKED" if value is None else str(len(value))

    print("SUBMISSION AUDIT")
    print("=" * 40)
    print(f"Artifact                 : {zip_path}")
    print(f"Question source          : {questions_path if questions_path else 'NOT PROVIDED'}")
    print(f"ZIP filename             : {state(checks['zip_filename'])}")
    print(f"ZIP readable             : {state(checks['zip_readable'])}")
    print(f"ZIP entries              : {zip_entries if zip_entries is not None else 'BLOCKED'}")
    print(f"Internal filename        : {state(checks['internal_filename'])}")
    print(f"UTF-8                    : {state(checks['utf8'])}")
    print(f"Valid JSON               : {state(checks['valid_json'])}")
    print(f"Root is JSON Object      : {state(checks['root_object'])}")
    print(f"Duplicate question IDs   : {count(duplicate_qids) if submission is not None else 'BLOCKED'}")
    print(f"Duplicate answer fields  : {count(duplicate_answers) if submission is not None else 'BLOCKED'}")
    print(f"Question ID type         : {state(checks['qid_type'])}")
    print(f"Value is Object          : {state(checks['value_object'])}")
    print(f"answer field exists      : {state(checks['answer_exists'])}")
    print(f"answer is string         : {state(checks['answer_string'])}")
    print(f"Empty answers            : {count(empty_qids) if submission is not None else 'BLOCKED'}")
    print(f"Extra fields             : {count(extra_field_qids) if submission is not None else 'BLOCKED'}")
    print(f"Expected IDs             : {count(expected_ids)}")
    print(f"Submission IDs           : {count(submission_ids)}")
    print(f"Missing IDs              : {count(missing_ids)}")
    print(f"Extra IDs                : {count(extra_ids)}")
    print(f"Unicode round-trip       : {state(checks['round_trip'])}")
    if control_warnings:
        print(f"WARNING unusual controls : {len(control_warnings)}; examples {examples(control_warnings)}")

    failed = bool(errors) or any(value is False for value in checks.values())
    verdict = "FAIL" if failed else ("BLOCKED" if blockers else "PASS")
    print("\nFINAL VERDICT:")
    print(verdict)
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
    if blockers:
        print("\nBLOCKERS:")
        for blocker in blockers:
            print(f"- {blocker}")
    return 0 if verdict == "PASS" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the final LegalQA submission ZIP")
    parser.add_argument("--zip", dest="zip_path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = parser.parse_args()
    raise SystemExit(audit(args.zip_path, args.questions))


if __name__ == "__main__":
    main()
