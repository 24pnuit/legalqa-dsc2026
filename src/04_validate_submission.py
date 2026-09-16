import json
import sys

PUBLIC_PATH = "data/raw/public-official.json"
SUBMISSION_PATH = "data/submission.json"


def main():
    with open(PUBLIC_PATH, encoding="utf-8") as f:
        public_qs = json.load(f)
    expected_ids = set(public_qs.keys())

    try:
        with open(SUBMISSION_PATH, encoding="utf-8") as f:
            submission = json.load(f)
    except json.JSONDecodeError as e:
        print(f"FAIL: submission.json is not valid JSON: {e}")
        sys.exit(1)

    errors = []

    got_ids = set(submission.keys())
    missing = expected_ids - got_ids
    extra = got_ids - expected_ids
    if missing:
        errors.append(f"Missing {len(missing)} question_id(s), e.g. {list(missing)[:5]}")
    if extra:
        errors.append(f"{len(extra)} unexpected extra question_id(s), e.g. {list(extra)[:5]}")

    n_null = 0
    n_not_str = 0
    n_missing_answer_field = 0
    n_empty = 0
    for qid, val in submission.items():
        if not isinstance(val, dict):
            errors.append(f"{qid}: value is not an object")
            continue
        if "answer" not in val:
            n_missing_answer_field += 1
            continue
        ans = val["answer"]
        if ans is None:
            n_null += 1
        elif not isinstance(ans, str):
            n_not_str += 1
        elif len(ans.strip()) == 0:
            n_empty += 1

    if n_missing_answer_field:
        errors.append(f"{n_missing_answer_field} entries missing 'answer' field")
    if n_null:
        errors.append(f"{n_null} entries have answer = null")
    if n_not_str:
        errors.append(f"{n_not_str} entries have non-string answer")
    if n_empty:
        errors.append(f"{n_empty} entries have empty-string answer")

    # UTF-8 encodability check (round trip)
    try:
        json.dumps(submission, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as e:
        errors.append(f"UTF-8 encode error: {e}")

    print(f"Expected IDs: {len(expected_ids)} | Submission IDs: {len(got_ids)}")
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    else:
        print("VALIDATION PASSED: all", len(got_ids), "IDs present, all answers valid non-empty strings, UTF-8 OK.")


if __name__ == "__main__":
    main()
