from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import bm25s
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from bm25_utils import tokenize_queries  # noqa: E402
from candidate_io import load_candidates_from_file  # noqa: E402
from submission_utils import validate_submission  # noqa: E402
from evaluate_local import eval_qa  # noqa: E402


def load_retrieval_module():
    path = SRC_DIR / "09_retrieve_rerank.py"
    spec = importlib.util.spec_from_file_location("retrieve_rerank", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def candidate(chunk_id: str, bm25_rank: int, rerank_rank=None) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "document_name": "Luật thử nghiệm",
        "document_name_origin": "source",
        "source_url": "https://example.invalid/law",
        "chapter": "Chương I",
        "section": "Mục 1",
        "article": "Điều 1",
        "article_title": "Phạm vi",
        "clause_start": "1",
        "clause_end": "1",
        "part": 1,
        "split_method": "article_clause",
        "text": "Nội dung căn cứ.",
        "search_text": "Tên văn bản Nội dung căn cứ.",
        "word_count": 3,
        "short_chunk_reason": None,
        "bm25_rank": bm25_rank,
        "bm25_score": 1.0,
        "rerank_rank": rerank_rank,
        "rerank_score": 0.5 if rerank_rank is not None else None,
    }


def test_candidate_loader_rejects_duplicate_qid(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    record = {"qid": "q1", "question": "Câu hỏi?", "candidates": [candidate("c1", 1)]}
    path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n"
        + json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate qid"):
        load_candidates_from_file(path)


def test_question_loader_rejects_duplicate_qid(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"qid":"q1","question":"Một?"}\n'
        '{"qid":"q1","question":"Hai?"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate question ID"):
        load_retrieval_module().load_questions(path)


def test_candidate_loader_validates_and_sorts_rerank(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    record = {
        "qid": "q1",
        "question": "Câu hỏi?",
        "candidates": [candidate("c2", 7, 2), candidate("c1", 3, 1)],
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = load_candidates_from_file(path)
    assert [item["chunk_id"] for item in loaded["q1"]["candidates"]] == ["c1", "c2"]


def test_retrieval_preserves_complete_chunk_metadata(tmp_path: Path) -> None:
    corpus = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "document_name": "Luật thử nghiệm",
            "document_name_origin": "source",
            "source_url": "https://example.invalid/law",
            "chapter": "Chương I",
            "section": "Mục 1",
            "article": "Điều 1",
            "article_title": "Phạm vi",
            "clause_start": "1",
            "clause_end": "1",
            "part": 1,
            "split_method": "article_clause",
            "text": "Quyền và nghĩa vụ thử nghiệm.",
            "search_text": "Luật thử nghiệm quyền và nghĩa vụ thử nghiệm.",
            "word_count": 6,
            "short_chunk_reason": None,
        }
    ]
    index_dir = tmp_path / "index"
    retriever = bm25s.BM25(method="lucene")
    retriever.index(tokenize_queries([corpus[0]["search_text"]]), show_progress=False)
    retriever.save(index_dir, corpus=corpus, show_progress=False)

    questions = tmp_path / "questions.json"
    output = tmp_path / "candidates.jsonl"
    write_json(questions, {"q1": {"question": "Quyền và nghĩa vụ?"}})
    completed = subprocess.run(
        [
            sys.executable,
            str(SRC_DIR / "09_retrieve_rerank.py"),
            "--questions",
            str(questions),
            "--index-dir",
            str(index_dir),
            "--output",
            str(output),
            "--no-rerank",
            "--retrieve-k",
            "1",
            "--keep-k",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    retrieved = result["candidates"][0]
    for field, value in corpus[0].items():
        assert retrieved[field] == value


def test_submission_validation_and_packaging_use_same_artifact(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    predictions = tmp_path / "predictions.json"
    archive_path = tmp_path / "submission.zip"
    write_json(questions, {"q1": {"question": "Câu hỏi?"}})
    write_json(predictions, {"q1": {"answer": "Câu trả lời."}})

    assert validate_submission(questions, predictions) == 1
    completed = subprocess.run(
        [
            sys.executable,
            str(SRC_DIR / "05_make_submission_zip.py"),
            "--questions",
            str(questions),
            "--pred",
            str(predictions),
            "--out",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["submission.json"]
        assert json.loads(archive.read("submission.json")) == {
            "q1": {"answer": "Câu trả lời."}
        }


def test_packager_rejects_wrong_id_set(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    predictions = tmp_path / "predictions.json"
    archive_path = tmp_path / "submission.zip"
    write_json(questions, {"q1": {"question": "Câu hỏi?"}})
    write_json(predictions, {"other": {"answer": "Sai ID."}})

    completed = subprocess.run(
        [
            sys.executable,
            str(SRC_DIR / "05_make_submission_zip.py"),
            "--questions",
            str(questions),
            "--pred",
            str(predictions),
            "--out",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode != 0
    assert "missing 1 IDs" in completed.stderr
    assert not archive_path.exists()


def test_evaluator_rejects_partial_prediction_set() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        eval_qa({"q1": {"answer": "Một"}}, {"q1": {"answer": "Một"}, "q2": {"answer": "Hai"}})
