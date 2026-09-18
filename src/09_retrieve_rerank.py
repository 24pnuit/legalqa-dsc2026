"""Retrieve BM25 candidates and rerank them with an optimized Cross-Encoder (bge-reranker-v2-m3).

This script combines BM25 retrieval with deep cross-encoder reranking, specifically
tuned for Vietnamese Legal QA tasks. It supports mmap BM25 loading, mixed precision (FP16),
and flexible max_length configuration for long legal documents.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Tuple

import bm25s
from bm25_utils import tokenize_queries

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"


def load_questions(path: Path) -> List[Tuple[str, str]]:
    """Read common LegalQA question JSON and JSONL representations."""
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            data: Any = [json.loads(line) for line in stream if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    questions: List[Tuple[str, str]] = []
    if isinstance(data, dict):
        records = data.items()
    elif isinstance(data, list):
        records = []
        for index, record in enumerate(data, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"Question record {index} must be an object")
            qid = next(
                (record[key] for key in ("qid", "id", "question_id") if record.get(key) is not None),
                None,
            )
            records.append((qid, record))
    else:
        raise ValueError("Questions must be a JSON object, a JSON list, or JSONL objects")

    seen: set[str] = set()
    for raw_qid, value in records:
        qid = str(raw_qid).strip() if raw_qid is not None else ""
        if not qid:
            raise ValueError("Every question must have a non-empty qid/id/question_id")
        if qid in seen:
            raise ValueError(f"Duplicate question ID: {qid}")
        seen.add(qid)

        question = value.get("question") if isinstance(value, dict) else value
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {qid} has empty or non-string text")
        questions.append((qid, question))

    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def make_reranker(model_name: str, device: str, max_length: int = 1024):
    """Factory to create an inference-optimized Reranker scoring function."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Reranking requires torch and transformers; use --no-rerank for BM25 baseline.") from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Reranker tokenizer and model: '{model_name}' on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()

    use_fp16 = device.startswith("cuda") and torch.cuda.is_available()

    def score(question: str, documents: List[str], batch_size: int) -> List[float]:
        if not documents:
            return []

        values: List[float] = []
        with torch.inference_mode():
            for start in range(0, len(documents), batch_size):
                passages = documents[start : start + batch_size]
                
                # Tokenize (Question, Legal Passage) pairs
                encoded = tokenizer(
                    [question] * len(passages),
                    passages,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}

                if use_fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        logits = model(**encoded).logits
                else:
                    logits = model(**encoded).logits

                if logits.shape[-1] == 1:
                    batch = logits[:, 0]
                else:
                    batch = logits[:, -1]

                values.extend(float(x) for x in batch.detach().cpu())
        return values

    return score, device


def retrieve_and_rerank(args: argparse.Namespace) -> None:
    questions = load_questions(Path(args.questions))
    if args.limit is not None:
        questions = questions[: args.limit]

    if args.retrieve_k < args.keep_k or args.retrieve_k < 1 or args.keep_k < 1:
        raise ValueError("Require 1 <= keep-k <= retrieve-k")

    print(f"Loading BM25 index from '{args.index_dir}' (mmap=True)...", flush=True)
    retriever = bm25s.BM25.load(args.index_dir, load_corpus=True, mmap=True)
    corpus = retriever.corpus

    print(f"Tokenizing {len(questions)} queries...", flush=True)
    query_tokens = tokenize_queries([question for _, question in questions])

    print(f"Retrieving Top-{args.retrieve_k} candidates via BM25...", flush=True)
    results, scores = retriever.retrieve(
        query_tokens, corpus=corpus, k=args.retrieve_k, show_progress=True
    )

    rerank_score_fn = None
    if not args.no_rerank:
        rerank_score_fn, actual_device = make_reranker(
            model_name=args.model, device=args.device, max_length=args.max_length
        )
        print(
            f"Reranker initialized on {actual_device} | batch_size={args.batch_size} | max_length={args.max_length}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Writing candidates and running Reranker pass...", flush=True)
    with output.open("w", encoding="utf-8") as stream:
        for row, (qid, question) in enumerate(questions):
            candidates: List[dict[str, Any]] = []
            
            # Format BM25 retrieved candidates
            for bm25_rank, (doc, score) in enumerate(zip(results[row], scores[row]), 1):
                item = dict(doc)
                item.update(
                    bm25_rank=bm25_rank,
                    bm25_score=float(score),
                    rerank_rank=None,
                    rerank_score=None,
                )
                candidates.append(item)

            # Apply Reranker if enabled
            if rerank_score_fn is not None:
                passages = [
                    c.get("text") or c.get("search_text") or c.get("content") or ""
                    for c in candidates
                ]
                values = rerank_score_fn(question, passages, args.batch_size)
                
                # Sort candidates by descending rerank score
                order = sorted(range(len(candidates)), key=lambda i: values[i], reverse=True)
                for rank, index in enumerate(order, 1):
                    candidates[index]["rerank_rank"] = rank
                    candidates[index]["rerank_score"] = values[index]

                candidates.sort(key=lambda c: c["rerank_rank"])

            # Slice down to top keep-k candidates
            candidates = candidates[: args.keep_k]
            
            stream.write(
                json.dumps(
                    {"qid": qid, "question": question, "candidates": candidates},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Success! Output saved to '{output}' ({len(questions)} questions, {args.keep_k} final candidates/question)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", "--bm25-index", dest="index_dir", default="data/bm25_index")
    parser.add_argument("--questions", "--question-path", required=True, help="Path to JSON/JSONL input questions")
    parser.add_argument("--output", required=True, help="Path to write JSONL retrieved results")
    parser.add_argument("--retrieve-k", type=int, default=30, help="Number of initial BM25 candidates to retrieve (Default: 30)")
    parser.add_argument("--keep-k", type=int, default=5, help="Top K candidates to keep after reranking (Default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for Cross-Encoder rerank inference")
    parser.add_argument("--max-length", type=int, default=1024, help="Max token length for reranker sequence (Default: 1024)")
    parser.add_argument("--model", default=RERANKER_NAME, help="Hugging Face model name for Reranker")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranking and rely purely on BM25")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of processed questions for fast debugging")
    
    args = parser.parse_args()
    retrieve_and_rerank(args)


if __name__ == "__main__":
    main()