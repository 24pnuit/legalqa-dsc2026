import json
import re
import unicodedata
import bm25s
import time

INDEX_DIR = "data/bm25_index"
PUBLIC_PATH = "PUBLIC_PATH = "data/raw/public-official.json"
OUT_PATH = "data/submission.json"

TOKEN_PATTERN = r"[0-9a-zA-ZÀ-ỹà-ỹ]+"
MAX_ANSWER_WORDS = 350  # roughly matches median reference answer length


def build_extractive_answer(top_docs):
    """Very simple extractive baseline: cite the source doc + article, then
    include the retrieved text, trimmed to a reasonable length."""
    parts = []
    for doc in top_docs:
        header = f"Theo {doc['document_name']}, {doc['article']}:"
        parts.append(header + "\n" + doc["text"])
    answer = "\n\n".join(parts)
    words = answer.split()
    if len(words) > MAX_ANSWER_WORDS:
        answer = " ".join(words[:MAX_ANSWER_WORDS])
    return answer


def main():
    print("Loading BM25 index...", flush=True)
    t0 = time.time()
    retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=True)
    chunks = retriever.corpus
    print(f"Loaded index with {len(chunks)} chunks in {time.time()-t0:.1f}s", flush=True)

    with open(PUBLIC_PATH, encoding="utf-8") as f:
        public_qs = json.load(f)
    print(f"Loaded {len(public_qs)} public questions", flush=True)

    submission = {}
    t0 = time.time()
    qids = list(public_qs.keys())
    questions = [public_qs[qid]["question"] for qid in qids]
    q_tokens = bm25s.tokenize(questions, stopwords=None, token_pattern=TOKEN_PATTERN, show_progress=False)
    results, scores = retriever.retrieve(q_tokens, corpus=chunks, k=3, show_progress=False)

    for i, qid in enumerate(qids):
        top_docs = list(results[i])
        answer = build_extractive_answer(top_docs)
        submission[qid] = {"answer": answer}

    print(f"Generated {len(submission)} answers in {time.time()-t0:.1f}s", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
