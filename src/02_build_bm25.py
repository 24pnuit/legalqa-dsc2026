import json
import re
import time
import unicodedata
import bm25s

CHUNKS_PATH = "data/chunks.jsonl"
INDEX_DIR = "data/bm25_index"

TOKEN_RE = re.compile(r"[0-9a-zA-ZÀ-ỹà-ỹ]+", re.UNICODE)


def tokenize(text: str):
    text = unicodedata.normalize("NFC", text.lower())
    return TOKEN_RE.findall(text)


def main():
    t0 = time.time()
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    corpus_texts = [c["text"] + " " + c.get("document_name", "") for c in chunks]
    corpus_tokens = bm25s.tokenize(
        corpus_texts, stopwords=None,
        token_pattern=r"[0-9a-zA-ZÀ-ỹà-ỹ]+", show_progress=False,
    )
    print(f"Tokenized in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    print(f"Built BM25 index in {time.time()-t0:.1f}s", flush=True)

    retriever.save(INDEX_DIR, corpus=chunks)
    print(f"Saved index to {INDEX_DIR}", flush=True)

    test_q = "Cơ sở sản xuất tôm sú giống có quyền và nghĩa vụ gì?"
    q_tokens = bm25s.tokenize(
        [test_q], stopwords=None,
        token_pattern=r"[0-9a-zA-ZÀ-ỹà-ỹ]+", show_progress=False,
    )
    t0 = time.time()
    results, scores = retriever.retrieve(q_tokens, corpus=chunks, k=3)
    print(f"Query scored in {time.time()-t0:.2f}s", flush=True)
    for doc, score in zip(results[0], scores[0]):
        print("---", doc["document_name"], doc["article"], "score=", round(float(score), 2))
        print(doc["text"][:200].replace("\n", " | "))


if __name__ == "__main__":
    main()
