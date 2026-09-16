import json
import re
import time
import sys
import unicodedata
from collections import Counter
import bm25s

INDEX_DIR = "data/bm25_index"
TRAIN_PATH = "data/raw/train.json"
OUT_PATH = "data/alignment_results.jsonl"
SUMMARY_PATH = "data/alignment_summary.json"

TOKEN_PATTERN = r"[0-9a-zA-ZÀ-ỹà-ỹ]+"
N = 4  # n-gram size for overlap-based alignment
TOP_K_ANSWER = 30
TOP_K_QUESTION = 50
COVERAGE_THRESHOLD = 0.5


def tokenize(text):
    text = unicodedata.normalize("NFC", text.lower())
    return re.findall(TOKEN_PATTERN, text)


def ngrams(tokens, n=N):
    if len(tokens) < n:
        return set([tuple(tokens)]) if tokens else set()
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def coverage(answer_ngrams, chunk_tokens):
    if not answer_ngrams:
        return 0.0
    chunk_ng = ngrams(chunk_tokens)
    if not chunk_ng:
        return 0.0
    inter = answer_ngrams & chunk_ng
    return len(inter) / len(answer_ngrams)


def union_coverage(answer_ngrams, list_of_chunk_tokens):
    if not answer_ngrams:
        return 0.0
    union_ng = set()
    for toks in list_of_chunk_tokens:
        union_ng |= ngrams(toks)
    inter = answer_ngrams & union_ng
    return len(inter) / len(answer_ngrams)


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 7000

    t0 = time.time()
    retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=True)
    chunks = retriever.corpus
    print(f"Loaded BM25 index ({len(chunks)} chunks) in {time.time()-t0:.1f}s", flush=True)

    with open(TRAIN_PATH, encoding="utf-8") as f:
        train = json.load(f)
    all_qids = list(train.keys())
    all_qids = all_qids[start:end]
    qids = all_qids
    questions = [train[q]["question"] for q in qids]
    answers = [train[q]["answer"] for q in qids]
    print(f"Processing batch [{start}:{end}] -> {len(qids)} examples", flush=True)

    # --- Step 1: retrieve candidate evidence chunks using the ANSWER as query ---
    t0 = time.time()
    ans_tokens_bm25 = bm25s.tokenize(answers, stopwords=None, token_pattern=TOKEN_PATTERN, show_progress=False)
    ans_results, ans_scores = retriever.retrieve(ans_tokens_bm25, corpus=chunks, k=TOP_K_ANSWER, show_progress=False)
    print(f"Answer-as-query retrieval done in {time.time()-t0:.1f}s", flush=True)

    # --- Step 2: retrieve using the QUESTION as query (for recall check later) ---
    t0 = time.time()
    q_tokens_bm25 = bm25s.tokenize(questions, stopwords=None, token_pattern=TOKEN_PATTERN, show_progress=False)
    q_results, q_scores = retriever.retrieve(q_tokens_bm25, corpus=chunks, k=TOP_K_QUESTION, show_progress=False)
    print(f"Question-as-query retrieval done in {time.time()-t0:.1f}s", flush=True)

    type_counts = Counter()
    coverage_singles = []
    recall_ranks_chunk = []   # rank (1-indexed) of gold chunk in question-retrieval, or None
    recall_ranks_doc = []     # rank (1-indexed) of gold DOCUMENT in question-retrieval, or None

    t0 = time.time()
    out_f = open(OUT_PATH, "a", encoding="utf-8")

    for i, qid in enumerate(qids):
        answer = answers[i]
        ans_toks = tokenize(answer)
        ans_ng = ngrams(ans_toks)

        candidates = list(ans_results[i])  # top-30 chunks by answer-as-query BM25
        cand_tokens = [tokenize(c["text"]) for c in candidates]
        cand_cov = [coverage(ans_ng, toks) for toks in cand_tokens]

        best_idx = max(range(len(candidates)), key=lambda j: cand_cov[j]) if candidates else None
        cov_single = cand_cov[best_idx] if best_idx is not None else 0.0
        best_chunk = candidates[best_idx] if best_idx is not None else None
        best_doc_id = best_chunk["document_id"] if best_chunk else None

        # top-3 same document as best_doc_id
        same_doc_idx = [j for j, c in enumerate(candidates) if c["document_id"] == best_doc_id][:3]
        cov_top3_samedoc = union_coverage(ans_ng, [cand_tokens[j] for j in same_doc_idx]) if same_doc_idx else cov_single

        # top-10 any document
        cov_top10_any = union_coverage(ans_ng, cand_tokens[:10])

        if cov_single >= COVERAGE_THRESHOLD:
            qtype = "Type1_single_passage"
        elif cov_top3_samedoc >= COVERAGE_THRESHOLD:
            qtype = "Type2_multi_passage_same_doc"
        elif cov_top10_any >= COVERAGE_THRESHOLD:
            qtype = "Type3_multi_document"
        else:
            qtype = "Type4_unaligned"

        type_counts[qtype] += 1
        coverage_singles.append(cov_single)

        # --- Recall@K check: does BM25-on-QUESTION retrieve the identified gold chunk/doc? ---
        rank_chunk = None
        rank_doc = None
        if best_chunk is not None and cov_single >= 0.3:  # only check recall when we have a reasonably confident gold chunk
            q_candidates = list(q_results[i])
            for rank, c in enumerate(q_candidates, start=1):
                if rank_chunk is None and c["chunk_id"] == best_chunk["chunk_id"]:
                    rank_chunk = rank
                if rank_doc is None and c["document_id"] == best_doc_id:
                    rank_doc = rank
                if rank_chunk is not None and rank_doc is not None:
                    break
        recall_ranks_chunk.append(rank_chunk)
        recall_ranks_doc.append(rank_doc)

        out_f.write(json.dumps({
            "qid": qid,
            "type": qtype,
            "coverage_single": round(cov_single, 3),
            "coverage_top3_samedoc": round(cov_top3_samedoc, 3),
            "coverage_top10_any": round(cov_top10_any, 3),
            "best_chunk_id": best_chunk["chunk_id"] if best_chunk else None,
            "best_document_name": best_chunk["document_name"] if best_chunk else None,
            "best_article": best_chunk["article"] if best_chunk else None,
            "rank_in_question_retrieval_chunk": rank_chunk,
            "rank_in_question_retrieval_doc": rank_doc,
        }, ensure_ascii=False) + "\n")

        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{len(qids)}", flush=True)

    out_f.close()
    print(f"Batch [{start}:{end}] done in {time.time()-t0:.1f}s", flush=True)
    print(f"Batch type counts: {dict(type_counts)}", flush=True)


if __name__ == "__main__":
    main()
