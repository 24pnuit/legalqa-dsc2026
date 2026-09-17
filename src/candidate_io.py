"""
Helper dùng chung cho cả nhánh Extractive (07) và Generator (08) của Nhi.

Mục tiêu: Nhi có thể phát triển ngay với BM25 top-3 hiện có, và khi Nhung bàn
giao `reranked_top3.jsonl` thì CHỈ cần đổi `--candidates_file`, không sửa gì
trong logic sinh answer.

Hỗ trợ 2 nguồn candidate:
  1. --candidates_file <path.jsonl>   (ưu tiên nếu có)
     Mỗi dòng đúng schema đã thống nhất với Nhung:
     {"qid": "...", "question": "...", "candidates": [
         {"chunk_id","document_id","document_name","article","text",
          "bm25_rank","bm25_score","rerank_rank","rerank_score"}, ...
     ]}
     Sắp xếp theo rerank_rank nếu có, else bm25_rank.

  2. --bm25_index <dir> + --public_path <public-official.json>  (fallback)
     Tự chạy BM25 k=3 trực tiếp bằng thư viện bm25s (giống 03 cũ), dùng khi
     Nhung CHƯA bàn giao file candidates.
"""
import json


def load_candidates_from_file(path):
    """Trả về dict: qid -> {"question": str, "candidates": [chunk,...] (đã sort)}"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cands = rec["candidates"]
            cands = sorted(
                cands,
                key=lambda c: c.get("rerank_rank") if c.get("rerank_rank") is not None
                else c.get("bm25_rank", 999),
            )
            out[rec["qid"]] = {"question": rec["question"], "candidates": cands}
    return out


def load_candidates_from_bm25_fallback(bm25_index_dir, public_path, k=3):
    """Fallback: tự retrieve BM25 top-k trực tiếp (dùng khi chưa có file candidates)."""
    import bm25s

    retriever = bm25s.BM25.load(bm25_index_dir, load_corpus=True)
    chunks = retriever.corpus

    with open(public_path, encoding="utf-8") as f:
        public_qs = json.load(f)

    qids = list(public_qs.keys())
    questions = [public_qs[qid]["question"] for qid in qids]
    q_tokens = bm25s.tokenize(
        questions, stopwords=None,
        token_pattern=r"[0-9a-zA-ZÀ-ỹà-ỹ]+", show_progress=False,
    )
    results, scores = retriever.retrieve(q_tokens, corpus=chunks, k=k, show_progress=False)

    out = {}
    for i, qid in enumerate(qids):
        cands = []
        for rank, (doc, score) in enumerate(zip(results[i], scores[i]), start=1):
            c = dict(doc)
            c["bm25_rank"] = rank
            c["bm25_score"] = float(score)
            cands.append(c)
        out[qid] = {"question": questions[i], "candidates": cands}
    return out


def load_candidates(args, k_fallback=3):
    if args.candidates_file:
        return load_candidates_from_file(args.candidates_file)
    assert args.bm25_index and args.public_path, (
        "Cần --candidates_file, hoặc cả --bm25_index và --public_path để chạy fallback BM25."
    )
    return load_candidates_from_bm25_fallback(args.bm25_index, args.public_path, k=k_fallback)


def add_candidate_source_args(parser):
    parser.add_argument("--candidates_file", default=None,
                         help="jsonl candidates từ Nhung (ưu tiên nếu có)")
    parser.add_argument("--bm25_index", default="data/bm25_index",
                         help="fallback: dùng khi chưa có candidates_file")
    parser.add_argument("--public_path", default="data/raw/public-official.json",
                         help="fallback: dùng khi chưa có candidates_file")
    return parser
