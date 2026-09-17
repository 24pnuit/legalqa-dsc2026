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


REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "document_id",
    "document_name",
    "document_name_origin",
    "source_url",
    "chapter",
    "section",
    "article",
    "article_title",
    "clause_start",
    "clause_end",
    "part",
    "split_method",
    "text",
    "search_text",
    "word_count",
    "short_chunk_reason",
}


def _validate_candidates_record(rec, line_number):
    if not isinstance(rec, dict):
        raise ValueError(f"Candidate line {line_number} must be an object")
    qid = rec.get("qid")
    question = rec.get("question")
    candidates = rec.get("candidates")
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError(f"Candidate line {line_number} has an invalid qid")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Candidate line {line_number} has an invalid question")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Candidate line {line_number} must contain candidates")

    chunk_ids = set()
    bm25_ranks = set()
    rerank_ranks = []
    for position, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError(f"{qid}: candidate {position} must be an object")
        missing = REQUIRED_CHUNK_FIELDS.difference(candidate)
        if missing:
            raise ValueError(
                f"{qid}: candidate {position} is missing metadata: "
                f"{', '.join(sorted(missing))}"
            )
        chunk_id = candidate.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"{qid}: candidate {position} has an invalid chunk_id")
        if chunk_id in chunk_ids:
            raise ValueError(f"{qid}: duplicate chunk_id {chunk_id}")
        chunk_ids.add(chunk_id)
        if not isinstance(candidate.get("text"), str) or not candidate["text"].strip():
            raise ValueError(f"{qid}: candidate {chunk_id} has empty text")
        if not isinstance(candidate.get("search_text"), str) or not candidate["search_text"].strip():
            raise ValueError(f"{qid}: candidate {chunk_id} has empty search_text")

        bm25_rank = candidate.get("bm25_rank")
        if not isinstance(bm25_rank, int) or isinstance(bm25_rank, bool) or bm25_rank < 1:
            raise ValueError(f"{qid}: candidate {chunk_id} has invalid bm25_rank")
        if bm25_rank in bm25_ranks:
            raise ValueError(f"{qid}: duplicate bm25_rank {bm25_rank}")
        bm25_ranks.add(bm25_rank)
        rerank_ranks.append(candidate.get("rerank_rank"))

    present_rerank = [rank for rank in rerank_ranks if rank is not None]
    if present_rerank:
        if len(present_rerank) != len(candidates):
            raise ValueError(f"{qid}: rerank_rank must be set for every candidate or none")
        if any(not isinstance(rank, int) or isinstance(rank, bool) for rank in present_rerank):
            raise ValueError(f"{qid}: rerank_rank values must be integers")
        if sorted(present_rerank) != list(range(1, len(candidates) + 1)):
            raise ValueError(f"{qid}: rerank_rank must be exactly 1..{len(candidates)}")

    return qid, question, candidates


def load_candidates_from_file(path):
    """Trả về dict: qid -> {"question": str, "candidates": [chunk,...] (đã sort)}"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid, question, cands = _validate_candidates_record(rec, line_number)
            if qid in out:
                raise ValueError(f"Duplicate qid in candidate file: {qid}")
            cands = sorted(
                cands,
                key=lambda c: c.get("rerank_rank") if c.get("rerank_rank") is not None
                else c.get("bm25_rank", 999),
            )
            out[qid] = {"question": question, "candidates": cands}
    if not out:
        raise ValueError(f"Candidate file is empty: {path}")
    return out


def load_candidates_from_bm25_fallback(bm25_index_dir, public_path, k=3):
    """Fallback: tự retrieve BM25 top-k trực tiếp (dùng khi chưa có file candidates)."""
    import bm25s

    from bm25_utils import tokenize_queries

    retriever = bm25s.BM25.load(bm25_index_dir, load_corpus=True, mmap=True)
    chunks = retriever.corpus

    with open(public_path, encoding="utf-8") as f:
        public_qs = json.load(f)

    qids = list(public_qs.keys())
    questions = [public_qs[qid]["question"] for qid in qids]
    q_tokens = tokenize_queries(questions)
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
