"""
Nhánh A của Nhi: Extractive sạch, 3 cấu hình E1/E2/E3.

Thay đổi so với 03_generate_extractive.py cũ:
  - XÓA hoàn toàn header tự sinh "Theo <document_name>, <article>:".
  - KHÔNG cắt cứng " ".join(words[:350]) — dùng truncate_at_boundary() để
    cắt tại ranh giới câu/khoản.
  - Sinh cả 3 config trong 1 lần chạy, ghi ra 3 file riêng để so sánh công
    bằng trên cùng validation set.

Chạy:
  python 07_generate_extractive_v2.py \
      --candidates_file data/experiments/<run_id>/reranked_top3.jsonl \
      --out_dir data/experiments/<run_id>

  # hoặc khi chưa có candidates từ Nhung, dùng fallback BM25 top-3:
  python 07_generate_extractive_v2.py \
      --bm25_index data/bm25_index --public_path data/raw/public-official.json \
      --out_dir data/experiments/dev
"""
import argparse
import json
import os

from candidate_io import load_candidates, add_candidate_source_args
from truncate_utils import truncate_at_boundary

E3_MIN_WORDS_TOP1 = 30   # nếu top-1 có ít hơn ngần này từ, ghép thêm top-2
E3_MAX_WORDS = 400       # chỉ truncate nếu vượt ngưỡng này, còn lại giữ nguyên


def join_chunks(chunks, sep="\n\n"):
    """Nối text các chunk lại, KHÔNG kèm header 'Theo ... :'."""
    return sep.join(c["text"].strip() for c in chunks if c.get("text", "").strip())


def build_e1(candidates):
    """Top-3, không header, tối đa 350 từ tại ranh giới câu/khoản."""
    text = join_chunks(candidates[:3])
    return truncate_at_boundary(text, max_words=350)


def build_e2(candidates):
    """Top-3, không header, tối đa 500 từ tại ranh giới câu/khoản."""
    text = join_chunks(candidates[:3])
    return truncate_at_boundary(text, max_words=500)


def build_e3(candidates):
    """Top-1 (hoặc top-2 nếu top-1 quá ngắn), không header,
    KHÔNG truncate nếu đủ ngắn — chỉ cắt khi vượt E3_MAX_WORDS."""
    if not candidates:
        return ""
    top1_text = candidates[0]["text"].strip()
    if len(top1_text.split()) < E3_MIN_WORDS_TOP1 and len(candidates) > 1:
        text = join_chunks(candidates[:2])
    else:
        text = top1_text

    if len(text.split()) > E3_MAX_WORDS:
        return truncate_at_boundary(text, max_words=E3_MAX_WORDS)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    add_candidate_source_args(ap)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_candidates(args)
    print(f"Loaded candidates cho {len(data)} câu hỏi")

    submissions = {"e1": {}, "e2": {}, "e3": {}}
    empty_count = {"e1": 0, "e2": 0, "e3": 0}

    for qid, rec in data.items():
        cands = rec["candidates"]
        ans1 = build_e1(cands)
        ans2 = build_e2(cands)
        ans3 = build_e3(cands)

        submissions["e1"][qid] = {"answer": ans1}
        submissions["e2"][qid] = {"answer": ans2}
        submissions["e3"][qid] = {"answer": ans3}

        for key, ans in [("e1", ans1), ("e2", ans2), ("e3", ans3)]:
            if not ans.strip():
                empty_count[key] += 1

    for key in ["e1", "e2", "e3"]:
        out_path = os.path.join(args.out_dir, f"pred_extractive_{key}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(submissions[key], f, ensure_ascii=False, indent=2)
        avg_words = sum(len(v["answer"].split()) for v in submissions[key].values()) / max(len(submissions[key]), 1)
        print(f"{key}: wrote {out_path} | avg_words={avg_words:.1f} | empty_answers={empty_count[key]}")

    # cảnh báo sớm nếu có answer rỗng — dấu hiệu candidates thiếu 'text' hoặc rỗng
    if any(empty_count.values()):
        print("\n⚠️  CẢNH BÁO: có answer rỗng — kiểm tra lại candidates_file / bm25 index trước khi nộp.")


if __name__ == "__main__":
    main()
