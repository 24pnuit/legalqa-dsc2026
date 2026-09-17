"""
Nhánh B của Nhi: Generator thật bằng Qwen3.5-4B (fallback Qwen3-4B-Instruct-2507).

Cấu hình theo đúng yêu cầu:
  temperature: 0.0 (tương đương do_sample=False, greedy decoding)
  do_sample: false
  max_new_tokens: 600
  thinking: false          -> enable_thinking=False khi apply_chat_template
  contexts: top_3_reranked -> đọc từ --candidates_file (ưu tiên) hoặc fallback BM25

Prompt bắt buộc model:
  - Chỉ dùng context được cung cấp.
  - Trả lời trực tiếp (không lặp lại câu hỏi).
  - KHÔNG tự thêm câu "Theo X, Điều Y".
  - Giữ nguyên số Điều/khoản/điểm, số tiền, thời hạn, tỷ lệ — không diễn giải lại số liệu.
  - Không thêm cảnh báo/giải thích ngoài phạm vi câu hỏi.
  - Tổng hợp nhiều chunk khi câu trả lời cần thông tin từ nhiều nguồn.
  - Nếu context không đủ để trả lời thì nói rõ KHÔNG đủ thông tin, không suy đoán.

Chạy thật (cần GPU + internet tới HuggingFace):
  python 08_generate_llm.py \
      --candidates_file data/experiments/<run_id>/reranked_top3.jsonl \
      --out_dir data/experiments/<run_id>

Chạy dry-run để test luồng I/O khi CHƯA có GPU/model (không gọi model thật):
  python 08_generate_llm.py --candidates_file fake_candidates.jsonl --out_dir ./out_test --mock
"""
import argparse
import json
import os
import re
import time

from candidate_io import load_candidates, add_candidate_source_args

MODEL_PRIMARY = "Qwen/Qwen3.5-4B"
MODEL_FALLBACK = "Qwen/Qwen3-4B-Instruct-2507"

MAX_NEW_TOKENS = 600
TOP_K_CONTEXT = 3

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Bạn CHỈ được dùng thông tin trong phần CONTEXT dưới đây "
    "để trả lời câu hỏi. Tuân thủ nghiêm ngặt các quy tắc sau:\n"
    "1. Trả lời trực tiếp vào câu hỏi, không lặp lại câu hỏi, không mở đầu bằng lời chào.\n"
    "2. KHÔNG tự thêm câu dạng \"Theo <tên văn bản>, Điều <số>:\" ở đầu câu trả lời.\n"
    "3. Giữ NGUYÊN VẸN mọi số Điều, khoản, điểm, số tiền, thời hạn, tỷ lệ phần trăm "
    "xuất hiện trong CONTEXT — không làm tròn, không diễn giải lại thành số khác.\n"
    "4. Không thêm cảnh báo, khuyến nghị, hay giải thích nằm ngoài phạm vi câu hỏi.\n"
    "5. Nếu câu trả lời cần thông tin từ nhiều đoạn CONTEXT khác nhau, hãy tổng hợp "
    "chúng lại thành một câu trả lời mạch lạc duy nhất.\n"
    "6. Nếu CONTEXT không chứa đủ thông tin để trả lời, hãy nói rõ là không đủ căn cứ "
    "trong tài liệu được cung cấp — TUYỆT ĐỐI không suy đoán hay bịa thông tin."
)

# safety net: nếu model vẫn lỡ sinh ra header dạng "Theo X, Điều Y:" ở đầu câu, cắt bỏ
FORBIDDEN_HEADER_RE = re.compile(r'^\s*Theo\s+[^,]+,\s*[^:]*:\s*', re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def build_user_prompt(question, chunks):
    context_block = "\n\n".join(
        f"[Đoạn {i+1}] {c['text'].strip()}" for i, c in enumerate(chunks)
    )
    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"CÂU HỎI: {question}\n\n"
        f"Trả lời:"
    )


def clean_output(text):
    text = THINK_BLOCK_RE.sub("", text).strip()
    text = FORBIDDEN_HEADER_RE.sub("", text).strip()
    return text


class MockModel:
    """Dùng để test luồng I/O khi chưa có GPU/model thật — KHÔNG dùng để nộp bài."""

    def generate_answer(self, question, chunks):
        return f"[MOCK] Trả lời dựa trên {len(chunks)} đoạn context cho câu hỏi: {question[:50]}..."


class QwenModel:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        model_name = MODEL_PRIMARY
        try:
            print(f"Đang tải {MODEL_PRIMARY} ...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True,
            )
        except Exception as e:
            print(f"Không tải được {MODEL_PRIMARY} ({e}) -> fallback {MODEL_FALLBACK}", flush=True)
            model_name = MODEL_FALLBACK
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto",
            )
        self.model_name = model_name
        print(f"Đã tải xong: {model_name}", flush=True)

    def generate_answer(self, question, chunks):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, chunks)},
        ]
        # enable_thinking=False: model non-thinking (Instruct-2507) bỏ qua kwarg này một
        # cách an toàn nếu template không tham chiếu tới nó -> luôn truyền được.
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,       # deterministic / greedy, tương đương temperature=0.0
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return raw


def main():
    ap = argparse.ArgumentParser()
    add_candidate_source_args(ap)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=TOP_K_CONTEXT)
    ap.add_argument("--mock", action="store_true",
                     help="Test luồng I/O không cần GPU/model thật — KHÔNG dùng để nộp bài")
    ap.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu (smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_candidates(args)
    qids = list(data.keys())
    if args.limit:
        qids = qids[:args.limit]
    print(f"Sẽ sinh answer cho {len(qids)} câu hỏi" + (" (MOCK MODE)" if args.mock else ""))

    model = MockModel() if args.mock else QwenModel()

    submission = {}
    log_path = os.path.join(args.out_dir, "generation_log.jsonl")
    log_f = open(log_path, "w", encoding="utf-8")

    total_time = 0.0
    n_empty = 0
    n_insufficient = 0

    for i, qid in enumerate(qids):
        rec = data[qid]
        question = rec["question"]
        chunks = rec["candidates"][:args.top_k]

        t0 = time.time()
        raw_output = model.generate_answer(question, chunks)
        elapsed = time.time() - t0
        total_time += elapsed

        answer = clean_output(raw_output)
        if not answer.strip():
            n_empty += 1
        if "không đủ" in answer.lower() and "thông tin" in answer.lower():
            n_insufficient += 1

        submission[qid] = {"answer": answer}

        log_f.write(json.dumps({
            "qid": qid,
            "question": question,
            "chunk_ids_used": [c.get("chunk_id") for c in chunks],
            "raw_output": raw_output,
            "clean_answer": answer,
            "latency_sec": round(elapsed, 3),
        }, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  đã xử lý {i+1}/{len(qids)} | avg {total_time/(i+1):.2f}s/câu", flush=True)

    log_f.close()

    out_path = os.path.join(args.out_dir, "pred_generator.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {out_path}")
    print(f"Wrote {log_path}")
    print(f"Tổng {len(qids)} câu | avg {total_time/max(len(qids),1):.2f}s/câu")
    print(f"Answer rỗng: {n_empty} | Trả lời 'không đủ thông tin': {n_insufficient}")


if __name__ == "__main__":
    main()
