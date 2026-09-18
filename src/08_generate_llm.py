"""Generate LegalQA answers with a Qwen instruct model (branch B).

This generator consumes the canonical candidate JSONL produced by
`09_retrieve_rerank.py`. It targets the Kaggle T4/P100 GPU environment:

- deterministic greedy decoding: temperature=0.0 / do_sample=False;
- thinking disabled if the loaded template supports it;
- periodic atomic checkpoints + --resume to survive a notebook session timeout;
- a --mock mode to smoke-test the I/O contract without a GPU/model.

Typical Kaggle command:
    python src/08_generate_llm.py --candidates_file data/candidates.jsonl \
        --out_dir data/out --model Qwen/Qwen3-4B-Instruct-2507 \
        --dtype float16 --save-every 25 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from candidate_io import add_candidate_source_args, load_candidates

MODEL_PRIMARY = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_FALLBACK = "Qwen/Qwen3-4B"
MAX_NEW_TOKENS = 600
TOP_K_CONTEXT = 3
DEFAULT_MAX_CONTEXT_WORDS = 1800
DEFAULT_SAVE_EVERY = 25

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Bạn CHỈ được dùng thông tin trong phần CONTEXT dưới đây "
    "để trả lời câu hỏi. Hãy tuân thủ nghiêm ngặt:\n"
    "1. Trả lời trực tiếp vào câu hỏi, không lặp lại câu hỏi, không mở đầu bằng lời chào.\n"
    "2. KHÔNG được tự thêm câu dạng \"Theo <tên văn bản>, Điều <số>:\" ở đầu câu trả lời.\n"
    "3. Giữ NGUYÊN VẸN mọi số Điều, khoản, điểm, số tiền, thời hạn, tỷ lệ phần trăm "
    "xuất hiện trong CONTEXT - không làm tròn, không diễn giải lại thành số khác.\n"
    "4. Không thêm cảnh báo, khuyến nghị, hay giải thích nằm ngoài phạm vi câu hỏi.\n"
    "5. Nếu cần dùng thông tin từ nhiều đoạn CONTEXT, hãy tổng hợp chúng thành một câu "
    "trả lời mạch lạc duy nhất.\n"
    "6. Nếu CONTEXT không đủ thông tin để trả lời, hãy nói rõ là không đủ căn cứ trong "
    "tài liệu được cung cấp - TUYỆT ĐỐI không suy đoán hay bịa thông tin."
)

FORBIDDEN_HEADER_RE = re.compile(r"^\s*Theo\s+[^,]+,\s*[^:]*:\s*", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _reconfigure_utf8():
    """Windows cp1252 stdout breaks when argparse/help prints Vietnamese text."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def trim_context(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def build_user_prompt(question: str, chunks, max_context_words: int) -> str:
    per_chunk = max(1, max_context_words // max(len(chunks), 1))
    parts = [
        f"[Đoạn {i+1}] {trim_context(c.get('text', ''), per_chunk)}"
        for i, c in enumerate(chunks)
    ]
    context_block = "\n\n".join(parts)
    return f"CONTEXT:\n{context_block}\n\nCÂU HỎI: {question}\n\nTrả lời:"


def clean_output(raw: str) -> str:
    cleaned = THINK_BLOCK_RE.sub("", raw).strip()
    return FORBIDDEN_HEADER_RE.sub("", cleaned).strip()


def write_checkpoint(path: Path, submission: dict) -> None:
    tmp = path.with_suffix(str(path.suffix) + ".tmp")
    tmp.write_text(json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class MockModel:
    def generate_answer(self, question: str, chunks) -> str:
        return f"[MOCK] {len(chunks)} context chunks: {question[:80]}..."


class QwenModel:
    def __init__(self, model_name: str | None, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        candidates = [
            name
            for name in (model_name, MODEL_PRIMARY, MODEL_FALLBACK)
            if name
        ]
        candidates = list(dict.fromkeys(candidates))
        last_error: Exception | None = None
        for name in candidates:
            try:
                print(f"Loading {name} (dtype={dtype})", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                model_kwargs = {"device_map": "auto", "torch_dtype": "auto"}
                if dtype != "auto":
                    model_kwargs["torch_dtype"] = getattr(torch, dtype)
                self.model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
                self.model_name = name
                print(f"Loaded {name}; device={self.model.device}", flush=True)
                return
            except Exception as exc:
                print(f"Could not load {name}: {exc}", flush=True)
                last_error = exc
        raise RuntimeError("Could not load any model candidate") from last_error

    def generate_answer(self, question, chunks, max_context_words, max_new_tokens) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, chunks, max_context_words)},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(
            [prompt], return_tensors="pt", truncation=True, max_length=8192
        ).to(self.model.device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    _reconfigure_utf8()
    p = argparse.ArgumentParser(description=__doc__)
    add_candidate_source_args(p)
    p.add_argument("--out_dir", required=True, help="Directory for pred_generator.json + log")
    p.add_argument("--top_k", type=int, default=TOP_K_CONTEXT)
    p.add_argument("--model", default=None, help="HF model id (default: project primary)")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--max-context-words", type=int, default=DEFAULT_MAX_CONTEXT_WORDS)
    p.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--mock", action="store_true", help="Avoid loading any model")
    p.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    return p.parse_args()


def main() -> None:
    _reconfigure_utf8()
    args = parse_args()
    if args.top_k < 1 or args.max_new_tokens < 1 or args.max_context_words < 1:
        raise ValueError("top_k, max-new-tokens and max-context-words must be positive")
    if args.save_every < 1:
        raise ValueError("save-every must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pred_generator.json"
    log_path = out_dir / "generation_log.jsonl"

    data = load_candidates(args)
    qids = list(data)
    if args.limit is not None:
        qids = qids[: args.limit]

    submission: dict[str, dict[str, str]] = {}
    log_mode = "w"
    if args.resume and out_path.exists():
        submission = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(submission, dict):
            raise ValueError(f"Invalid checkpoint (expected a JSON object): {out_path}")
        qids = [qid for qid in qids if qid not in submission]
        print(f"Resume: skipping {len(submission)} existing answers", flush=True)
        log_mode = "a"
    elif not args.resume and out_path.exists():
        print(f"Overwriting {out_path} (use --resume to continue)", flush=True)

    print(f"Generating {len(qids)} answers" + (" (MOCK)" if args.mock else ""), flush=True)
    model = MockModel() if args.mock else QwenModel(args.model, args.dtype)

    total_time = 0.0
    n_empty = 0
    n_insufficient = 0

    with open(log_path, log_mode, encoding="utf-8") as log_f:
        if log_mode == "a":
            log_f.write(json.dumps({"resume_batch": True}, ensure_ascii=False) + "\n")
        for i, qid in enumerate(qids, start=1):
            rec = data[qid]
            question = rec["question"]
            chunks = rec["candidates"][: args.top_k]

            t0 = time.perf_counter()
            if args.mock:
                raw_output = model.generate_answer(question, chunks)
            else:
                raw_output = model.generate_answer(
                    question, chunks, args.max_context_words, args.max_new_tokens
                )
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            answer = clean_output(raw_output)
            if not answer.strip():
                n_empty += 1
            if "không đủ" in answer.lower() and "thông tin" in answer.lower():
                n_insufficient += 1

            submission[qid] = {"answer": answer}
            log_f.write(
                json.dumps(
                    {
                        "qid": qid,
                        "question": question,
                        "chunk_ids_used": [c.get("chunk_id") for c in chunks],
                        "raw_output": raw_output,
                        "clean_answer": answer,
                        "latency_sec": round(elapsed, 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_f.flush()

            if i % args.save_every == 0:
                write_checkpoint(out_path, submission)
                print(f"Checkpoint: {len(submission)} answers", flush=True)
            if i % 25 == 0:
                print(f"Processed {i}/{len(qids)} | avg {total_time / i:.2f}s/q", flush=True)

    write_checkpoint(out_path, submission)
    print(f"Wrote {out_path}", flush=True)
    print(f"Wrote {log_path}", flush=True)
    print(f"Total: {len(submission)} | empty {n_empty} | insufficient {n_insufficient}", flush=True)


if __name__ == "__main__":
    main()
