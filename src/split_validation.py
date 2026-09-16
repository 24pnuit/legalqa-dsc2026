import json
from pathlib import Path
import random

TRAIN_PATH = Path("data/raw/train.json")
OUTPUT_DIR = Path("data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(TRAIN_PATH, "r", encoding="utf-8") as f:
  data = json.load(f)

# Cố định random seed để không bị xáo trộn qua các lần chạy
random.seed(42)

# train.json dạng dict {qid: {"question": ..., "answer": ...}} hoặc list
if isinstance(data, dict):
  all_qids = sorted(list(data.keys()))
  random.shuffle(all_qids)
  val_qids = all_qids[:1000]

  val_queries = {qid: {"question": data[qid]["question"]} for qid in val_qids}
  val_reference = {qid: {"answer": data[qid]["answer"]} for qid in val_qids}
else:
  random.shuffle(data)
  val_set = data[:1000]
  val_qids = [item["id"] for item in val_set]
  val_queries = {
      item["id"]: {"question": item["question"]} for item in val_set
  }
  val_reference = {item["id"]: {"answer": item["answer"]} for item in val_set}

# 1. File chỉ chứa ID (file này SẼ ĐẨY LÊN GITHUB)
with open(OUTPUT_DIR / "validation_ids.json", "w", encoding="utf-8") as f:
  json.dump(val_qids, f, ensure_ascii=False, indent=2)

# 2. File câu hỏi dùng để suy luận thử nghiệm (local)
with open(OUTPUT_DIR / "val_queries.json", "w", encoding="utf-8") as f:
  json.dump(val_queries, f, ensure_ascii=False, indent=2)

# 3. File đáp án chuẩn của chuyên gia để chấm điểm (local)
with open(OUTPUT_DIR / "val_reference.json", "w", encoding="utf-8") as f:
  json.dump(val_reference, f, ensure_ascii=False, indent=2)

print(f"[*] Đã tạo thành công tập Validation gồm {len(val_qids)} câu tại {OUTPUT_DIR}")