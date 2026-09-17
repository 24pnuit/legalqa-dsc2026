# Pipeline LegalQA (BM25 + extractive/generator)

## Corpus và BM25 index

Corpus chuẩn hiện tại là `data/processed/chunks.jsonl`. File này đã được tạo
bởi pipeline xử lý dữ liệu mới và có hai trường nội dung riêng biệt:

- `search_text`: nội dung dành cho BM25 indexing.
- `text`: nội dung gốc dành cho bước chọn evidence và sinh câu trả lời.

Không chạy `src/01_chunk_corpus.py` trước bước build dưới đây. Script đó chỉ
được giữ lại để tái hiện baseline cũ và sẽ tạo corpus khác schema.

```powershell
# CPU/retrieval/evaluation dependencies
python -m pip install -r requirements.txt

# Chỉ cần cho reranker và generator
python -m pip install -r requirements-model.txt

python src/02_build_bm25.py `
  --chunks data/processed/chunks.jsonl `
  --index-dir data/bm25_index
```

Builder đọc JSONL tuần tự, index đúng `search_text`, và lưu toàn bộ record vào
`data/bm25_index/corpus.jsonl`. Vì vậy kết quả retrieve vẫn có `chunk_id`,
`text`, `source_url`, tên văn bản, Điều/Khoản và metadata còn lại. File
`manifest.json` cạnh index ghi hash corpus, số chunk, cấu hình tokenizer/BM25,
phiên bản thư viện, commit, thời gian và peak RAM của lần build.

Các consumer load index với `mmap=True` để không nạp toàn bộ sparse matrix và
corpus hơn 1 GiB vào RAM. Tokenizer query dùng chung cấu hình với builder:
Unicode NFC, lowercase, không bỏ stopword, giữ chữ số và dấu tiếng Việt.

## Pipeline hiện tại

Pipeline canonical là:

`data/processed/chunks.jsonl` → `02_build_bm25.py` → `09_retrieve_rerank.py`
→ `07_generate_extractive_v2.py` / `08_generate_llm.py` → `evaluate_local.py`
→ `04_validate_submission.py` → `05_make_submission_zip.py`

`09_retrieve_rerank.py` lấy BM25 top-20 rồi rerank toàn bộ bằng
`BAAI/bge-reranker-v2-m3`, giữ top-3 và bảo toàn toàn bộ record chunk, gồm
`text`, `search_text`, thông tin nguồn và metadata cấu trúc. Dùng
`--no-rerank` để chạy baseline không cần PyTorch/model.

```powershell
python src/09_retrieve_rerank.py `
  --questions data/splits/val_queries.json `
  --index-dir data/bm25_index `
  --output data/experiments/val/candidates.jsonl
```

Sau đó truyền file candidates vào bước 07 hoặc 08 bằng `--candidates_file`.
Loader sẽ từ chối qid/chunk trùng, text rỗng và rank không hợp lệ.

Chấm local và ghi metrics:

```powershell
python src/evaluate_local.py `
  --ref data/splits/val_reference.json `
  --pred data/experiments/val/pred_generator.json `
  --out data/experiments/val/metrics_generator.json
```

METEOR cần WordNet/OMW. Cài một lần bằng
`python -m nltk.downloader wordnet omw-1.4`, hoặc cho phép evaluator tải bằng
`--download-nltk-data`. Import module không tự truy cập mạng.

Validator và packager đều nhận cùng file câu hỏi và prediction. Packager sẽ
từ chối tạo ZIP nếu validation thất bại:

```powershell
python src/04_validate_submission.py `
  --questions data/raw/public-official.json `
  --pred data/experiments/public_final/pred_generator.json

python src/05_make_submission_zip.py `
  --questions data/raw/public-official.json `
  --pred data/experiments/public_final/pred_generator.json `
  --out outputs/submission_final.zip
```

ZIP chứa đúng một file tên `submission.json`.

Ví dụ fallback BM25 cho nhánh extractive mới:

```powershell
python src/09_retrieve_rerank.py `
  --questions data/raw/public-official.json `
  --output data/experiments/dev/reranked_top3.jsonl `
  --no-rerank
```

Toàn bộ `data/`, `outputs/`, `*.jsonl` và file index lớn đã được `.gitignore`
loại khỏi Git; không thêm cưỡng bức các artifact này vào commit.

## Kiểm tra trước khi push

```powershell
python -m compileall -q src tests
python -m pytest -q
git diff --check
git status --short
```
