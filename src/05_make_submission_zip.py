import zipfile
import os

SRC = "data/submission.json"
OUT_DIR = "outputs"
OUT_ZIP = os.path.join(OUT_DIR, "submission.zip")

os.makedirs(OUT_DIR, exist_ok=True)
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(SRC, arcname="submission.json")

print("Wrote", OUT_ZIP)
