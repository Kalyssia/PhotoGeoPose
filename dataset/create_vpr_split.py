"""
This file prepares the dataset for training and evaluation by COPYING the relevant images 
in separate directories for database and queries.
 
It relies on the JSON annotation files to determine which images belong to the
training and validation splits.
"""
# create_vpr_split.py
import json
from pathlib import Path
import shutil

IMAGE_DIR = Path("/scratch/users/agraillet/images")
OUT_ROOT = Path("/scratch/users/akayembe/my_dataset")
DB_DIR = OUT_ROOT / "database"
Q_DIR = OUT_ROOT / "queries"

DB_DIR.mkdir(parents=True, exist_ok=True)
Q_DIR.mkdir(parents=True, exist_ok=True)

def copy_split(annotations_path: Path, out_dir: Path):
    with annotations_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    for x in items:
        src = IMAGE_DIR / f"{x['id']}.jpg"
        if src.exists():
            dst = out_dir / f"{x['id']}.jpg"
            if not dst.exists():
                shutil.copy(src, dst)

copy_split(Path("dataset/splits/annotations_train.json"), DB_DIR)
copy_split(Path("dataset/splits/annotations_val.json"), Q_DIR)
print("Done.")