import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_splits(data_dir=DATA_DIR):
    # train/val/test with columns URL, label (0 = phishing, 1 = legitimate)
    return {s: pd.read_csv(Path(data_dir) / f"{s}.csv") for s in ["train", "val", "test"]}


def load_config(data_dir=DATA_DIR):
    with open(Path(data_dir) / "config.json") as f:
        config = json.load(f)
    with open(Path(data_dir) / "vocab.json") as f:
        vocab = json.load(f)
    return config, vocab
