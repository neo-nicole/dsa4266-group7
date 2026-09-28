# DSA4266 Group 7: Adversarial ML for Phishing URL Detection

How vulnerable are URL-based phishing detectors (char-CNN, char-BiLSTM, URLBERT) to adversarial URL edits, and does adversarial training help?

Dataset: [PhiUSIIL Phishing URL](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset), URL only. Label `0` = phishing (positive class), `1` = legitimate.

## Structure
```
data/
  raw/                  original PhiUSIIL csv
  train.csv, val.csv, test.csv, vocab.json, config.json   cleaned splits (from cleaning_and_eda)
notebooks/              one notebook per step / model
  cleaning_and_eda.ipynb
src/                    shared code
  load_data.py          load splits, config, vocab
  evaluate.py           metrics (clean + adversarial), save predictions
results/                metrics, plots, prediction csvs (created by model notebooks)
```

## Setup
```
pip install -r requirements.txt
python src/evaluate.py   # self-check
```

In a notebook:
```python
import sys; sys.path.append("../src")
from load_data import load_splits
from evaluate import evaluate
```

Predictions are saved as `URL,label,p_phish` (`evaluate.save_preds`), so models can be compared on the same files.
