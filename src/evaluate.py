import re

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from validate_adv import parts

PHISH = 0  # positive class


def evaluate(labels, p_phish, threshold=0.5):
    # predict phishing when p_phish >= threshold
    labels = np.asarray(labels)
    pred = np.where(np.asarray(p_phish) >= threshold, PHISH, 1)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, pos_label=PHISH, zero_division=0)),
        "recall": float(recall_score(labels, pred, pos_label=PHISH, zero_division=0)),
        "f1": float(f1_score(labels, pred, pos_label=PHISH, zero_division=0)),
        # threshold-free, scored on p_phish with phishing as positive
        "roc_auc": float(roc_auc_score(labels == PHISH, p_phish)),
        "pr_auc": float(average_precision_score(labels == PHISH, p_phish)),
        "confusion_matrix": confusion_matrix(labels, pred, labels=[PHISH, 1]).tolist(),  # rows = true [phish, legit]
    }


def adversarial_metrics(labels, p_clean, p_adv, threshold=0.5):
    # rows aligned: p_adv[i] is the score of the attacked version of URL i
    labels, p_clean, p_adv = map(np.asarray, (labels, p_clean, p_adv))
    phish = labels == PHISH
    attempts = phish & (p_clean >= threshold)  # phishing that was detected before the attack
    return {
        "attack_success_rate": float((p_adv[attempts] < threshold).mean()) if attempts.any() else float("nan"),
        "n_attempts": int(attempts.sum()),
        "confidence_reduction": float((p_clean[phish] - p_adv[phish]).mean()),
    }


def deltas(m_orig, m_adv):
    # change from test_original to test_original + test_adversarial (both from evaluate())
    return {f"delta_{k}": m_adv[k] - m_orig[k] for k in ["accuracy", "recall", "f1"]}


def adversarial_rows(adv):
    # rows of an attack csv (columns in attack-rules.md) that get appended to the clean set: edited, not duplicate
    return adv[(adv.n_edits > 0) & ~adv.duplicate]


def adversarial_recall(p_adv, threshold=0.5):
    # recall on the adversarial rows alone (all phishing): share of attacked URLs the model still detects
    p_adv = np.asarray(p_adv)
    return float((p_adv >= threshold).mean()) if len(p_adv) else float("nan")


def edited_parts(orig, adv):
    # which URL parts the attack changed, e.g. "subdomain" or "path+query" ("none" if unchanged)
    def split(url):
        host, reg, *_ = parts(url)
        rest = re.sub(r"^([a-zA-Z][a-zA-Z0-9+.\-]*://)?[^/?#]*", "", url)  # drop scheme + host
        path, query, fragment = re.fullmatch(r"([^?#]*)(\?[^#]*)?(#.*)?", rest).groups()
        return {"subdomain": host[:len(host) - len(reg)], "path": path, "query": query, "fragment": fragment}

    a, b = split(orig), split(adv)
    return "+".join(k for k in a if a[k] != b[k]) or "none"


def attack_summary(adv, k=3, threshold=0.5):
    # counts for an attack csv (columns in attack-rules.md); success rates are over all attacked URLs
    edited, flipped = adv.n_edits > 0, adv.p_adv < threshold
    where = pd.Series([edited_parts(o, a) for o, a in zip(adv.orig_URL[edited], adv.URL[edited])], dtype=str)
    return {
        "attacked": len(adv),
        "flipped": int((edited & flipped).sum()),
        "edited_not_flipped": int((edited & ~flipped).sum()),
        "not_edited": int((~edited).sum()),
        "duplicates_dropped": int((edited & adv.duplicate).sum()),
        # cumulative: share of attacked URLs flipped with <= j edits
        "success_within_edits": {j: float((edited & flipped & (adv.n_edits <= j)).mean()) for j in range(1, k + 1)},
        "edited_parts": {p: int(n) for p, n in where.value_counts().items()},
        "flipped_parts": {p: int(n) for p, n in where[flipped[edited].values].value_counts().items()},
    }


def path_baseline(urls):
    # reference rule: phishing if URL has a path/query/fragment (all legit URLs are bare domains), returns p_phish (0 or 1)
    return pd.Series(urls).str.contains(r"[/?#]").astype(float).to_numpy()


def save_preds(path, urls, labels, p_phish):
    # common format for all models
    pd.DataFrame({"URL": urls, "label": labels, "p_phish": p_phish}).to_csv(path, index=False)


if __name__ == "__main__":
    # toy check (hand-computed)
    y = [0, 0, 0, 1, 1]
    p = [0.9, 0.8, 0.2, 0.6, 0.1]  # TP=2, FN=1, FP=1, TN=1
    m = evaluate(y, p)
    assert m["confusion_matrix"] == [[2, 1], [1, 1]]
    assert np.isclose(m["recall"], 2 / 3) and np.isclose(m["precision"], 2 / 3) and np.isclose(m["accuracy"], 3 / 5)
    assert np.isclose(m["roc_auc"], 5 / 6) and np.isclose(m["pr_auc"], (1 + 1 + 3 / 4) / 3)

    a = adversarial_metrics(y, p, [0.3, 0.9, 0.1, 0.6, 0.1])  # 2 attempts, 1 success
    assert a["n_attempts"] == 2 and a["attack_success_rate"] == 0.5
    assert np.isclose(a["confidence_reduction"], (0.6 - 0.1 + 0.1) / 3)

    assert deltas({"accuracy": .9, "recall": .8, "f1": .85}, {"accuracy": .8, "recall": .5, "f1": .6})["delta_recall"] == .5 - .8
    assert adversarial_recall([0.9, 0.3, 0.6, 0.1]) == 0.5

    assert edited_parts("a.b.com/x?q=1#f", "a.b.com/x?q=1#f") == "none"
    assert edited_parts("a.b.com/x?q=1#f", "c.b.com/x?q=1#f") == "subdomain"
    assert edited_parts("a.b.com/x?q=1#f", "a.b.com/y?q=2#f") == "path+query"
    assert edited_parts("a.b.com/xy", "a.b.com/x?y") == "path+query"  # '?' turns part of the path into a query
    assert edited_parts("b.com", "b.com/a#b") == "path+fragment"

    adv = pd.DataFrame({"orig_URL": ["a.b.com", "b.com/x", "b.com/y", "c.com/z"],
                        "URL": ["e.b.com", "b.com/xy", "b.com/y", "c.com/zz"],
                        "p_clean": [0.9, 0.8, 0.7, 0.9], "p_adv": [0.2, 0.6, 0.7, 0.4],
                        "n_edits": [1, 1, 0, 3], "duplicate": [False, False, False, True]})
    s = attack_summary(adv)
    assert (s["flipped"], s["edited_not_flipped"], s["not_edited"], s["duplicates_dropped"]) == (2, 1, 1, 1)
    assert s["success_within_edits"] == {1: 0.25, 2: 0.25, 3: 0.5}
    assert s["edited_parts"] == {"path": 2, "subdomain": 1} and s["flipped_parts"] == {"subdomain": 1, "path": 1}
    assert list(adversarial_rows(adv).URL) == ["e.b.com", "b.com/xy"]

    # sanity check: path rule baseline on test
    from load_data import load_splits
    test = load_splits()["test"]
    m = evaluate(test.label, path_baseline(test.URL))
    print("path baseline:", {k: round(v, 4) for k, v in m.items() if k != "confusion_matrix"})
    assert round(m["accuracy"], 4) == 0.621 and round(m["f1"], 4) == 0.1716
    print("all checks passed")
