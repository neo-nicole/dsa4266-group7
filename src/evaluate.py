import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

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


def regex_baseline(urls):
    # reference rule: legit only if bare https://www.<domain>, returns p_phish (0 or 1)
    return (~pd.Series(urls).str.match(r"^https://www\.[^/?#]+$")).astype(float).to_numpy()


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

    a = adversarial_metrics(y, p, [0.3, 0.9, 0.1, 0.6, 0.1])  # 2 attempts, 1 success
    assert a["n_attempts"] == 2 and a["attack_success_rate"] == 0.5
    assert np.isclose(a["confidence_reduction"], (0.6 - 0.1 + 0.1) / 3)

    # sanity check: regex rule baseline on test
    from load_data import load_splits
    test = load_splits()["test"]
    m = evaluate(test.label, regex_baseline(test.URL))
    print("regex baseline:", {k: round(v, 4) for k, v in m.items() if k != "confusion_matrix"})
    assert round(m["accuracy"], 4) == 0.9976 and round(m["f1"], 4) == 0.9971
    print("all checks passed")
