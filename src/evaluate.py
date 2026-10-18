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


def web_app(urls):
    # hosted on web.app: only in test (all phishing), never in train/val, so reported as its own group
    return np.array([parts(u)[0].endswith(".web.app") for u in urls], dtype=bool)


def robust_recall(p_clean, p_adv, threshold=0.5):
    # share of phishing urls detected both before and after the attack (p_adv = p_clean if not attacked)
    # base = all phishing urls, so it is comparable across models (unlike adversarial_recall)
    p_clean, p_adv = np.asarray(p_clean), np.asarray(p_adv)
    return float(((p_clean >= threshold) & (p_adv >= threshold)).mean()) if len(p_clean) else float("nan")


def group_evaluation(test, p_test, adv, p_adv_clean, p_adv, threshold=0.5):
    # metrics on test_original and test_original + test_adversarial, for all / web.app / other urls
    # test: URL, label; adv: attack csv (all attacked rows); p_*: this model's p_phish on test.URL, adv.orig_URL, adv.URL
    def metrics(labels, p):
        if len(set(labels)) == 2:
            return evaluate(labels, p, threshold)
        assert set(labels) == {PHISH}  # web.app: no legit urls, so only recall means anything
        return {k: float("nan") for k in ["accuracy", "precision", "f1", "roc_auc", "pr_auc"]} | {
            "recall": adversarial_recall(p, threshold)}

    test = test[["URL", "label"]].assign(p=np.asarray(p_test), web=web_app(test.URL))
    adv = adv.assign(p_clean=np.asarray(p_adv_clean), p_adv=np.asarray(p_adv), web=web_app(adv.orig_URL))
    rows = []
    for group in ["all", "web.app", "other"]:
        t, a = (test, adv) if group == "all" else (test[test.web == (group == "web.app")], adv[adv.web == (group == "web.app")])
        e = adversarial_rows(a)
        m_orig = metrics(t.label, t.p)
        m_both = metrics(pd.concat([t.label, e.label]), np.concatenate([t.p, e.p_adv]))
        ph = t[t.label == PHISH]
        ph_adv = ph.URL.map(dict(zip(a.orig_URL, a.p_adv))).fillna(ph.p)  # not attacked: score on the original
        rows += [{"group": group, "test_set": "test_original", **m_orig},
                 {"group": group, "test_set": "test_original + test_adversarial", **m_both,
                  **adversarial_metrics(a.label, a.p_clean, a.p_adv, threshold),
                  "adversarial_recall": adversarial_recall(e.p_adv, threshold),
                  "robust_recall": robust_recall(ph.p, ph_adv, threshold), **deltas(m_orig, m_both)}]
    return pd.DataFrame(rows)


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

    assert list(web_app(["a.web.app/x", "https://b.web.app", "web.app.evil.com"])) == [True, True, False]
    assert robust_recall([0.9, 0.8, 0.2, 0.7], [0.3, 0.8, 0.2, 0.6]) == 0.5

    # groups: web.app = a, b (phishing); other = c (phishing), d, e (legit, e is a false positive)
    test = pd.DataFrame({"URL": ["a.web.app", "b.web.app/x", "c.com/p", "d.com", "e.com"], "label": [0, 0, 0, 1, 1]})
    adv = pd.DataFrame({"orig_URL": ["a.web.app", "b.web.app/x", "c.com/p"], "URL": ["ab.web.app", "b.web.app/x", "c.com/q"],
                        "label": 0, "n_edits": [1, 0, 1], "duplicate": False})
    g = group_evaluation(test, [0.9, 0.8, 0.9, 0.2, 0.6], adv, [0.9, 0.8, 0.9], [0.3, 0.8, 0.7]).set_index(["group", "test_set"])
    assert len(g) == 6
    w, o, al = (g.loc[(grp, "test_original + test_adversarial")] for grp in ["web.app", "other", "all"])
    assert g.loc[("web.app", "test_original"), "recall"] == 1 and np.isclose(w.recall, 2 / 3) and np.isnan(w.f1)
    assert w.robust_recall == 0.5 and w.attack_success_rate == 0.5 and np.isclose(w.delta_recall, -1 / 3)
    assert o.recall == 1 and o.precision == 2 / 3 and o.robust_recall == 1 and o.attack_success_rate == 0
    assert np.isclose(al.robust_recall, 2 / 3) and np.isclose(al.attack_success_rate, 1 / 3)

    # sanity check: path rule baseline on test
    from load_data import load_splits
    test = load_splits()["test"]
    m = evaluate(test.label, path_baseline(test.URL))
    print("path baseline:", {k: round(v, 4) for k, v in m.items() if k != "confusion_matrix"})
    assert round(m["accuracy"], 4) == 0.621 and round(m["f1"], 4) == 0.1716
    print("all checks passed")
