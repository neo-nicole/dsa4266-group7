import argparse
import json
import re
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from tqdm.auto import tqdm

from evaluate import (
    adversarial_metrics,
    adversarial_recall,
    adversarial_rows,
    attack_summary,
    deltas,
    evaluate,
    save_preds,
)
from load_data import load_config, load_splits
from validate_adv import SCHEME, drop_invalid_originals, edit_distance, invalid_reason, parts


HOST_CHARS = string.ascii_lowercase + string.digits + "-."
PATH_CHARS = HOST_CHARS + "_~/=&?#"
PHISH = 0


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, pad_idx, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward_embeddings(self, embedded, lengths):
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(representation))

    def forward(self, ids, lengths):
        return self.forward_embeddings(self.embedding(ids), lengths)


def load_model(checkpoint, vocab_size, pad_idx, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    embedding_dim = state["embedding.weight"].shape[1]
    hidden_dim = state["lstm.weight_ih_l0"].shape[0] // 4
    layers = [
        int(match.group(1))
        for key in state
        if (match := re.search(r"lstm\.weight_ih_l(\d+)$", key))
    ]
    num_layers = max(layers) + 1
    model = BiLSTMClassifier(
        vocab_size,
        pad_idx,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout=0.3,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_urls(urls, char_to_idx, unk_idx, pad_idx, max_len):
    sequences = [
        torch.tensor(
            [char_to_idx.get(character, unk_idx) for character in url[:max_len]],
            dtype=torch.long,
        )
        for url in urls
    ]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    ids = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
    return ids, lengths


def make_predictor(model, char_to_idx, unk_idx, pad_idx, max_len, device):
    @torch.no_grad()
    def p_phish(urls, batch_size=1024):
        probabilities = np.zeros(len(urls), dtype=np.float32)
        for start in range(0, len(urls), batch_size):
            batch_urls = urls[start:start + batch_size]
            ids, lengths = encode_urls(batch_urls, char_to_idx, unk_idx, pad_idx, max_len)
            logits = model(ids.to(device), lengths.to(device)).float()
            probabilities[start:start + len(batch_urls)] = (
                torch.softmax(logits, dim=-1)[:, PHISH].cpu().numpy()
            )
        return probabilities

    return p_phish


def domain_span(url):
    host, registered_domain, *_ = parts(url)
    scheme_match = SCHEME.match(url)
    start = scheme_match.end() if scheme_match else 0
    authority = re.match(r"[^/?#]*", url[start:]).group()
    start += authority.rfind("@") + 1
    return start + len(host) - len(registered_domain), start + len(host)


def candidate_urls(url, positions):
    domain_start, domain_end = domain_span(url)
    candidates = {url + character for character in PATH_CHARS}
    candidates.update({url + "/" + character for character in PATH_CHARS})

    for position in positions:
        for insertion_position in {position, position + 1}:
            if domain_start <= insertion_position < domain_end:
                continue
            characters = HOST_CHARS if insertion_position < domain_end else PATH_CHARS
            candidates.update(
                url[:insertion_position] + character + url[insertion_position:]
                for character in characters
            )

        if not domain_start <= position < domain_end:
            characters = HOST_CHARS if position < domain_end else PATH_CHARS
            candidates.update(
                url[:position] + character + url[position + 1:]
                for character in characters
            )
            candidates.add(url[:position] + url[position + 1:])

    candidates.discard(url)
    return list(candidates)


def token_scores(
    urls,
    model,
    embedding,
    valid_embedding_ids,
    char_to_idx,
    unk_idx,
    pad_idx,
    max_len,
    device,
):
    ids, lengths = encode_urls(urls, char_to_idx, unk_idx, pad_idx, max_len)
    ids = ids.to(device)
    lengths = lengths.to(device)
    embedded = embedding[ids].detach().requires_grad_()
    labels = torch.zeros(len(urls), dtype=torch.long, device=device)

    # cuDNN does not support RNN backward in eval mode; disabling it keeps dropout off.
    with torch.backends.cudnn.flags(enabled=False):
        logits = model.forward_embeddings(embedded, lengths)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        loss.backward()

    gradients = embedded.grad
    candidate_embeddings = embedding[valid_embedding_ids]
    scores = (gradients @ candidate_embeddings.T).max(dim=-1).values
    scores -= (gradients * embedded).sum(dim=-1)
    return scores.detach().cpu().tolist(), lengths.detach().cpu().tolist()


def attack_batch(
    urls,
    probabilities,
    model,
    embedding,
    valid_embedding_ids,
    char_to_idx,
    unk_idx,
    pad_idx,
    max_len,
    budget,
    top_positions,
    device,
    p_phish,
):
    adversarial_urls = list(urls)
    current_probabilities = list(probabilities)
    finished = [False] * len(urls)

    for _ in range(budget):
        live = [
            index
            for index in range(len(urls))
            if not finished[index] and current_probabilities[index] >= 0.5
        ]
        if not live:
            break

        scores, lengths = token_scores(
            [adversarial_urls[index] for index in live],
            model,
            embedding,
            valid_embedding_ids,
            char_to_idx,
            unk_idx,
            pad_idx,
            max_len,
            device,
        )
        candidate_groups = []
        for index, score, length in zip(live, scores, lengths):
            domain_start, domain_end = domain_span(adversarial_urls[index])
            editable_positions = [
                position
                for position in range(length)
                if not domain_start <= position < domain_end
            ]
            selected_positions = sorted(
                editable_positions,
                key=lambda position: -score[position],
            )[:top_positions]
            possible = candidate_urls(adversarial_urls[index], selected_positions)
            candidate_groups.append(
                [
                    candidate
                    for candidate in possible
                    if not invalid_reason(urls[index], candidate, budget)
                ]
            )

        flattened = [candidate for group in candidate_groups for candidate in group]
        candidate_probabilities = p_phish(flattened) if flattened else np.array([])
        offset = 0
        for index, candidates in zip(live, candidate_groups):
            probabilities_for_url = candidate_probabilities[offset:offset + len(candidates)]
            offset += len(candidates)
            if len(candidates) and probabilities_for_url.min() < current_probabilities[index]:
                best = int(probabilities_for_url.argmin())
                adversarial_urls[index] = candidates[best]
                current_probabilities[index] = float(probabilities_for_url[best])
            else:
                finished[index] = True

    return adversarial_urls


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BiLSTM adversarial URL examples")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--frac", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-positions", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/bilstm_model1.pt"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--partial", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.frac <= 1:
        raise ValueError("--frac must be in (0, 1]")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config, vocab = load_config()
    max_len = config["max_len"]
    budget = config["budget"]
    char_to_idx = {
        character: index
        for index, character in enumerate(vocab["index_to_string"])
    }
    pad_idx = vocab["pad_idx"]
    unk_idx = vocab["unk_idx"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    splits = load_splits()
    model = load_model(args.checkpoint, len(char_to_idx), pad_idx, device)
    embedding = model.embedding.weight
    valid_embedding_ids = torch.tensor(
        [
            index
            for character, index in char_to_idx.items()
            if character not in {"<PAD>", "<UNK>"}
        ],
        dtype=torch.long,
        device=device,
    )
    p_phish = make_predictor(model, char_to_idx, unk_idx, pad_idx, max_len, device)

    phishing = splits[args.split][splits[args.split].label == PHISH].copy()
    probabilities = p_phish(phishing.URL.tolist())
    targets = phishing[probabilities >= 0.5].copy()
    targets["p_clean"] = probabilities[probabilities >= 0.5]
    if args.split == "train":
        targets = targets.sample(frac=args.frac, random_state=args.seed)
    if args.n is not None:
        targets = targets.sample(min(args.n, len(targets)), random_state=args.seed)

    output = args.output or Path("data") / f"bilstm_adv_{args.split}.csv"
    partial = args.partial or output.with_name(output.stem + "_partial.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    completed = (
        pd.read_csv(partial, keep_default_na=False)
        if partial.exists()
        else pd.DataFrame(columns=["orig_URL", "URL"])
    )
    todo = targets[~targets.URL.isin(completed.orig_URL)]
    start_time = time.time()
    for start in tqdm(range(0, len(todo), args.batch_size), desc=f"attack {args.split}"):
        chunk = todo.iloc[start:start + args.batch_size]
        adversarial = attack_batch(
            chunk.URL.tolist(),
            chunk.p_clean.tolist(),
            model,
            embedding,
            valid_embedding_ids,
            char_to_idx,
            unk_idx,
            pad_idx,
            max_len,
            budget,
            args.top_positions,
            device,
            p_phish,
        )
        pd.DataFrame({"orig_URL": chunk.URL.values, "URL": adversarial}).to_csv(
            partial,
            mode="a",
            header=not partial.exists(),
            index=False,
        )

    attack_map = dict(pd.read_csv(partial, keep_default_na=False).values) if partial.exists() else {}
    adversarial_df = pd.DataFrame(
        {
            "orig_URL": targets.URL.values,
            "URL": [attack_map.get(url, url) for url in targets.URL],
            "label": PHISH,
            "p_clean": targets.p_clean.values,
        }
    )
    adversarial_df["p_adv"] = p_phish(adversarial_df.URL.tolist())
    adversarial_df["n_edits"] = [
        edit_distance(original, edited)
        for original, edited in zip(adversarial_df.orig_URL, adversarial_df.URL)
    ]
    adversarial_df["flipped"] = adversarial_df.p_adv < 0.5
    reasons = pd.Series(
        [
            invalid_reason(original, edited, budget)
            for original, edited in zip(adversarial_df.orig_URL, adversarial_df.URL)
        ]
    )
    if not (reasons == "").all():
        raise RuntimeError(f"invalid adversarial URLs: {reasons[reasons != ''].value_counts().to_dict()}")
    adversarial_df["duplicate"] = (adversarial_df.n_edits > 0) & (
        adversarial_df.URL.isin(splits[args.split].URL)
        | adversarial_df.URL.duplicated()
    )

    summary = {
        "model": "bilstm",
        "split": args.split,
        "phishing": int(len(phishing)),
        "missed_by_model": int((probabilities < 0.5).sum()),
        **attack_summary(adversarial_df, budget),
        "n_edits": {
            int(count): int(number)
            for count, number in adversarial_df.n_edits.value_counts().sort_index().items()
        },
        "sec_per_url": round((time.time() - start_time) / max(1, len(todo)), 4),
    }
    adversarial_df.to_csv(output, index=False)
    if partial.exists():
        partial.unlink()
    with (args.results_dir / f"bilstm_adv_{args.split}_summary.json").open("w") as file:
        json.dump(summary, file, indent=1)

    if args.split == "test":
        test = splits["test"]
        clean_probabilities = p_phish(test.URL.tolist())
        edited = adversarial_rows(adversarial_df)
        combined = pd.concat(
            [test[["URL", "label"]], edited[["URL", "label"]]],
            ignore_index=True,
        )
        combined_probabilities = np.concatenate(
            [clean_probabilities, edited.p_adv.values]
        )
        clean_metrics = evaluate(test.label, clean_probabilities)
        combined_metrics = evaluate(combined.label, combined_probabilities)
        metrics = adversarial_metrics(
            adversarial_df.label,
            adversarial_df.p_clean,
            adversarial_df.p_adv,
        )
        metrics["adversarial_recall"] = adversarial_recall(edited.p_adv)
        metrics.update(deltas(clean_metrics, combined_metrics))
        save_preds(
            args.results_dir / "bilstm_test_adversarial.csv",
            combined.URL,
            combined.label,
            combined_probabilities,
        )
        with (args.results_dir / "bilstm_test_adversarial.json").open("w") as file:
            json.dump({**combined_metrics, **metrics}, file, indent=1)

    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
