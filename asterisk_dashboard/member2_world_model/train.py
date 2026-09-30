import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.data import load_with_packet_features, make_sequences
from src.models import GRUWorldModel, TemporalTransformerWorldModel


def seed_all(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "support": {"benign": int((y == 0).sum()), "attack": int((y == 1).sum())},
    }


def roc_auc(y, scores):
    """ROC-AUC using the rank/Mann-Whitney formulation; no sklearn required."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    pos_ranks = ranks[y == 1].sum()
    auc = (pos_ranks - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


class TorchLogisticRegression(torch.nn.Module):
    """A plain linear logistic-regression baseline implemented in PyTorch."""
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_logistic(X, y, device, epochs=200, lr=1e-2):
    model = TorchLogisticRegression(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device)
    )
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        opt.step()
    return model


def predict_model(model, xs, device, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            xb = torch.tensor(xs[i:i + batch], dtype=torch.float32, device=device)
            _, logits = model(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


def make_dataset_sequences(X, A, L, mean, std, lookback, horizon):
    Xn = ((X - mean) / std).astype(np.float32)
    return make_sequences(Xn, A, L, lookback, horizon)


def eval_horizons(probs, attack, prefix):
    results = {}
    for h in (1, 3, 5):
        j = h - 1
        p = probs[:, j]
        m = binary_metrics(attack[:, j], p)
        auc = roc_auc(attack[:, j], p)
        if auc is not None:
            m["roc_auc"] = auc
        results[f"{prefix}_horizon_{h}"] = m
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data2017", default="data/cic2017_processed/windows.csv")
    ap.add_argument("--data2018", default="data/synthetic2018/windows.csv")
    ap.add_argument("--packet2017", default="data/packet_features/synthetic2017_packets.csv")
    ap.add_argument("--packet2018", default="data/packet_features/synthetic2018_packets.csv")
    ap.add_argument("--model", choices=["transformer", "gru"], default="transformer")
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_fraction", type=float, default=0.10)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()
    seed_all()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    r17, X17, A17, L17, F17 = load_with_packet_features(args.data2017, args.packet2017)
    r18, X18, A18, L18, F18 = load_with_packet_features(args.data2018, args.packet2018)

    common = [c for c in F17 if c in F18]
    if len(common) < 10:
        raise ValueError("Too few common state features between datasets.")
    X17 = X17[:, [F17.index(c) for c in common]]
    X18 = X18[:, [F18.index(c) for c in common]]
    print(f"Features: {len(common)} ({len(common)-33} packet-level development features)")

    # Chronological 70/30 split within EACH dataset. This avoids mixing temporal
    # sequences across the 2017/2018 boundary and gives two independent test reports.
    cut17 = int(len(X17) * 0.70)
    cut18 = int(len(X18) * 0.70)
    train_parts = [(X17[:cut17], A17[:cut17], L17[:cut17]),
                   (X18[:cut18], A18[:cut18], L18[:cut18])]
    test_parts = [(X17[cut17:], A17[cut17:], L17[cut17:]),
                  (X18[cut18:], A18[cut18:], L18[cut18:])]

    X_train_raw = np.concatenate([p[0] for p in train_parts], axis=0)
    mean = X_train_raw.mean(0)
    std = np.where(X_train_raw.std(0) < 1e-6, 1.0, X_train_raw.std(0))

    seq_train = [make_dataset_sequences(*p, mean, std, args.lookback, args.horizon)
                 for p in train_parts]
    seq_test = [make_dataset_sequences(*p, mean, std, args.lookback, args.horizon)
                for p in test_parts]

    xtr = np.concatenate([s[0] for s in seq_train])
    ytr = np.concatenate([s[1] for s in seq_train])
    atr = np.concatenate([s[2] for s in seq_train])
    ltr = np.concatenate([s[3] for s in seq_train])
    x17e, y17e, a17e, l17e = seq_test[0]
    x18e, y18e, a18e, l18e = seq_test[1]

    if len(xtr) == 0 or len(x17e) == 0 or len(x18e) == 0:
        raise ValueError("Not enough sequence data after splitting.")

    # Chronological validation tail from training sequences only.
    n_val = max(1, int(len(xtr) * args.val_fraction))
    x_fit, y_fit, a_fit = xtr[:-n_val], ytr[:-n_val], atr[:-n_val]
    x_val, y_val, a_val = xtr[-n_val:], ytr[-n_val:], atr[-n_val:]

    ds = TensorDataset(torch.tensor(x_fit), torch.tensor(y_fit), torch.tensor(a_fit))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        pin_memory=(device.type == "cuda"))

    if args.model == "transformer":
        model = TemporalTransformerWorldModel(xtr.shape[-1], horizon=args.horizon).to(device)
    else:
        model = GRUWorldModel(xtr.shape[-1], horizon=args.horizon).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    state_loss = torch.nn.SmoothL1Loss()
    pos = max(float(a_fit.sum()), 1.0)
    neg = max(float(a_fit.size - a_fit.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos] * args.horizon, device=device)
    risk_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb, ab in loader:
            xb, yb, ab = xb.to(device), yb.to(device), ab.to(device)
            opt.zero_grad(set_to_none=True)
            pred_state, risk_logits = model(xb)
            loss = 0.65 * state_loss(pred_state, yb) + 0.35 * risk_loss(risk_logits, ab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(x_val, dtype=torch.float32, device=device)
            yv = torch.tensor(y_val, dtype=torch.float32, device=device)
            av = torch.tensor(a_val, dtype=torch.float32, device=device)
            ps, rl = model(xv)
            val_loss = 0.65 * state_loss(ps, yv) + 0.35 * risk_loss(rl, av)
            val_loss = float(val_loss.item())
        print(f"Epoch {epoch:02d}/{args.epochs} train_loss={np.mean(losses):.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.5f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    ckpt_path = out / f"{args.model}_world_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "features": common,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "model": args.model,
        "mean": mean,
        "std": std,
        "packet_features": common[-10:],
    }, ckpt_path)

    probs17 = predict_model(model, x17e, device, args.batch)
    probs18 = predict_model(model, x18e, device, args.batch)

    results = {
        "evaluation_protocol": {
            "train_fraction_per_dataset": 0.70,
            "test_fraction_per_dataset": 0.30,
            "datasets": ["CIC-IDS2017", "synthetic-2018-development"],
            "packet_features": common[-10:],
            "note": "Synthetic packet and synthetic-2018 results are development-only."
        },
        "validation": {"best_loss": best_val},
    }
    results.update(eval_horizons(probs17, a17e, "cic2017"))
    results.update(eval_horizons(probs18, a18e, "synthetic2018"))

    # Required Logistic Regression baseline, implemented in PyTorch to avoid the
    # Windows Application Control DLL issue affecting scikit-learn.
    lr_model = train_logistic(x_fit[:, -1, :], a_fit[:, 0].astype(np.float32), device)
    lr_model.eval()
    with torch.no_grad():
        p17 = torch.sigmoid(lr_model(torch.tensor(x17e[:, -1, :], dtype=torch.float32, device=device))).cpu().numpy()
        p18 = torch.sigmoid(lr_model(torch.tensor(x18e[:, -1, :], dtype=torch.float32, device=device))).cpu().numpy()
    results["logistic_baseline_cic2017_horizon_1"] = binary_metrics(a17e[:, 0], p17)
    results["logistic_baseline_cic2017_horizon_1"]["roc_auc"] = roc_auc(a17e[:, 0], p17)
    results["logistic_baseline_synthetic2018_horizon_1"] = binary_metrics(a18e[:, 0], p18)
    results["logistic_baseline_synthetic2018_horizon_1"]["roc_auc"] = roc_auc(a18e[:, 0], p18)
    torch.save({"state_dict": lr_model.state_dict(), "features": common,
                "mean": mean, "std": std}, out / "logistic_baseline.pt")

    with open(out / f"{args.model}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nTest results:")
    print(json.dumps(results, indent=2))
    print("\nSaved model to", ckpt_path)
    print("Saved baseline to", out / "logistic_baseline.pt")


if __name__ == "__main__":
    main()
