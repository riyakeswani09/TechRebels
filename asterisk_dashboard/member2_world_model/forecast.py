import argparse
import numpy as np
import torch
from src.data import load_with_packet_features
from src.models import TemporalTransformerWorldModel, GRUWorldModel

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/synthetic2018/windows.csv")
ap.add_argument("--packet", default="data/packet_features/synthetic2018_packets.csv")
ap.add_argument("--model", default="artifacts/transformer_world_model.pt")
args = ap.parse_args()

ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
rows, X, labels, attack, features = (None, None, None, None, None)
rows, X, attack, labels, features = load_with_packet_features(args.data, args.packet)
idx = [features.index(c) for c in ckpt["features"]]
X = X[:, idx]
X = ((X - ckpt["mean"]) / ckpt["std"]).astype(np.float32)
lookback = ckpt["lookback"]
horizon = ckpt["horizon"]

if ckpt["model"] == "transformer":
    model = TemporalTransformerWorldModel(X.shape[-1], horizon=horizon)
else:
    model = GRUWorldModel(X.shape[-1], horizon=horizon)
model.load_state_dict(ckpt["state_dict"])
model.eval()

with torch.no_grad():
    future_state, logits = model(torch.tensor(X[-lookback:]).unsqueeze(0))
    probs = torch.sigmoid(logits).squeeze(0).numpy()
    future_state = future_state.squeeze(0).numpy()

print("\n=== NETWORK ATTACK FORECAST ===")
for i, p in enumerate(probs, 1):
    print(f"Next window +{i}: {p:.1%} attack probability")
print("\nForecasted state vectors:", future_state.shape)
print("Top-changing features relative to current state:")
cur = X[-1]
for step in range(min(horizon, 5)):
    delta = np.abs(future_state[step] - cur)
    top = np.argsort(delta)[-5:][::-1]
    print(f"\n+{step+1}:")
    for j in top:
        print(f"  {features[idx[j]] if False else ckpt['features'][j]}  Δ={delta[j]:.3f}")
