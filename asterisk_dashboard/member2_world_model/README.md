# Member 2 — AI World Model MVP v3

## What is included now

This version completes the three requested Member 2 upgrades:

1. **Flow-level + packet-level features** in one temporal state vector.
2. **Two-dataset evaluation** with chronological 70/30 holdouts for CIC-IDS2017 and synthetic-2018 development data.
3. **Logistic Regression baseline** implemented in PyTorch, avoiding the Windows Application Control DLL issue that blocks scikit-learn on the current laptop.

The main model remains a Temporal Transformer:
- 10 past windows as input
- 5 future windows predicted directly
- future-state regression + attack-risk prediction
- CUDA support for the RTX 3050
- validation loss + early stopping

## Packet-level integration

`src/data.py` aggregates packet observations into the same 60-second windows as the flow/state data. Expected packet CSV columns:

`timestamp, ttl, tcp_window, payload_len, is_retransmission, src_port, dst_port`

Ten packet-level aggregate features are appended to the 33 flow/state features:

- packet_count
- packet_ttl_mean / std
- tcp_window_mean / std
- payload_len_mean / std
- packet_retransmission_rate
- unique_src_ports
- unique_dst_ports

The included packet CSVs are **synthetic development data** generated from the window labels. They demonstrate the integration path but must not be presented as real packet captures.

When real PCAP-derived packet features are available, generate a packet-feature CSV with the same columns and pass it with `--packet2017` / `--packet2018`.

## Evaluation

Training uses the first 70% of each dataset chronologically. Testing uses the final 30% of each dataset separately. Sequences are created independently per dataset, so a sequence never crosses the 2017/2018 boundary.

Metrics are reported separately for:
- CIC-IDS2017 held-out region
- synthetic-2018 held-out region

Metrics at horizons +1, +3, +5:
- precision
- recall
- F1
- false-positive rate
- ROC-AUC

The synthetic-2018 and synthetic packet results are development-only.

## Logistic Regression baseline

The required simple baseline is a standard linear logistic-regression classifier using the latest state in each input sequence. It is implemented as a single `torch.nn.Linear` layer trained with binary cross entropy. This is mathematically a logistic regression model and avoids scikit-learn's blocked native DLL on the current Windows environment.

## Run

From the project folder:

```powershell
python train.py --model transformer --epochs 50
```

Expected new inputs:

```text
data/cic2017_processed/windows.csv
data/synthetic2018/windows.csv
data/packet_features/synthetic2017_packets.csv
data/packet_features/synthetic2018_packets.csv
```

Forecast:

```powershell
python forecast.py --model artifacts/transformer_world_model.pt --data data/synthetic2018/windows.csv --packet data/packet_features/synthetic2018_packets.csv
```

## Important

The existing 50-epoch checkpoint from the previous version is compatible with the old 33-feature pipeline, **not** this new 43-feature packet-integrated model. Keep the old checkpoint if you want to preserve the earlier experiment. The new training command creates a new 43-feature checkpoint.

Do not describe the synthetic datasets as real CIC-IDS2018 or real PCAP evidence in the SIH submission.
