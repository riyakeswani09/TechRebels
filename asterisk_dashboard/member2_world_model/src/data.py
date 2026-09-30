import csv
from collections import defaultdict
from datetime import datetime
import numpy as np

META = {
    "window_id","segment","window_start","n_flows","attack_ratio",
    "window_label","label_id","is_train"
}

PACKET_FEATURE_NAMES = [
    "packet_count",
    "packet_ttl_mean",
    "packet_ttl_std",
    "tcp_window_mean",
    "tcp_window_std",
    "payload_len_mean",
    "payload_len_std",
    "packet_retransmission_rate",
    "unique_src_ports",
    "unique_dst_ports",
]


def load_windows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []
    if not rows:
        raise ValueError(f"{path}: no rows found")
    if "window_label" not in columns:
        raise ValueError(f"{path}: expected window_label")

    feature_cols = [
        c for c in columns
        if c not in META and not c.startswith("cnt_")
    ]
    X = np.asarray(
        [[float(row[c]) if row[c] not in (None, "") else 0.0 for c in feature_cols]
         for row in rows], dtype=np.float32
    )
    labels = np.asarray([str(row["window_label"]) for row in rows], dtype=str)
    attack = (np.char.upper(labels) != "BENIGN").astype(np.float32)
    return rows, X, attack, labels, feature_cols


def _parse_dt(value):
    value = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(value)


def aggregate_packet_features(packet_csv, window_rows, window_seconds=60):
    """Aggregate packet-level observations into the same temporal windows as flow data.

    Expected packet columns: timestamp, ttl, tcp_window, payload_len,
    is_retransmission, src_port, dst_port.
    """
    if not packet_csv:
        return np.zeros((len(window_rows), len(PACKET_FEATURE_NAMES)), dtype=np.float32)

    with open(packet_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        packets = list(reader)
        cols = reader.fieldnames or []
    required = {"timestamp", "ttl", "tcp_window", "payload_len",
                "is_retransmission", "src_port", "dst_port"}
    missing = required - set(cols)
    if missing:
        raise ValueError(f"{packet_csv}: missing packet columns {sorted(missing)}")

    buckets = defaultdict(lambda: {
        "n": 0, "ttl": [], "tcp": [], "payload": [], "rtx": 0,
        "src": set(), "dst": set()
    })
    epoch = datetime(1970, 1, 1)
    for row in packets:
        ts = _parse_dt(row["timestamp"])
        bucket = int((ts - epoch).total_seconds() // window_seconds)
        b = buckets[bucket]
        b["n"] += 1
        b["ttl"].append(float(row["ttl"]))
        b["tcp"].append(float(row["tcp_window"]))
        b["payload"].append(float(row["payload_len"]))
        b["rtx"] += int(float(row["is_retransmission"]))
        b["src"].add(str(row["src_port"]))
        b["dst"].add(str(row["dst_port"]))

    out = np.zeros((len(window_rows), len(PACKET_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(window_rows):
        ts = _parse_dt(row["window_start"])
        bucket = int((ts - epoch).total_seconds() // window_seconds)
        b = buckets.get(bucket)
        if not b or b["n"] == 0:
            continue
        out[i] = [
            b["n"],
            float(np.mean(b["ttl"])), float(np.std(b["ttl"])),
            float(np.mean(b["tcp"])), float(np.std(b["tcp"])),
            float(np.mean(b["payload"])), float(np.std(b["payload"])),
            float(b["rtx"] / b["n"]),
            float(len(b["src"])), float(len(b["dst"])),
        ]
    return out


def load_with_packet_features(window_path, packet_path=None):
    rows, X, attack, labels, feature_cols = load_windows(window_path)
    if packet_path:
        P = aggregate_packet_features(packet_path, rows)
        X = np.concatenate([X, P], axis=1)
        feature_cols = feature_cols + PACKET_FEATURE_NAMES
    return rows, X, attack, labels, feature_cols


def make_sequences(X, attack, labels, lookback=10, horizon=5):
    xs, ys, y_attack, y_labels = [], [], [], []
    for end in range(lookback, len(X) - horizon + 1):
        xs.append(X[end-lookback:end])
        ys.append(X[end:end+horizon])
        y_attack.append(attack[end:end+horizon])
        y_labels.append(labels[end:end+horizon])
    return (np.asarray(xs, np.float32), np.asarray(ys, np.float32),
            np.asarray(y_attack, np.float32), np.asarray(y_labels, object))
