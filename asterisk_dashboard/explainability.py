"""
explainability.py  -  Member 4 (Tech Rebels, SIH 2026)  [corrected version]

Same interface as before, so app.py and static/index.html work unchanged:
    ex = Explainer(feature_names, stages, k_steps=3)
    ex.fit(X_train_windows, y_train_stage, normal_class=0)
    out = ex.explain(window, pred_probs, stage, risk)

Shapes:
    X windows : (n_samples, window_size, n_features)
    y_stage   : (n_samples,) integer index into `stages`
    window    : (window_size, n_features)
    pred_probs: 1D array, one probability per stage (same order as `stages`)
    stage     : stage NAME string (must be one of `stages`)
    risk      : "LOW" / "MEDIUM" / "HIGH"

What was fixed vs. the old version
    1. Benign predictions no longer say "Abnormal ..." / "attack likely" - they
       report a normal-activity message instead.
    2. A feature is only called abnormal if |z| >= Z_ABNORMAL (default 2.0);
       otherwise the reason says it is within the normal range.
    3. Feature meanings are matched at word starts ("ack" no longer matches "backward") on the cleaned name (suffix "__mean"/"__std"
       is stripped, "std" -> variability). Specific keys (syn, port, iat, ...)
       win over generic ones; "Fwd Packet Length" no longer becomes "traffic
       from source".
    4. Duplicate meanings are removed from reasons and from the warning text.
    5. z-score uses a safe std floor (no huge z for constant features).
    6. Column-vector safe SHAP handling for every shap version (list / 3-D array).
    7. Stage not in the list is handled safely; probs length is validated.
    8. Horizon text uses k_steps set in the constructor (no hard-coded "3").
"""
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

Z_ABNORMAL = 2.0          # |z| at/above this => "abnormal"
Z_MILD = 1.0              # |z| between MILD and ABNORMAL => "slightly"

# ---------- 1. Plain-English feature meanings ----------
# Matched by substring on the cleaned lowercase name. ORDER MATTERS: first match wins,
# so put specific keys before generic ones.
FEATURE_MEANING = [
    ("syn",       "SYN packet activity (possible scan or SYN flood)"),
    ("rst",       "Connection resets (possible port probing)"),
    ("fin",       "Connection-teardown flags"),
    ("ack",       "ACK flag activity (handshake / session behaviour)"),
    ("destination port", "Destination ports contacted (port sweep or targeted service)"),
    ("port",      "Ports being contacted (port sweep)"),
    ("protocol",  "Mix of network protocols in use"),
    ("login",     "Repeated login attempts (possible brute force)"),
    ("failed",    "Repeated failed attempts (possible brute force)"),
    ("iat",       "Timing between packets (automated tool behaviour)"),
    ("idle",      "Idle time between bursts of activity"),
    ("init_win",  "TCP initial window size (client/tool fingerprint)"),
    ("header",    "Packet header size (crafted or unusual packets)"),
    ("flow duration", "Connection duration"),
    ("duration",  "Connection duration"),
    ("bytes/s",   "Data rate (bytes per second)"),
    ("packets/s", "Packet rate (packets per second)"),
    ("flow_count", "Number of flows (connection volume)"),
    ("count",     "Number of connections to the same host/service"),
    ("backward",  "Response traffic from the target"),
    ("bwd",       "Response traffic from the target"),
    ("packet length", "Packet size"),
    ("packet size",   "Average packet size"),
    ("length",    "Amount of data sent"),
    ("fwd",       "Traffic from the source towards the target"),
    ("bytes",     "Data volume transferred"),
    ("pkts",      "Packet count"),
    ("packet",    "Packet count / size"),
    ("flow",      "Flow behaviour"),
    ("rate",      "Traffic rate"),
]

# ---------- 2. Stage -> remediation (app.py adds/renames keys as needed) ----------
REMEDIATION = {
    "Normal":          ["No action needed", "Continue routine monitoring"],
    "Benign":          ["No action needed", "Continue routine monitoring"],
    "Reconnaissance":  ["Enable detailed logging", "Monitor the source IP",
                        "Hide unnecessary service banners"],
    "Scanning":        ["Rate-limit the source IP", "Block the IP at the firewall",
                        "Close unused ports"],
    "Exploitation":    ["Isolate the target host", "Patch the exposed service",
                        "Force credential reset"],
    "Post-Compromise": ["Disconnect affected systems", "Run forensic analysis",
                        "Revoke keys and tokens"],
}
URGENCY = {"LOW": "MONITOR", "MEDIUM": "INVESTIGATE", "HIGH": "ACT NOW"}
NORMAL_NAMES = {"normal", "benign"}


def _clean(name):
    """'Flow Bytes/s__mean' -> ('Flow Bytes/s', 'mean')"""
    if "__" in name:
        base, stat = name.rsplit("__", 1)
        return base, stat.lower()
    return name, ""


class Explainer:
    def __init__(self, feature_names, stages, k_steps=3):
        self.features = list(feature_names)
        self.stages = list(stages)
        self.k = k_steps
        self.rf = None
        self.tree_exp = None
        self.mu = self.sd = None
        self.global_imp = None

    # ---- window (T,F) -> summary vector (3F): mean, max, last ----
    @staticmethod
    def _summ(W):
        W = np.asarray(W, dtype=float)
        if W.ndim == 2:
            W = W[None]
        return np.concatenate([W.mean(1), W.max(1), W[:, -1, :]], axis=1)

    def fit(self, X_windows, y_stage, normal_class=0, max_bg=100):
        """Train the surrogate model that mimics the AI model's stage output."""
        X_windows = np.asarray(X_windows, dtype=float)
        y = np.asarray(y_stage)
        X = self._summ(X_windows)
        self.rf = RandomForestClassifier(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=42,
            class_weight="balanced").fit(X, y)          # rare attack classes matter
        benign = X_windows[y == normal_class][:, -1, :]
        if len(benign) < 5:
            benign = X_windows[:, -1, :]
        self.mu = benign.mean(0)
        # safe floor: 5% of the overall spread (or tiny absolute) avoids huge z on ~constant features
        overall = X_windows[:, -1, :].std(0)
        self.sd = np.maximum(benign.std(0), np.maximum(0.05 * overall, 1e-6))
        F = len(self.features)
        gi = self.rf.feature_importances_
        self.global_imp = gi[:F] + gi[F:2 * F] + gi[2 * F:]
        self.tree_exp = shap.TreeExplainer(self.rf) if HAS_SHAP else None
        return self

    def _z(self, window):
        return (np.asarray(window, dtype=float)[-1] - self.mu) / self.sd

    # ---- per-prediction feature contributions (length F, non-negative) ----
    def _contrib(self, window, class_idx):
        F = len(self.features)
        if self.tree_exp is not None:
            s = self._summ(window)
            sv = self.tree_exp.shap_values(s)
            classes = list(self.rf.classes_)
            ci = classes.index(class_idx) if class_idx in classes else 0   # map stage id -> column
            if isinstance(sv, list):
                sv = np.asarray(sv[ci])[0]                  # older shap: list per class
            else:
                sv = np.asarray(sv)
                sv = sv[0, :, ci] if sv.ndim == 3 else sv[0]  # newer shap: (1, feats, classes)
            sv = np.abs(sv)
            return sv[:F] + sv[F:2 * F] + sv[2 * F:]
        return self.global_imp * np.abs(self._z(window))       # fallback

    def _meaning(self, name):
        base, stat = _clean(name)
        n = base.lower()
        text = None
        for key, t in FEATURE_MEANING:
            # match at a word start so "ack" does not hit "backward"
            if re.search(r"(?<![a-z0-9])" + re.escape(key), n):
                text = t
                break
        if text is None:
            text = base
        if stat == "std":
            return f"Fluctuation in {text[0].lower() + text[1:]}"
        return text

    @staticmethod
    def _describe(z):
        a = abs(z)
        d = "above" if z > 0 else "below"
        if a >= Z_ABNORMAL:
            return f"abnormal: {a:.1f}\u03c3 {d} normal"
        if a >= Z_MILD:
            return f"slightly {d} normal ({a:.1f}\u03c3)"
        return f"within normal range ({a:.1f}\u03c3)"

    def explain(self, window, pred_probs, stage, risk, top_n=3):
        window = np.asarray(window, dtype=float)
        pred_probs = np.asarray(pred_probs, dtype=float)
        if len(pred_probs) != len(self.stages):
            raise ValueError(f"pred_probs has {len(pred_probs)} entries, expected {len(self.stages)}")
        risk = str(risk).upper()
        if stage in self.stages:
            cidx = self.stages.index(stage)
        else:
            cidx = int(np.argmax(pred_probs))
            stage = self.stages[cidx]
        is_normal = stage.lower() in NORMAL_NAMES

        contrib = self._contrib(window, cidx)
        total = contrib.sum() + 1e-12
        z = self._z(window)
        order = np.argsort(contrib)[::-1]

        top_features = [(self.features[i], round(float(contrib[i] / total), 3))
                        for i in order[:top_n]]

        # reasons: top features, with de-duplicated meanings
        reasons, seen, key_meanings = [], set(), []
        for i in order:
            if len(reasons) >= top_n:
                break
            name = self.features[i]
            m = self._meaning(name)
            if m in seen:
                continue
            seen.add(m)
            reasons.append(f"{m} ({_clean(name)[0]} is {self._describe(float(z[i]))})")
            if abs(z[i]) >= Z_ABNORMAL:
                key_meanings.append(m)

        prob = float(pred_probs[cidx]) * 100
        if is_normal:
            warning = (f"\u2705 {risk} RISK [{URGENCY.get(risk, '')}]: traffic looks normal "
                       f"for the next {self.k} steps ({prob:.0f}% confidence).")
        else:
            ind = "; ".join(key_meanings) if key_meanings else \
                  "no single feature is strongly abnormal - the pattern across several features drives this"
            warning = (f"\u26a0 {risk} RISK [{URGENCY.get(risk, '')}]: {stage} activity likely "
                       f"within the next {self.k} steps ({prob:.0f}% confidence). "
                       f"Key indicators: {ind}.")

        actions = list(REMEDIATION.get(stage, ["Investigate manually"]))[:3]
        if not is_normal and any("port" in f.lower() for f, _ in top_features):
            actions.append("Review firewall rules for port sweeps")

        return {"top_features": top_features, "reasons": reasons,
                "warning": warning, "actions": actions,
                "urgency": URGENCY.get(risk, ""),
                "method": "SHAP (surrogate RandomForest)" if self.tree_exp is not None
                          else "Importance x deviation (surrogate RandomForest)"}

    # ---- for the dashboard ----
    def plot_top_features(self, result):
        import plotly.graph_objects as go
        names = [f for f, _ in result["top_features"]][::-1]
        vals = [v for _, v in result["top_features"]][::-1]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h"))
        fig.update_layout(title="Top contributing features",
                          xaxis_title="Share of contribution", height=300)
        return fig

    def global_importance(self, top_n=10):
        idx = np.argsort(self.global_imp)[::-1][:top_n]
        return [(self.features[i], round(float(self.global_imp[i]), 3)) for i in idx]


# ---------------- self-test with synthetic data ----------------
if __name__ == "__main__":
    import json
    rng = np.random.default_rng(0)
    feats = ["SYN Flag Count__mean", "Destination Port__std", "Flow Duration__mean",
             "Total Fwd Packets__mean", "Flow Bytes/s__mean", "Idle Mean__mean"]
    stages = ["Benign", "Reconnaissance", "Initial Access"]
    n, T, F = 600, 10, len(feats)
    y = rng.integers(0, 3, n)
    X = rng.normal(0, 1, (n, T, F))
    X[y == 1, :, 0] += 4
    X[y == 1, :, 1] += 4
    ex = Explainer(feats, stages).fit(X, y)
    w = X[np.where(y == 1)[0][0]]
    print(json.dumps(ex.explain(w, [0.05, 0.9, 0.05], "Reconnaissance", "HIGH"), indent=2))
    wb = X[np.where(y == 0)[0][0]]
    print(json.dumps(ex.explain(wb, [0.9, 0.05, 0.05], "Benign", "LOW"), indent=2))
    print("Global:", ex.global_importance(3))
