"""Integration layer: Member1 data -> Member2 world model -> Member3 stage/risk -> Member4 explanation -> Member5 UI."""
import json, sys
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, send_from_directory

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "attack_progression"))      # Member 3 uses flat imports
sys.path.insert(0, str(ROOT / "member2_world_model"))
from analyzer import analyze_attack                       # Member 3
from stages import STAGE_NAMES, ATTACK_STAGES, BENIGN_STAGE
from label_mapping import LABEL_TO_STAGE
import explainability as xp                               # Member 4

# ---------- Member 1 outputs ----------
D = ROOT / "data"
Xtr, Xte = np.load(D / "X_train.npy"), np.load(D / "X_test.npy")
ytr, yte = np.load(D / "y_train.npy"), np.load(D / "y_test.npy")
FEATS = json.load(open(D / "feature_names.json"))
LABELS = {v: k for k, v in json.load(open(D / "label_map.json")).items()}

def stage_of(label_id):
    return LABEL_TO_STAGE.get(LABELS[int(label_id)], "UNKNOWN")

# ---------- Member 4 surrogate, trained to forecast the stage at the furthest horizon ----------
PRETTY = {BENIGN_STAGE: "Normal", "UNKNOWN": "Other attack (unmapped)",
          **{k: STAGE_NAMES[v] for k, v in ATTACK_STAGES.items()}}
tr_stage = np.array([stage_of(v) for v in ytr[:, -1]])
present = [s for s in [BENIGN_STAGE, *ATTACK_STAGES, "UNKNOWN"] if s in set(tr_stage)]
stage_idx = {s: i for i, s in enumerate(present)}         # contiguous ids -> SHAP class index stays valid
names = [PRETTY[s] for s in present]

xp.REMEDIATION.update({
    "Normal": xp.REMEDIATION["Normal"],
    "Other attack (unmapped)": ["Rate-limit or scrub the traffic source", "Check DoS/DDoS protections", "Escalate to SOC"],
    "Reconnaissance": xp.REMEDIATION["Reconnaissance"] + ["Rate-limit the source IP"],
    "Initial Access": ["Lock accounts under brute force", "Patch the exposed service", "Force credential reset"],
    "Lateral Movement": ["Isolate the affected segment", "Audit internal credentials", "Restrict east-west traffic"],
    "Command & Control": ["Block outbound C2 destinations", "Isolate the infected host", "Run forensic analysis"],
    "Exfiltration": ["Cut egress for affected hosts", "Revoke keys and tokens", "Start incident response"],
})
EX = xp.Explainer(FEATS, names).fit(Xtr, np.array([stage_idx[s] for s in tr_stage]), normal_class=stage_idx[BENIGN_STAGE])

# ---------- Member 2 world model (needs PyTorch; falls back to surrogate so the UI still runs) ----------
WM, WM_NOTE = None, "Surrogate fallback (PyTorch not installed)"
try:
    import torch
    from src.models import TemporalTransformerWorldModel
    ck = torch.load(ROOT / "member2_world_model/artifacts/transformer_world_model.pt", map_location="cpu", weights_only=False)
    m = TemporalTransformerWorldModel(len(ck["features"]), horizon=ck["horizon"]); m.load_state_dict(ck["state_dict"]); m.eval()
    col = {f: i for i, f in enumerate(FEATS)}
    WM, WM_NOTE = (m, ck), "Temporal Transformer world model"
except Exception as e:
    WM_NOTE = f"Surrogate fallback ({type(e).__name__})"

def attack_risk(window):
    """Attack probability for the next 5 windows."""
    if WM:
        m, ck = WM
        x = np.stack([window[:, col[f]] if f in col else np.zeros(len(window)) for f in ck["features"]], 1)  # packet feats absent -> 0
        x = ((x - ck["mean"]) / ck["std"]).astype(np.float32)
        with torch.no_grad():
            return torch.sigmoid(m(torch.tensor(x)[None])[1]).squeeze(0).numpy().tolist()
    p = EX.rf.predict_proba(EX._summ(window))[0]
    base = 1 - (p[list(EX.rf.classes_).index(stage_idx[BENIGN_STAGE])] if stage_idx[BENIGN_STAGE] in EX.rf.classes_ else 0)
    return [float(np.clip(base * (0.97 ** k), 0, 1)) for k in range(5)]

def probs_full(window):
    p = EX.rf.predict_proba(EX._summ(window))[0]
    full = np.zeros(len(names)); full[EX.rf.classes_.astype(int)] = p
    return full

app = Flask(__name__, static_folder="static")

@app.get("/")
def index(): return send_from_directory("static", "index.html")

@app.get("/api/meta")
def meta():
    return jsonify(n=len(Xte), world_model=WM_NOTE, explain=("SHAP" if EX.tree_exp else "Importance x deviation"),
                   stages=[STAGE_NAMES[v] for v in ATTACK_STAGES.values()], n_features=len(FEATS))

@app.get("/api/timeline")
def timeline():
    return jsonify([{"i": i, "truth": LABELS[int(y[0])], "stage": PRETTY[stage_of(y[0])]} for i, y in enumerate(yte)])

@app.get("/api/predict/<int:i>")
def predict(i):
    i = max(0, min(i, len(Xte) - 1)); w = Xte[i]
    pf = probs_full(w); k = int(np.argmax(pf)); pred_stage = present[k]
    cur_stage = stage_of(yte[i][0])
    ana = analyze_attack([LABELS[int(v)] for v in (yte[i][0], yte[i][-1])])          # Member 3
    if "error" in ana: ana = {"transitions": [], "progression_distance": None, "risk_level": "UNDETERMINED"}
    from risk import assess_risk
    rk = assess_risk(cur_stage, pred_stage)                                           # current -> forecast
    level = rk["risk_level"]
    ex = EX.explain(w, pf, names[k], level if level in ("LOW", "MEDIUM", "HIGH") else "LOW")
    return jsonify(i=i, truth=[LABELS[int(v)] for v in yte[i]], current=PRETTY[cur_stage], predicted=names[k],
                   confidence=float(pf[k]), stage_probs=dict(zip(names, map(float, pf))), risk=level,
                   distance=rk["progression_distance"], transitions=ana["transitions"],
                   attack_curve=attack_risk(w), explanation=ex)

if __name__ == "__main__":
    app.run(debug=False, port=5000)
