# Asterisk Attack Forecast Console
Run: `pip install -r requirements.txt && python app.py` then open http://localhost:5000

Pipeline (app.py): Member 1 windows (`data/`) -> Member 2 Transformer (`member2_world_model/`, PyTorch) -> Member 3 stage/risk (`attack_progression/`) -> Member 4 SHAP explainer (`explainability.py`) -> UI (`static/index.html`).
Without PyTorch/SHAP the app falls back to the surrogate model / importance x deviation (shown in the header).
Note: the checkpoint expects 43 features (33 flow + 10 packet); packet features are zero-filled until Member 1's PCAP output is connected.
