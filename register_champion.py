"""
register_champion.py
────────────────────
Registers both sentiment models in the MLflow Model Registry and assigns
the 'champion' and 'challenger' aliases based on the decision made in
evaluate.py.

Pre-requisite
-------------
Run evaluate.py first so that outputs/run_ids.json exists.

Run
---
    python register_champion.py
"""

import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

TRACKING_URI          = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-model"
OUT_DIR               = Path("outputs")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # ── Load run IDs written by evaluate.py ───────────────────────────────────
    ids_path = OUT_DIR / "run_ids.json"
    if not ids_path.exists():
        raise FileNotFoundError(
            f"{ids_path} not found. Run evaluate.py first."
        )
    with open(ids_path) as f:
        run_ids = json.load(f)

    run_id_a = run_ids["run_id_A"]
    run_id_b = run_ids["run_id_B"]
    winner   = run_ids["winner"]

    print(f"Run A : {run_id_a}")
    print(f"Run B : {run_id_b}")
    print(f"Winner: Model {winner}\n")

    # ── Create registered model (idempotent) ──────────────────────────────────
    try:
        client.create_registered_model(
            name=REGISTERED_MODEL_NAME,
            description=(
                "Sentiment classification on SST-2.  "
                "Champion: distilbert-base-uncased-finetuned-sst-2-english.  "
                "Challenger: textattack/bert-base-uncased-SST-2."
            ),
        )
        print(f"Created registered model '{REGISTERED_MODEL_NAME}'.")
    except mlflow.exceptions.MlflowException:
        print(f"Registered model '{REGISTERED_MODEL_NAME}' already exists – skipping creation.")

    # ── Register both model versions ──────────────────────────────────────────
    print("\nRegistering Model A …")
    mv_a = mlflow.register_model(
        model_uri=f"runs:/{run_id_a}/model",
        name=REGISTERED_MODEL_NAME,
    )
    print(f"  Model A → version {mv_a.version}")

    print("Registering Model B …")
    mv_b = mlflow.register_model(
        model_uri=f"runs:/{run_id_b}/model",
        name=REGISTERED_MODEL_NAME,
    )
    print(f"  Model B → version {mv_b.version}")

    # ── Assign aliases based on the decision ──────────────────────────────────
    if winner == "A":
        champion_ver   = mv_a.version
        challenger_ver = mv_b.version
    else:
        champion_ver   = mv_b.version
        challenger_ver = mv_a.version

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion",   str(champion_ver))
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", str(challenger_ver))

    print(f"\nAlias 'champion'   → version {champion_ver}  (Model {winner})")
    print(
        f"Alias 'challenger' → version {challenger_ver}"
        f"  (Model {'B' if winner == 'A' else 'A'})"
    )
    print(
        f"\nDone.  Load champion with:\n"
        f"  mlflow.pyfunc.load_model('models:/{REGISTERED_MODEL_NAME}@champion')"
    )
    print("Next step: python predict_from_mlflow.py\n")


if __name__ == "__main__":
    main()
