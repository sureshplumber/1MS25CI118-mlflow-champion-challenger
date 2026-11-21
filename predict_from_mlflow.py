"""
predict_from_mlflow.py
──────────────────────
Loads the Champion (and Challenger) from the MLflow Model Registry by alias
URI and runs predictions on new, unseen data.

IMPORTANT: This script does NOT import or call the Hugging Face pipeline
directly.  All prediction goes through the MLflow pyfunc wrapper.  The
Hugging Face model ID is read back from the run's logged parameters so the
script works correctly even after the alias is flipped to a different version.

Pre-requisites
--------------
Run evaluate.py and register_champion.py first.

Run
---
    python predict_from_mlflow.py

Stretch goal (alias flip demo)
-------------------------------
In the MLflow UI (or CLI), reassign the 'champion' alias to the other version,
then rerun this script without editing it.  The printed model ID and version
number will change automatically.
"""

from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

TRACKING_URI          = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-model"
NEW_DATA_PATH         = Path("data/new_data.csv")
OUT_DIR               = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_model_info(alias: str) -> dict:
    """
    Returns registry metadata for a given alias without hard-coding any
    version number or run ID.
    """
    mv      = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    run     = client.get_run(mv.run_id)
    hf_id   = run.data.params.get("hf_model_id", "unknown")
    return {
        "alias":        alias,
        "version":      mv.version,
        "run_id":       mv.run_id,
        "hf_model_id":  hf_id,
    }


def load_and_predict(alias: str, texts: list) -> pd.DataFrame:
    uri   = f"models:/{REGISTERED_MODEL_NAME}@{alias}"
    model = mlflow.pyfunc.load_model(uri)      # all HF calls are inside the pyfunc
    return model.predict(pd.DataFrame({"text": texts}))


def print_banner(info: dict):
    print(f"\n{'=' * 60}")
    print(f"  Loaded  : {info['alias'].upper()}")
    print(f"  Registry version : {info['version']}")
    print(f"  Run ID           : {info['run_id']}")
    print(f"  HF model ID      : {info['hf_model_id']}")
    print(f"{'=' * 60}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not NEW_DATA_PATH.exists():
        raise FileNotFoundError(f"{NEW_DATA_PATH} not found.")

    new_df = pd.read_csv(NEW_DATA_PATH)
    texts  = new_df["text"].tolist()

    all_preds = []

    for alias in ("champion", "challenger"):
        try:
            info = get_model_info(alias)
            print_banner(info)

            preds = load_and_predict(alias, texts)
            preds.insert(0, "id", new_df["id"].values)
            preds["expected_label"]    = new_df["expected_label"].values
            preds["model_alias"]       = alias
            preds["registry_version"]  = info["version"]
            preds["hf_model_id"]       = info["hf_model_id"]

            print(f"\n{alias.capitalize()} predictions:")
            display = preds[["id", "text", "expected_label", "prediction_name", "score"]].copy()
            display.columns = ["id", "text", "expected", "predicted", "confidence"]
            print(display.to_string(index=False))

            all_preds.append(preds)

        except Exception as exc:
            print(f"\n  [{alias}] Could not load — {exc}")

    if not all_preds:
        print("\nNo predictions were produced. Check the registry and tracking server.")
        return

    combined = pd.concat(all_preds, ignore_index=True)
    out_path = OUT_DIR / "new_data_predictions.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nAll predictions saved → {out_path}\n")


if __name__ == "__main__":
    main()
