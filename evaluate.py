"""
evaluate.py
───────────
Champion-Challenger sentiment analysis evaluation with MLflow.

Pre-requisites
--------------
1. MLflow tracking server must be running:
       python -m mlflow server --host 127.0.0.1 --port 5000
2. Replace STUDENT_USN below with your actual USN.

Run
---
    python evaluate.py

What it does
------------
  • Downloads 200 SST-2 validation examples (seeded, reproducible).
  • Saves them to data/eval_inputs.csv.
  • Runs both models on the identical inputs.
  • Logs one nested MLflow run per model and one parent summary run.
  • Saves artifacts (predictions, metrics, plots) under outputs/.
  • Writes outputs/run_ids.json for use by register_champion.py.
"""

import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – must come before pyplot
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

from model_wrapper import HuggingFaceSentimentModel

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration  – edit STUDENT_USN before your first run
# ══════════════════════════════════════════════════════════════════════════════
TRACKING_URI          = "http://127.0.0.1:5000"
EXPERIMENT_NAME       = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"
TASK                  = "sentiment"

EVAL_SAMPLES = 200
SEED         = 42
MAX_LENGTH   = 512
BATCH_SIZE   = 16

STUDENT_USN  = "1MS25CI118"   

DATASET_NAME = "glue/sst2, validation split"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_A = {
    "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
    "role":        "champion",
    "run_name":    "A_sentiment_distilbert-sst2",
    "short_name":  "distilbert-sst2",
}
MODEL_B = {
    "hf_model_id": "textattack/bert-base-uncased-SST-2",
    "role":        "challenger",
    "run_name":    "B_sentiment_bert-sst2",
    "short_name":  "bert-sst2",
}

# ── Champion-Challenger rule (written before the final run) ──────────────────
PRIMARY_METRIC = "macro_f1"   # higher is better
MARGIN         = 0.01         # Challenger must exceed Champion by ≥ 1 pp
# Tie-breaker: within the margin → prefer the faster model (avg_latency_ms)
# Guardrail   : reject any model whose latency > 5× the Champion's

OUT_DIR  = Path("outputs")
DATA_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Load evaluation data
# ══════════════════════════════════════════════════════════════════════════════
def load_eval_data() -> pd.DataFrame:
    print("Loading SST-2 validation set from HuggingFace …")
    dataset = load_dataset("nyu-mll/glue", "sst2", split="validation")
    rng     = np.random.default_rng(SEED)
    indices = sorted(
        rng.choice(len(dataset), size=min(EVAL_SAMPLES, len(dataset)), replace=False).tolist()
    )
    subset = dataset.select(indices)
    df = pd.DataFrame(
        {
            "sentence":   subset["sentence"],
            "label":      subset["label"],        # 0 = negative, 1 = positive
            "label_name": [
                "positive" if l == 1 else "negative" for l in subset["label"]
            ],
        }
    )
    out = DATA_DIR / "eval_inputs.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} examples → {out}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  2. Run predictions
# ══════════════════════════════════════════════════════════════════════════════
def run_predictions(model_cfg: dict, df: pd.DataFrame):
    print(f"\nRunning predictions with '{model_cfg['hf_model_id']}' …")
    pipe = pipeline(
        "text-classification",
        model=model_cfg["hf_model_id"],
        device=0 if DEVICE == "cuda" else -1,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    texts = df["sentence"].tolist()

    t0        = time.perf_counter()
    raw_preds = pipe(texts, batch_size=BATCH_SIZE)
    elapsed   = time.perf_counter() - t0

    pred_labels, pred_scores = [], []
    for r in raw_preds:
        lbl = r["label"].upper()
        pred_labels.append(1 if ("POSITIVE" in lbl or lbl == "LABEL_1") else 0)
        pred_scores.append(r["score"])

    preds_df = df.copy()
    preds_df["prediction"]       = pred_labels
    preds_df["prediction_name"]  = [
        "positive" if p == 1 else "negative" for p in pred_labels
    ]
    preds_df["confidence"]  = pred_scores
    preds_df["correct"]     = (preds_df["prediction"] == preds_df["label"]).astype(int)

    y_true = df["label"].tolist()
    metrics = {
        "accuracy":          float(accuracy_score(y_true, pred_labels)),
        "macro_f1":          float(f1_score(y_true, pred_labels, average="macro")),
        "macro_precision":   float(precision_score(y_true, pred_labels, average="macro")),
        "macro_recall":      float(recall_score(y_true, pred_labels, average="macro")),
        "avg_latency_ms":    float(elapsed / len(texts) * 1000),
        "total_inference_s": float(elapsed),
        "eval_samples":      float(len(texts)),
    }
    print(
        f"  accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}"
        f"  avg_latency={metrics['avg_latency_ms']:.1f} ms"
    )
    return preds_df, metrics


# ══════════════════════════════════════════════════════════════════════════════
#  3. Plots
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(preds_df: pd.DataFrame, short_name: str) -> Path:
    cm   = confusion_matrix(preds_df["label"], preds_df["prediction"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix – {short_name}")
    path = OUT_DIR / f"confusion_matrix_{short_name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path


def plot_metrics_comparison(metrics_a: dict, metrics_b: dict) -> Path:
    keys   = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    labels = ["Accuracy", "Macro-F1", "Macro-Prec.", "Macro-Rec."]
    vals_a = [metrics_a[k] for k in keys]
    vals_b = [metrics_b[k] for k in keys]
    x, w   = np.arange(len(labels)), 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ba = ax.bar(x - w / 2, vals_a, w, label="A: distilbert-sst2", color="#4C72B0")
    bb = ax.bar(x + w / 2, vals_b, w, label="B: bert-sst2",       color="#DD8452")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Champion vs Challenger – All Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(ba, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bb, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "metrics_comparison.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  4. Log one model run (nested inside the parent summary run)
# ══════════════════════════════════════════════════════════════════════════════
def log_model_run(model_cfg: dict, preds_df: pd.DataFrame, metrics: dict) -> str:
    lib_versions = json.dumps(
        {
            "transformers": transformers.__version__,
            "torch":        torch.__version__,
            "mlflow":       mlflow.__version__,
        }
    )
    short = model_cfg["short_name"]

    with mlflow.start_run(run_name=model_cfg["run_name"], nested=True) as run:
        run_id = run.info.run_id

        # ── Parameters ────────────────────────────────────────────────────────
        mlflow.log_params(
            {
                "hf_model_id":  model_cfg["hf_model_id"],
                "task":         TASK,
                "role":         model_cfg["role"],
                "max_length":   MAX_LENGTH,
                "batch_size":   BATCH_SIZE,
                "eval_samples": EVAL_SAMPLES,
                "seed":         SEED,
                "device":       DEVICE,
            }
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        mlflow.log_metrics(metrics)

        # ── Tags ──────────────────────────────────────────────────────────────
        mlflow.set_tags(
            {
                "student_usn":      STUDENT_USN,
                "library_versions": lib_versions,
                "dataset_name":     DATASET_NAME,
            }
        )

        # ── Prediction CSV artifact ───────────────────────────────────────────
        pred_path = OUT_DIR / f"predictions_{short}.csv"
        preds_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(str(pred_path), artifact_path="predictions")

        # ── Shared eval inputs ────────────────────────────────────────────────
        mlflow.log_artifact(str(DATA_DIR / "eval_inputs.csv"), artifact_path="data")

        # ── Metrics summary JSON ──────────────────────────────────────────────
        metrics_path = OUT_DIR / f"metrics_summary_{short}.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "model":        model_cfg["hf_model_id"],
                    "role":         model_cfg["role"],
                    "eval_samples": EVAL_SAMPLES,
                    **metrics,
                    "direction": {
                        "accuracy":          "higher is better",
                        "macro_f1":          "higher is better",
                        "macro_precision":   "higher is better",
                        "macro_recall":      "higher is better",
                        "avg_latency_ms":    "lower is better",
                        "total_inference_s": "lower is better",
                    },
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        # ── Confusion matrix plot ─────────────────────────────────────────────
        cm_path = plot_confusion_matrix(preds_df, short)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # ── Log model (custom pyfunc wrapper) ─────────────────────────────────
        # model_wrapper.py is bundled via code_paths so the artifact is
        # self-contained and loadable with any runs:/ or models:/ URI.
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=HuggingFaceSentimentModel(model_cfg["hf_model_id"]),
            code_paths=["model_wrapper.py"],
            pip_requirements=[
                f"transformers=={transformers.__version__}",
                f"torch=={torch.__version__}",
                "pandas",
            ],
            input_example=pd.DataFrame({"text": ["This movie is great!"]}),
        )

        print(f"  MLflow run '{model_cfg['run_name']}' logged (run_id={run_id})")
        return run_id


# ══════════════════════════════════════════════════════════════════════════════
#  5. Champion-Challenger decision
# ══════════════════════════════════════════════════════════════════════════════
def make_decision(
    metrics_a: dict, metrics_b: dict,
    preds_a: pd.DataFrame, preds_b: pd.DataFrame,
) -> dict:
    val_a = metrics_a[PRIMARY_METRIC]
    val_b = metrics_b[PRIMARY_METRIC]
    delta = val_b - val_a          # positive → Challenger (B) is better

    # Guardrail: reject if latency is > 5× the Champion's
    latency_ratio = metrics_b["avg_latency_ms"] / max(metrics_a["avg_latency_ms"], 1e-9)
    guardrail_failed = latency_ratio > 5.0

    if guardrail_failed:
        winner = "A"
        reason = (
            f"Challenger (B) latency is {latency_ratio:.1f}× the Champion's "
            f"(guardrail: > 5×). Champion (A) retains title."
        )
    elif delta >= MARGIN:
        winner = "B"
        reason = (
            f"Challenger (B) beats Champion (A) by {delta:.4f} on {PRIMARY_METRIC} "
            f"(≥ margin of {MARGIN}). Challenger promoted."
        )
    elif delta <= -MARGIN:
        winner = "A"
        reason = (
            f"Champion (A) retains title; Challenger (B) is worse "
            f"by {abs(delta):.4f} on {PRIMARY_METRIC}."
        )
    else:
        # Within margin → prefer faster model (tie-breaker)
        if metrics_b["avg_latency_ms"] < metrics_a["avg_latency_ms"]:
            winner = "B"
            reason = (
                f"Scores within margin (Δ={delta:+.4f}). "
                f"Tie-breaker: Challenger (B) is faster "
                f"({metrics_b['avg_latency_ms']:.1f} ms vs "
                f"{metrics_a['avg_latency_ms']:.1f} ms)."
            )
        else:
            winner = "A"
            reason = (
                f"Scores within margin (Δ={delta:+.4f}). "
                f"Tie-breaker: Champion (A) is not slower; prefer stability."
            )

    # Disagreement examples (qualitative analysis)
    merged = preds_a[["sentence", "label", "prediction"]].rename(
        columns={"prediction": "pred_A"}
    ).copy()
    merged["pred_B"]          = preds_b["prediction"].values
    merged["label_name"]      = merged["label"].map({0: "negative", 1: "positive"})
    merged["pred_A_name"]     = merged["pred_A"].map({0: "negative", 1: "positive"})
    merged["pred_B_name"]     = merged["pred_B"].map({0: "negative", 1: "positive"})
    merged["A_correct"]       = (merged["pred_A"] == merged["label"]).astype(int)
    merged["B_correct"]       = (merged["pred_B"] == merged["label"]).astype(int)
    disagree = merged[merged["pred_A"] != merged["pred_B"]].head(5).copy()
    disagree_path = OUT_DIR / "disagreements.csv"
    disagree.to_csv(disagree_path, index=False)
    print(f"  Saved {len(disagree)} disagreement examples → {disagree_path}")

    return {
        "winner":               winner,
        "reason":               reason,
        "delta_primary_metric": float(delta),
        "latency_ratio":        float(latency_ratio),
        "guardrail_failed":     guardrail_failed,
        "disagreements_path":   str(disagree_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\nExperiment : {EXPERIMENT_NAME}  (id={exp.experiment_id})")
    print(f"Tracking   : {TRACKING_URI}")
    print(f"Device     : {DEVICE}\n")

    # 1. Load shared evaluation set
    eval_df = load_eval_data()

    # 2. Predict with both models
    preds_a, metrics_a = run_predictions(MODEL_A, eval_df)
    preds_b, metrics_b = run_predictions(MODEL_B, eval_df)

    # 3. Comparison bar chart (needs both metrics)
    comp_path = plot_metrics_comparison(metrics_a, metrics_b)
    print(f"\nSaved comparison plot → {comp_path}")

    # 4. Parent (summary) run  +  nested model runs
    with mlflow.start_run(run_name=f"compare_{TASK}") as parent_run:
        parent_run_id = parent_run.info.run_id

        run_id_a = log_model_run(MODEL_A, preds_a, metrics_a)
        run_id_b = log_model_run(MODEL_B, preds_b, metrics_b)

        decision = make_decision(metrics_a, metrics_b, preds_a, preds_b)

        # Summary-run parameters, metrics, tags
        mlflow.log_params(
            {
                "primary_metric": PRIMARY_METRIC,
                "margin":         MARGIN,
                "tie_breaker":    "prefer_faster",
                "guardrail":      "latency_ratio_max_5x",
            }
        )
        mlflow.log_metrics(
            {
                "delta_primary_metric":    decision["delta_primary_metric"],
                f"A_{PRIMARY_METRIC}":     metrics_a[PRIMARY_METRIC],
                f"B_{PRIMARY_METRIC}":     metrics_b[PRIMARY_METRIC],
                "A_avg_latency_ms":        metrics_a["avg_latency_ms"],
                "B_avg_latency_ms":        metrics_b["avg_latency_ms"],
                "latency_ratio_B_over_A":  decision["latency_ratio"],
            }
        )
        mlflow.set_tags(
            {
                "winner":      decision["winner"],
                "student_usn": STUDENT_USN,
                "dataset_name": DATASET_NAME,
            }
        )

        # Artifacts in summary run
        mlflow.log_artifact(str(comp_path),                        artifact_path="plots")
        mlflow.log_artifact(str(decision["disagreements_path"]),   artifact_path="qualitative")
        mlflow.log_artifact(str(DATA_DIR / "eval_inputs.csv"),     artifact_path="data")

    # 5. Print summary
    print(f"\n{'=' * 65}")
    print(f"  DECISION : {decision['reason']}")
    print(f"  Winner   : Model {decision['winner']}")
    print(f"  Δ {PRIMARY_METRIC} (B − A) : {decision['delta_primary_metric']:+.4f}")
    print(f"  Summary run ID : {parent_run_id}")
    print(f"{'=' * 65}\n")

    # 6. Save run IDs for register_champion.py
    run_ids = {
        "run_id_A":        run_id_a,
        "run_id_B":        run_id_b,
        "winner":          decision["winner"],
        "experiment_name": EXPERIMENT_NAME,
    }
    out = OUT_DIR / "run_ids.json"
    with open(out, "w") as f:
        json.dump(run_ids, f, indent=2)
    print(f"Run IDs saved → {out}")
    print("Next step: python register_champion.py\n")


if __name__ == "__main__":
    main()
