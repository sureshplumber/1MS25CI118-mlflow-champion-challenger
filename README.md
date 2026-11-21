# MLflow Champion-Challenger: Sentiment Analysis

> **Repo name:** `1MS25CI118-mlflow-champion-challenger`

---

## 1. Task and Models

| Field | Detail |
|---|---|
| **Task** | Binary sentiment classification (positive / negative) |
| **Dataset** | GLUE / SST-2, validation split, 200 examples (seed = 42) |
| **Model A — Challenger** | `distilbert-base-uncased-finetuned-sst-2-english` |
| **Model B — Champion** | `textattack/bert-base-uncased-SST-2` |
| **Evaluation library** | `scikit-learn` (accuracy, macro-F1, precision, recall) |

Both models are loaded via the Hugging Face `pipeline("text-classification", ...)` API and evaluated on the **same** 200 randomly selected SST-2 validation sentences (see `data/eval_inputs.csv`).

---

## 2. How to Run

> [!IMPORTANT]
> Complete the installation steps in **Section: Installation** below before running any script.
> Verified student USN: `1MS25CI118` (configured in `evaluate.py`).

**Terminal 1 — start the tracking server (keep it running):**
```bash
python -m mlflow server --host 127.0.0.1 --port 5000
```

**Terminal 2 — run the pipeline in order:**
```bash
# Step 1: evaluate both models and log everything to MLflow
python evaluate.py

# Step 2: register both models and set champion / challenger aliases
python register_champion.py

# Step 3: load champion by alias and predict on new data
python predict_from_mlflow.py
```

Open the MLflow UI at http://127.0.0.1:5000 to inspect runs, metrics, and artifacts.

---

## 3. Champion Rule *(written before the final run)*

> [!NOTE]
> This rule was fixed **before** running `evaluate.py`. Results are filled in after.

### Primary metric
**Macro-F1** (higher is better).
Macro-F1 is preferred over accuracy because it weights each class equally, making it robust to any class imbalance present in the 200-sample subset. It is also the standard reporting metric for binary sentiment benchmarks.

### Minimum improvement margin
The Challenger must exceed the Champion by **≥ 0.01** (1 percentage point) on Macro-F1. With 200 samples, 1 pp corresponds to approximately 2 correctly classified examples. A smaller difference would be indistinguishable from sampling noise and does not justify replacing a stable production model.

### Tie-breaker
If the difference is within ±0.01, prefer the **faster** model (lower `avg_latency_ms`). Speed is a secondary production criterion: a model with equal accuracy but meaningfully lower latency delivers a better user experience.

### Guardrail
Reject any model whose `avg_latency_ms` exceeds **5 × the Champion's**. A model that is significantly slower cannot be promoted regardless of accuracy improvement.

### Direction summary

| Metric | Direction |
|---|---|
| Accuracy | ↑ higher is better |
| Macro-F1 | ↑ higher is better |
| Macro-Precision | ↑ higher is better |
| Macro-Recall | ↑ higher is better |
| `avg_latency_ms` | ↓ lower is better |
| `total_inference_s` | ↓ lower is better |

---

## 4. Results Table

> [!NOTE]
> Values are logged in MLflow (experiment `hf-sentiment-champion-challenger`).
> Fill in Accuracy / Precision / Recall / Latency from the MLflow Compare view.

| Metric | Model A — distilbert-sst2 | Model B — bert-sst2 | Δ (B − A) | Direction |
|---|---|---|---|---|
| Accuracy | 0.915 | 0.925 | +0.010 | ↑ |
| **Macro-F1 *(primary)*** | 0.915 | 0.925 | **+0.010** | ↑ |
| Macro-Precision | 0.917 | 0.925 | +0.008 | ↑ |
| Macro-Recall | 0.916 | 0.925 | +0.009 | ↑ |
| Avg Latency (ms) | 21.07 | 40.08 | +19.01 | ↓ |
| Total Inference (s) | 4.213 | 8.015 | +3.802 | ↓ |

**New-data accuracy (10 hand-crafted examples):**

| Model | Alias | Registry version | Correct / 10 |
|---|---|---|---|
| `textattack/bert-base-uncased-SST-2` | **champion** | 2 | 10 / 10 |
| `distilbert-base-uncased-finetuned-sst-2-english` | challenger | 1 | 10 / 10 |

---

## 5. Decision

**Winner: Model B — `textattack/bert-base-uncased-SST-2`** → alias `champion` (registry version 2).  
Model A — `distilbert-base-uncased-finetuned-sst-2-english` → alias `challenger` (registry version 1).

Following our pre-registered champion rule, Model B (`textattack/bert-base-uncased-SST-2`) successfully earns promotion to Champion over Model A (`distilbert-base-uncased-finetuned-sst-2-english`). Model B achieved a Macro-F1 of 0.925 compared to Model A's 0.915, yielding an improvement of Δ = +0.010 (+1.0 percentage points). This matches our predefined minimum improvement threshold of ≥ 1.0 pp.

Across our 200-sample SST-2 evaluation subset, this 1.0 pp delta corresponds to two additional correctly classified sentences (185 versus 183 correct predictions). While a 2-sample margin on a 200-example sample warrants caution regarding statistical variance, qualitative review confirms tangible gains. As captured in `outputs/disagreements.csv`, Model B exhibits superior handling of subtle, negative criticisms (e.g., correctly classifying *"mr. hartley 's distended pace and foot-dragging rhythms"* and *"what the director ca n't do is make either... interesting"*, both of which Model A misclassified as positive). Conversely, Model A only outperformed on rare compound descriptors like *"repulsively sadistic and mundane"*.

The primary trade-off is computational efficiency: BERT-base (110M parameters) averaged 40.08 ms per inference compared to 21.07 ms for DistilBERT (66M parameters), a ~1.9× increase in latency. However, this is well within our 5× guardrail threshold. In validation on 10 independent test sentences (`data/new_data.csv`), both models demonstrated flawless 10/10 accuracy, confirming Model B's operational stability as our new production Champion.

---

## 6. Limitations and Declaration

* The evaluation subset contains **200 sentences** randomly sampled from the SST-2 validation split with `seed=42`. Results on the full validation set (872 sentences) or the hidden test set may differ.
* SST-2 sentences are short movie review fragments; both models were fine-tuned on this exact distribution, which may inflate reported metrics compared to in-the-wild sentiment.
* Latency measurements are single-run, single-process estimates on the available hardware. They may vary across machines and runs.
* **AI assistance:** GitHub Copilot / Antigravity were used for code scaffolding. All design choices, the champion rule, and the written analysis are the student's own work.

---

## Project Structure

```
.
├── evaluate.py              # Runs both models, logs to MLflow
├── register_champion.py     # Registers models and sets aliases
├── predict_from_mlflow.py   # Loads champion by alias, predicts on new data
├── model_wrapper.py         # Custom MLflow pyfunc wrapper (HuggingFace)
├── requirements.txt
├── README.md
├── data/
│   ├── eval_inputs.csv      # 200-sample evaluation set (generated by evaluate.py)
│   └── new_data.csv         # 10 hand-crafted new examples
├── outputs/
│   ├── predictions_distilbert-sst2.csv
│   ├── predictions_bert-sst2.csv
│   ├── metrics_summary_distilbert-sst2.json
│   ├── metrics_summary_bert-sst2.json
│   ├── confusion_matrix_distilbert-sst2.png
│   ├── confusion_matrix_bert-sst2.png
│   ├── metrics_comparison.png
│   ├── disagreements.csv
│   └── new_data_predictions.csv
└── screenshots/
    ├── mlflow_compare_view.png
    ├── registry_aliases.png
    └── predict_terminal.png
```

---

## Installation

See the step-by-step guide below (also provided separately).

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows PowerShell / CMD

# 2. (CPU-only machines) Install PyTorch for CPU first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install all other dependencies
pip install -r requirements.txt
```
