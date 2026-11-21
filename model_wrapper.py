"""
model_wrapper.py
────────────────
Custom MLflow PythonModel that wraps a Hugging Face text-classification
pipeline.  This file is bundled with every logged model (via code_paths)
so it is fully self-contained inside the MLflow artifact store.

Input  : pd.DataFrame with column 'text'  (one row per example)
Output : pd.DataFrame with columns
         [text, prediction, prediction_name, score]
         where  prediction      ∈ {0, 1}  (0 = negative, 1 = positive)
                prediction_name ∈ {"negative", "positive"}
                score           – the model's softmax confidence
"""

import pandas as pd
import mlflow.pyfunc


class HuggingFaceSentimentModel(mlflow.pyfunc.PythonModel):
    """
    Wraps a Hugging Face text-classification pipeline as an MLflow pyfunc.

    The HuggingFace model is NOT imported at the module level; it is loaded
    lazily inside load_context so the class can be pickled/unpickled without
    requiring the pipeline at import time.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id  # stored in the pickle; no HF import here
        self._pipe = None

    # ------------------------------------------------------------------
    def load_context(self, context):          # called by MLflow at load time
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            "text-classification",
            model=self.model_id,
            device=device,
            truncation=True,
            max_length=512,
        )

    # ------------------------------------------------------------------
    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """
        Accept a DataFrame (column 'text'), a dict {"text": [...]}, or a list
        and return a prediction DataFrame.
        """
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        elif isinstance(model_input, dict):
            raw = model_input.get("text", [])
            texts = [raw] if isinstance(raw, str) else list(raw)
        else:
            texts = list(model_input)

        raw_preds = self._pipe(texts)

        rows = []
        for text, r in zip(texts, raw_preds):
            lbl = r["label"].upper()
            # Handle both "POSITIVE"/"NEGATIVE" and "LABEL_1"/"LABEL_0" conventions
            pred = 1 if ("POSITIVE" in lbl or lbl == "LABEL_1") else 0
            rows.append(
                {
                    "text":            text,
                    "prediction":      pred,
                    "prediction_name": "positive" if pred == 1 else "negative",
                    "score":           round(r["score"], 6),
                }
            )
        return pd.DataFrame(rows)
