"""
Train a simple Iris flower classifier using scikit-learn
and push it to Hugging Face Hub.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train(output_dir: str = "model_output", n_estimators: int = 100, test_size: float = 0.2):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────────────
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # ── 2. Pre-process ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── 3. Train ────────────────────────────────────────────────────────────────
    print(f"Training RandomForestClassifier (n_estimators={n_estimators})...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # ── 4. Evaluate ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ── 5. Save artefacts ───────────────────────────────────────────────────────
    # Model
    model_path = output_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {model_path}")

    # Scaler  (must be saved alongside the model)
    scaler_path = output_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {scaler_path}")

    # Metrics
    metrics = {
        "accuracy":    round(accuracy, 4),
        "n_estimators": n_estimators,
        "test_size":    test_size,
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "class_names":   class_names,
        "per_class":     {
            cls: {k: round(v, 4) for k, v in report[cls].items()}
            for cls in class_names
        },
    }
    metrics_path = output_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved → {metrics_path}")

    # Model card (README.md)
    readme = f"""---
language: en
license: mit
tags:
  - sklearn
  - iris
  - classification
  - random-forest
---

# 🌸 Iris Flower Classifier

A simple Random Forest classifier trained on the classic Iris dataset.

## Model Details

| Property       | Value                    |
|----------------|--------------------------|
| Algorithm      | Random Forest            |
| n_estimators   | {n_estimators}           |
| Test Accuracy  | {accuracy:.4f}           |
| Train samples  | {len(X_train)}           |
| Test samples   | {len(X_test)}            |

## Classes

The model predicts one of three Iris species:
- `setosa`
- `versicolor`
- `virginica`

## Usage

```python
import pickle, numpy as np

with open("model.pkl",  "rb") as f: model  = pickle.load(f)
with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)

# sepal length, sepal width, petal length, petal width  (all in cm)
X = np.array([[5.1, 3.5, 1.4, 0.2]])
X_scaled = scaler.transform(X)
prediction = model.predict(X_scaled)
print(prediction)   # e.g. [0]  →  setosa
```

## Per-class Metrics

| Class       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
{chr(10).join(f"| {cls:<11} | {report[cls]['precision']:.4f}    | {report[cls]['recall']:.4f} | {report[cls]['f1-score']:.4f}    |" for cls in class_names)}
"""
    (output_path / "README.md").write_text(readme)
    print(f"README saved → {output_path / 'README.md'}")

    return metrics


def push_to_hf(output_dir: str = "model_output", repo_id: str = None):
    from huggingface_hub import HfApi, create_repo, upload_folder

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    if not repo_id:
        raise ValueError("--repo-id is required for pushing to Hugging Face.")

    print(f"\nPushing to Hugging Face Hub → {repo_id}")
    api = HfApi()

    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")

    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"✅  Upload complete!  https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris classifier and optionally push to HF")
    parser.add_argument("--output-dir",     default="model_output", help="Directory for model artefacts")
    parser.add_argument("--n-estimators",   type=int, default=100,  help="Number of trees in the forest")
    parser.add_argument("--test-size",      type=float, default=0.2, help="Fraction of data used for testing")
    parser.add_argument("--push-to-hub",    action="store_true",    help="Push model to Hugging Face Hub")
    parser.add_argument("--repo-id",        default=None,           help="HF repo id, e.g. username/iris-classifier")
    args = parser.parse_args()

    train(
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        test_size=args.test_size,
    )

    if args.push_to_hub:
        push_to_hf(output_dir=args.output_dir, repo_id=args.repo_id)
