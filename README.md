# 🌸 Iris Classifier — ML + GitHub Actions + Hugging Face

A minimal end-to-end example: train a scikit-learn model locally **or** via GitHub Actions, then push the artefacts to the Hugging Face Hub automatically.

---

## Project Structure

```
.
├── train.py                              # Training + HF upload script
├── requirements.txt                      # Python dependencies
└── .github/
    └── workflows/
        └── train_and_push.yml            # GitHub Actions workflow
```

---

## Quickstart (local)

```bash
# 1 — install deps
pip install -r requirements.txt

# 2 — train (artefacts saved to model_output/)
python train.py --n-estimators 100 --test-size 0.2

# 3 — train AND push to HF Hub
export HF_TOKEN=hf_...
python train.py --push-to-hub --repo-id your-username/iris-classifier
```

---

## GitHub Actions

### One-time setup

1. Go to **Settings → Secrets → Actions** in your repository.
2. Add a secret named **`HF_TOKEN`** with your Hugging Face access token  
   (create one at <https://huggingface.co/settings/tokens>).

### Trigger a run manually

1. Open the **Actions** tab in your repository.
2. Select **"Train & Push to Hugging Face"**.
3. Click **"Run workflow"** and fill in the inputs:

| Input | Description | Default |
|-------|-------------|---------|
| `n_estimators` | Number of trees in the Random Forest | `100` |
| `test_size` | Fraction of data used for testing | `0.2` |
| `repo_id` | HF repo, e.g. `your-username/iris-classifier` | *(required)* |
| `run_name` | Friendly label for this run | `iris-run` |

The workflow will also run automatically whenever you push changes to `train.py` or `requirements.txt` on `main`.

---

## What gets pushed to Hugging Face

| File | Description |
|------|-------------|
| `model.pkl` | Trained `RandomForestClassifier` |
| `scaler.pkl` | Fitted `StandardScaler` (needed for inference) |
| `metrics.json` | Accuracy + per-class metrics |
| `README.md` | Auto-generated model card |
