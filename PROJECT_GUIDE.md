# Musanze Cooperative Harvest & Dispatch Decision Pipeline

**Course:** SWE 3513 — Artificial Intelligence  
**Assignment:** 1  
**Group:** AI-G01  

---

## What This Project Does

This pipeline helps the fictional **Musanze HarvestLink Cooperative** make three operational decisions from farm collection data:

| # | Decision | ML Task | Model | Key Metric |
|---|----------|---------|-------|------------|
| 1 | Estimate harvest weight | Regression | Linear Regression (NumPy gradient descent) | MAE=51 kg, R²=0.995 |
| 2 | Flag consignments needing dispatch attention | Binary Classification | Logistic Regression | Accuracy=1.0, F1=1.0 |
| 3 | Group collection points with similar profiles | Clustering | K-Means (k=3, silhouette=0.58) | 3 clusters |

---

## Input Data

**File:** `data/AI_A1_G01.csv` (30 farms, group-specific fictional data)

| Feature | Type | Description |
|---------|------|-------------|
| `record_id` | text | Unique identifier (not used as feature) |
| `plot_area_ha` | numeric | Farm area in hectares |
| `rainfall_mm` | numeric | Recent rainfall estimate |
| `soil_ph` | numeric | Soil acidity measure |
| `seed_kg` | numeric | Seed quantity |
| `distance_km` | numeric | Distance to collection point |
| `arrival_hour` | numeric | Planned arrival hour (24h) |
| `actual_yield_kg` | target | Regression target |
| `dispatch_attention` | target | Classification target (0/1) |

---

## Project Structure

```
AI_A1_G01/
├── run_all.py              # Full pipeline: train all models
├── predict.py              # Single-record prediction CLI
├── requirements.txt        # Dependencies
├── README.md               # Setup & commands
├── PROJECT_GUIDE.md        # This file
├── src/
│   ├── data_pipeline.py    # Load, validate, vectorize, SHA-256
│   ├── regression.py       # NumPy gradient descent (no sklearn)
│   ├── classification.py   # LogisticRegression + metrics
│   └── clustering.py       # KMeans k=2..5, silhouette, PCA
├── data/
│   └── AI_A1_G01.csv
├── artifacts/              # Generated outputs
│   ├── data_report.json
│   ├── regression_metrics.json
│   ├── regression_loss.png
│   ├── classification_metrics.json
│   ├── confusion_matrix.png
│   ├── clustering_metrics.json
│   ├── clusters.csv
│   └── cluster_plot.png
├── models/                 # Saved models
│   ├── regression_model.npz
│   ├── classification_model.joblib
│   └── clustering_model.npz
└── evidence/               # Submission evidence
    ├── AI_USE.md
    ├── TEST_LOG.md
    └── AI_A1_G01_DEMO.md
```

---

## Quick Start

### 1. Activate Virtual Environment
```powershell
cd C:\Users\Ali\AI_A1_G01
.venv\Scripts\activate
```

### 2. Run Full Pipeline
```powershell
python run_all.py --data data/AI_A1_G01.csv --output artifacts/ --group AI-G01
```

**Output:**
```
Loading data from data\AI_A1_G01.csv...
Data shape: (30, 9)
Generating data report... SHA-256: 70480135477a038edb41ec136b249f606176e616acf8c64a523773133f52c762
Training regression model... MAE=50.96, RMSE=59.15, R2=0.9949
Training classification model... Acc=1.0000, F1=1.0000
Training clustering model... k=3, Silhouette=0.5801
Pipeline complete! Artifacts saved to: artifacts
Models saved to: models
```

### 3. Run Prediction on New Data

**Option A: Python one-liner (avoids PowerShell JSON escaping)**
```powershell
python -c "
import sys; sys.path.insert(0, '.')
from src.regression import load_regression_model, predict_regression
from predict import load_classification_model, load_clustering_model, predict_classification, predict_clustering
from pathlib import Path; import numpy as np

X = np.array([[1.2, 81, 5.7, 210, 14, 9]], dtype=float)  # plot_area, rainfall, ph, seed, distance, hour

# Regression
w, m, s = load_regression_model(Path('models/regression_model.npz'))
print('Yield (kg):', round(predict_regression(X, w, m, s)[0], 1))

# Classification
c = load_classification_model(Path('models'))
pred, prob = predict_classification(X, c['model'], c['scaler'])
print('Dispatch flag:', pred[0], '| Probability:', round(prob[0], 2))

# Clustering
cl = load_clustering_model(Path('models'))
cluster = predict_clustering(X, cl['cluster_centers'], cl['scaler_mean'], cl['scaler_std'])
print('Cluster:', cluster[0])
"
```

**Output:**
```
Yield (kg): 2684.4
Dispatch flag: 0 | Probability: 0.46
Cluster: 1
```

**Option B: JSON file (for complex inputs)**
```powershell
# Create input.json
@'
{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}
'@ | Out-File -Encoding utf8 input.json

python predict.py --record "$(Get-Content input.json -Raw)"
```

---

## Generated Artifacts

| File | Description |
|------|-------------|
| `artifacts/data_report.json` | Row count, feature count, missing values, descriptive stats, SHA-256 fingerprint |
| `artifacts/regression_metrics.json` | Weights, scaler params, MAE, RMSE, R², learning rate, epochs |
| `artifacts/regression_loss.png` | Training/validation loss curves |
| `artifacts/classification_metrics.json` | Accuracy, Precision, Recall, F1, confusion matrix |
| `artifacts/confusion_matrix.png` | Heatmap visualization |
| `artifacts/clustering_metrics.json` | Silhouette scores for k=2..5, best k, inertia |
| `artifacts/clusters.csv` | `record_id,cluster_label` for all 30 farms |
| `artifacts/cluster_plot.png` | PCA 2D projection colored by cluster |

---

## Prediction Input Validation

`predict.py` validates every request:

| Input | Result |
|-------|--------|
| All 6 fields present, numeric | ✅ Returns predictions |
| Missing fields | ❌ `{"error": "Missing required fields: ['soil_ph', 'seed_kg', ...]"}` |
| Non-numeric value | ❌ `{"error": "Field 'plot_area_ha' must be numeric, got str"}` |
| Invalid JSON | ❌ `{"error": "Invalid JSON: ..."}` |
| Model files missing | ❌ `{"error": "Model not found: ..."}` |

---

## Technical Constraints Satisfied

- ✅ **Regression from scratch** — NumPy batch gradient descent, no sklearn estimator
- ✅ **Fixed seed (42)** — Recorded in all metrics files
- ✅ **No data leakage** — Scalers fit only on training data
- ✅ **No hardcoded outputs** — Works on hidden test data with same schema
- ✅ **Stratified split** — Classification uses stratified train/test
- ✅ **Silhouette-based k selection** — k=2..5 evaluated, best chosen
- ✅ **Cautious clustering interpretation** — Not claimed as real-world categories
- ✅ **Clean environment run** — Pipeline executes from fresh venv

---

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0  (v1.7.2 used — pre-built wheel for Windows compatibility)
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

Install: `pip install -r requirements.txt`

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: sklearn` | Run `pip install --ignore-installed --no-deps scikit-learn==1.7.2` |
| `DLL load failed` | System Application Control policy blocks compiled DLLs; use pre-built wheel (v1.7.2) |
| PowerShell JSON escaping | Use `python -c "..."` or save JSON to file |
| `predict.py` unrecognized arguments | Use Python -c method above or escape quotes carefully |

---

## Submission Files (for assessor)

| File | Purpose |
|------|---------|
| `AI_A1_G01.zip` | Full project (excludes .venv, __pycache__, .ipynb_checkpoints) |
| `AI_A1_G01_UIUX.pdf` | 5–7 page dashboard design |
| `AI_A1_G01_CONTRIBUTIONS.pdf` | Signed member contributions |
| Moodle text | Group code, verification code, SHA-256, GitHub URL, commit hash, demo video link |

---

## Demo Video Requirements (90 seconds max)

1. Show group code, terminal clock, Git commit hash, dataset SHA-256
2. Run `python run_all.py ...`
3. Show generated JSON, CSV, PNG, model files
4. Open one metrics JSON and one plot
5. Explain one model result in own words
6. Run `predict.py` with valid input
7. Run `predict.py` with invalid input (show error)
8. Show UI/UX PDF and contributions PDF filenames