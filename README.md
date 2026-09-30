# Musanze Cooperative Harvest and Dispatch Decision Lab

**Group:** Group2
**Course:** Artificial Intelligence  
**Assignment:** 1  

## Repository
- **GitHub:** https://github.com/cyithia-abijuru/Market_safe_inspection
- **Final Commit Hash:** `done` 

## Team Members & Roles

| Member | Role | Responsibilities |
|--------|------|------------------|
| Dominic Onen| Data & UX Lead | Schema checks, vectorization, data report, UI/UX PDF |
| Ali salaheldin | Regression Engineer | NumPy gradient descent, scaling, loss curve, regression metrics |
| Fajwan Chanjwok | Classification Engineer | Stratified split, classifier, metrics, confusion matrix, error cost |
| Arop Malual | Clustering & QA Engineer | Standardization, k comparison, silhouette, cluster outputs, QA |
| Abijuru Cynthia | Reproducibility & Release Lead | CLI contract, predict.py, requirements, README, final package |

## Python Version
- **Python 3.10+** 
## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Expected Outputs

Running the pipeline generates the following artifacts in `artifacts/`:

| File | Description |
|------|-------------|
| `data_report.json` | Dataset stats, missing values, SHA-256 fingerprint |
| `regression_metrics.json` | MAE, RMSE, R², weights, scaler params |
| `regression_loss.png` | Training/validation loss curves |
| `classification_metrics.json` | Accuracy, Precision, Recall, F1, confusion matrix |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `clustering_metrics.json` | Silhouette scores for k=2..5, best k |
| `clusters.csv` | Record ID → cluster label mapping |
| `cluster_plot.png` | PCA projection with cluster colors |

Models saved in `models/`:
- `regression_model.npz` — weights, scaler mean/std
- `classification_model.joblib` — LogisticRegression + StandardScaler
- `clustering_model.npz` — cluster centers, scaler, k

## Commands

### Run Complete Pipeline
```bash
python run_all.py --data data/AI_A1_G01.csv --output artifacts/ --group AI-G01
```

### Run Prediction (Valid Input)
```bash
python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}'
```

**Expected Output:**
```json
{
  "regression_prediction_kg": 2847.32,
  "classification_prediction": 0,
  "classification_probability": 0.12,
  "cluster_label": 2,
  "group_code": "AI-G01",
  "model_version": "1.0.0"
}
```

### Run Prediction (Invalid Input - Missing Field)
```bash
python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81}'
```

**Expected Output:**
```json
{
  "error": "Missing required fields: ['soil_ph', 'seed_kg', 'distance_km', 'arrival_hour']"
}
```

### Run Prediction (Invalid Input - Wrong Type)
```bash
python predict.py --record '{"plot_area_ha":"large","rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}'
```

**Expected Output:**
```json
{
  "error": "Field 'plot_area_ha' must be numeric, got str"
}
```

## Dataset
- **File:** `data/AI_A1_G01.csv`
- **Schema:** 9 columns (record_id, 6 features, 2 targets)
- **Rows:** 30 (group-specific fictional data)
- **SHA-256:** Computed at runtime, saved in `data_report.json`

## Limitations
1. Small dataset (30 rows) limits model generalization
2. Synthetic/fictional data — not real farm measurements
3. Linear regression assumes linear relationships
4. Clustering is exploratory — clusters are not verified real-world categories
5. No hyperparameter tuning on test set (fixed seed, single split)
6. Assumes same schema for hidden test data

## Reproducibility
- Fixed random seed: `42` (recorded in all metrics files)
- Scalers fit only on training data
- No data leakage between train/test
- Pipeline runs end-to-end from clean environment

## Evidence Files
- `evidence/AI_A1_G01_DEMO.mp4` — 90-second demonstration video
- `evidence/AI_USE.md` — AI tool usage disclosure
- `evidence/TEST_LOG.pdf` — Test execution log
- `AI_A1_G01_CONTRIBUTIONS.pdf` — Signed contribution statements
- `AI_A1_G01_UIUX.pdf` — 5-7 page UI/UX design document

## License
Academic assignment — Musanze HarvestLink Cooperative (fictional)

# AI_A1_G02
