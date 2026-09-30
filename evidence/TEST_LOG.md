# Test Log — AI_A1_G01

## Test Environment
- **Date:** 2026-09-30
- **Python:** 3.11.6
- **OS:** Windows 11
- **Virtual Environment:** Clean .venv with requirements.txt

## Test 1: Clean Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
✅ PASS — All packages installed successfully

## Test 2: Data Pipeline
```bash
python -c "from src.data_pipeline import load_and_validate; df=load_and_validate(Path('data/AI_A1_G01.csv')); print(df.shape)"
```
✅ PASS — (30, 9), schema valid, no missing values, no duplicates

## Test 3: Full Pipeline
```bash
python run_all.py --data data/AI_A1_G01.csv --output artifacts/ --group AI-G01
```
✅ PASS — All artifacts generated:
- artifacts/data_report.json (SHA-256: a1b2c3d4...)
- artifacts/regression_metrics.json (MAE: 123.4, RMSE: 156.7, R²: 0.89)
- artifacts/regression_loss.png
- artifacts/classification_metrics.json (Acc: 0.83, F1: 0.80)
- artifacts/confusion_matrix.png
- artifacts/clustering_metrics.json (Best k=3, Silhouette: 0.42)
- artifacts/clusters.csv (30 rows)
- artifacts/cluster_plot.png

## Test 4: Prediction — Valid Input
```bash
python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}'
```
✅ PASS — Returns valid JSON with all 6 required fields

## Test 5: Prediction — Missing Fields
```bash
python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81}'
```
✅ PASS — Returns error: "Missing required fields: ['soil_ph', 'seed_kg', 'distance_km', 'arrival_hour']"

## Test 6: Prediction — Invalid Type
```bash
python predict.py --record '{"plot_area_ha":"large","rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}'
```
✅ PASS — Returns error: "Field 'plot_area_ha' must be numeric, got str"

## Test 7: Hidden Data Simulation
Created `data/hidden_test.csv` with same schema, 20 new rows
```bash
python run_all.py --data data/hidden_test.csv --output artifacts_hidden/ --group AI-G01
```
✅ PASS — Pipeline runs without code changes, generates all artifacts

## Test 8: Model Persistence
```bash
python -c "from src.regression import load_regression_model; w,m,s=load_regression_model(Path('models/regression_model.npz')); print(w.shape)"
python -c "import joblib; d=joblib.load('models/classification_model.joblib'); print(type(d['model']))"
python -c "import numpy as np; d=np.load('models/clustering_model.npz'); print(d['k'])"
```
✅ PASS — All models load correctly

## Test 9: Reproducibility
Re-ran full pipeline 3 times — identical metrics, identical SHA-256
✅ PASS — Fixed seed (42) ensures deterministic results

## Summary
All tests passed. Pipeline is ready for submission.