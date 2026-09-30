import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

output_path = Path('AI_A1_G01_TECHNICAL_REPORT.pdf')

with PdfPages(output_path) as pdf:
    # Page 1: TEAM STRUCTURE & PAGE OWNERSHIP
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.1, 0.95, 'TEAM STRUCTURE & PAGE OWNERSHIP', fontsize=16, weight='bold')
    
    y = 0.90
    fig.text(0.1, y, 'Group: AI-G01 | Course: SWE 3513 - Artificial Intelligence | Assignment 1', fontsize=10); y -= 0.04
    
    fig.text(0.1, y, 'Member Roles & Page Ownership:', fontsize=12, weight='bold'); y -= 0.03
    
    team = [
        ['Member 1', 'Data & UX Lead', 'Schema Validation, Feature Vectorization, Data Report, Dashboard PDF (Pages 1-2, 5)', 'src/data_pipeline.py, artifacts/data_report.json, AI_A1_G01_UIUX.pdf'],
        ['Member 2', 'Regression Engineer', 'NumPy Gradient Descent, Scaling, Loss Curve, Regression Metrics, Derivation Notes', 'src/regression.py, artifacts/regression_metrics.json, artifacts/regression_loss.png'],
        ['Member 3', 'Classification Engineer', 'Stratified Split, Logistic Regression, Metrics, Confusion Matrix, Error Cost Analysis', 'src/classification.py, artifacts/classification_metrics.json, artifacts/confusion_matrix.png'],
        ['Member 4', 'Clustering & QA Engineer', 'Standardization, k Comparison, Silhouette, Cluster Outputs, Cross-Pipeline QA', 'src/clustering.py, artifacts/clustering_metrics.json, artifacts/clusters.csv, artifacts/cluster_plot.png'],
        ['Member 5', 'Reproducibility & Release Lead', 'CLI Contract, Prediction Command, Requirements, README, Evidence, Final Package', 'run_all.py, predict.py, requirements.txt, README.md, evidence/'],
    ]
    
    for row in team:
        fig.text(0.1, y, f'Role: {row[1]} ({row[0]})', fontsize=10, weight='bold'); y -= 0.02
        fig.text(0.15, y, f'Responsibilities: {row[2]}', fontsize=9); y -= 0.02
        fig.text(0.15, y, f'Key Files: {row[3]}', fontsize=9, family='monospace'); y -= 0.03
    
    y -= 0.02
    fig.text(0.1, y, 'Commit Requirements: Each member >= 2 meaningful commits to their assigned modules', fontsize=10, weight='bold')
    fig.text(0.1, y-0.02, 'Verification: Each member must explain/modify their section during live verification', fontsize=10)
    
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Page 2: SYSTEM OVERVIEW & OPERATIONAL DECISIONS
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.1, 0.95, 'SYSTEM OVERVIEW & OPERATIONAL DECISIONS', fontsize=16, weight='bold')
    
    y = 0.90
    fig.text(0.1, y, 'Stakeholder: Musanze HarvestLink Cooperative Operations Team', fontsize=10); y -= 0.03
    
    fig.text(0.1, y, 'Three Operational Decisions:', fontsize=12, weight='bold'); y -= 0.03
    
    decisions = [
        ['1', 'Estimate Expected Harvest Weight', 'Regression', 'Linear Regression (NumPy GD)', 'actual_yield_kg (kg)', 'MAE=50.96, RMSE=59.15, R2=0.995'],
        ['2', 'Flag Consignments Needing Dispatch Attention', 'Binary Classification', 'Logistic Regression', 'dispatch_attention (0/1)', 'Accuracy=1.0, F1=1.0, Prob threshold=0.5'],
        ['3', 'Group Collection Points with Similar Profiles', 'Unsupervised Clustering', 'K-Means (k=3)', '6 input features', 'Silhouette=0.58, Inertia=156.7'],
    ]
    
    for d in decisions:
        fig.text(0.1, y, f'Decision {d[0]}: {d[1]}', fontsize=11, weight='bold'); y -= 0.02
        fig.text(0.15, y, f'Task Type: {d[2]} | Model: {d[3]}', fontsize=9); y -= 0.02
        fig.text(0.15, y, f'Target: {d[4]} | Key Metrics: {d[5]}', fontsize=9); y -= 0.03
    
    y -= 0.02
    fig.text(0.1, y, 'System Flow:', fontsize=12, weight='bold'); y -= 0.02
    flow = [
        'CSV Upload (schema validated) -> Data Quality Report (SHA-256, stats, missing/duplicates)',
        '-> Feature Vectorization (6 features, standardized) -> Model Inference (3 models)',
        '-> Results: Yield prediction (kg) + Dispatch flag (0/1 + prob) + Cluster label (0/1/2)',
        '-> Human Decision: Operations team reviews, applies domain knowledge, confirms dispatch',
    ]
    for f in flow:
        fig.text(0.15, y, f'{f}', fontsize=9); y -= 0.02
    
    y -= 0.02
    fig.text(0.1, y, 'Input Schema: 9 columns (record_id + 6 features + 2 targets), 30 rows, no missing values, no duplicates', fontsize=9)
    fig.text(0.1, y-0.02, 'Dataset SHA-256: 70480135477a038edb41ec136b249f606176e616acf8c64a523773133f52c762', fontsize=9, family='monospace')
    
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Page 3: SCHEMA VALIDATION SPECIFICATION
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.1, 0.95, 'SCHEMA VALIDATION SPECIFICATION', fontsize=16, weight='bold')
    
    y = 0.90
    fig.text(0.1, y, 'Implementation: src/data_pipeline.py -> load_and_validate()', fontsize=10); y -= 0.03
    
    fig.text(0.1, y, 'Required Columns (exact names, exact order not enforced):', fontsize=11, weight='bold'); y -= 0.025
    cols = [
        ['record_id', 'text', 'Unique row identifier', 'Never used as model feature'],
        ['plot_area_ha', 'numeric', 'Farm area (hectares)', 'Must be numeric'],
        ['rainfall_mm', 'numeric', 'Recent rainfall estimate', 'Must be numeric'],
        ['soil_ph', 'numeric', 'Soil acidity measure', 'Must be numeric'],
        ['seed_kg', 'numeric', 'Seed quantity (kg)', 'Must be numeric'],
        ['distance_km', 'numeric', 'Distance to collection point', 'Must be numeric'],
        ['arrival_hour', 'numeric', 'Planned arrival hour (24h)', 'Must be numeric'],
        ['actual_yield_kg', 'numeric target', 'Regression target', 'Must be numeric'],
        ['dispatch_attention', 'binary target', 'Classification target', 'Must be 0 or 1 only'],
    ]
    
    for c in cols:
        fig.text(0.15, y, f'{c[0]:<20} {c[1]:<15} {c[2]:<30} {c[3]}', fontsize=9, family='monospace'); y -= 0.022
    
    y -= 0.02
    fig.text(0.1, y, 'Validation Rules:', fontsize=11, weight='bold'); y -= 0.02
    rules = [
        'All 9 columns present; missing columns raise ValueError with column names',
        'No unexpected/extra columns allowed',
        'Features + regression target must be numeric dtype',
        'dispatch_attention must contain only {0, 1}',
        'Missing values reported per column (count)',
        'Duplicate rows checked on features + targets (excluding record_id)',
        'SHA-256 computed on raw CSV file bytes for audit trail',
    ]
    for r in rules:
        fig.text(0.15, y, f'{r}', fontsize=9); y -= 0.02
    
    y -= 0.02
    fig.text(0.1, y, 'Output: data_report.json containing:', fontsize=11, weight='bold'); y -= 0.02
    outputs = ['Row count, Feature count (6)', 'Missing values per column', 'Duplicate row count',
               'Descriptive statistics (mean, std, min, max, median per column)', 'Group code (AI-G01)', 'Dataset SHA-256 fingerprint', 'Random seed (42)']
    for o in outputs:
        fig.text(0.15, y, f'{o}', fontsize=9); y -= 0.02
    
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Page 4: FEATURE VECTORIZATION STRATEGY
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.1, 0.95, 'FEATURE VECTORIZATION STRATEGY', fontsize=16, weight='bold')
    
    y = 0.90
    fig.text(0.1, y, 'Feature Matrix Construction:', fontsize=11, weight='bold'); y -= 0.025
    
    steps = [
        '1. Load CSV with pandas, validate schema',
        '2. Separate identifier column (record_id) from features and targets',
        '3. Feature columns (6): plot_area_ha, rainfall_mm, soil_ph, seed_kg, distance_km, arrival_hour',
        '4. Convert to NumPy array: X = df[FEATURE_COLS].values.astype(float) -> shape (n_samples, 6)',
        '5. Targets: y_reg = df[actual_yield_kg].values.astype(float), y_clf = df[dispatch_attention].values.astype(int)',
        '6. Train/test split: 80/20 stratified for classification, random for regression (seed=42)',
    ]
    for s in steps:
        fig.text(0.15, y, s, fontsize=9); y -= 0.022
    
    y -= 0.02
    fig.text(0.1, y, 'Standardization (Supervised Models):', fontsize=11, weight='bold'); y -= 0.025
    std_steps = [
        'Scaler fitted ONLY on training data: mean = X_train.mean(axis=0), std = X_train.std(axis=0)',
        'std[std == 0] = 1.0 to avoid division by zero',
        'Transform: X_scaled = (X - mean) / std',
        'Same scaler applied to test data and prediction inputs',
        'Parameters saved in model artifacts for reproducibility',
    ]
    for s in std_steps:
        fig.text(0.15, y, f'{s}', fontsize=9); y -= 0.02
    
    y -= 0.02
    fig.text(0.1, y, 'Clustering Standardization:', fontsize=11, weight='bold'); y -= 0.025
    fig.text(0.15, y, 'StandardScaler fitted on ALL data (unsupervised, no train/test split)', fontsize=9); y -= 0.02
    fig.text(0.15, y, 'Targets (actual_yield_kg, dispatch_attention) EXCLUDED from clustering features', fontsize=9, weight='bold'); y -= 0.02
    fig.text(0.15, y, 'Scaler params saved in clustering_model.npz for prediction', fontsize=9); y -= 0.02
    
    y -= 0.02
    fig.text(0.1, y, 'No Hardcoded Values:', fontsize=11, weight='bold'); y -= 0.02
    fig.text(0.15, y, 'All dimensions derived from data shape at runtime', fontsize=9); y -= 0.02
    fig.text(0.15, y, 'Column names from FEATURE_COLS constant', fontsize=9); y -= 0.02
    fig.text(0.15, y, 'Works on hidden test data with same schema', fontsize=9); y -= 0.02
    
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Page 5: MATHEMATICAL DERIVATION NOTES
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.1, 0.95, 'MATHEMATICAL DERIVATION NOTES', fontsize=16, weight='bold')
    
    y = 0.90
    fig.text(0.1, y, 'Linear Regression with Batch Gradient Descent (src/regression.py)', fontsize=11, weight='bold'); y -= 0.03
    
    fig.text(0.1, y, 'Model: y = Xw + b  ->  with bias trick: y = X_bias @ w_bias', fontsize=10); y -= 0.02
    fig.text(0.1, y, 'where X_bias = [1, X] shape (n, d+1), w_bias = [b, w] shape (d+1)', fontsize=10); y -= 0.03
    
    fig.text(0.1, y, 'Loss Function (MSE):', fontsize=11, weight='bold'); y -= 0.02
    fig.text(0.15, y, 'L(w) = (1/n) * ||y - Xw||^2 = (1/n) * sum((y_i - x_i^T w)^2)', fontsize=10); y -= 0.025
    
    fig.text(0.1, y, 'Gradient:', fontsize=11, weight='bold'); y -= 0.02
    fig.text(0.15, y, 'dL/dw = (-2/n) * X^T @ (y - Xw) = (-2/n) * X^T @ residuals', fontsize=10); y -= 0.025
    
    fig.text(0.1, y, 'Batch Gradient Descent Update:', fontsize=11, weight='bold'); y -= 0.02
    fig.text(0.15, y, 'w <- w - alpha * dL/dw', fontsize=10); y -= 0.02
    fig.text(0.15, y, 'alpha = learning_rate (0.01), epochs = 2000, tolerance = 1e-6', fontsize=9); y -= 0.03
    
    fig.text(0.1, y, 'Algorithm:', fontsize=11, weight='bold'); y -= 0.02
    algo = [
        '1. Add bias column: X_bias = hstack([ones(n,1), X_scaled])',
        '2. Initialize weights: w ~ N(0, 0.01^2) with seed=42',
        '3. For each epoch: y_pred = X_bias @ w; loss = MSE(y, y_pred);',
        '   grad = (-2/n) * X_bias.T @ (y - y_pred); w -= lr * grad',
        '4. Track train_loss and val_loss per epoch',
        '5. Early stop if |prev_loss - curr_loss| < tolerance',
    ]
    for a in algo:
        fig.text(0.15, y, a, fontsize=9); y -= 0.02
    
    y -= 0.02
    fig.text(0.1, y, 'Metrics:', fontsize=11, weight='bold'); y -= 0.02
    fig.text(0.15, y, 'MAE = mean(|y - y_pred|)', fontsize=9); y -= 0.02
    fig.text(0.15, y, 'RMSE = sqrt(mean((y - y_pred)^2))', fontsize=9); y -= 0.02
    fig.text(0.15, y, 'R2 = 1 - SS_res/SS_tot where SS_res = sum((y - y_pred)^2), SS_tot = sum((y - mean(y))^2)', fontsize=9); y -= 0.02
    
    y -= 0.02
    fig.text(0.1, y, 'Classification (Logistic Regression):', fontsize=11, weight='bold'); y -= 0.02
    fig.text(0.15, y, 'P(y=1|x) = 1 / (1 + exp(-(w^T x + b)))', fontsize=10); y -= 0.02
    fig.text(0.15, y, 'Trained with sklearn LogisticRegression(max_iter=1000, random_state=42)', fontsize=9); y -= 0.02
    fig.text(0.15, y, 'Cost Analysis: False Negative (miss dispatch) > False Positive (extra attention)', fontsize=9); y -= 0.02
    
    y -= 0.02
    fig.text(0.1, y, 'Clustering (K-Means):', fontsize=11, weight='bold'); y -= 0.02
    fig.text(0.15, y, 'Minimize: sum_{i=1}^n ||x_i - c_{z_i}||^2 where z_i = argmin_k ||x_i - c_k||^2', fontsize=9); y -= 0.02
    fig.text(0.15, y, 'k evaluated 2..5 via silhouette_score; best k=3 (sil=0.58)', fontsize=9); y -= 0.02
    
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Page 6: CORE PIPELINE IMPLEMENTATION
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.1, 0.95, 'CORE PIPELINE IMPLEMENTATION', fontsize=16, weight='bold')
    
    y = 0.90
    fig.text(0.1, y, 'Entry Point: run_all.py', fontsize=11, weight='bold'); y -= 0.025
    
    pipeline = [
        'def main():',
        '  1. Parse CLI args: --data, --output, --group',
        '  2. Load & validate CSV -> DataFrame (30, 9)',
        '  3. Generate data_report.json (stats, SHA-256)',
        '  4. Separate features/targets: X (30,6), y_reg (30,), y_clf (30,)',
        '  5. Train regression: train_regression(X, y_reg) -> weights, scaler, metrics, loss_history',
        '  6. Save regression artifacts: metrics.json, loss.png, model.npz',
        '  7. Train classification: train_classification(X, y_clf) -> model, scaler, metrics, CM',
        '  8. Save classification artifacts: metrics.json, cm.png, model.joblib',
        '  9. Train clustering: train_clustering(X) -> labels, centers, scaler, k_eval',
        ' 10. Save clustering artifacts: metrics.json, clusters.csv, plot.png, model.npz',
        ' 11. Print summary metrics to stdout',
    ]
    for p in pipeline:
        fig.text(0.1, y, p, fontsize=9, family='monospace'); y -= 0.02
    
    y -= 0.02
    fig.text(0.1, y, 'Key Implementation Details:', fontsize=11, weight='bold'); y -= 0.025
    details = [
        'Fixed SEED=42 used in: np.random.seed, train_test_split, KMeans, LogisticRegression, PCA',
        'Scalers fitted on training data only (supervised); all data for clustering',
        'Regression: gradient_descent() returns (weights, train_losses, val_losses)',
        'Classification: stratified split preserves class distribution',
        'Clustering: silhouette_score for k=2..5, automatic best-k selection',
        'All artifacts saved to --output directory; models to models/',
        'SHA-256 computed via hashlib.sha256 on raw file bytes',
    ]
    for d in details:
        fig.text(0.15, y, f'{d}', fontsize=9); y -= 0.02
    
    y -= 0.02
    fig.text(0.1, y, 'Prediction CLI: predict.py', fontsize=11, weight='bold'); y -= 0.025
    pred_details = [
        'Accepts --record JSON with 6 feature fields',
        'Validates: all fields present, all numeric types',
        'Loads all 3 models from models/ directory',
        'Returns JSON: regression_prediction_kg, classification_prediction,',
        '         classification_probability, cluster_label, group_code, model_version',
        'Rejects invalid input with clear error message',
    ]
    for d in pred_details:
        fig.text(0.15, y, f'{d}', fontsize=9); y -= 0.02
    
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Page 7: UNIT TESTING SUIT
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.1, 0.95, 'UNIT TESTING SUIT', fontsize=16, weight='bold')
    
    y = 0.90
    fig.text(0.1, y, 'Test Strategy: Each module has if __name__ == "__main__" block for isolated testing', fontsize=10); y -= 0.03
    fig.text(0.1, y, 'Integration: Full pipeline test via run_all.py in clean venv', fontsize=10); y -= 0.03
    
    fig.text(0.1, y, 'Module-Level Tests:', fontsize=11, weight='bold'); y -= 0.025
    modules = [
        ('data_pipeline.py', [
            'Load CSV -> validate schema (30, 9)',
            'Check missing values -> all zeros',
            'Check duplicates -> 0',
            'Separate features -> X(30,6), y_reg(30,), y_clf(30,)',
            'Generate data_report.json -> valid JSON with all required keys',
        ]),
        ('regression.py', [
            'Train on X, y_reg -> returns weights(7), metrics, loss_history',
            'Metrics: MAE, RMSE, R2 computed on test set',
            'Loss decreases monotonically (verified)',
            'Save/load model -> weights match',
            'Predict on new sample -> returns scalar',
            'Cross-check: vs sklearn.LinearRegression on same split',
        ]),
        ('classification.py', [
            'Stratified split -> class distribution preserved',
            'Train LogisticRegression -> converged (max_iter=1000)',
            'Metrics: Acc, Prec, Rec, F1, Confusion Matrix',
            'Error cost analysis: FN > FP documented',
            'Save/load model -> predictions match',
        ]),
        ('clustering.py', [
            'Evaluate k=2,3,4,5 -> silhouette scores computed',
            'Best k selected automatically (max silhouette)',
            'KMeans fitted -> labels for all 30 samples',
            'PCA plot generated -> 2D projection',
            'Save/load model -> cluster assignment matches',
        ]),
    ]
    
    for mod, tests in modules:
        fig.text(0.1, y, mod, fontsize=10, weight='bold'); y -= 0.02
        for t in tests:
            fig.text(0.15, y, f'{t}', fontsize=9); y -= 0.018
        y -= 0.015
    
    y -= 0.02
    fig.text(0.1, y, 'Integration Tests (from TEST_LOG.md):', fontsize=11, weight='bold'); y -= 0.025
    integration = [
        'Clean venv setup + pip install -r requirements.txt',
        'Full pipeline run -> all 8 artifacts + 3 models generated',
        'Regression metrics match expected ranges',
        'Classification metrics: Acc=1.0, F1=1.0',
        'Clustering: k=3, silhouette=0.58',
        'Prediction valid input -> returns all 6 output fields',
        'Prediction missing fields -> JSON error with field names',
        'Prediction wrong type -> JSON error with field name',
        'Hidden data simulation -> pipeline runs without code changes',
        'Reproducibility: 3 runs -> identical metrics, identical SHA-256',
    ]
    for t in integration:
        fig.text(0.15, y, f'{t}', fontsize=9); y -= 0.018
    
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Page 8: DECISION DASHBOARD TERMINAL INTERFACE
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.1, 0.95, 'DECISION DASHBOARD TERMINAL INTERFACE', fontsize=16, weight='bold')
    
    y = 0.90
    fig.text(0.1, y, 'CLI Commands:', fontsize=11, weight='bold'); y -= 0.025
    
    cmds = [
        ('Full Pipeline:', 'python run_all.py --data data/AI_A1_G01.csv --output artifacts/ --group AI-G01'),
        ('Prediction:', 'python predict.py --record \'{"plot_area_ha":1.2,"rainfall_mm":81,...}\''),
        ('  Validated via Python -c to avoid shell escaping issues', ''),
    ]
    for label, cmd in cmds:
        fig.text(0.1, y, label, fontsize=10, weight='bold'); y -= 0.02
        if cmd:
            fig.text(0.15, y, cmd, fontsize=8, family='monospace'); y -= 0.02
        y -= 0.015
    
    y -= 0.02
    fig.text(0.1, y, 'Pipeline Output (stdout):', fontsize=11, weight='bold'); y -= 0.025
    stdout = [
        'Loading data from data\\AI_A1_G01.csv...',
        'Data shape: (30, 9)',
        'Generating data report...',
        'Data report saved. SHA-256: 70480135477a038edb41ec136b249f606176e616acf8c64a523773133f52c762',
        'Training regression model...',
        'Regression metrics: MAE=50.96, RMSE=59.15, R2=0.9949',
        'Training classification model...',
        'Classification metrics: Acc=1.0000, F1=1.0000',
        'Error cost: False Negative more costly than False Positive...',
        'Training clustering model...',
        'Clustering: k=3, Silhouette=0.5801',
        'Interpretation: Clusters are statistical groupings...',
        'Pipeline complete! Artifacts saved to: artifacts',
        'Models saved to: models',
    ]
    for s in stdout:
        fig.text(0.15, y, s, fontsize=8, family='monospace'); y -= 0.018
    
    y -= 0.02
    fig.text(0.1, y, 'Prediction Output (JSON):', fontsize=11, weight='bold'); y -= 0.025
    pred_out = {
        "regression_prediction_kg": 2684.4,
        "classification_prediction": 0,
        "classification_probability": 0.46,
        "cluster_label": 1,
        "group_code": "AI-G01",
        "model_version": "1.0.0"
    }
    import json
    fig.text(0.15, y, json.dumps(pred_out, indent=2), fontsize=8, family='monospace')
    
    y -= 0.12
    fig.text(0.1, y, 'Validation Error Examples:', fontsize=11, weight='bold'); y -= 0.025
    fig.text(0.15, y, '{"error": "Missing required fields: [\'soil_ph\', \'seed_kg\', \'distance_km\', \'arrival_hour\']"}', fontsize=8, family='monospace'); y -= 0.02
    fig.text(0.15, y, '{"error": "Field \'plot_area_ha\' must be numeric, got str"}', fontsize=8, family='monospace'); y -= 0.02
    fig.text(0.15, y, '{"error": "Invalid JSON: Expecting property name..."}', fontsize=8, family='monospace')
    
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Page 9: VERSIONED COMMIT HISTORY
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.1, 0.95, 'VERSIONED COMMIT HISTORY', fontsize=16, weight='bold')
    
    y = 0.90
    fig.text(0.1, y, 'Git Repository: [GitHub URL]', fontsize=10); y -= 0.02
    fig.text(0.1, y, 'Final Commit Hash: [to be filled at submission]', fontsize=10); y -= 0.04
    
    fig.text(0.1, y, 'Required: Each member >= 2 meaningful commits to their modules', fontsize=10, weight='bold'); y -= 0.03
    
    fig.text(0.1, y, 'Expected Commit Pattern:', fontsize=11, weight='bold'); y -= 0.025
    commits = [
        'Member 1 (Data & UX):',
        '  - feat(data): add schema validation and SHA-256 computation',
        '  - feat(data): implement data_report.json generation',
        '  - docs(uiux): add dashboard design PDF pages 1-2, 5',
        '',
        'Member 2 (Regression):',
        '  - feat(reg): implement NumPy batch gradient descent',
        '  - feat(reg): add feature standardization and loss tracking',
        '  - feat(reg): implement MAE/RMSE/R2 metrics',
        '',
        'Member 3 (Classification):',
        '  - feat(clf): add stratified split and LogisticRegression',
        '  - feat(clf): implement confusion matrix and metrics',
        '  - docs(clf): add error cost analysis (FN > FP)',
        '',
        'Member 4 (Clustering & QA):',
        '  - feat(cluster): implement k=2..5 evaluation with silhouette',
        '  - feat(cluster): add PCA visualization and cluster CSV',
        '  - test(qa): cross-pipeline validation and TEST_LOG.md',
        '',
        'Member 5 (Reproducibility):',
        '  - feat(cli): implement run_all.py and predict.py contracts',
        '  - feat(cli): add input validation and model loading',
        '  - docs: requirements.txt, README.md, evidence files',
        '  - chore: final package assembly and submission prep',
    ]
    for c in commits:
        if c == '':
            y -= 0.01
        else:
            fig.text(0.15 if not c.endswith(':') else 0.1, y, c, fontsize=9, family='monospace'); y -= 0.018
    
    y -= 0.02
    fig.text(0.1, y, 'Verification Requirements:', fontsize=11, weight='bold'); y -= 0.025
    verify = [
        'GitHub commit hash in Moodle must match submitted ZIP',
        'Dataset must not be renamed, replaced, extended, or modified',
        'SHA-256 in Moodle must match run_all.py output',
        'README.md must contain: setup/execution commands, Python version, members, roles, repo URL, commit hash, expected outputs, limitations',
        'requirements.txt must contain every external dependency',
        'AI_A1_G01_CONTRIBUTIONS.pdf must be signed by every member',
        'AI_USE.md must disclose every generative AI tool used, purpose, prompts, affected files, verification',
        'All artifacts must be generated by submitted code (not manual)',
    ]
    for v in verify:
        fig.text(0.15, y, f'{v}', fontsize=9); y -= 0.018
    
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

print(f'Technical report saved to {output_path}')