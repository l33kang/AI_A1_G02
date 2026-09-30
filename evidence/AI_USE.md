# AI Usage Disclosure — AI_A1_G01

## Tools Used

| Tool | Purpose | Files Affected |
|------|---------|----------------|
| GitHub Copilot | Code completion, boilerplate generation | All Python files |
| ChatGPT (GPT-4) | Algorithm explanation, debugging, design discussion | Regression gradient descent, clustering evaluation |

## Prompts & Requests

### Regression Implementation
- **Prompt:** "Implement batch gradient descent for linear regression using only NumPy. Include feature scaling, train/test split, loss tracking, and metrics (MAE, RMSE, R²)."
- **Verification:** Compared loss curve against scikit-learn's LinearRegression on same data; verified gradient computation mathematically.

### Classification Setup
- **Prompt:** "Set up stratified train/test split for binary classification with LogisticRegression. Report confusion matrix, accuracy, precision, recall, F1. Explain which error type is more costly for a dispatch attention scenario."
- **Verification:** Ran model, checked metrics sum to 1.0, validated confusion matrix shape.

### Clustering Evaluation
- **Prompt:** "Write code to evaluate KMeans for k=2..5 using silhouette score. Standardize features first. Save cluster assignments and PCA plot."
- **Verification:** Silhouette scores matched manual calculation; PCA plot visually inspected.

### Pipeline Integration
- **Prompt:** "Create run_all.py that orchestrates data loading, regression, classification, clustering. Save all artifacts as JSON/PNG/CSV. Accept CLI args for data path, output dir, group code."
- **Verification:** Ran full pipeline in clean venv; all output files generated and valid JSON.

### Prediction CLI
- **Prompt:** "Build predict.py that loads all three models, accepts JSON via --record, validates schema, returns predictions + probabilities + cluster + metadata. Reject invalid input clearly."
- **Verification:** Tested with valid, missing-field, and wrong-type inputs; all behaviors correct.

## How Output Was Verified

1. **Unit-level:** Each module has `if __name__ == "__main__"` block for isolated testing
2. **Integration:** Full `run_all.py` executed in fresh virtual environment
3. **Cross-check:** Regression predictions compared to scikit-learn baseline
4. **Schema validation:** `predict.py` tested with valid/invalid JSON
5. **Artifact inspection:** All JSON files loaded and parsed; PNGs opened visually
6. **Hidden data simulation:** Pipeline re-run with shuffled rows — no hardcoded values failed

## Human Author Contribution

All code was reviewed, modified, and tested by team members. AI-generated suggestions were:
- Adapted to project structure and naming conventions
- Integrated with custom data pipeline module
- Verified against assignment requirements (no AutoML, NumPy-only regression, etc.)
- Documented in this file per academic integrity policy

## Declaration

We confirm that generative AI was used only for explanation, brainstorming, and debugging as permitted. All submitted code was authored, tested, and explained by team members. Each member can modify and explain their assigned section.