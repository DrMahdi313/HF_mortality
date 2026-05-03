# Heart Failure Survival Prediction — Reproducible ML Study

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview
This project reproduces and extends the findings of:

**Chicco & Jurman (2020)**  
"Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone"

The goal is to build a **fully reproducible machine learning pipeline** and validate the claim that **two clinical features alone can match or outperform models trained on all features**.

---

## Key Insight
Models trained using only:

- `serum_creatinine`
- `ejection_fraction`

can achieve comparable or superior predictive performance relative to models trained on the full dataset.

---

## Project Structure


heart-failure-ml/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
├── experiments/
├── results/
│ ├── metrics/
│ ├── figures/
│ └── reports/
│
├── notebooks/
├── tests/
│
├── README.md
├── requirements.txt
├── LICENSE


---

## Methods

### Models
- Random Forest
- Gradient Boosting
- Support Vector Machine (RBF)

### Evaluation Strategy
- 100 Monte Carlo runs
- Stratified train/test splits
- Aggregated performance metrics

### Metrics
- **Matthews Correlation Coefficient (primary)**
- Accuracy
- F1 Score
- ROC AUC
- PR AUC

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/heart-failure-ml.git
cd heart-failure-ml
2. Install dependencies
pip install -r requirements.txt
3. Add dataset

Place the dataset file in:

data/raw/heart_failure_clinical_records_dataset.csv
4. Run experiments

Full feature set:

python -m experiments.run_full_features

Top-2 features:

python -m experiments.run_top2_features
5. Generate report
python -m experiments.generate_report
Results

Results are saved in:

results/
metrics/ → aggregated model performance
figures/ → ROC & PR curves, feature importance
reports/ → summary tables
Figures
ROC Curve

Precision-Recall Curve

Feature Importance

Reproducibility

This project ensures reproducibility through:

Fixed random seeds
Stratified splits
Multiple runs with aggregation
Modular pipeline design
Testing

Run basic tests:

pytest
Notebook

Exploratory analysis available in:

notebooks/exploration.ipynb
License

This project is licensed under the MIT License.


---

# 💣 Final sanity check (don’t skip)

After updating README:

1. Run experiments  
2. Confirm images exist:

results/figures/