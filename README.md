# Heart Failure Mortality Prediction

A reproducible machine learning study using the UCI Heart Failure Clinical Records dataset to evaluate mortality classification models with admission-time clinical features.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project investigates whether machine learning models can classify mortality outcomes among patients with heart failure using clinical features available at admission.

The analysis compares models trained on the full set of admission-time predictors against models trained using only two features:

- `serum_creatinine`
- `ejection_fraction`

The two-feature comparison is motivated by the findings of Chicco and Jurman (2020), who studied the predictive value of these variables in the same dataset.

This repository is an independent machine learning implementation and comparative experiment, not a clinical prediction system.

## Dataset

The project uses the **Heart Failure Clinical Records Dataset** from the UCI Machine Learning Repository.

- **Observations:** 299 patients
- **Target:** `DEATH_EVENT`
- **Outcome:** Whether the patient died during the follow-up period
- **Predictors:** Clinical and demographic variables recorded in the dataset

Dataset source: [UCI Machine Learning Repository — Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)

### Preventing target leakage

The dataset includes a `time` variable representing the follow-up period in days. This information is not available at the time of admission and is therefore excluded from all model predictors.

Including it would introduce information that would not be available at the intended prediction point and could lead to misleading performance estimates.

The prediction target is `DEATH_EVENT`; `time` is excluded from the feature matrix throughout the experiments.

## Models

Five classification algorithms are evaluated:

| Model | Implementation |
|---|---|
| Random Forest | `RandomForestClassifier` |
| Gradient Boosting | `GradientBoostingClassifier` |
| Support Vector Machine | RBF kernel with feature scaling |
| XGBoost | `XGBClassifier` |
| Multi-layer Perceptron | `MLPClassifier` with feature scaling |

Standardization is applied through scikit-learn pipelines for the SVM and MLP models.

## Evaluation methodology

### Main experiments

The full-feature and two-feature experiments use **Repeated Stratified 5-Fold Cross-Validation**, with 20 repeats.

This produces 100 validation folds per model. Stratification preserves the class proportions across folds, while fixed random seeds support reproducibility.

Performance is evaluated using:

- **Matthews Correlation Coefficient (MCC)**
- Accuracy
- F1 score
- ROC AUC
- Average precision (PR AUC)

For each metric, the mean and standard deviation across validation folds are reported. These standard deviations describe fold-level variation; they are not confidence intervals.

### Feature ablation

A separate ablation experiment removes one admission-time feature at a time and compares the resulting performance against a baseline using all admission-time features.

To keep this experiment computationally manageable, it uses 5-fold stratified cross-validation with 3 repeats.

Paired fold-level metric differences are also saved for comparison.

## Interpretability and visualizations

The full-feature experiment generates:

- ROC curve
- Precision-recall curve
- Calibration curve
- Random Forest feature importance, aggregated across CV folds
- SHAP summary plot for a Random Forest fitted on the complete dataset

The SHAP analysis is descriptive: it explains the fitted full-data Random Forest model. It should not be interpreted as independent or out-of-sample validation.

## Repository structure

```text
HF_mortality/
│
├── data/
│   └── raw/
│       └── heart_failure_clinical_records_dataset.csv
│
├── experiments/
│   ├── run_full_features.py
│   ├── run_top2_features.py
│   ├── run_ablation.py
│   └── generate_report.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── top2_features.json
│   └── summary.csv
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── interpretation.py
│   ├── models.py
│   ├── reporting.py
│   ├── train.py
│   ├── utils.py
│   └── visualization.py
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Installation and usage

### 1. Clone the repository

```bash
git clone https://github.com/DrMahdi313/HF_mortality.git
cd HF_mortality
```

### 2. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) and place the CSV file at:

```text
data/raw/heart_failure_clinical_records_dataset.csv
```

### 5. Run the experiments

Run the scripts from the repository root.

Full-feature experiment:

```bash
python .\experiments\run_full_features.py
```

Two-feature experiment:

```bash
python .\experiments\run_top2_features.py
```

Feature ablation experiment:

```bash
python .\experiments\run_ablation.py
```

On Windows PowerShell, the three experiments can also be run sequentially:

```powershell
python .\experiments\run_full_features.py && python .\experiments\run_top2_features.py && python .\experiments\run_ablation.py
```

### 6. Generate the summary report

After the full-feature and two-feature experiments have completed:

```bash
python .\experiments\generate_report.py
```

This generates:

```text
results/summary.csv
```

The ablation experiment separately generates:

```text
results/metrics/ablation.json
results/metrics/ablation_comparison.csv
```

## Outputs

| Output | Description |
|---|---|
| `results/metrics/full_features.json` | Aggregated metrics for all models using admission-time predictors |
| `results/top2_features.json` | Aggregated metrics using serum creatinine and ejection fraction |
| `results/summary.csv` | Combined full-feature and two-feature metric summary |
| `results/metrics/ablation.json` | Aggregated ablation results and paired comparisons |
| `results/metrics/ablation_comparison.csv` | Feature-removal comparison by model and metric |
| `results/figures/roc_rf.png` | Random Forest ROC curve |
| `results/figures/pr_rf.png` | Random Forest precision-recall curve |
| `results/figures/calibration_rf.png` | Random Forest calibration curve |
| `results/figures/feature_importance_rf.png` | Random Forest feature importance |
| `results/figures/shap_summary.png` | Random Forest SHAP summary plot |

## Tests

Run the test suite from the repository root:

```bash
pytest -v
```

The tests cover data loading, target and follow-up-time exclusion, reproducible splitting, model construction, scaling pipelines, and the cross-validation experiment's feature selection and output structure.

## Limitations

This project has important limitations:

- The dataset contains only 299 observations, which is small for training and comparing multiple machine learning models.
- Repeated cross-validation provides repeated estimates on the same dataset; it does not replace external validation on an independent population.
- Fold-level metric variation should not be interpreted as a confidence interval or proof of statistical significance.
- The experiments do not establish that any model is suitable for clinical decision-making.
- The SHAP analysis describes a model fitted on the available dataset and does not establish causal relationships.
- Model performance on this dataset should not be assumed to generalize to other hospitals, populations, or clinical settings.

**This project is for educational and portfolio purposes only. It is not intended for medical diagnosis, prognosis, or clinical decision-making.**

## References

1. Chicco, D., & Jurman, G. (2020). *Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone.* BMC Medical Informatics and Decision Making, 20, 16. [https://doi.org/10.1186/s12911-020-1023-5](https://doi.org/10.1186/s12911-020-1023-5)
2. Ahmad, T., et al. (2017). *Survival analysis of heart failure patients: A case study.* PLOS ONE. Dataset available through the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.