from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier    # Added XGBoost classifier
from sklearn.neural_network import MLPClassifier    # Added MLPClassifier for neural network model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_models():
    return {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "svm_rbf": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                kernel="rbf",
                probability=True,
                random_state=42
                ))
        ]),
        "xgboost": XGBClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss"
            ),
        "mlp": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(32, 16),
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                max_iter=2000))
        ])
    }
