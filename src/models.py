from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier    # Added XGBoost classifier
from sklearn.neural_network import MLPClassifier    # Added MLPClassifier for neural network model


def get_models():
    return {
        "random_forest": RandomForestClassifier(n_estimators=200),
        "gradient_boosting": GradientBoostingClassifier(),
        "svm_rbf": SVC(kernel="rbf", probability=True),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss"
            ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=500)
    }
