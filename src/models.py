from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def get_models():
    return {
        "random_forest": RandomForestClassifier(n_estimators=200),
        "gradient_boosting": GradientBoostingClassifier(),
        "svm_rbf": SVC(kernel="rbf", probability=True)
    }
