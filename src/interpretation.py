import shap

def compute_shap(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # binary classification handling
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values