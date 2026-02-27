import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np

FEATURES = [
    'order_frequency', 'avg_order_value', 'is_premium', 'rating_normalized',
    'hour_sin', 'hour_cos', 'is_lunch', 'is_dinner', 'cart_size',
    'has_main', 'has_side', 'has_dessert', 'has_drink',
    'city_encoded', 'cuisine_encoded', 'category_encoded'
]

TARGET = 'addon_accepted'

def train_baseline(df):
    """
    Baseline model: predict average probability.
    """
    avg_prob = df[TARGET].mean()
    return lambda X: np.full(len(X), avg_prob)

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    if callable(model):  # Baseline
        y_pred_prob = model(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Precision@5: top 5% predictions
    threshold = np.percentile(y_pred_prob, 95)
    y_pred_top5 = (y_pred_prob >= threshold).astype(int)
    precision_at_5 = precision_score(y_test, y_pred_top5)

    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'precision_at_5': precision_at_5
    }

def train_and_evaluate(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {}

    # Baseline
    baseline = train_baseline(pd.concat([X_train, y_train], axis=1))
    models['baseline'] = evaluate_model(baseline, X_test, y_test)

    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    models['logistic_regression'] = evaluate_model(lr_model, X_test, y_test)

    # Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train)
    models['gradient_boosting'] = evaluate_model(gb_model, X_test, y_test)

    return models, lr_model, gb_model  # Return trained models for inference
