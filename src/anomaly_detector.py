import numpy as np
import pickle
import os
from sklearn.ensemble import IsolationForest

# Features used by the Isolation Forest (rolling means smooth out single-point noise)
FEATURES = [
    'furnace_temperature_rolling_mean',
    'steam_pressure_rolling_mean',
    'steam_flow_rolling_mean',
    'vibration_rolling_mean',
    'furnace_temperature_residual',
]

# Number of most-recent readings scored together (pattern, not single point)
SCORING_WINDOW = 10

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'anomaly_detector.pkl')


def train_anomaly_detector(df_normal, model_path=MODEL_PATH):
    """Train Isolation Forest on NORMAL operating data only."""
    X = df_normal[FEATURES].fillna(0)
    clf = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    clf.fit(X)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    return clf


def score_rows(df_proc, model_path=MODEL_PATH):
    """Return a 0-1 anomaly score for every row (used for charts and evaluation)."""
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    decision = clf.decision_function(df_proc[FEATURES].fillna(0))
    # decision > 0 => normal, < 0 => anomalous. Map smoothly to 0..1.
    return 1.0 / (1.0 + np.exp(20 * decision))


def predict_anomaly(df_proc, model_path=MODEL_PATH):
    """
    Score the most recent SCORING_WINDOW readings and average them,
    so a single noisy reading does not flip the result.
    """
    if not os.path.exists(model_path):
        return {"ml_anomaly": False, "anomaly_score": 0.0}

    window = df_proc.tail(SCORING_WINDOW)
    scores = score_rows(window, model_path)
    mean_score = float(np.mean(scores))

    return {
        "ml_anomaly": bool(mean_score > 0.5),
        "anomaly_score": mean_score,
    }
