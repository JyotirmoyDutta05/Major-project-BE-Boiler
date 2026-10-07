import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
import pickle

# Features used by the Autoencoder
FEATURES = [
    'furnace_temperature_rolling_mean',
    'steam_pressure_rolling_mean',
    'steam_flow_rolling_mean',
    'vibration_rolling_mean',
    'furnace_temperature_residual',
]

SCORING_WINDOW = 10
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'anomaly_detector.keras')
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'scaler.pkl')

def build_autoencoder(input_dim):
    """Builds a simple TensorFlow Autoencoder"""
    model = Sequential([
        # Encoder
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        # Decoder
        layers.Dense(16, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_anomaly_detector(df_normal, model_path=MODEL_PATH):
    """Train TensorFlow Autoencoder on NORMAL operating data."""
    X = df_normal[FEATURES].fillna(0).values
    
    # We must scale data for neural networks (manual standard scaler)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X_scaled = (X - mean) / std
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
        
    model = build_autoencoder(X.shape[1])
    # Suppress training output (verbose=0)
    model.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)
    
    model.save(model_path)
    return model

def score_rows(df_proc, model_path=MODEL_PATH):
    """Return a 0-1 anomaly score for every row using reconstruction error."""
    if not os.path.exists(model_path) or not os.path.exists(SCALER_PATH):
        return np.zeros(len(df_proc))
        
    model = tf.keras.models.load_model(model_path)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    X = df_proc[FEATURES].fillna(0).values
    X_scaled = (X - scaler['mean']) / scaler['std']
    
    X_pred = model.predict(X_scaled, verbose=0)
    # Mean Squared Error for each row
    mse = np.mean(np.square(X_scaled - X_pred), axis=1)
    
    # Map MSE to a 0-1 score smoothly. 
    # Normal data usually has MSE < 0.5. Anomalous data has MSE > 2.0.
    # Using a sigmoid function centered at 1.0
    scores = 1.0 / (1.0 + np.exp(-3 * (mse - 1.0)))
    return scores

def predict_anomaly(df_proc, model_path=MODEL_PATH):
    """
    Score the most recent SCORING_WINDOW readings and average them.
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
