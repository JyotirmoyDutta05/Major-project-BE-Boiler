import pandas as pd

def analyze(df_proc):
    latest = df_proc.iloc[-1]
    
    finding = "Sensors appear reliable."
    severity = "NORMAL"
    confidence = 0.99
    score = 0.0
    
    if latest['vibration_is_nan']:
        finding = "Vibration sensor reporting missing values."
        severity = "WARNING"
        confidence = 0.95
        score = 0.8
        
    if len(df_proc) > 10:
        recent_temp_std = df_proc['furnace_temperature'].tail(10).std()
        if pd.isna(recent_temp_std) or recent_temp_std < 0.01:
            finding = "Temperature sensor appears stuck."
            severity = "CRITICAL"
            confidence = 0.90
            score = 0.9
            
    return {
        "agent": "sensor_agent",
        "finding": finding,
        "severity": severity,
        "confidence": confidence,
        "score": score
    }
