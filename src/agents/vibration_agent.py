def analyze(df_proc):
    """
    Vibration Agent: Analyzes vibration level and sustained trends.
    Uses rolling mean to avoid false positives from single-point noise.
    Normal vibration is ~2.5 mm/s with std 0.3.
    """
    latest = df_proc.iloc[-1]
    
    finding = "Normal"
    severity = "NORMAL"
    confidence = 0.90
    score = 0.0
    
    vib_rolling_mean = latest.get('vibration_rolling_mean', latest['vibration'])
    
    # Normal vibration is ~2.5, warning at sustained >3.5, critical at >5.0
    if vib_rolling_mean > 3.5:
        finding = "Increasing vibration detected."
        severity = "WARNING"
        confidence = 0.82
        score = 0.6
        if vib_rolling_mean > 5.0:
            finding = "Vibration critically elevated."
            severity = "CRITICAL"
            confidence = 0.92
            score = 0.9
            
    return {
        "agent": "vibration_agent",
        "finding": finding,
        "severity": severity,
        "confidence": confidence,
        "score": score
    }
