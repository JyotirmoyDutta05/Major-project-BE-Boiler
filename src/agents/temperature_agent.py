def analyze(df_proc):
    """
    Temperature Agent: Analyzes furnace and steam temperature for anomalies.
    Uses rolling mean trends rather than single-point deltas to avoid
    false positives from normal sensor noise.
    """
    latest = df_proc.iloc[-1]
    
    finding = "Normal"
    severity = "NORMAL"
    confidence = 0.95
    score = 0.0
    
    # Use rolling mean to detect sustained trends, not single-point noise
    temp_rolling_mean = latest.get('furnace_temperature_rolling_mean', latest['furnace_temperature'])
    residual_std = latest.get('furnace_temperature_residual_rolling_std', 0)
    
    # Check for sustained high temperature (normal ~1050 °C)
    is_high_temp = temp_rolling_mean > 1100
    # Instability measured after removing load-driven variation (normal residual std ~10)
    is_trending_up = residual_std > 25
    
    if is_high_temp or is_trending_up:
        finding = "Persistent abnormal temperature trend detected."
        severity = "WARNING"
        confidence = 0.87
        score = 0.6
        if temp_rolling_mean > 1200:
            finding = "Temperature critically above expected operating range."
            severity = "CRITICAL"
            confidence = 0.95
            score = 0.9
            
    return {
        "agent": "temperature_agent",
        "finding": finding,
        "severity": severity,
        "confidence": confidence,
        "score": score
    }
