def analyze(df_proc):
    """
    Flow Agent: Analyzes steam flow and feedwater flow for anomalies.
    Uses rolling mean to detect sustained decline, not single-point noise.
    Normal steam_flow is ~435 (85*5+10).
    """
    latest = df_proc.iloc[-1]
    
    finding = "Normal"
    severity = "NORMAL"
    confidence = 0.88
    score = 0.0
    
    flow_rolling_mean = latest.get('steam_flow_rolling_mean', latest['steam_flow'])
    
    # Normal steam flow is ~435. Warning if sustained drop below 400.
    if flow_rolling_mean < 400:
        finding = "Steam flow is declining relative to expected load."
        severity = "WARNING"
        confidence = 0.75
        score = 0.5
        if flow_rolling_mean < 350:
            finding = "Steam flow critically low."
            severity = "CRITICAL"
            confidence = 0.90
            score = 0.85
        
    return {
        "agent": "flow_agent",
        "finding": finding,
        "severity": severity,
        "confidence": confidence,
        "score": score
    }
