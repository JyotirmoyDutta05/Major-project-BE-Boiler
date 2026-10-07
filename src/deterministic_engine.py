def run_engineering_rules(df_proc):
    """
    Deterministic Engineering Engine: transparent rule-based detection.
    Uses rolling statistics to avoid false positives from single-point noise.
    All thresholds are calibrated against the synthetic data generator's normal ranges:
      - furnace_temperature: ~1050 (85*10 + 200), std ~10
      - vibration: ~2.5, std ~0.3
      - steam_flow: ~435 (85*5 + 10), std ~5
    """
    latest = df_proc.iloc[-1]
    
    rules_triggered = []
    engineering_findings = []
    
    # Use rolling means for sustained-trend detection
    temp_rmean = latest.get('furnace_temperature_rolling_mean', latest['furnace_temperature'])
    temp_rstd = latest.get('furnace_temperature_rolling_std', 0)
    vib_rmean = latest.get('vibration_rolling_mean', latest['vibration'])
    flow_rmean = latest.get('steam_flow_rolling_mean', latest['steam_flow'])
    
    # 1. Temperature rules (normal ~1050)
    if temp_rmean > 1100:
        rules_triggered.append("TEMP_RISING")
        engineering_findings.append("Temperature trend is abnormally high")
    
    if temp_rmean > 1200:
        rules_triggered.append("TEMP_HIGH")
        engineering_findings.append("Furnace temperature is significantly above expected range")
        
    if latest.get('furnace_temperature_residual_rolling_std', 0) > 25:
        rules_triggered.append("TEMP_UNSTABLE")
        engineering_findings.append("Temperature is unstable relative to boiler load")
        
    # 2. Vibration rules (normal ~2.5)
    if vib_rmean > 3.5:
        rules_triggered.append("VIBRATION_HIGH")
        engineering_findings.append("Vibration level is high")
        
    if vib_rmean > 5.0:
        rules_triggered.append("VIBRATION_CRITICAL")
        engineering_findings.append("Vibration is critically elevated")
        
    # 3. Steam Flow rules (normal ~435)
    if flow_rmean < 400:
        rules_triggered.append("STEAM_FLOW_DECLINING")
        engineering_findings.append("Steam flow is experiencing unexpected reduction")
        
    # 4. Sensor Reliability rules
    if latest.get('vibration_is_nan', False):
        rules_triggered.append("SENSOR_FAULT_VIBRATION")
        engineering_findings.append("Vibration sensor reported missing values (NaN)")
        
    # Sensor stuck check
    if len(df_proc) > 10:
        recent_temp_std = df_proc['furnace_temperature'].tail(10).std()
        if recent_temp_std < 0.01:
            rules_triggered.append("SENSOR_STUCK_TEMP")
            engineering_findings.append("Temperature sensor appears stuck (constant value)")
            
    # 5. Pressure rules
    if latest['steam_pressure'] > 160:
        rules_triggered.append("PRESSURE_HIGH")
        engineering_findings.append("Steam pressure is above normal operating range")
    
    if latest['steam_pressure'] < 100:
        rules_triggered.append("PRESSURE_LOW")
        engineering_findings.append("Steam pressure is below normal operating range")
            
    # Calculate rule risk score based on number and severity of rules
    risk_score = min(1.0, len(rules_triggered) * 0.20)
    
    return {
        "rules_triggered": rules_triggered,
        "engineering_findings": engineering_findings,
        "rule_risk_score": risk_score
    }
