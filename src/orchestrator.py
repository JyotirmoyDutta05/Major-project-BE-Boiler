from src.agents import temperature_agent, vibration_agent, flow_agent, sensor_agent

def run_orchestrator(df_proc, engineering_results, ml_results):
    # Run agents
    temp_result = temperature_agent.analyze(df_proc)
    vib_result = vibration_agent.analyze(df_proc)
    flow_result = flow_agent.analyze(df_proc)
    sens_result = sensor_agent.analyze(df_proc)
    
    agents_results = [temp_result, vib_result, flow_result, sens_result]
    
    # Evidence fusion
    # Weights: deterministic(0.3), ML(0.3), Temp(0.15), Vib(0.15), Flow(0.1)
    # Note: sensor fault overrides general risk
    
    deterministic_score = engineering_results['rule_risk_score']
    ml_score = ml_results['anomaly_score']
    
    fused_risk = (
        0.30 * deterministic_score +
        0.30 * ml_score +
        0.15 * temp_result['score'] +
        0.15 * vib_result['score'] +
        0.10 * flow_result['score']
    )
    
    # Check if it's primarily a sensor fault
    if sens_result['score'] > 0.5:
        primary_issue = "SENSOR_FAULT"
        final_risk = sens_result['score'] # Override risk
    else:
        final_risk = fused_risk
        if fused_risk > 0.75:
            primary_issue = "CRITICAL"
        elif fused_risk > 0.4:
            primary_issue = "EARLY_DEGRADATION"
        else:
            primary_issue = "NORMAL"
            
    # Calculate health score (0-100)
    health_score = int(max(0, min(100, 100 - (final_risk * 100))))
    
    # Determine risk level
    if health_score >= 75: risk_level = "NORMAL"
    elif health_score >= 50: risk_level = "WARNING"
    elif health_score >= 25: risk_level = "HIGH"
    else: risk_level = "CRITICAL"
    
    confidence = (temp_result['confidence'] + vib_result['confidence'] + flow_result['confidence']) / 3.0
    
    return {
        "health_score": health_score,
        "risk_score": final_risk,
        "risk_level": risk_level,
        "confidence": confidence,
        "possible_fault": primary_issue,
        "agent_findings": agents_results
    }
