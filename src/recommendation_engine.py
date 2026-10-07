def get_recommendation(orchestrator_results):
    risk_level = orchestrator_results['risk_level']
    possible_fault = orchestrator_results['possible_fault']
    
    if possible_fault == "SENSOR_FAULT":
        return "Verify sensor health/calibration before concluding equipment degradation."
        
    if risk_level == "NORMAL":
        return "Continue normal monitoring."
    elif risk_level == "WARNING" or possible_fault == "EARLY_DEGRADATION":
        return "Increase monitoring frequency and schedule engineering inspection during the next suitable maintenance window."
    elif risk_level == "HIGH" or risk_level == "CRITICAL":
        return "Escalate for immediate engineering assessment according to plant safety procedures."
        
    return "Continue monitoring."
