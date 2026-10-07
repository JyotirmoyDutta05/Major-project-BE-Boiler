import json
import os
import requests

API_URL = "https://api.openai.com/v1/chat/completions" # Or your local Micro-LM URL

def get_explanation(orchestrator_results, deterministic_findings, use_real_llm=False):
    """
    Translates orchestrator results into a human-readable engineering explanation.
    """
    api_key = os.environ.get("LLM_API_KEY")
    if use_real_llm and api_key:
        return _call_real_llm_api(orchestrator_results, deterministic_findings, api_key)
    else:
        return _simulate_explanation(orchestrator_results, deterministic_findings)

def _call_real_llm_api(orchestrator_results, deterministic_findings, api_key):
    prompt = f"""
    You are an expert boiler diagnostics system. Explain the following system state in engineering language.
    Do NOT invent numerical data or facts not provided here. Do NOT override the deterministic findings.
    
    System State:
    Health: {orchestrator_results['health_score']}/100
    Risk: {orchestrator_results['risk_level']}
    Possible Fault: {orchestrator_results['possible_fault']}
    Findings: {deterministic_findings}
    """
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo", # Changed to gpt-3.5-turbo as it is available on all keys
        "messages": [
            {"role": "system", "content": "You are a thermal power plant decision-support AI."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        # If API fails (e.g. 404 model not found, invalid key, no internet), seamlessly fail over
        return _simulate_explanation(orchestrator_results, deterministic_findings)

def _simulate_explanation(orchestrator_results, deterministic_findings):
    
    fault = orchestrator_results['possible_fault']
    risk = orchestrator_results['risk_level']
    health = orchestrator_results['health_score']
    
    explanation = f"**System Assessment**: The boiler health score is {health}/100, indicating a {risk} risk level.\n\n"
    
    if fault == "NORMAL":
        explanation += "The system is operating within normal parameters. No significant anomalies detected."
    elif fault == "EARLY_DEGRADATION":
        explanation += "The system has detected an early degradation pattern. "
        if deterministic_findings:
            explanation += f"This is evidenced by: {', '.join(deterministic_findings)}. "
        explanation += "These subtle deviations suggest potential wear or inefficiency before traditional alarms trigger."
    elif fault == "CRITICAL":
        explanation += "A severe anomaly has been detected. "
        if deterministic_findings:
            explanation += f"Key indicators include: {', '.join(deterministic_findings)}. "
        explanation += "This represents a significant deviation from normal operating conditions."
    elif fault == "SENSOR_FAULT":
        explanation += "The system has identified a potential sensor reliability issue rather than an equipment failure. "
        if deterministic_findings:
            explanation += f"Specifically: {', '.join(deterministic_findings)}. "
            
    explanation += "\n\n**Confidence**: {:.1f}%".format(orchestrator_results['confidence'] * 100)
    
    return explanation
