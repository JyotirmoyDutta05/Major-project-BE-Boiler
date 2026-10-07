# BHEL-Optimize

**BHEL-Optimize: TensorFlow Agents for Autonomous Boiler Lifecycle Monitoring in Thermal Power Plants**

## Important Data Policy
**This prototype uses synthetic boiler sensor data. BHEL documents were used purely for domain reference.**
This system is an AI-based boiler health monitoring and decision-support prototype. It does NOT replace DCS/SCADA systems and it does NOT contain proprietary BHEL plant sensor data.

## Architecture
- **Synthetic Data Engine**: Generates time series data.
- **Data Preprocessing**: Features engineering, moving averages, handling missing data.
- **Deterministic Engineering Engine**: Rule-based detection.
- **ML Model**: Isolation Forest for anomaly detection.
- **Specialized Agents**: Temperature, Vibration, Flow, Sensor agents.
- **Orchestrator**: Fuses evidence to compute health and risk.
- **Recommendation & API Layer**: Generates human-readable explanations based on orchestrator results.

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```
