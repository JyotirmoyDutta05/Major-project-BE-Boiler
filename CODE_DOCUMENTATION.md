# BHEL-Optimize — Complete Code Documentation

> **Prototype using synthetic boiler sensor data; BHEL documents used for domain reference.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [How the Data is Simulated](#2-how-the-data-is-simulated)
3. [File-by-File Explanation](#3-file-by-file-explanation)
4. [Data Flow Pipeline](#4-data-flow-pipeline)
5. [Scenario Descriptions](#5-scenario-descriptions)
6. [Threshold Reference Table](#6-threshold-reference-table)

---

## 1. Project Overview

BHEL-Optimize is an AI-based boiler health monitoring and decision-support prototype. It demonstrates how machine learning and rule-based engineering logic can work together to detect early boiler degradation — even before traditional threshold alarms trigger.

**It does NOT replace existing DCS/SCADA/RMDS systems.** It is an intelligent orchestration layer that sits on top of existing monitoring infrastructure.

### Architecture Flow

```
Synthetic Data Engine  →  Preprocessing Pipeline  →  Deterministic Engine + ML Model
                                                              ↓
                                                      Specialized Agents
                                                              ↓
                                                        Orchestrator
                                                              ↓
                                                      Evidence Fusion
                                                              ↓
                                                  Health Score + Risk Level
                                                              ↓
                                                 Maintenance Recommendation
                                                              ↓
                                                    API/LLM Explanation
                                                              ↓
                                                       Streamlit UI
```

---

## 2. How the Data is Simulated

### Why Synthetic Data?

Real BHEL boiler sensor data is proprietary and confidential. We do NOT have access to it. Instead, we generate synthetic data that is **engineered to behave plausibly** according to established boiler physics and the relationships described in BHEL documentation.

### How the Synthetic Data Engine Works

The file `src/data_generator.py` creates time-series data for **14 sensors** plus **3 label columns** (anomaly_label, fault_type, severity).

#### Base Relationships (NORMAL scenario)

All sensors are **correlated to boiler load** — just like a real boiler:

| Sensor | Formula | Normal Value |
|--------|---------|-------------|
| `boiler_load_percent` | Random ~85% ± 2% | ~85% |
| `furnace_temperature` | load × 10 + 200 + noise(σ=10) | ~1050 °C (total spread ≈ ±22 because load also varies) |
| `steam_pressure` | load × 1.5 + 20 + noise(σ=2) | ~147 bar |
| `steam_temperature` | furnace × 0.4 + 50 + noise(σ=5) | ~470 °C |
| `steam_flow` | load × 5 + 10 + noise(σ=5) | ~435 t/h |
| `feedwater_flow` | steam_flow × 1.05 + noise(σ=1) | ~459 t/h |
| `fuel_flow` | load × 2 + 1 + noise(σ=0.5) | ~171 kg/s |
| `oxygen_percent` | Random ~3.5% ± 0.2% | ~3.5% |
| `co2_percent` | 21 - O₂ + noise(σ=0.1) | ~17.5% |
| `vibration` | Random ~2.5 ± 0.3 mm/s | ~2.5 mm/s |
| `fan_speed` | load × 10 + 50 + noise(σ=5) | ~900 RPM |
| `pump_pressure` | load × 2 + 10 + noise(σ=1) | ~180 bar |
| `valve_position` | load × 0.8 + 5 + noise(σ=2) | ~73% |

#### Why These Relationships?

- **Higher boiler load → higher temperature, pressure, flow** — this is how real boilers work.
- **Feedwater flow slightly exceeds steam flow** — compensating for blowdown losses.
- **O₂ and CO₂ are inversely related** — combustion chemistry.
- **Vibration is independent** — mechanical vibration doesn't directly scale with load in normal conditions.

#### How Anomalies Are Injected

For non-NORMAL scenarios, the engine modifies the second half of the dataset (from point 500 onwards in a 1000-point series):

**EARLY_DEGRADATION:**
```python
furnace_temperature += gradual_ramp * 80    # Up to +80°C
vibration           += gradual_ramp * 2.0   # Up to +2 mm/s
steam_flow          -= gradual_ramp * 50    # Down by -50 t/h
```
The ramp is `np.linspace(0, 1, 500)` — a smooth, gradual increase. The values initially stay WITHIN normal alarm thresholds. This is the key demo: the AI catches it before simple threshold alarms would.

**CRITICAL:**
```python
furnace_temperature += random(200, σ=20)    # Sudden +200°C
vibration           += random(5.0, σ=1.0)   # Sudden +5 mm/s
steam_pressure      += random(40, σ=10)     # Sudden +40 bar
steam_flow          -= random(150, σ=20)    # Sudden -150 t/h
```

**SENSOR_FAULT:**
```python
steam_pressure[500:]       += noise(σ=6)    # Noisy pressure sensor
furnace_temperature[600]    = 2500        # One impossible spike
furnace_temperature[800:]   = constant    # Stuck sensor until the end
vibration[900:]             = NaN         # Vibration sensor drops out
```
The faults are placed at the **end** of the series on purpose: the analysis looks at the most recent readings (like a live monitor would), so the faults have to be present "now".

### Where the Data is Stored

When data is generated, it is automatically saved to the `data/` directory:
```
data/synthetic_boiler_data_normal.csv
data/synthetic_boiler_data_early_degradation.csv
data/synthetic_boiler_data_critical.csv
data/synthetic_boiler_data_sensor_fault.csv
```

These CSV files can be opened in Excel, pandas, or any data tool.

### Live Streaming Mode

The function `generate_live_point()` creates ONE data point at a time, simulating a real-time sensor feed. It uses the same physics relationships. The app adds a new point every 2 seconds and re-runs the whole pipeline (preprocessing → rules → ML → agents → orchestrator) on the latest 200 points.

- Steps 0–30: always normal (warm-up).
- After step 30 the selected scenario's fault is injected.
- EARLY_DEGRADATION ramps up slowly over 100 steps, so it is typically flagged around step ~80 — while furnace temperature is still below the 1100 °C hard rule. This shows pattern detection beating a simple threshold.
- CRITICAL and SENSOR_FAULT are flagged within a couple of steps of injection.

---

## 3. File-by-File Explanation

### `app.py` — Main Streamlit Application

**Purpose:** The front-end UI that ties everything together.

**Two modes:**
1. **Static Mode** — Generates 1000 data points at once, runs the full pipeline, displays 7 tabs.
2. **Live Streaming Mode** — Generates one point every 2 seconds, updates charts in real time.

**Tabs (Static Mode):**
| Tab | What it shows |
|-----|--------------|
| Executive Overview | Health Score, Risk Level, Confidence, Sensor readings, System status |
| Sensor Monitoring | Interactive time-series graph for any sensor, anomaly regions highlighted |
| AI Detection | ML anomaly score + Deterministic rule results side by side |
| Agent Investigation | Each agent's finding, severity, and confidence with progress bars |
| Orchestrator | Reasoning flow diagram + fusion weights + raw JSON output |
| Maintenance | Health score, evidence, recommended action, AI explanation |
| Model Performance | Model info, features used, current prediction |

**What the Graphs Do:**
- **Static Mode (Sensor Monitoring Tab):** This tab provides an interactive line chart (built with Plotly) that lets you visualize the entire 1000-point time-series for any selected sensor. Crucially, if you select an anomalous scenario, it overlays a **transparent red block ("Anomaly Region")** over the exact timeframe where the synthetic faults were injected. This allows you to visually compare normal operating behavior against the degraded behavior, demonstrating exactly what the AI is analyzing.
- **Live Streaming Mode:** This mode displays four continuous, auto-updating line charts for the most critical metrics: *Furnace Temperature*, *Vibration*, *Steam Flow*, and *Steam Pressure*. These graphs simulate a DCS (Distributed Control System) operator screen, showing how the sensor values fluctuate in real-time as new data points arrive every 2 seconds. When an anomaly is injected (after step 30), you can watch the trends shift live on these graphs while the Health Score reacts.

---

### `src/data_generator.py` — Synthetic Data Engine

**Purpose:** Generates all boiler sensor data.

**Key functions:**
- `generate_scenario_data(scenario, num_samples=1000)` — Generates full dataset, saves CSV.
- `generate_live_point(scenario, step)` — Generates one point for live mode.

**Inputs:** Scenario name (string), number of samples.
**Outputs:** pandas DataFrame with 17 columns.

---

### `src/preprocessing.py` — Data Preprocessing Pipeline

**Purpose:** Transforms raw sensor data into analysis-ready features.

**What it does:**
1. **Flags NaN values** — Creates `vibration_is_nan` column before filling.
2. **Forward-fills missing data** — So rolling calculations work.
3. **Calculates deltas** — `diff()` for rate-of-change on temperature, pressure, flow, vibration.
4. **Computes rolling statistics** — 10-point rolling mean and standard deviation.
5. **Creates load-normalized features** — Temperature and flow divided by load percentage.

**Output columns added:**
```
furnace_temperature_delta, steam_pressure_delta, steam_flow_delta, vibration_delta
furnace_temperature_rolling_mean, furnace_temperature_rolling_std
steam_pressure_rolling_mean, steam_pressure_rolling_std
steam_flow_rolling_mean, steam_flow_rolling_std
vibration_rolling_mean, vibration_rolling_std
load_normalized_temperature, load_normalized_steam_flow
furnace_temperature_residual            (actual temp − temp expected from load)
furnace_temperature_residual_rolling_std (instability after removing load effect)
vibration_is_nan
```

---

### `src/deterministic_engine.py` — Rule-Based Engineering Engine

**Purpose:** Transparent, explainable detection using hard engineering knowledge.

**Uses rolling means** (not single-point deltas) to avoid false alarms from normal noise.

**Rules:**
| Rule Code | Condition | Meaning |
|-----------|-----------|---------|
| `TEMP_RISING` | Rolling mean temp > 1100°C | Temperature abnormally high |
| `TEMP_HIGH` | Rolling mean temp > 1200°C | Temperature significantly above range |
| `TEMP_UNSTABLE` | Rolling std of load-corrected temperature residual > 25 | Temperature unstable relative to load |
| `VIBRATION_HIGH` | Rolling mean vib > 3.5 mm/s | Elevated vibration |
| `VIBRATION_CRITICAL` | Rolling mean vib > 5.0 mm/s | Critical vibration |
| `STEAM_FLOW_DECLINING` | Rolling mean flow < 400 t/h | Unexpected flow reduction |
| `PRESSURE_HIGH` | Pressure > 160 bar | Over-pressure |
| `PRESSURE_LOW` | Pressure < 100 bar | Under-pressure |
| `SENSOR_FAULT_VIBRATION` | vibration_is_nan = True | Missing sensor data |
| `SENSOR_STUCK_TEMP` | Std of last 10 temps < 0.01 | Stuck sensor |

**Output:**
```json
{
  "rules_triggered": ["TEMP_RISING", "VIBRATION_HIGH"],
  "engineering_findings": ["Temperature trend is abnormally high", "Vibration level is high"],
  "rule_risk_score": 0.40
}
```

---

### `src/anomaly_detector.py` — ML Anomaly Detection

**Purpose:** Uses Isolation Forest to detect multivariate anomalies.

**How it works:**
1. **Training:** Fits on NORMAL data only (learns what "healthy" looks like).
2. **Prediction:** Scores the latest 10 readings and averages them, so one noisy reading cannot flip the result. The raw Isolation Forest decision value is mapped to 0–1 with a smooth sigmoid.
   - Score near 0 = normal (NORMAL scenario scores ~0.02)
   - Score above 0.5 = anomalous
3. **Features:** rolling means of `furnace_temperature`, `steam_pressure`, `steam_flow`, `vibration`, plus `furnace_temperature_residual`. Rolling means are used instead of single-point deltas because deltas are dominated by sensor noise.

**Why Isolation Forest?**
- It's unsupervised — no need for labeled anomaly data.
- It's lightweight — runs in milliseconds.
- It works well for detecting deviations from a known "normal" baseline.

**Saved model:** `models/anomaly_detector.pkl`

---

### `src/agents/temperature_agent.py`

**Purpose:** Specialized agent analyzing furnace/steam temperature.

**Logic:** Checks rolling mean temperature:
- > 1100°C → WARNING (score 0.6)
- > 1200°C → CRITICAL (score 0.9)
- Otherwise → NORMAL (score 0.0)

---

### `src/agents/vibration_agent.py`

**Purpose:** Specialized agent analyzing vibration levels.

**Logic:** Checks rolling mean vibration:
- > 3.5 mm/s → WARNING (score 0.6)
- > 5.0 mm/s → CRITICAL (score 0.9)
- Otherwise → NORMAL (score 0.0)

---

### `src/agents/flow_agent.py`

**Purpose:** Specialized agent analyzing steam flow.

**Logic:** Checks rolling mean steam flow:
- < 400 t/h → WARNING (score 0.5)
- < 350 t/h → CRITICAL (score 0.85)
- Otherwise → NORMAL (score 0.0)

---

### `src/agents/sensor_agent.py`

**Purpose:** Checks sensor reliability — detects if sensors themselves are malfunctioning.

**Logic:**
- If vibration reported NaN → WARNING (score 0.8)
- If temperature std over last 10 readings < 0.01 → CRITICAL (score 0.9, stuck sensor)
- Otherwise → NORMAL (score 0.0)

---

### `src/orchestrator.py` — Central Orchestrator

**Purpose:** Collects results from ALL components and fuses them into a final assessment.

**Evidence Fusion Formula:**
```
final_risk = 0.30 × deterministic_score
           + 0.30 × ML_anomaly_score
           + 0.15 × temperature_agent_score
           + 0.15 × vibration_agent_score
           + 0.10 × flow_agent_score
```

**Special case:** If sensor_agent detects a sensor fault (score > 0.5), it overrides the general risk — because a sensor fault means we can't trust the other readings.

**Health Score:** `100 - (risk × 100)`, clamped to 0–100.

**Risk Level Mapping:**
| Health Score | Risk Level |
|-------------|-----------|
| 75–100 | NORMAL |
| 50–74 | WARNING |
| 25–49 | HIGH |
| 0–24 | CRITICAL |

---

### `src/recommendation_engine.py` — Maintenance Recommendations

**Purpose:** Maps risk levels to safe, actionable recommendations.

| Condition | Recommendation |
|-----------|---------------|
| SENSOR_FAULT | Verify sensor health/calibration before concluding equipment degradation. |
| NORMAL | Continue normal monitoring. |
| EARLY_DEGRADATION / WARNING | Increase monitoring frequency and schedule engineering inspection during the next suitable maintenance window. |
| HIGH / CRITICAL | Escalate for immediate engineering assessment according to plant safety procedures. |

---

### `src/api_client.py` — LLM Explanation Layer

**Purpose:** Translates the orchestrator's structured JSON results into a natural-language explanation.

**Two modes:**
1. **Real LLM** — Sends a structured prompt (NOT raw sensor data) to the OpenAI API. The prompt explicitly instructs the LLM: do not invent data, do not override findings.
2. **Simulated** — If no API key is available, generates a template-based explanation locally.

**The API key is read from the `.env` file** — never hardcoded or displayed in the UI.

---

## 4. Data Flow Pipeline

```
Step 1: data_generator.py
    Generates 1000 timestamped rows of 14 correlated sensor values.
    Injects scenario-specific anomalies.
    Saves CSV to data/ directory.
                ↓
Step 2: preprocessing.py
    Fills NaN values (forward-fill).
    Calculates deltas (rate of change).
    Computes 10-point rolling mean and standard deviation.
    Creates load-normalized features.
                ↓
Step 3a: deterministic_engine.py
    Applies engineering rules using rolling statistics.
    Produces list of triggered rules + risk score.
                ↓
Step 3b: anomaly_detector.py (runs in parallel with 3a)
    Isolation Forest trained on NORMAL data.
    Scores the latest data point as normal/anomalous.
                ↓
Step 4: agents/ (temperature, vibration, flow, sensor)
    Each agent independently analyzes its domain.
    Produces finding, severity, confidence, score.
                ↓
Step 5: orchestrator.py
    Collects deterministic + ML + agent scores.
    Applies weighted evidence fusion.
    Computes Health Score and Risk Level.
                ↓
Step 6: recommendation_engine.py
    Maps risk level to safe maintenance recommendation.
                ↓
Step 7: api_client.py
    Sends structured summary to LLM for natural-language explanation.
    LLM explains — it does NOT calculate or diagnose independently.
                ↓
Step 8: app.py (Streamlit UI)
    Displays everything across 7 tabs or live dashboard.
```

---

## 5. Scenario Descriptions

### NORMAL
- All sensors within normal ranges.
- Small random fluctuations (realistic measurement noise).
- No rules triggered, ML says normal, all agents report NORMAL.
- Health Score: ~99 (verified).

### EARLY_DEGRADATION
- First half: completely normal.
- Second half: gradual ramp — temperature slowly rises, vibration slowly increases, steam flow slowly decreases.
- The changes are **correlated** (they happen together, like real wear).
- Values initially stay **within** traditional alarm thresholds.
- The AI detects the **pattern** before a simple threshold alarm would.
- Health Score drops from ~100 to ~40-60.

### CRITICAL
- First half: normal.
- Second half: sudden, large deviations across multiple sensors.
- Temperature jumps +200°C, vibration jumps +5 mm/s, flow drops -150 t/h.
- Every rule triggers, ML flags anomaly, all agents report WARNING/CRITICAL.
- Health Score: ~5 (verified).

### SENSOR_FAULT
- First half: normal.
- Second half: temperature sensor stuck (constant value for 100 points), then vibration sensor goes NaN (missing for 100 points).
- The sensor agent catches this and the system flags it as a sensor issue — NOT equipment failure.
- Health Score: ~10 (verified; low because sensor trust is lost, recommendation is to verify sensors — not to assume equipment damage).

---

## 6. Threshold Reference Table

| Parameter | Normal Range | WARNING Threshold | CRITICAL Threshold |
|-----------|-------------|-------------------|-------------------|
| Furnace Temperature | ~1050 ± 22 °C | > 1100 °C (rolling) | > 1200 °C (rolling) |
| Vibration | ~2.5 ± 0.3 mm/s | > 3.5 mm/s (rolling) | > 5.0 mm/s (rolling) |
| Steam Flow | ~435 ± 5 t/h | < 400 t/h (rolling) | < 350 t/h (rolling) |
| Steam Pressure | ~147 ± 2 bar | > 160 bar | < 100 bar |
| Temp residual std | ~10 | > 25 (instability) | — |
| Sensor Stuck | Std > 0 | — | Std < 0.01 |

All thresholds are **prototype values** and are NOT BHEL proprietary thresholds.

---

*Document generated for BHEL-Optimize prototype.*
