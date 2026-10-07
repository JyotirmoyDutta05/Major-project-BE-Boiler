import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ============================================================
# SYNTHETIC BOILER DATA GENERATOR
# ============================================================
# This module generates synthetic time-series boiler sensor data.
# The data is NOT real BHEL data. It is engineered to behave
# plausibly according to established boiler engineering principles.
#
# KEY RELATIONSHIPS (all derived from boiler_load_percent as base):
#   furnace_temperature  = load * 10 + 200   (normal ~1050°C)
#   steam_pressure       = load * 1.5 + 20   (normal ~147 bar)
#   steam_temperature    = furnace * 0.4 + 50 (normal ~470°C)
#   steam_flow           = load * 5 + 10     (normal ~435 t/h)
#   feedwater_flow       = steam_flow * 1.05  (slightly more than steam)
#   fuel_flow            = load * 2 + 1      (normal ~171 kg/s)
#   oxygen_percent       = ~3.5%             (independent)
#   co2_percent          = 21 - O2           (~17.5%)
#   vibration            = ~2.5 mm/s         (independent)
#   fan_speed            = load * 10 + 50    (normal ~900 RPM)
#   pump_pressure        = load * 2 + 10     (normal ~180 bar)
#   valve_position       = load * 0.8 + 5    (normal ~73%)
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def generate_scenario_data(scenario, num_samples=1000, save_csv=True):
    """
    Generate synthetic boiler sensor data for a given scenario.
    
    Scenarios:
      NORMAL            - Stable operation, small realistic fluctuations.
      EARLY_DEGRADATION - Gradual temperature rise, vibration increase, steam flow decline.
      CRITICAL          - Significant anomalies across multiple sensors.
      SENSOR_FAULT      - Stuck sensor, NaN values simulating sensor failures.
    """
    np.random.seed(42)
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_samples)]
    
    # ---- BASE NORMAL VALUES (correlated to boiler load) ----
    boiler_load_percent = np.random.normal(85, 2, num_samples)
    furnace_temperature = boiler_load_percent * 10 + np.random.normal(200, 10, num_samples)
    steam_pressure = boiler_load_percent * 1.5 + np.random.normal(20, 2, num_samples)
    steam_temperature = furnace_temperature * 0.4 + np.random.normal(50, 5, num_samples)
    steam_flow = boiler_load_percent * 5 + np.random.normal(10, 5, num_samples)
    feedwater_flow = steam_flow * 1.05 + np.random.normal(2, 1, num_samples)
    fuel_flow = boiler_load_percent * 2 + np.random.normal(1, 0.5, num_samples)
    oxygen_percent = np.random.normal(3.5, 0.2, num_samples)
    co2_percent = 21 - oxygen_percent + np.random.normal(0, 0.1, num_samples)
    vibration = np.random.normal(2.5, 0.3, num_samples)
    fan_speed = boiler_load_percent * 10 + np.random.normal(50, 5, num_samples)
    pump_pressure = boiler_load_percent * 2 + np.random.normal(10, 1, num_samples)
    valve_position = boiler_load_percent * 0.8 + np.random.normal(5, 2, num_samples)
    
    anomaly_label = np.zeros(num_samples)
    fault_type = ['NORMAL'] * num_samples
    severity = np.zeros(num_samples)
    
    # ---- INJECT SCENARIO-SPECIFIC ANOMALIES ----
    if scenario == 'EARLY_DEGRADATION':
        # Gradual degradation from halfway point — subtle, correlated changes.
        mid_point = num_samples // 2
        degradation_factor = np.linspace(0, 1, num_samples - mid_point)
        
        furnace_temperature[mid_point:] += degradation_factor * 80   # +80°C max
        vibration[mid_point:]           += degradation_factor * 2.0  # +2 mm/s max
        steam_flow[mid_point:]          -= degradation_factor * 50   # -50 t/h max
        
        anomaly_label[mid_point:] = 1
        for i in range(mid_point, num_samples):
            fault_type[i] = 'EARLY_DEGRADATION'
            severity[i] = degradation_factor[i - mid_point] * 50
            
    elif scenario == 'CRITICAL':
        mid_point = num_samples // 2
        
        furnace_temperature[mid_point:] += np.random.normal(200, 20, num_samples - mid_point)
        vibration[mid_point:]           += np.random.normal(5.0, 1.0, num_samples - mid_point)
        steam_pressure[mid_point:]      += np.random.normal(40, 10, num_samples - mid_point)
        steam_flow[mid_point:]          -= np.random.normal(150, 20, num_samples - mid_point)
        
        anomaly_label[mid_point:] = 1
        for i in range(mid_point, num_samples):
            fault_type[i] = 'CRITICAL'
            severity[i] = 100
            
    elif scenario == 'SENSOR_FAULT':
        mid_point = num_samples // 2
        # Noisy sensor: extra noise on pressure in the second half
        steam_pressure[mid_point:] += np.random.normal(0, 6, num_samples - mid_point)
        # Sudden unrealistic spike (single impossible reading)
        furnace_temperature[mid_point + 100] = 2500.0
        # Stuck temperature sensor: constant value through the end of the series
        stuck_start = int(num_samples * 0.8)
        furnace_temperature[stuck_start:] = furnace_temperature[stuck_start - 1]
        # Vibration sensor drops out (NaN) for the final readings
        nan_start = int(num_samples * 0.9)
        vibration[nan_start:] = np.nan
        
        anomaly_label[mid_point:] = 1
        for i in range(mid_point, num_samples):
            fault_type[i] = 'SENSOR_FAULT'
            severity[i] = 75

    df = pd.DataFrame({
        'timestamp': timestamps,
        'boiler_load_percent': boiler_load_percent,
        'furnace_temperature': furnace_temperature,
        'steam_pressure': steam_pressure,
        'steam_temperature': steam_temperature,
        'steam_flow': steam_flow,
        'feedwater_flow': feedwater_flow,
        'fuel_flow': fuel_flow,
        'oxygen_percent': oxygen_percent,
        'co2_percent': co2_percent,
        'vibration': vibration,
        'fan_speed': fan_speed,
        'pump_pressure': pump_pressure,
        'valve_position': valve_position,
        'anomaly_label': anomaly_label,
        'fault_type': fault_type,
        'severity': severity
    })
    
    # Save to data/ directory
    if save_csv:
        os.makedirs(DATA_DIR, exist_ok=True)
        csv_path = os.path.join(DATA_DIR, f'synthetic_boiler_data_{scenario.lower()}.csv')
        df.to_csv(csv_path, index=False)
    
    return df


def generate_live_point(scenario, step, history_df=None):
    """
    Generate a SINGLE new data point for live/streaming simulation.
    This is called repeatedly by the Streamlit live-mode to simulate
    a continuously arriving sensor feed.
    
    Args:
        scenario: The operating scenario (NORMAL, EARLY_DEGRADATION, etc.)
        step: Current time step index (used to compute degradation progression)
        history_df: Previous DataFrame to append to (optional)
    
    Returns:
        A single-row DataFrame representing the new sensor reading.
    """
    np.random.seed(step)  # Reproducible but varying per step
    if history_df is not None and len(history_df) > 0:
        now = history_df['timestamp'].iloc[-1] + timedelta(minutes=1)
    else:
        now = datetime(2026, 1, 1, 0, 0, 0)
    
    load = np.random.normal(85, 2)
    temp = load * 10 + np.random.normal(200, 10)
    pressure = load * 1.5 + np.random.normal(20, 2)
    steam_temp = temp * 0.4 + np.random.normal(50, 5)
    flow = load * 5 + np.random.normal(10, 5)
    feed = flow * 1.05 + np.random.normal(2, 1)
    fuel = load * 2 + np.random.normal(1, 0.5)
    o2 = np.random.normal(3.5, 0.2)
    co2 = 21 - o2 + np.random.normal(0, 0.1)
    vib = np.random.normal(2.5, 0.3)
    fan = load * 10 + np.random.normal(50, 5)
    pump = load * 2 + np.random.normal(10, 1)
    valve = load * 0.8 + np.random.normal(5, 2)
    
    anomaly = 0
    fault = 'NORMAL'
    sev = 0.0
    
    # Inject anomaly based on scenario and step
    degradation_start = 30  # After 30 live steps, begin anomaly
    
    if scenario == 'EARLY_DEGRADATION' and step > degradation_start:
        progress = min(1.0, (step - degradation_start) / 100.0)
        temp += progress * 80
        vib += progress * 2.0
        flow -= progress * 50
        anomaly = 1
        fault = 'EARLY_DEGRADATION'
        sev = progress * 50
        
    elif scenario == 'CRITICAL' and step > degradation_start:
        temp += np.random.normal(200, 20)
        vib += np.random.normal(5.0, 1.0)
        pressure += np.random.normal(40, 10)
        flow -= np.random.normal(150, 20)
        anomaly = 1
        fault = 'CRITICAL'
        sev = 100
        
    elif scenario == 'SENSOR_FAULT' and step > degradation_start:
        # Temperature sensor freezes at its last reported value
        if history_df is not None and len(history_df) > 0:
            temp = float(history_df['furnace_temperature'].iloc[-1])
        # Vibration sensor intermittently drops out
        if step % 4 == 0:
            vib = np.nan
        anomaly = 1
        fault = 'SENSOR_FAULT'
        sev = 75
    
    new_row = pd.DataFrame([{
        'timestamp': now,
        'boiler_load_percent': load,
        'furnace_temperature': temp,
        'steam_pressure': pressure,
        'steam_temperature': steam_temp,
        'steam_flow': flow,
        'feedwater_flow': feed,
        'fuel_flow': fuel,
        'oxygen_percent': o2,
        'co2_percent': co2,
        'vibration': vib,
        'fan_speed': fan,
        'pump_pressure': pump,
        'valve_position': valve,
        'anomaly_label': anomaly,
        'fault_type': fault,
        'severity': sev
    }])
    
    return new_row


if __name__ == "__main__":
    for s in ['NORMAL', 'EARLY_DEGRADATION', 'CRITICAL', 'SENSOR_FAULT']:
        df = generate_scenario_data(s)
        print(f"{s} data shape: {df.shape}, saved to data/")
