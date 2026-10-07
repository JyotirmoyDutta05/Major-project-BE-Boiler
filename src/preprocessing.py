import pandas as pd
import numpy as np

def preprocess_data(df):
    df_proc = df.copy()
    
    # Missing value handling (fill with last valid observation, but keep track of nans for sensor fault logic)
    df_proc['vibration_is_nan'] = df_proc['vibration'].isna()
    df_proc.ffill(inplace=True)
    df_proc.bfill(inplace=True)
    
    # Calculate deltas (rate of change)
    sensors = ['furnace_temperature', 'steam_pressure', 'steam_flow', 'vibration']
    for sensor in sensors:
        df_proc[f'{sensor}_delta'] = df_proc[sensor].diff().fillna(0)
        
    # Rolling statistics
    window_size = 10
    for sensor in sensors:
        df_proc[f'{sensor}_rolling_mean'] = df_proc[sensor].rolling(window=window_size).mean().fillna(df_proc[sensor])
        df_proc[f'{sensor}_rolling_std'] = df_proc[sensor].rolling(window=window_size).std().fillna(0)
        
    # Load normalized features
    df_proc['load_normalized_temperature'] = df_proc['furnace_temperature'] / df_proc['boiler_load_percent']
    df_proc['load_normalized_steam_flow'] = df_proc['steam_flow'] / df_proc['boiler_load_percent']
    
    # Load-corrected residual: deviation of temperature from what the current load predicts.
    # Removes the normal load-driven swing so instability checks only see abnormal variation.
    expected_temp = df_proc['boiler_load_percent'] * 10 + 200
    df_proc['furnace_temperature_residual'] = df_proc['furnace_temperature'] - expected_temp
    df_proc['furnace_temperature_residual_rolling_std'] = (
        df_proc['furnace_temperature_residual'].rolling(window=window_size).std().fillna(0)
    )
    
    return df_proc
