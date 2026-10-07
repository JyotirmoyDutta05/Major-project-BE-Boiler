# Data Dictionary — Synthetic Boiler Sensor Data

All numerical data in this project is **synthetically generated**. It is NOT real BHEL plant data.

## Columns

| Column | Type | Unit | Normal Range | Description |
|--------|------|------|-------------|-------------|
| `timestamp` | datetime | — | — | Simulated timestamp, 1-minute intervals |
| `boiler_load_percent` | float | % | 81–89 | Boiler load as percentage of capacity |
| `furnace_temperature` | float | °C | 1030–1070 | Furnace combustion temperature |
| `steam_pressure` | float | bar | 143–151 | Main steam pressure |
| `steam_temperature` | float | °C | 460–480 | Main steam temperature |
| `steam_flow` | float | t/h | 425–445 | Main steam flow rate |
| `feedwater_flow` | float | t/h | 448–470 | Boiler feedwater flow rate |
| `fuel_flow` | float | kg/s | 169–173 | Fuel input flow rate |
| `oxygen_percent` | float | % | 3.1–3.9 | Flue gas oxygen content |
| `co2_percent` | float | % | 17.1–17.9 | Flue gas CO₂ content |
| `vibration` | float | mm/s | 1.9–3.1 | Bearing/structural vibration |
| `fan_speed` | float | RPM | 885–915 | Induced/forced draft fan speed |
| `pump_pressure` | float | bar | 178–182 | Boiler feed pump discharge pressure |
| `valve_position` | float | % | 69–77 | Control valve opening position |
| `anomaly_label` | int | — | 0 or 1 | 0=normal, 1=anomalous |
| `fault_type` | string | — | — | NORMAL, EARLY_DEGRADATION, CRITICAL, SENSOR_FAULT |
| `severity` | float | — | 0–100 | Severity of injected anomaly |
