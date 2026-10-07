import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator import generate_scenario_data, generate_live_point
from src.preprocessing import preprocess_data
from src.deterministic_engine import run_engineering_rules
from src.anomaly_detector import train_anomaly_detector, predict_anomaly, FEATURES, SCORING_WINDOW
from src.orchestrator import run_orchestrator
from src.recommendation_engine import get_recommendation
from src.api_client import get_explanation

st.set_page_config(page_title="BHEL-Optimize Prototype", layout="wide")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("BHEL-Optimize")
st.sidebar.caption("Prototype using synthetic boiler sensor data; BHEL documents used for domain reference.")

scenario = st.sidebar.selectbox("Select operating scenario:", 
                                ["NORMAL", "EARLY_DEGRADATION", "CRITICAL", "SENSOR_FAULT"])

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Mode")
sim_mode = st.sidebar.radio("Data Mode:", ["Static (Full Dataset)", "Live Streaming"])

if st.sidebar.button("▶ Run BHEL-Optimize Analysis", use_container_width=True):
    st.session_state['run_analysis'] = True
if 'run_analysis' not in st.session_state:
    st.session_state['run_analysis'] = False

# ============================================================
# DATA GENERATION & MODEL
# ============================================================
@st.cache_data
def get_data(scenario_name):
    df = generate_scenario_data(scenario_name, save_csv=True)
    df_proc = preprocess_data(df)
    return df, df_proc

@st.cache_resource
def get_model():
    df_normal = generate_scenario_data("NORMAL", save_csv=False)
    df_proc_normal = preprocess_data(df_normal)
    clf = train_anomaly_detector(df_proc_normal)
    return clf

clf = get_model()

# ============================================================
# LIVE STREAMING MODE
# ============================================================
if sim_mode == "Live Streaming":
    st.title("🔴 BHEL-Optimize — Live Boiler Monitoring")
    st.info("Simulating a continuously arriving sensor feed. New data point every 2 seconds.")
    
    # Initialize live state
    if 'live_step' not in st.session_state or st.session_state.get('live_scenario') != scenario:
        st.session_state['live_step'] = 0
        st.session_state['live_scenario'] = scenario
        # Seed with 20 normal points for rolling stats
        seed_df = generate_scenario_data("NORMAL", num_samples=20, save_csv=False)
        st.session_state['live_df'] = seed_df

    # Layout
    col_health, col_risk, col_conf, col_step = st.columns(4)
    health_placeholder = col_health.empty()
    risk_placeholder = col_risk.empty()
    conf_placeholder = col_conf.empty()
    step_placeholder = col_step.empty()
    
    st.markdown("---")
    st.subheader("Live Sensor Monitoring")
    live_sensor_col = st.selectbox("Select Sensor to View (Live):", 
                              ['furnace_temperature', 'steam_pressure', 'steam_flow', 
                               'vibration', 'fuel_flow', 'oxygen_percent', 
                               'feedwater_flow', 'co2_percent', 'boiler_load_percent'])
    
    live_chart = st.empty()
    
    findings_placeholder = st.empty()
    agents_placeholder = st.empty()
    
    # Stream loop
    for i in range(300):
        step = st.session_state['live_step']
        new_point = generate_live_point(scenario, step, history_df=st.session_state['live_df'])
        st.session_state['live_df'] = pd.concat([st.session_state['live_df'], new_point], ignore_index=True)
        
        # Keep last 200 points for performance
        if len(st.session_state['live_df']) > 200:
            st.session_state['live_df'] = st.session_state['live_df'].iloc[-200:]
        
        live_df = st.session_state['live_df']
        live_proc = preprocess_data(live_df)
        
        eng_results = run_engineering_rules(live_proc)
        ml_results = predict_anomaly(live_proc)
        orch_results = run_orchestrator(live_proc, eng_results, ml_results)
        
        # Update metrics
        health_placeholder.metric("Health Score", f"{orch_results['health_score']}/100")
        risk_placeholder.metric("Risk Level", orch_results['risk_level'])
        conf_placeholder.metric("Confidence", f"{orch_results['confidence']*100:.1f}%")
        step_placeholder.metric("Live Step", f"{step}")
        
        # Update chart
        fig = px.line(live_df, x='timestamp', y=live_sensor_col, 
                      title=f"🔴 Live {live_sensor_col.replace('_', ' ').title()}")
        fig.update_layout(xaxis_title="Time", yaxis_title=live_sensor_col.replace('_', ' ').title())
        live_chart.plotly_chart(fig, use_container_width=True, key=f"live_{step}")
        
        # Update findings
        if eng_results['rules_triggered']:
            findings_placeholder.error(f"⚠ Rules Triggered: {', '.join(eng_results['rules_triggered'])}")
        else:
            findings_placeholder.success("✅ All systems normal — no engineering rules triggered.")
        
        # Update agent summary
        agent_text = ""
        for a in orch_results['agent_findings']:
            icon = "🟢" if a['severity'] == "NORMAL" else ("🟡" if a['severity'] == "WARNING" else "🔴")
            agent_text += f"{icon} **{a['agent'].replace('_',' ').title()}**: {a['finding']} (confidence: {a['confidence']*100:.0f}%)  \n"
        agents_placeholder.markdown(agent_text)
        
        st.session_state['live_step'] += 1
        time.sleep(2)

# ============================================================
# STATIC MODE (original full-dataset analysis)
# ============================================================
else:
    df, df_proc = get_data(scenario)
    
    # Run the full pipeline
    eng_results = run_engineering_rules(df_proc)
    ml_results = predict_anomaly(df_proc)
    orch_results = run_orchestrator(df_proc, eng_results, ml_results)
    recommendation = get_recommendation(orch_results)
    explanation = get_explanation(orch_results, eng_results['engineering_findings'], use_real_llm=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Executive Overview", "📈 Sensor Monitoring", "🤖 AI Detection", 
        "🔍 Agent Investigation", "🧠 Orchestrator", "🛠 Maintenance", "📉 Model Performance"
    ])
    
    # ---- TAB 1: EXECUTIVE OVERVIEW ----
    with tab1:
        st.title("BHEL-Optimize — Executive Overview")
        st.warning("AI-based boiler health monitoring and decision-support prototype. Does not replace DCS/SCADA/RMDS systems.")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Boiler Health Score", f"{orch_results['health_score']}/100")
        col2.metric("Risk Level", orch_results['risk_level'])
        col3.metric("Current Scenario", scenario)
        col4.metric("Confidence", f"{orch_results['confidence']*100:.1f}%")
        
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Active Sensor Summary")
            latest = df_proc.iloc[-1]
            st.write(f"🌡️ Furnace Temperature: **{latest['furnace_temperature']:.1f} °C**")
            st.write(f"🔵 Steam Pressure: **{latest['steam_pressure']:.1f} bar**")
            st.write(f"💨 Steam Flow: **{latest['steam_flow']:.1f} t/h**")
            st.write(f"📳 Vibration: **{latest['vibration']:.2f} mm/s**")
            st.write(f"⚡ Boiler Load: **{latest['boiler_load_percent']:.1f}%**")
            st.write(f"🔥 Fuel Flow: **{latest['fuel_flow']:.1f} kg/s**")
            st.write(f"💧 Feedwater Flow: **{latest['feedwater_flow']:.1f} t/h**")
            st.write(f"🌬️ O₂: **{latest['oxygen_percent']:.2f}%** | CO₂: **{latest['co2_percent']:.2f}%**")
        
        with col_right:
            st.subheader("System Status")
            if orch_results['possible_fault'] == "NORMAL":
                st.success("✅ System operating normally.")
            elif orch_results['possible_fault'] == "EARLY_DEGRADATION":
                st.warning("⚠️ Early degradation pattern detected.")
            elif orch_results['possible_fault'] == "CRITICAL":
                st.error("🚨 Critical anomaly detected!")
            elif orch_results['possible_fault'] == "SENSOR_FAULT":
                st.info("🔧 Sensor reliability issue detected.")
            
            if eng_results['engineering_findings']:
                st.markdown("**Engineering Findings:**")
                for finding in eng_results['engineering_findings']:
                    st.write(f"- {finding}")
            else:
                st.write("No engineering rules triggered.")
    
    # ---- TAB 2: SENSOR MONITORING ----
    with tab2:
        st.header("Sensor Monitoring")
        sensor_col = st.selectbox("Select Sensor to View:", 
                                  ['furnace_temperature', 'steam_pressure', 'steam_flow', 
                                   'vibration', 'fuel_flow', 'oxygen_percent', 
                                   'feedwater_flow', 'co2_percent', 'boiler_load_percent'])
        
        fig = px.line(df, x='timestamp', y=sensor_col, 
                      title=f"{sensor_col.replace('_', ' ').title()} over Time")
        fig.update_layout(xaxis_title="Time", yaxis_title=sensor_col.replace('_', ' ').title())
        
        # Highlight anomaly region
        if scenario != "NORMAL":
            anomaly_points = df[df['anomaly_label'] == 1]
            if not anomaly_points.empty:
                fig.add_vrect(x0=anomaly_points['timestamp'].iloc[0], 
                              x1=anomaly_points['timestamp'].iloc[-1],
                              fillcolor="red", opacity=0.15, line_width=0,
                              annotation_text="Anomaly Region", annotation_position="top left")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Displaying {len(df)} data points. Data saved at: `data/synthetic_boiler_data_{scenario.lower()}.csv`")
    
    # ---- TAB 3: AI DETECTION ----
    with tab3:
        st.header("AI Detection")
        
        col_ml, col_det = st.columns(2)
        with col_ml:
            st.subheader("ML Anomaly Detection (Isolation Forest)")
            st.metric("Anomaly Score", f"{ml_results['anomaly_score']:.3f}")
            if ml_results['ml_anomaly']:
                st.error("🔴 ML Model: Anomaly Detected")
            else:
                st.success("🟢 ML Model: Normal")
        
        with col_det:
            st.subheader("Deterministic Rule Results")
            if eng_results['rules_triggered']:
                for rule in eng_results['rules_triggered']:
                    st.write(f"⚠️ {rule}")
            else:
                st.success("✅ No engineering rules triggered.")
    
    # ---- TAB 4: AGENT INVESTIGATION ----
    with tab4:
        st.header("Agent Investigation")
        for agent_res in orch_results['agent_findings']:
            sev = agent_res['severity']
            icon = "🟢" if sev == "NORMAL" else ("🟡" if sev == "WARNING" else "🔴")
            with st.expander(f"{icon} {agent_res['agent'].replace('_', ' ').title()} — {sev}", expanded=True):
                st.write(f"**Finding:** {agent_res['finding']}")
                st.progress(agent_res['confidence'], text=f"Confidence: {agent_res['confidence']*100:.1f}%")
    
    # ---- TAB 5: ORCHESTRATOR ----
    with tab5:
        st.header("Orchestrator — Evidence Fusion")
        st.markdown("""
        ```
        Anomaly Detected
              ↓
        Orchestrator activates relevant agents
              ↓
        Temperature Agent  ←→  Vibration Agent  ←→  Flow Agent  ←→  Sensor Agent
              ↓
        Evidence collected from all agents
              ↓
        Evidence fused (weighted scoring)
              ↓
        Risk calculated → Health Score computed
              ↓
        Diagnosis generated
        ```
        """)
        
        st.subheader("Fusion Weights (Prototype)")
        st.write("- Deterministic Engine: **30%**")
        st.write("- ML Anomaly Score: **30%**")
        st.write("- Temperature Agent: **15%**")
        st.write("- Vibration Agent: **15%**")
        st.write("- Flow Agent: **10%**")
        
        with st.expander("📋 Raw Orchestrator Output (JSON)", expanded=False):
            st.json(orch_results)
    
    # ---- TAB 6: MAINTENANCE RECOMMENDATION ----
    with tab6:
        st.header("Maintenance Recommendation")
        
        st.markdown(f"### Boiler Health: {orch_results['health_score']}/100")
        st.markdown(f"**Risk Level:** {orch_results['risk_level']}")
        st.markdown(f"**Possible Condition:** {orch_results['possible_fault']}")
        
        if eng_results['engineering_findings']:
            st.markdown("**Evidence:**")
            for finding in eng_results['engineering_findings']:
                st.markdown(f"- {finding}")
        
        st.markdown(f"### Recommended Action")
        st.info(recommendation)
        
        st.markdown("---")
        st.subheader("AI Explanation")
        st.write(explanation)
    
    # ---- TAB 7: MODEL PERFORMANCE ----
    with tab7:
        st.header("Model Performance")
        st.write("**Model:** TensorFlow Autoencoder (Keras)")
        st.write("**Training Data:** NORMAL scenario (1000 samples)")
        st.write("**Features used:** " + ", ".join(f"`{f}`" for f in FEATURES))
        st.write(f"**Scoring:** average anomaly score over the latest {SCORING_WINDOW} readings")
        
        st.markdown("---")
        st.subheader("Current Prediction")
        st.write(f"- Anomaly Score: **{ml_results['anomaly_score']:.3f}**")
        st.write(f"- Classification: **{'🔴 Anomaly' if ml_results['ml_anomaly'] else '🟢 Normal'}**")
        
        st.markdown("---")
        st.caption("This is a prototype with synthetic data. In production, model metrics (accuracy, precision, recall, F1, confusion matrix) would be computed from a labeled validation set.")
