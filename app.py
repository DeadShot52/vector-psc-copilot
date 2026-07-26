import streamlit as st
import pandas as pd
from fpdf import FPDF
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import ast
import os

# Page configuration
st.set_page_config(
    page_title="Vector OS - PSC Copilot",
    page_icon="⚓",
    layout="wide"
)

# Header Section
st.title("⚓ VECTOR OS: PREDICTIVE PSC INTELLIGENCE")
st.markdown("Enter an IMO number to query the offline database and calculate detention probabilities.")

# Load Fleet Data cleanly
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("fleet_data.csv")
        # Convert IMO numbers to clean strings (removes .0 and leading/trailing spaces)
        df["IMO_Number"] = df["IMO_Number"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        return df
    except Exception as e:
        st.error(f"Error loading fleet_data.csv: {e}")
        return pd.DataFrame()

df = load_data()

# Dynamic sample hint for the input label
if not df.empty and "IMO_Number" in df.columns:
    sample_imo = df["IMO_Number"].iloc[0]
else:
    sample_imo = "9338632"

# User Manual Text Input
imo_input = st.text_input(f"Vessel IMO Number (Try: {sample_imo})", value="").strip()

if st.button("Query Database"):
    if not imo_input:
        st.warning("Please enter an IMO number.")
    else:
        if not df.empty and "IMO_Number" in df.columns:
            # Match exact string IMO
            vessel_match = df[df["IMO_Number"] == imo_input]
            
            if not vessel_match.empty:
                vessel = vessel_match.iloc[0]
                st.success(f"Vessel Record Found: {vessel.get('Vessel_Name', 'Vessel')} (IMO: {imo_input})")
                
                st.divider()
                
                # Display Key Vessel Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Vessel Name", str(vessel.get("Vessel_Name", "N/A")))
                    st.metric("Vessel Type", str(vessel.get("Vessel_Type", "Bulk Carrier")))
                with col2:
                    st.metric("Flag State", str(vessel.get("Flag_State", "N/A")))
                    st.metric("Year Built", str(vessel.get("Year_Built", "N/A")))
                with col3:
                    risk_val = vessel.get("Risk_Score", 68.5)
                    st.metric("Predicted PSC Risk", f"{risk_val}%")
                    st.metric("Risk Level", "HIGH" if float(risk_val) > 50 else "LOW")
                
                st.divider()
                
                # ROI & Detention Risk Exposure Calculator
                st.subheader("💰 Financial Risk Exposure Engine")
                detention_cost_per_day = 50000
                risk_pct = float(risk_val) / 100.0
                expected_exposure = detention_cost_per_day * risk_pct
                
                st.warning(f"**Expected Daily Detention Exposure:** ${expected_exposure:,.2f} / day")
                st.info("💡 **Commercial Impact:** Preventing a single 24-hour detention ($50,000 loss) pays for Vector OS across a 10-vessel fleet for over 14 years.")
                
            else:
                st.error(f"IMO '{imo_input}' not found in offline database.")
                sample_list = ", ".join(df["IMO_Number"].head(3).tolist())
                st.info(f"💡 Try typing one of these valid IMOs currently in your dataset: **{sample_list}**")
        else:
            st.error("Fleet dataset is empty or missing 'IMO_Number' column.")
                
