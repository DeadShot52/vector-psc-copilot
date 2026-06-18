import streamlit as st

# 1. Page Configuration & Custom Theme
st.set_page_config(page_title="Vector PSC Copilot", page_icon="⚓", layout="wide")

# Custom CSS to force an enterprise dark maritime dashboard appearance
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    h1, h2, h3 { color: #E63946 !important; font-family: 'Courier New', Courier, monospace; }
    .stAlert { border-left: 4px solid #E63946; }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar Configuration: Vessel Profile Inputs
with st.sidebar:
    st.title("⚓ VECTOR PSC COPILOT")
    st.caption("Predictive Detention Intelligence")
    st.markdown("---")
    
    st.subheader("Vessel Profile Configuration")
    vessel_type = st.selectbox("Vessel Hull Type", ["Bulk Carrier", "Oil Tanker", "Container Ship"])
    vessel_age = st.slider("Vessel Age (Years Since Built)", 0, 30, 12)
    destination_port = st.selectbox("Destination Port Authority", ["USCG (USA)", "AMSA (Australia)", "MPA (Singapore)", "Paris MoU (Europe)"])
    
    st.markdown("---")
    st.caption("Intelligence Layer Connection: Standard")

# 3. Main Operational Dashboard Layout
st.title("Pre-Arrival PSC Risk Predictor")
st.markdown("Analyze localized vessel metrics against historical Port State Control data matrices to eliminate detention events.")

if st.button("Generate Predictive PSC Checklist", type="primary"):
    with st.spinner("Analyzing historical detention databases..."):
        # Temporary placeholder text until we connect the Groq brain in the next step
        st.info("System Ready. Connect the core AI reasoning engine to execute calculations.")

st.markdown("---")
st.subheader("SMS Verification Mode")
st.markdown("Paste active daily orders or draft engineering procedures to run an automated deficiency check.")
st.text_area("Paste SMS Segment / Operational Text here:", height=100, placeholder="e.g., Chief Engineer instructions for testing the emergency fire pump...")
st.button("Audit Document against Target Port Criteria")
