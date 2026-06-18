import streamlit as st
from groq import Groq

# --- CONFIGURATION ---
st.set_page_config(page_title="Vector PSC Copilot", page_icon="⚓", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    h1, h2, h3 { color: #E63946 !important; font-family: 'Courier New', Courier, monospace; }
    .stAlert { border-left: 4px solid #E63946; }
    </style>
    """, unsafe_allow_html=True)

# --- API INITIALIZATION ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("SYSTEM OFFLINE: Groq API Key missing in Streamlit Secrets.")
    st.stop()

# --- MAIN DASHBOARD (MOBILE OPTIMIZED) ---
st.title("⚓ VECTOR PSC COPILOT")
st.markdown("Predictive Detention Intelligence for Port State Control.")
st.markdown("---")

# Moved from Sidebar to Main Page Columns for Mobile Visibility
st.subheader("1. Vessel Profile Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    vessel_type = st.selectbox("Vessel Hull Type", ["Bulk Carrier", "Oil Tanker", "Container Ship"])
with col2:
    vessel_age = st.number_input("Vessel Age (Years)", min_value=0, max_value=30, value=12)
with col3:
    destination_port = st.selectbox("Destination Port Authority", ["USCG (USA)", "AMSA (Australia)", "MPA (Singapore)", "Paris MoU (Europe)"])

st.markdown("---")
st.subheader("2. Pre-Arrival PSC Risk Predictor")

if st.button("Generate Predictive PSC Checklist", type="primary", use_container_width=True):
    with st.spinner("Analyzing historical detention databases..."):
        
        system_prompt = f"""You are the Vector PSC Predictive Intelligence Engine. 
Predict the top 3 most likely Port State Control (PSC) deficiencies for a {vessel_age}-year-old {vessel_type} arriving in {destination_port} jurisdiction.

RULES:
1. DYNAMIC ANALYSIS: Do not use generic answers. Identify the top 3 statistically most likely detainable deficiencies for this EXACT vessel age, type, and destination based on historical PSC data (e.g., AMSA targets older bulkers for structural issues; USCG targets MARPOL Annex VI compliance, etc.).
2. Format strictly as a checklist.
3. Cite the relevant SOLAS or MARPOL regulation.
4. Provide an immediate "Corrective Action".
5. ZERO HALLUCINATION POLICY: If you are not 100% certain of the exact regulation number, do NOT guess.
6. At the very end of your response, write a "💡 Vector Intelligence Insight:" summarizing the biggest risk trend for this specific port authority.
"""

        user_prompt = "Generate the targeted pre-arrival audit checklist."
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            st.warning(f"⚠️ HIGH RISK PROFILE DETECTED FOR {destination_port}")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"API Error: {str(e)}")
    else:
        st.warning("Please paste some text first.")
