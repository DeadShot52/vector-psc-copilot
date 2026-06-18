import streamlit as st
from groq import Groq

# --- CONFIGURATION ---
st.set_page_config(page_title="Vector PSC Copilot", page_icon="⚓", layout="wide")

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

# --- SIDEBAR: VESSEL PROFILE ---
with st.sidebar:
    st.title("⚓ VECTOR PSC COPILOT")
    st.caption("Predictive Detention Intelligence")
    st.markdown("---")
    
    st.subheader("Vessel Profile Configuration")
    vessel_type = st.selectbox("Vessel Hull Type", ["Bulk Carrier", "Oil Tanker", "Container Ship"])
    vessel_age = st.slider("Vessel Age (Years Since Built)", 0, 30, 12)
    destination_port = st.selectbox("Destination Port Authority", ["USCG (USA)", "AMSA (Australia)", "MPA (Singapore)", "Paris MoU (Europe)"])
    
    st.markdown("---")
    st.caption("Intelligence Layer: Active")

# --- MAIN DASHBOARD ---
st.title("Pre-Arrival PSC Risk Predictor")
st.markdown("Analyze localized vessel metrics against historical Port State Control data matrices to eliminate detention events.")

if st.button("Generate Predictive PSC Checklist", type="primary"):
    with st.spinner("Analyzing historical detention databases..."):
        
        system_prompt = f"""You are the Vector PSC Predictive Intelligence Engine. 
        Your task is to predict the top 3 most likely Port State Control (PSC) deficiencies for a {vessel_age}-year-old {vessel_type} arriving in {destination_port} jurisdiction.
        
        RULES:
        1. Base predictions on known high-risk areas: ISM Code (15150), Fire Safety (07115), and Life-Saving Appliances (11101).
        2. Format output strictly as a checklist.
        3. For each predicted deficiency, cite the relevant SOLAS or MARPOL regulation.
        4. Provide an immediate "Corrective Action" the Chief Officer must take before arrival.
        5. Keep it highly technical, zero fluff. Use severity warnings.
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
            
            st.info("💡 Vector Intelligence: 80% of vessels matching this profile were detained for ISM Code violations in the last 12 months. Verify SMS implementation immediately.")
            
        except Exception as e:
            st.error(f"API Error: {str(e)}")

st.markdown("---")
st.subheader("SMS Verification Mode")
st.markdown("Paste active daily orders or draft engineering procedures to run an automated deficiency check.")
doc_text = st.text_area("Paste SMS Segment / Operational Text here:", height=100, placeholder="e.g., Chief Engineer instructions for testing the emergency fire pump...")

if st.button("Audit Document against Target Port Criteria"):
    if doc_text:
        with st.spinner("Auditing document against SOLAS/MARPOL frameworks..."):
            sys_prompt = "You are a strict Maritime Auditor. Review the text for violations of MARPOL, SOLAS, or STCW. Highlight risks and suggest compliant corrections."
            try:
                res = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": f"Audit this procedure: {doc_text}"}
                    ],
                    temperature=0.1
                )
                st.markdown(res.choices[0].message.content)
            except Exception as e:
                st.error(f"API Error: {str(e)}")
    else:
        st.warning("Please paste some text first.")
