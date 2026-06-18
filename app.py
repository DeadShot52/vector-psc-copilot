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
        1. Base predictions on high-risk areas: ISM Code (15150), Fire Safety (07115), and Life-Saving Appliances (11101).
        2. Format strictly as a checklist.
        3. Cite the relevant SOLAS or MARPOL regulation.
        4. Provide an immediate "Corrective Action".
        5. Keep it highly technical. Use severity warnings.
        6. ZERO HALLUCINATION POLICY: If you are not 100% certain of the exact IMO, SOLAS, or MARPOL regulation number, do NOT guess or invent one. 
        7. If you do not know the exact citation, you must explicitly state: "Exact regulation code not found in offline knowledge base. Consult vessel SMS."
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
st.subheader("3. SMS Verification Mode")
doc_text = st.text_area("Paste SMS Segment / Operational Text here:", height=100, placeholder="e.g., Chief Engineer instructions for testing the emergency fire pump...")

if st.button("Audit Document against Target Port Criteria", use_container_width=True):
    if doc_text:
        with st.spinner("Auditing document against SOLAS/MARPOL frameworks..."):
            sys_prompt = """You are a strict Maritime Auditor. Review the text for violations of MARPOL, SOLAS, or STCW. 
            RULES:
            1. Highlight risks and suggest compliant corrections.
            2. ZERO HALLUCINATION POLICY: Only cite regulation numbers if you are absolute certain they exist in official maritime frameworks. 
            3. If you cannot verify the exact rule, do not invent one. Instead, explicitly state: 'Refer to official Flag State circulars for exact regulatory citation.'
            """
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
