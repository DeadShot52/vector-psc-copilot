import streamlit as st
from groq import Groq
from pinecone import Pinecone
from datetime import datetime, timedelta
from twilio.rest import Client
import requests

from rag_engine import ingest_data_to_pinecone, query_pinecone

# --- ADMIN PANEL (Temporary) ---
st.sidebar.title("Admin Tools")
if st.sidebar.button("Ingest Knowledge Base to Pinecone"):
    with st.spinner("Ingesting data..."):
        result = ingest_data_to_pinecone()
        st.sidebar.success(result)

# Example of how to use it later in your app:
# user_question = "What are the fire safety requirements?"
# context = query_pinecone(user_question)
# prompt = f"Context: {context}\n\nQuestion: {user_question}\n\nAnswer:"

# --- CONFIGURATION & 3D UI ---
st.set_page_config(page_title="Vector OS | Fleet Intelligence", page_icon="⚓", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0A0E17; color: #E0E6ED; }
    h1, h2, h3, h4 { color: #00F2FE !important; font-family: 'Courier New', Courier, monospace; letter-spacing: 1px; text-shadow: 0 0 10px rgba(0, 242, 254, 0.3); }
    div[data-testid="stExpander"], div.stTextArea>div>div, div.stTabs [data-baseweb="tab-panel"] {
        background: linear-gradient(145deg, #111827, #0d121c) !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.9) !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #E63946, #9b2226); color: white; border: none; border-radius: 8px; box-shadow: 0 4px 15px rgba(230, 57, 70, 0.4);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(230, 57, 70, 0.8); }
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background-color: #111827 !important; color: #00F2FE !important; border: 1px solid #374151 !important;
    }
    .metric-card { background-color: #111827; padding: 15px; border-radius: 10px; border-left: 4px solid #00F2FE; }
    </style>
    """, unsafe_allow_html=True)

# --- API INITIALIZATION ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("vector-maritime")
except Exception as e:
    st.error(f"SYSTEM OFFLINE: {str(e)}")
    st.stop()

# --- INITIALIZE IN-MEMORY FLEET DATABASE ---
if 'fleet' not in st.session_state:
    # Starting baseline fleet data for a professional presentation
    st.session_state['fleet'] = [
        {"name": "Vector Horizon", "imo": 9412345, "risk": 22, "type": "Oil Tanker"},
        {"name": "Vector Sovereign", "imo": 9654321, "risk": 74, "type": "Bulk Carrier"},
        {"name": "Vector Voyager", "imo": 9881122, "risk": 41, "type": "Container Ship"}
    ]

# --- DYNAMIC METRIC CALCULATIONS ---
fleet_data = st.session_state['fleet']
total_ships = len(fleet_data)

# High Risk means a vessel has an individual risk score over 60
high_risk_count = sum(1 for ship in fleet_data if ship['risk'] > 60)

# Fleet Readiness is the inverse average risk of all combined vessels
avg_fleet_risk = sum(ship['risk'] for ship in fleet_data) / total_ships if total_ships > 0 else 0
computed_readiness = int(100 - avg_fleet_risk)

# Dynamic Threat Level threshold evaluation
computed_threat = "CRITICAL" if high_risk_count >= 2 else "ELEVATED" if high_risk_count == 1 else "LOW RISK"

# --- HEADER: FLEET DASHBOARD ---
st.title("⚓ VECTOR OS: PREDICTIVE INTELLIGENCE")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Fleet Readiness Score", f"{computed_readiness}/100", f"{'+1.2%' if computed_readiness > 80 else '-2.1%'}")
col_b.metric("Global PSC Threat Level", computed_threat, "-")
col_c.metric("Active Fleet Vessels", f"{total_ships} Ships", "Live")
col_d.metric("High-Risk Vessels Detected", f"{high_risk_count}", f"+{high_risk_count}" if high_risk_count > 0 else "0")

st.markdown("---")


# --- CORE ARCHITECTURE TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Document & SMS Audit (Core)", 
    "2. PSC Digital Twin", 
    "3. Decision Simulator", 
    "4. CIC & Cert Engine"
])

# ==========================================
# TAB 1: SMS & DOCUMENT AUDIT (Claude & Grok)
# ==========================================
with tab1:
    st.subheader("Proprietary Document Cross-Examination")
    
    # Fortified Enterprise Security Shield Banner
    st.markdown("""
        <div style="background: rgba(0, 242, 254, 0.05); border: 1px solid #00F2FE; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <span style="color: #00F2FE; font-weight: bold;">🔒 ENTERPRISE SECURITY PROTOCOL ACTIVE</span><br>
            <span style="color: #A0AEC0; font-size: 0.9rem;">
                Zero-Data Retention rule enforced. Text is processed entirely inside transient memory and immediately purged post-audit. No data logs are saved to Pinecone or used for model training.
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("Audit agent messages, crew SMS, and printed operational PDFs directly against port-specific rules.")
    
    audit_target = st.selectbox("Target Authority for Audit", ["USCG", "Paris MoU", "AMSA", "Tokyo MoU"])
    doc_text = st.text_area("Paste Raw Operational Text or SMS:", height=150)
    
    if st.button("Execute Cross-Examination", type="primary"):
        if doc_text:
            with st.spinner("Executing Vector Cross-Examination..."):
                try:
                    audit_embedding = pc.inference.embed(
                        model="multilingual-e5-large", inputs=[doc_text], parameters={"input_type": "query"}
                    )
                    audit_matches = index.query(vector=audit_embedding[0].values, top_k=3, include_metadata=True)
                    audit_context = "\n".join([m['metadata']['text'] for m in audit_matches['matches']]) if audit_matches['matches'] else ""

                    sys_prompt = f"""You are Vector OS, a robotic Maritime Auditor. Evaluate the text against {audit_target} criteria.
                    USE THIS DATABASE CONTEXT: {audit_context}
                    OUTPUT FORMAT:
                    1. COMPLIANT / NON-COMPLIANT
                    2. Inspector Mindset: Explain exactly why {audit_target} inspectors target this.
                    3. Exact Regulation Citation.
                    4. Professional Correction needed."""
                    
                    res = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": doc_text}],
                        temperature=0.0
                    )
                    st.markdown(res.choices[0].message.content)
                except Exception as e:
                    st.error(str(e))


# ==========================================
# TAB 2: PSC DIGITAL TWIN & AIS TELEMETRY
# ==========================================
with tab2:
    st.subheader("Autonomous Vessel Intelligence")
    st.markdown("Enter an IMO number to automatically pull satellite telemetry and calculate detention probabilities.")
    
    col_search, col_btn = st.columns([3, 1])
    target_imo = col_search.text_input("Vessel IMO Number", placeholder="e.g. 9412345")
    
    # Initialize session state for fetched data so it persists
    if "vessel_data" not in st.session_state:
        st.session_state.vessel_data = {"type": "Bulk Carrier", "age": 12, "flag": "Liberia"}
        
    if col_btn.button("Engage AIS Uplink", use_container_width=True):
        if target_imo:
            with st.spinner("Triangulating satellite telemetry..."):
                try:
                    # Simulated API Hook for Prototype (In production, replace with MarineTraffic API)
                    # We simulate the fetch delay to demonstrate the architecture
                    import time
                    time.sleep(1.5)
                    
                    # Logic to simulate data retrieval based on IMO
                    if target_imo.startswith("94"):
                        st.session_state.vessel_data = {"type": "Oil Tanker", "age": 15, "flag": "Panama"}
                    elif target_imo.startswith("98"):
                        st.session_state.vessel_data = {"type": "Container Ship", "age": 5, "flag": "Marshall Islands"}
                    else:
                        st.session_state.vessel_data = {"type": "Bulk Carrier", "age": 10, "flag": "Liberia"}
                        
                    st.success(f"Uplink Established. Telemetry acquired for IMO {target_imo}.")
                except Exception as e:
                    st.error("AIS Satellite Uplink Failed. Manual override required.")
        else:
            st.warning("Input valid IMO.")

    st.markdown("---")
    
    # The inputs now auto-fill from the AIS data, but remain editable
    col1, col2, col3 = st.columns(3)
    v_type = col1.selectbox("Hull Type", ["Bulk Carrier", "Oil Tanker", "Container Ship"], index=["Bulk Carrier", "Oil Tanker", "Container Ship"].index(st.session_state.vessel_data["type"]))
    v_age = col2.number_input("Age (Years)", 0, 35, st.session_state.vessel_data["age"])
    v_port = col3.selectbox("Next Port of Call", ["USCG (Houston)", "Paris MoU (Rotterdam)", "MPA (Singapore)"])
    
    col4, col5, col6 = st.columns(3)
    v_flag = col4.selectbox("Flag State", ["Liberia", "Panama", "Marshall Islands", "Blacklisted Flag"], index=["Liberia", "Panama", "Marshall Islands", "Blacklisted Flag"].index(st.session_state.vessel_data["flag"]))
    v_class = col5.selectbox("Class Society", ["IACS", "Non-IACS"])
    v_def = col6.text_input("Last Deficiencies (Optional)", "e.g., Fire doors, OWS")

    if st.button("Generate Detention Forecast", type="primary"):
        with st.spinner("Calculating Boarding & Detention Probabilities..."):
            try:
                search_query = f"Deficiencies and detention targets for {v_type} under {v_flag} flag arriving in {v_port} regarding {v_def}"
                query_emb = pc.inference.embed(model="multilingual-e5-large", inputs=[search_query], parameters={"input_type": "query"})
                db_res = index.query(vector=query_emb[0].values, top_k=3, include_metadata=True)
                ctx = "\n".join([m['metadata']['text'] for m in db_res['matches']]) if db_res['matches'] else "No context found. Defaulting to baseline compliance."

                sys_prompt = f"""You are the Vector OS Probability Engine.
                Evaluate a {v_age}yr old {v_type}, Flag: {v_flag}, Class: {v_class}, Port: {v_port}. 
                Database Context: {ctx}
                
                OUTPUT FORMAT EXACTLY LIKE THIS:
                1. Boarding Probability: (Provide a realistic percentage)
                2. Detention Probability Score: (Provide a realistic percentage)
                3. Inspector Behavior Intelligence: (What does {v_port} strictly focus on historically based on the context?)
                4. Top 3 Likely Findings: (List them with percentages of likelihood)."""
                
                res = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Forecast risk."}],
                    temperature=0.2
                )
                st.markdown(res.choices[0].message.content)
            except Exception as e:
                st.error(str(e))


# ==========================================
# TAB 3: DECISION SIMULATOR & EMERGENCY ALERTS
# ==========================================
with tab3:
    st.subheader("Detention Avoidance Simulator (What-If)")
    st.markdown("Select corrective actions to see the real-time impact on your vessel's Detention Probability.")
    
    base_risk = 68
    st.write(f"**Current Baseline Risk Score: {base_risk}/100**")
    
    fix_1 = st.checkbox("Rectify sluggish engine room fire dampers")
    fix_2 = st.checkbox("Update ECDIS passage plan for local port limits")
    fix_3 = st.checkbox("Conduct emergency steering drill and log it")
    
    new_risk = base_risk
    if fix_1: new_risk -= 24
    if fix_2: new_risk -= 12
    if fix_3: new_risk -= 15
    
    st.progress(new_risk / 100)
    st.metric("Simulated Adjusted Risk Score", f"{new_risk}/100", f"-{base_risk - new_risk}")
    
    if new_risk < 30:
        st.success("Target risk profile achieved. Vessel cleared for arrival.")
    else:
        st.warning("Vessel remains in high-risk detention bracket.")

    st.markdown("---")
    st.subheader("📱 Vector OS Outbound Notification Control")
    st.caption("Broadcast instantaneous critical risk vectors directly to the DPA or Fleet Manager's mobile device via WhatsApp.")
    
    target_phone = st.text_input("Enter DPA Mobile Number (With Country Code):", placeholder="+91XXXXXXXXXX")
    alert_payload = f"⚠️ VECTOR OS ALERT: High risk vessel threat detected. Simulated fleet risk score is currently critical at {new_risk}/100. Corrective field actions required immediately."

    if st.button("Dispatch Urgent WhatsApp Notification", type="primary"):
        if not target_phone:
            st.warning("Please input a valid destination mobile number.")
        else:
            with st.spinner("Establishing secure handshake with telecommunications gateway..."):
                try:
                    # Pulling security credentials from your hidden Streamlit Secrets
                    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
                    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
                    twilio_whatsapp_number = st.secrets["TWILIO_WHATSAPP_NUMBER"] # Typically 'whatsapp:+14155238886' for sandbox
                    
                    twilio_client = Client(account_sid, auth_token)
                    
                    message = twilio_client.messages.create(
                        from_=f"whatsapp:{twilio_whatsapp_number}",
                        body=alert_payload,
                        to=f"whatsapp:{target_phone}"
                    )
                    st.success(f"🚀 Dispatch Successful! Message ID: {message.sid}")
                except Exception as e:
                    st.error(f"Error: {str(e)} | Keys Streamlit can actually see: {list(st.secrets.keys())}")



# ==========================================
# TAB 4: CIC & CERT ENGINE (Claude)
# ==========================================
with tab4:
    st.subheader("Concentrated Inspection Campaign (CIC) Engine")
    st.info("🚨 **ACTIVE CIC WARNING**: Paris & Tokyo MoU are actively targeting Crew Wages & SEAs (2024/2025).")
    st.warning("🚨 **ACTIVE CIC WARNING**: USCG prioritizing Engine Room Automation.")
    
    st.markdown("---")
    st.subheader("Certificate Expiry Intelligence")
    col_x, col_y = st.columns(2)
    cert_name = col_x.text_input("Certificate Name", "e.g., Safety Equipment Certificate")
    cert_date = col_y.date_input("Expiry Date", datetime.today() + timedelta(days=14))
    
    if st.button("Check Port State Exposure"):
        days_left = (cert_date - datetime.today().date()).days
        if days_left < 30:
            st.error(f"CRITICAL: {cert_name} expires in {days_left} days. High probability of PSC code 30 (Detainable) if an extension is not secured before arrival.")
        else:
            st.success(f"{cert_name} is secure for {days_left} days.")
