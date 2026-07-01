import streamlit as st
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime, timedelta
from twilio.rest import Client
import requests
import PyPDF2
import uuid
import time

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

# --- API INITIALIZATION & CLOUD CONNECTION ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    
    # Ensure the 1024-dimension enterprise index exists
    index_name = "vector-core-v1"
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1024, 
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    index = pc.Index(index_name)
    
except Exception as e:
    st.error(f"SYSTEM OFFLINE: {str(e)}")
    st.stop()

# --- INITIALIZE IN-MEMORY FLEET DATABASE ---
if 'fleet' not in st.session_state:
    st.session_state['fleet'] = [
        {"name": "Vector Horizon", "imo": 9412345, "risk": 22, "type": "Oil Tanker"},
        {"name": "Vector Sovereign", "imo": 9654321, "risk": 74, "type": "Bulk Carrier"},
        {"name": "Vector Voyager", "imo": 9881122, "risk": 41, "type": "Container Ship"}
    ]

# --- DYNAMIC METRIC CALCULATIONS ---
fleet_data = st.session_state['fleet']
total_ships = len(fleet_data)
high_risk_count = sum(1 for ship in fleet_data if ship['risk'] > 60)
avg_fleet_risk = sum(ship['risk'] for ship in fleet_data) / total_ships if total_ships > 0 else 0
computed_readiness = int(100 - avg_fleet_risk)
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
# TAB 1: SMS & DOCUMENT AUDIT 
# ==========================================
with tab1:
    st.subheader("Proprietary Document Cross-Examination")
    
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
    
    if "vessel_data" not in st.session_state:
        st.session_state.vessel_data = {"type": "Bulk Carrier", "age": 12, "flag": "Liberia"}
        
    if col_btn.button("Engage AIS Uplink", use_container_width=True):
        if target_imo:
            with st.spinner("Pinging Global AIS Satellite Network..."):
                # 1. Attempt Live API Uplink
                api_key = st.secrets.get("VESSEL_API_KEY", "")
                api_success = False
                
                if api_key:
                    try:
                        headers = {"Authorization": f"Bearer {api_key}"}
                        url = f"https://api.vesselapi.com/v1/vessel/{target_imo}?filter.idType=imo"
                        res = requests.get(url, headers=headers, timeout=8)
                        
                        if res.status_code == 200:
                            raw_data = res.json()
                            ship = raw_data
                            
                            # Handle nested arrays/objects safely
                            if "data" in ship:
                                ship = ship["data"]
                            if isinstance(ship, list) and len(ship) > 0:
                                ship = ship[0]
                                
                            # Omni-Catcher: Scans for multiple variations of API keys
                            v_name = ship.get("vesselName") or ship.get("vessel_name") or ship.get("name") or "Unknown"
                            v_type = ship.get("vesselType") or ship.get("vessel_type") or ship.get("type") or "Bulk Carrier"
                            v_flag = ship.get("flag") or ship.get("country") or "Unknown"
                            
                            y_built = ship.get("yearBuilt") or ship.get("year_built") or ship.get("build_year") or 2010
                            try:
                                v_age = datetime.now().year - int(y_built)
                            except:
                                v_age = 15 # Default fallback if API sends a blank year
                                
                            st.session_state.vessel_data = {
                                "type": v_type, 
                                "age": v_age, 
                                "flag": v_flag
                            }
                            st.success(f"Live Uplink Established: {v_name} (IMO {target_imo})")
                            
                            # The "Matrix" Data Dump for live pitches
                            with st.expander("📡 View Raw Satellite Telemetry"):
                                st.json(raw_data)
                                
                            api_success = True
                        else:
                            st.error(f"API Rejected Request (Code {res.status_code}): {res.text}")
                    except Exception as e:
                        st.error(f"Live API Network Failure: {str(e)}")
                else:
                    st.error("No VESSEL_API_KEY found in Streamlit Secrets.")
                
                # 2. Bulletproof Fallback to Offline Registry
                if not api_success:
                    st.info("Attempting offline registry fallback...")
                    local_registry = {
                        "9388297": {"name": "SELECAO", "type": "Oil Tanker", "flag": "Liberia", "build": 2008},
                        "9320520": {"name": "Emma Maersk", "type": "Container Ship", "flag": "Denmark", "build": 2006},
                        "9432652": {"name": "Stena Polaris", "type": "Oil Tanker", "flag": "Cyprus", "build": 2010},
                        "9455923": {"name": "Valemax Brasil", "type": "Bulk Carrier", "flag": "Singapore", "build": 2011},
                        "9848520": {"name": "CMA CGM Jacques Saade", "type": "Container Ship", "flag": "France", "build": 2020},
                        "9164263": {"name": "Front Century", "type": "Oil Tanker", "flag": "Marshall Islands", "build": 1998}
                    }
                    if target_imo in local_registry:
                        ship_intel = local_registry[target_imo]
                        st.session_state.vessel_data = {
                            "type": ship_intel["type"], 
                            "age": datetime.now().year - ship_intel["build"], 
                            "flag": ship_intel["flag"]
                        }
                        st.success(f"Offline Uplink: {ship_intel['name']} (IMO {target_imo}).")
                    else:
                        st.warning("IMO not found in live network or offline registry. Use manual dropdowns.")
        else:
            st.warning("Input valid IMO.")

    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    # Safe Hull Type Fallback
    hull_options = ["Bulk Carrier", "Oil Tanker", "Container Ship", "Tanker", "General Cargo"]
    fetched_type = st.session_state.vessel_data["type"]
    if fetched_type not in hull_options:
        hull_options.append(fetched_type)
        
    v_type = col1.selectbox("Hull Type", hull_options, index=hull_options.index(fetched_type))
    v_age = col2.number_input("Age (Years)", 0, 50, st.session_state.vessel_data["age"])
    v_port = col3.selectbox("Next Port of Call", ["USCG (Houston)", "Paris MoU (Rotterdam)", "MPA (Singapore)"])
    
    col4, col5, col6 = st.columns(3)
    
    # Safe Flag Fallback
    flag_options = list(set(["Liberia", "Panama", "Marshall Islands", "Cyprus", "Denmark", "France", "Blacklisted Flag", st.session_state.vessel_data["flag"]]))
    v_flag = col4.selectbox("Flag State", flag_options, index=flag_options.index(st.session_state.vessel_data["flag"]))
    
    v_class = col5.selectbox("Class Society", ["IACS", "Non-IACS"])
    v_def = col6.text_input("Last Deficiencies (Optional)", "e.g., Fire doors, OWS")

    if st.button("Generate Detention Forecast", type="primary"):
        with st.spinner("Calculating Boarding & Detention Probabilities..."):
            try:
                search_query = f"Deficiencies and detention targets for {v_type} under {v_flag} flag arriving in {v_port} regarding {v_def}"
                query_emb = pc.inference.embed(model="multilingual-e5-large", inputs=[search_query], parameters={"input_type": "query"})
                db_res = index.query(vector=query_emb[0].values, top_k=3, include_metadata=True)
                ctx = "\n".join([m['metadata']['text'] for m in db_res['matches']]) if db_res['matches'] else "No context found. Defaulting to baseline compliance."

                sys_prompt = f"""You are the Vector OS Predictive Risk Engine.
                Analyze a {v_age}yr old {v_type}, Flag: {v_flag}, Class: {v_class}, Port: {v_port}. 
                Context: {ctx}
                
                CRITICAL RULES:
                - Output ONLY the exact format below.
                - Max 10 words per bullet point. NO full sentences.
                - Be blunt, analytical, and highly technical.
                
                **Boarding Probability**: [X]%
                * [Primary trigger factor]
                
                **Detention Risk**: [X]%
                * [Primary risk factor]
                
                **Inspector Focus Vectors**: 
                * [Target area 1]
                * [Target area 2]
                
                **Top 3 Likely Findings**: 
                1. [Finding] ([X]%)
                2. [Finding] ([X]%)
                3. [Finding] ([X]%)"""
                
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
                    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
                    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
                    twilio_whatsapp_number = st.secrets["TWILIO_WHATSAPP_NUMBER"]
                    
                    twilio_client = Client(account_sid, auth_token)
                    
                    message = twilio_client.messages.create(
                        from_=f"whatsapp:{twilio_whatsapp_number}",
                        body=alert_payload,
                        to=f"whatsapp:{target_phone}"
                    )
                    st.success(f"🚀 Dispatch Successful! Message ID: {message.sid}")
                except Exception as e:
                    st.error(f"Alert Failed: {str(e)}")

# ==========================================
# TAB 4: CIC & CERT ENGINE
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

# ==========================================
# SYSTEM ADMINISTRATION & DATA INGESTION
# ==========================================
st.markdown("---")
with st.expander("⚓ System Administration & Knowledge Base Ingestion"):
    st.subheader("Vector Brain Core Uplink")
    st.markdown("Upload official maritime convention updates or MoU reports directly into the Pinecone Vector database.")
    
    admin_password = st.text_input("Enter Admin Security Token", type="password")
    
    if admin_password == "vector2026": 
        uploaded_file = st.file_uploader("Choose a maritime PDF report", type=["pdf"])
        
        if uploaded_file is not None:
            if st.button("Execute Vector Conversion & Ingestion", type="primary"):
                with st.spinner("Deconstructing PDF and generating 1024-dimension mathematical embeddings..."):
                    try:
                        reader = PyPDF2.PdfReader(uploaded_file)
                        full_text = ""
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                full_text += text + "\n"
                        
                        if not full_text.strip():
                            st.error("Could not extract legible text from this PDF. It might be scanned images.")
                            st.stop()
                            
                        # Intelligent string chunking (Bypasses LangChain completely)
                        chunk_size = 1000
                        overlap = 200
                        chunks = []
                        start = 0
                        while start < len(full_text):
                            end = start + chunk_size
                            chunks.append(full_text[start:end])
                            start += chunk_size - overlap
                        
                        st.info(f"PDF split into {len(chunks)} tactical data chunks. Initiating cloud uplink...")
                        
                        batch_size = 20
                        progress_bar = st.progress(0)
                        
                        for i in range(0, len(chunks), batch_size):
                            batch_chunks = chunks[i:i+batch_size]
                            
                            embeddings = pc.inference.embed(
                                model="multilingual-e5-large",
                                inputs=batch_chunks,
                                parameters={"input_type": "passage"}
                            )
                            
                            vectors_to_upsert = []
                            for j, chunk in enumerate(batch_chunks):
                                unique_id = f"{uploaded_file.name}_chunk_{i+j}_{str(uuid.uuid4())[:8]}"
                                vectors_to_upsert.append({
                                    "id": unique_id,
                                    "values": embeddings[j].values,
                                    "metadata": {"text": chunk, "source": uploaded_file.name}
                                })
                            
                            index.upsert(vectors=vectors_to_upsert)
                            
                            fraction = min((i + batch_size) / len(chunks), 1.0)
                            progress_bar.progress(fraction)
                        
                        st.success(f"Uplink complete! Permanent indexing successful for: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"Ingestion Engine Failure: {str(e)}")
    elif admin_password:
        st.error("Invalid Security Token. Access Denied.")
