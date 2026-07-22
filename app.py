import streamlit as st
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import requests
import PyPDF2
import uuid

def calculate_vector_risk_score(vessel_age, flag_state, target_port, past_deficiencies_count, vessel_type):
    risk_score = 0
    risk_factors = []

    # 1. Vessel Age Multiplier (Max 20 pts)
    if vessel_age >= 15:
        risk_score += 20
        risk_factors.append(f"Critical Age ({vessel_age} years) -> +20 pts")
    elif vessel_age >= 10:
        risk_score += 10
        risk_factors.append(f"Moderate Age ({vessel_age} years) -> +10 pts")

    # 2. Flag State Matrix (Dictionary Lookup - Max 30 pts)
    # This maps specific flags to their Paris/Tokyo MOU BGW status
    flag_bgw_mapping = {
        'Togo': 'BLACK', 'Comoros': 'BLACK', 
        'Panama': 'GREY', 'Liberia': 'GREY',
        'Marshall Islands': 'WHITE'
    }
    flag_status = flag_bgw_mapping.get(flag_state, 'WHITE') # Defaults to White if not found
    
    if flag_status == 'BLACK':
        risk_score += 30
        risk_factors.append(f"Black-Listed Flag ({flag_state}) -> +30 pts")
    elif flag_status == 'GREY':
        risk_score += 15
        risk_factors.append(f"Grey-Listed Flag ({flag_state}) -> +15 pts")

    # 3. Port Stringency Matrix (Max 20 pts)
    strict_ports = ['Rotterdam', 'Singapore', 'Houston', 'Brisbane']
    if target_port in strict_ports:
        risk_score += 20
        risk_factors.append(f"High-Stringency Target Port ({target_port}) -> +20 pts")

    # 4. Vessel Type Hazard Multiplier (Max 15 pts)
    vessel_type_weights = {
        'Chemical Tanker': 15,
        'Gas Carrier': 15,
        'Bulk Carrier': 10,
        'Container': 5
    }
    type_weight = vessel_type_weights.get(vessel_type, 0)
    if type_weight > 0:
        risk_score += type_weight
        risk_factors.append(f"High-Scrutiny Vessel Type ({vessel_type}) -> +{type_weight} pts")

    # 5. Historical Deficiency Penalty (Max 15 pts)
    if past_deficiencies_count >= 5:
        risk_score += 15
        risk_factors.append(f"High Deficiency History ({past_deficiencies_count} found) -> +15 pts")

    # Ensure score does not accidentally exceed 100
    risk_score = min(risk_score, 100)

    # 6. Final Category Determination
    if risk_score >= 75:
        category = "CRITICAL RISK - High Probability of Detention"
    elif risk_score >= 45:
        category = "MODERATE RISK - Heightened Inspection Probability"
    else:
        category = "LOW RISK - Standard Clearance Expected"

    return risk_score, category, risk_factors

@st.cache_data
def load_vessel_database():
    # This reads your local CSV file into a pandas DataFrame
    df = pd.read_csv("fleet_data.csv")
    return df

# Load the database into memory
vessel_db = load_vessel_database()


# --- CONFIGURATION & UI ---
st.set_page_config(page_title="VectorPrime | PSC Intelligence", page_icon="⚓", layout="wide")

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
    </style>
    """, unsafe_allow_html=True)

# --- API INITIALIZATION ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    
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


# ==========================================
# CORE APPLICATION: PSC DIGITAL TWIN
# ==========================================
st.title("⚓ VECTORPRIME: PREDICTIVE PSC INTELLIGENCE")
st.markdown("Enter an IMO number to automatically pull satellite telemetry and calculate detention probabilities.")

col_search, col_btn = st.columns([3, 1])
target_imo = col_search.text_input("Vessel IMO Number", placeholder="e.g. 9987213")

if "vessel_data" not in st.session_state:
    st.session_state.vessel_data = {"type": "Bulk Carrier", "age": 12, "flag": "Liberia"}
    
if col_btn.button("Engage AIS Uplink", use_container_width=True):
    if target_imo:
        with st.spinner("Pinging Global AIS Satellite Network..."):
            api_key = st.secrets.get("VESSEL_API_KEY", "")
            api_success = False
            
            if api_key:
                try:
                    headers = {"Authorization": f"Bearer {api_key}"}
                    url = f"https://api.vesselapi.com/v1/vessel/{target_imo}?filter.idType=imo"
                    res = requests.get(url, headers=headers, timeout=8)
                    
                    if res.status_code == 200:
                        raw_data = res.json()
                        if "vessel" in raw_data:
                            ship = raw_data["vessel"]
                        else:
                            ship = raw_data
                            
                        v_name = ship.get("name", "Unknown")
                        v_type = ship.get("vessel_type", "Bulk Carrier")
                        v_flag = ship.get("country", "Unknown")
                        
                        y_built = ship.get("year_built", 2010)
                        try:
                            v_age = datetime.now().year - int(y_built)
                        except:
                            v_age = 15 
                            
                        st.session_state.vessel_data = {
                            "type": v_type, 
                            "age": v_age, 
                            "flag": v_flag
                        }
                        st.success(f"Live Uplink Established: {v_name} (IMO {target_imo})")
                        
                        with st.expander("📡 View Raw Satellite Telemetry"):
                            st.json(raw_data)
                            
                        api_success = True
                    else:
                        st.error(f"API Rejected Request (Code {res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Live API Network Failure: {str(e)}")
            else:
                st.error("No VESSEL_API_KEY found in Streamlit Secrets.")
            
            # Offline Fallback
            if not api_success:
                st.info("Attempting offline registry fallback...")
                local_registry = {
                    "9388297": {"name": "SELECAO", "type": "Oil Tanker", "flag": "Liberia", "build": 2008},
                    "9320520": {"name": "Emma Maersk", "type": "Container Ship", "flag": "Denmark", "build": 2006},
                    "9987213": {"name": "CMA CGM VENDOME", "type": "Container Ship", "flag": "Singapore", "build": 2025}
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
                    st.warning("IMO not found in live network. Use manual dropdowns below.")
    else:
        st.warning("Input valid IMO.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

hull_options = ["Bulk Carrier", "Oil Tanker", "Container Ship", "Tanker", "General Cargo"]
fetched_type = st.session_state.vessel_data["type"]
if fetched_type not in hull_options:
    hull_options.append(fetched_type)
    
v_type = col1.selectbox("Hull Type", hull_options, index=hull_options.index(fetched_type))
v_age = col2.number_input("Age (Years)", 0, 50, st.session_state.vessel_data["age"])
v_port = col3.selectbox("Next Port of Call", ["USCG (Houston)", "Paris MoU (Rotterdam)", "MPA (Singapore)"])

col4, col5, col6 = st.columns(3)

flag_options = list(set(["Liberia", "Panama", "Marshall Islands", "Cyprus", "Denmark", "France", "Singapore", "Blacklisted Flag", st.session_state.vessel_data["flag"]]))
v_flag = col4.selectbox("Flag State", flag_options, index=flag_options.index(st.session_state.vessel_data["flag"]))

v_class = col5.selectbox("Class Society", ["IACS", "Non-IACS"])
v_def = col6.text_input("Last Deficiencies (Optional)", "e.g., Fire doors, OWS")

if st.button("Generate Detention Forecast", type="primary"):
    with st.spinner("Analyzing Database & Generating Risk Matrices..."):
        try:
            # 1. DETERMINISTIC MATH ENGINE
            base_boarding = 20
            base_detention = 5
            
            if v_age <= 5:
                age_penalty = v_age * 1.2
            elif v_age <= 15:
                age_penalty = 6 + ((v_age - 5) * 1.8)
            else:
                age_penalty = 24 + ((v_age - 15) * 2.5)
                
            age_penalty = min(age_penalty, 45)
            
            flag_penalty = 35 if v_flag == "Blacklisted Flag" else (12 if v_flag in ["Panama", "Liberia", "Cyprus"] else 0)
            class_penalty = 30 if v_class == "Non-IACS" else 0
            
            matrix_penalty = 5
            if "USCG" in v_port and v_type in ["Oil Tanker", "Tanker"]: matrix_penalty = 18
            elif "Paris" in v_port and v_type == "Bulk Carrier": matrix_penalty = 15
            elif "MPA" in v_port and v_type == "Container Ship": matrix_penalty = 10
            
            calc_boarding = min(int(base_boarding + age_penalty + flag_penalty + matrix_penalty), 99)
            calc_detention = min(int(base_detention + (age_penalty * 0.7) + flag_penalty + class_penalty + (matrix_penalty * 0.5)), 99)

            # 2. RAG DATABASE RETRIEVAL
            search_query = f"Deficiencies and detention targets for {v_type} under {v_flag} flag arriving in {v_port} regarding {v_def}"
            query_emb = pc.inference.embed(model="multilingual-e5-large", inputs=[search_query], parameters={"input_type": "query"})
            db_res = index.query(vector=query_emb[0].values, top_k=3, include_metadata=True)
            ctx = "\n".join([m['metadata']['text'] for m in db_res['matches']]) if db_res['matches'] else "No context found. Defaulting to baseline compliance."

            # 3. FORECAST GENERATION
            sys_prompt = f"""You are the VectorPrime Predictive Risk Engine, built for Technical Superintendents.
            Analyze a {v_age}yr old {v_type}, Flag: {v_flag}, Class: {v_class}, Port: {v_port}. Context: {ctx}
            
            CRITICAL TECHNICAL RULES:
            1. NO GENERALITIES: Name exact components.
            2. USE STRICT MATH: Boarding Probability is STRICTLY {calc_boarding}%. Detention Risk is STRICTLY {calc_detention}%. Do not change these numbers.
            3. DETAILED ACTIONABILITY: Include specific regulatory code links and exact mechanical vulnerabilities.
            4. DO NOT INVENT PAST DEFICIENCIES. Predict future risk based on Port trends.
            
            OUTPUT FORMAT EXACTLY LIKE THIS:
            **Boarding Probability**: {calc_boarding}%
            * **Risk Driver:** [Specific technical explanation]
            
            **Detention Risk**: {calc_detention}%
            * **Risk Driver:** [Specific mechanical/systemic exposure]
            
            **Inspector Focus Vectors (Based on {v_port})**: 
            * **[System Name]** -> [Specific component targeted] | *Target Reason:* [Why]
            * **[System Name]** -> [Specific component targeted] | *Target Reason:* [Why]
            
            **Top 3 High-Probability Findings & Corrective Actions**: 
            1. 🔴 **Deficiency Item:** [Name specific hardware/component]
               * **Code:** [Exact Code]
               * **The Vulnerability:** [Exactly what fails]
               * **DPA Action Directive:** [Precise maintenance instruction]
            2. 🟡 **Deficiency Item:** [Name specific hardware/component]
               * **Code:** [Exact Code]
               * **The Vulnerability:** [Exactly what fails]
               * **DPA Action Directive:** [Precise maintenance instruction]
            3. 🟡 **Deficiency Item:** [Name specific hardware/component]
               * **Code:** [Exact Code]
               * **The Vulnerability:** [Exactly what fails]
               * **DPA Action Directive:** [Precise maintenance instruction]"""
            
            res_forecast = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Forecast risk."}],
                temperature=0.0
            )
            st.markdown(res_forecast.choices[0].message.content)
            
            
            # 4. TAILORED SHIPBOARD CHECKLIST GENERATOR (JSON Format for Interactive UI)
            checklist_prompt = f"""You are a strict Master Mariner generating a pre-arrival checklist.
            Vessel: {v_age}yr old {v_type}. Port: {v_port}.
            
            RULES:
            1. Output ONLY a raw, valid Python list of strings. Do not use markdown. Do not include introductory text.
            2. Provide exactly 6 highly specific, operational action items based on {v_port} vulnerabilities for this ship.
            3. Example format: ["Test emergency fire pump suction valve", "Verify OWS 15ppm alarm", "Check fire damper coaming flaps"]"""
            
            res_checklist = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": checklist_prompt}, {"role": "user", "content": "Generate checklist."}],
                temperature=0.0
            )
            
            # Safely parse the AI output into a Python list
            import ast
            try:
                raw_output = res_checklist.choices[0].message.content.strip()
                # Clean up if the AI accidentally adds markdown code blocks
                if raw_output.startswith("```"):
                    raw_output = raw_output.split("\n", 1)[1].rsplit("\n", 1)[0]
                checklist_items = ast.literal_eval(raw_output)
            except:
                # Fallback if parsing fails
                checklist_items = ["Verify OWS calibration", "Test emergency fire pump", "Inspect fire dampers", "Check ECDIS passage plan"]

            # --- THE INTERACTIVE COMPLIANCE WORKFLOW ---
            st.markdown("---")
            st.subheader("📋 Digital SMS Compliance Workflow")
            st.info("Execute checklist, document findings, and submit to DPA for arrival clearance.")
            
            with st.form("compliance_report_form"):
                st.markdown("#### Engine & Deck Pre-Arrival Audit")
                
                # Dynamically generate checkboxes and comment fields based on AI output
                for i, item in enumerate(checklist_items):
                    col_check, col_comment = st.columns([2, 2])
                    with col_check:
                        st.checkbox(f"**{item}**", key=f"check_{i}")
                    with col_comment:
                        st.text_input("Deficiency/Action Taken (Optional)", key=f"comment_{i}", placeholder="e.g., Valve greased, operating normally.")
                
                st.markdown("---")
                st.markdown("#### Final Observations & Master's Authorization")
                
                # Compulsory Fleet Feedback
                overall_comments = st.text_area("Chief Engineer / Chief Officer Final Observations (Required)", placeholder="Summarize overall readiness state...")
                
                # Authorization Block
                col_sig, col_pin = st.columns(2)
                master_sig = col_sig.text_input("Master's E-Signature")
                master_pin = col_pin.text_input("Master's Authorization PIN", type="password")
                
                submit_report = st.form_submit_button("Transmit DPA Arrival Report", type="primary", use_container_width=True)
                
                if submit_report:
                    if not overall_comments or not master_sig or not master_pin:
                        st.error("Submission Failed: Final observations, E-Signature, and Master's PIN are compulsory.")
                    elif master_pin != "1234": # Hardcoded PIN for the beta demo
                        st.error("Submission Failed: Invalid Master's Authorization PIN.")
                    else:
                        st.success("✅ Report securely transmitted to DPA Dashboard.")
                        st.balloons()
            
            # --- VERIFICATION AUDIT TRAIL ---
            with st.expander("🔍 Verify AI Intelligence (Raw Database Context)"):
                st.info("VectorPrime generated this forecast strictly from the following encrypted data chunks extracted from your uploaded maritime PDFs:")
                st.text(ctx)
                
        except Exception as e:
            st.error(str(e))



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
