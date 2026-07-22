import streamlit as st
import pandas as pd
from fpdf import FPDF
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import PyPDF2
import uuid
import ast

# ==========================================
# 1. CORE ENGINE & OFFLINE PIPELINE
# ==========================================
def calculate_vector_risk_score(vessel_age, flag_state, target_port, past_deficiencies_count, vessel_type):
    risk_score = 0
    risk_factors = []

    # 1. Vessel Age Multiplier
    if vessel_age >= 15:
        risk_score += 20
        risk_factors.append(f"Critical Age ({vessel_age} years) -> +20 pts")
    elif vessel_age >= 10:
        risk_score += 10
        risk_factors.append(f"Moderate Age ({vessel_age} years) -> +10 pts")

    # 2. Flag State Matrix
    flag_bgw_mapping = {
        'Togo': 'BLACK', 'Comoros': 'BLACK', 
        'Panama': 'GREY', 'Liberia': 'GREY',
        'Marshall Islands': 'WHITE'
    }
    flag_status = flag_bgw_mapping.get(flag_state, 'WHITE') 
    
    if flag_status == 'BLACK':
        risk_score += 30
        risk_factors.append(f"Black-Listed Flag ({flag_state}) -> +30 pts")
    elif flag_status == 'GREY':
        risk_score += 15
        risk_factors.append(f"Grey-Listed Flag ({flag_state}) -> +15 pts")

    # 3. Port Stringency Matrix
    strict_ports = ['Rotterdam', 'Singapore', 'Houston', 'Brisbane']
    if any(strict_port in target_port for strict_port in strict_ports):
        risk_score += 20
        risk_factors.append(f"High-Stringency Target Port ({target_port}) -> +20 pts")

    # 4. Vessel Type Hazard Multiplier
    vessel_type_weights = {'Chemical Tanker': 15, 'Gas Carrier': 15, 'Bulk Carrier': 10, 'Container': 5}
    type_weight = vessel_type_weights.get(vessel_type, 0)
    if type_weight > 0:
        risk_score += type_weight
        risk_factors.append(f"High-Scrutiny Vessel Type ({vessel_type}) -> +{type_weight} pts")

    # 5. Historical Deficiency Penalty
    if past_deficiencies_count >= 5:
        risk_score += 15
        risk_factors.append(f"High Deficiency History ({past_deficiencies_count} found) -> +15 pts")

    risk_score = min(risk_score, 100)

    if risk_score >= 75:
        category = "CRITICAL RISK - High Probability of Detention"
    elif risk_score >= 45:
        category = "MODERATE RISK - Heightened Inspection Probability"
    else:
        category = "LOW RISK - Standard Clearance Expected"

    return risk_score, category, risk_factors

@st.cache_data
def load_vessel_database():
    try:
        return pd.read_csv("fleet_data.csv")
    except FileNotFoundError:
        # Fallback if the harvester hasn't been run yet
        return pd.DataFrame(columns=['IMO_Number', 'Vessel_Name', 'Vessel_Age', 'Flag_State', 'Vessel_Type', 'Past_Deficiencies'])

def generate_pdf_report(v_name, imo, score, category, ai_forecast):
    """Generates a downloadable PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="VECTORPRIME: PSC INTELLIGENCE REPORT", ln=True, align='C')
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Vessel: {v_name} (IMO: {imo})", ln=True)
    pdf.cell(200, 10, txt=f"Vector OS Risk Score: {score}/100", ln=True)
    pdf.cell(200, 10, txt=f"Category: {category}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="AI Predictive Forecast:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    # Clean text to prevent FPDF encoding errors with LLM emojis
    clean_text = ai_forecast.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 8, txt=clean_text)
    
    return pdf.output(dest="S").encode('latin-1')

vessel_db = load_vessel_database()

# ==========================================
# 2. CONFIGURATION & UI
# ==========================================
st.set_page_config(page_title="VectorPrime | PSC Intelligence", page_icon="⚓", layout="wide")

# ... [Keep your existing CSS markdown here for brevity] ...

try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index_name = "vector-core-v1"
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"SYSTEM OFFLINE: {str(e)}")
    st.stop()

# ==========================================
# 3. CORE APPLICATION: PSC DIGITAL TWIN
# ==========================================
st.title("⚓ VECTORPRIME: PREDICTIVE PSC INTELLIGENCE")
st.markdown("Enter an IMO number to query the offline database and calculate detention probabilities.")

if "vessel_data" not in st.session_state:
    st.session_state.vessel_data = None

col_search, col_btn = st.columns([3, 1])
target_imo = col_search.text_input("Vessel IMO Number", placeholder="e.g. 9123456")

# The New Offline Database Lookup
if col_btn.button("Query Database", use_container_width=True):
    if target_imo:
        try:
            target_vessel = vessel_db[vessel_db['IMO_Number'] == int(target_imo)]
            if not target_vessel.empty:
                st.session_state.vessel_data = {
                    "name": target_vessel['Vessel_Name'].values[0],
                    "type": target_vessel['Vessel_Type'].values[0], 
                    "age": int(target_vessel['Vessel_Age'].values[0]), 
                    "flag": target_vessel['Flag_State'].values[0],
                    "deficiencies": int(target_vessel['Past_Deficiencies'].values[0])
                }
                st.success(f"Vessel Found: {st.session_state.vessel_data['name']} (IMO {target_imo})")
            else:
                st.error("IMO not found in offline database. Ensure Harvester script has been run.")
        except ValueError:
            st.error("Please enter a valid numeric IMO Number.")
    else:
        st.warning("Input valid IMO.")

st.markdown("---")

# Only show the rest of the app if a vessel was found
if st.session_state.vessel_data:
    col1, col2, col3 = st.columns(3)
    v_type = col1.selectbox("Hull Type", [st.session_state.vessel_data["type"]], index=0)
    v_age = col2.number_input("Age (Years)", value=st.session_state.vessel_data["age"], disabled=True)
    v_port = col3.selectbox("Next Port of Call", ["USCG (Houston)", "Paris MoU (Rotterdam)", "MPA (Singapore)"])

    col4, col5, col6 = st.columns(3)
    v_flag = col4.selectbox("Flag State", [st.session_state.vessel_data["flag"]], index=0)
    v_class = col5.selectbox("Class Society", ["IACS", "Non-IACS"])
    v_def = col6.number_input("Past Deficiencies Count", value=st.session_state.vessel_data["deficiencies"], disabled=True)

    if st.button("Generate Detention Forecast", type="primary"):
        with st.spinner("Analyzing Database & Generating Risk Matrices..."):
            try:
                # 1. USE THE NEW PROPRIETARY RISK ENGINE
                score, category, factors = calculate_vector_risk_score(v_age, v_flag, v_port, v_def, v_type)
                
                st.metric(label="Vector OS Risk Index", value=f"{score}/100", delta=category, delta_color="inverse")
                with st.expander("View Mathematical Risk Drivers"):
                    for factor in factors:
                        st.warning(factor)

                # 2. RAG DATABASE RETRIEVAL
                search_query = f"Deficiencies for {v_type} under {v_flag} flag arriving in {v_port}"
                query_emb = pc.inference.embed(model="multilingual-e5-large", inputs=[search_query], parameters={"input_type": "query"})
                db_res = index.query(vector=query_emb[0].values, top_k=3, include_metadata=True)
                ctx = "\n".join([m['metadata']['text'] for m in db_res['matches']]) if db_res['matches'] else "No context found."

                # 3. FORECAST GENERATION
                sys_prompt = f"""You are the VectorPrime Predictive Risk Engine. Analyze a {v_age}yr old {v_type}, Flag: {v_flag}, Port: {v_port}. Context: {ctx}
                Provide a highly technical list of Inspector Focus Vectors and Top 3 Corrective Actions."""
                
                res_forecast = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Forecast risk."}],
                    temperature=0.0
                )
                
                ai_output = res_forecast.choices[0].message.content
                st.markdown(ai_output)
                
                # 4. GENERATE AND DOWNLOAD PDF
                pdf_bytes = generate_pdf_report(
                    st.session_state.vessel_data['name'], 
                    target_imo, score, category, ai_output
                )
                
                st.download_button(
                    label="📄 Download Official DPA Risk Report",
                    data=pdf_bytes,
                    file_name=f"VectorPrime_Report_{target_imo}.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
