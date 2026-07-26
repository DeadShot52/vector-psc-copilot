import streamlit as st
import pandas as pd
from fpdf import FPDF
from groq import Groq
import os
from datetime import datetime
# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Vector OS | Predictive PSC Intelligence",
    page_icon="⚓",
    layout="wide"
)
# Minimalist B2B CSS
st.markdown("""
    <style>
    .stButton>button {width: 100%; font-weight: bold;}
    .report-table {font-size: 14px;}
    </style>
""", unsafe_allow_html=True)
# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_fleet_data():
    try:
        df = pd.read_csv("fleet_data.csv")
        df["IMO_Number"] = df["IMO_Number"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        return df
    except Exception as e:
        st.error(f"Error loading fleet_data.csv: {e}")
        return pd.DataFrame()
df = load_fleet_data()
# ==========================================
# MAIN UI CONTROLS (Moved from Sidebar for Mobile)
# ==========================================
st.title("⚓ VECTOR OS: PSC INTELLIGENCE")
st.markdown("Data-centric target deficiency prediction.")
# Configuration Box
with st.container(border=True):
    st.subheader("⚙️ Voyage Parameters")
    
    sample_imo = df["IMO_Number"].iloc[0] if not df.empty and "IMO_Number" in df.columns else "9438456"
    
    col_a, col_b = st.columns(2)
    with col_a:
        imo_input = st.text_input("Vessel IMO Number", value=sample_imo).strip()
    with col_b:
        port_options = {
            "Rotterdam (Paris MOU)": {"multiplier": 1.1, "regime": "Paris MOU"},
            "Singapore (Tokyo MOU)": {"multiplier": 1.0, "regime": "Tokyo MOU"},
            "Houston (USCG)": {"multiplier": 1.35, "regime": "US Coast Guard"}
        }
        selected_port = st.selectbox("Next Port of Call", list(port_options.keys()))
        port_info = port_options[selected_port]
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.text_input("Groq API Key (Optional)", type="password", placeholder="Enter key for live AI analytics")
    run_audit = st.button("RUN PREDICTIVE AUDIT", type="primary")
# ==========================================
# DYNAMIC ENGINE (MINIMAL DATA FORMAT)
# ==========================================
def get_data_centric_priorities(vessel_type, vessel_age, target_port, regime, groq_key):
    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            prompt = f"""
            Vessel: {vessel_age}-year-old {vessel_type}. Port: {target_port} ({regime}).
            Identify TOP 3 PSC detention risks. 
            RULES: NO conversational filler. NO paragraphs. 
            Format EXACTLY as a Markdown table: 

| Code | Vulnerability | Immediate Action |
| :--- | :--- | :--- |

            """
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception:
            pass # Fallback below
    # Minimal Fallback Table
    return f"""

| Code | High-Risk Vulnerability | Corrective Action |
| :--- | :--- | :--- |
| **LSA 01** | Lifeboat on-load release mechanism failure | Drop test mechanism & verify overhaul certs |
| **MAR 02** | 15 PPM OWS 3-way valve malfunction | Flush optical sensor, test auto-recirculation |
| **FIRE 03** | Funnel fire dampers frozen / corroded | De-rust linkages & physically test closure |

    """
# ==========================================
# DASHBOARD LOGIC
# ==========================================
if run_audit or imo_input:
    if df.empty:
        st.error("Database unavailable.")
    else:
        vessel_match = df[df["IMO_Number"] == imo_input]
        
        if vessel_match.empty:
            st.error("IMO not found.")
        else:
            vessel = vessel_match.iloc[0]
            
            # Clean Specs Extraction
            vessel_name = str(vessel.get("Vessel_Name", "Unknown Vessel"))
            vessel_type = str(vessel.get("Vessel_Type", "Bulk Carrier"))
            
            raw_year = vessel.get("Year_Built", 2014)
            year_built = int(float(raw_year)) if pd.notna(raw_year) and str(raw_year).upper() != "N/A" else 2014
            vessel_age = datetime.now().year - year_built
            
            base_risk = float(vessel.get("Risk_Score", 68.5))
            adjusted_risk = min(round(base_risk * port_info["multiplier"], 1), 98.0)
            
            st.divider()
            # --- 1. METRICS (Clean Data View) ---
            st.subheader(f"📊 Fleet Data: {vessel_name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Type", vessel_type)
            c2.metric("Age", f"{vessel_age} Years")
            c3.metric("Regime", port_info["regime"])
            c4.metric("Risk Index", f"{adjusted_risk}%", delta=f"{adjusted_risk - base_risk:.1f}% port impact", delta_color="inverse")
            # --- 2. DATA-CENTRIC PRIORITIES ---
            st.subheader("🚨 Priority Target Matrix")
            with st.spinner("Compiling target matrix..."):
                findings_table = get_data_centric_priorities(vessel_type, vessel_age, selected_port, port_info["regime"], groq_api_key)
            st.markdown(findings_table)
            st.divider()
            # --- 3. FIX: Q&A SECTION WITH DEDICATED SUBMIT BUTTON ---
            st.subheader("Terminal: Regulation Query")
            
            with st.form("qna_form"):
                st.caption("Enter deficiency code or regulation focus (e.g., 'Emergency generator SOLAS Ch II-1'):")
                query_input = st.text_input("Query:", label_visibility="collapsed")
                
                # Dedicated button inside the form
                submit_query = st.form_submit_button("Query Vector AI")
                
                if submit_query and query_input:
                    if groq_api_key:
                        try:
                            client = Groq(api_key=groq_api_key)
                            with st.spinner("Processing..."):
                                response = client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[
                                        {"role": "system", "content": "You are a raw data terminal. Output strict, technical maritime compliance data. Use bullets. NO conversational filler. NO introductions. NO conclusions."},
                                        {"role": "user", "content": query_input}
                                    ],
                                    temperature=0.1,
                                    max_tokens=400
                                )
                                st.info(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"API Error: {e}")
                    else:
                        st.warning("Please enter a Groq API Key above to run live queries.")
            st.divider()
            # --- 4. FINANCIAL ENGINE ---
            expected_exposure = 50000 * (adjusted_risk / 100.0)
            st.error(f"**Calculated Detention Exposure:** ${expected_exposure:,.2f} / day")
            
            # --- 5. FIX: PDF GENERATOR BYTE ERROR ---
            def create_pdf():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 8, f"Vector OS - Audit Briefing: {vessel_name}", ln=True)
                pdf.set_font("Arial", '', 10)
                pdf.cell(0, 6, f"Port: {selected_port} | Risk Score: {adjusted_risk}%", ln=True)
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(0, 6, "See web dashboard for granular deficiency matrix.", ln=True)
                
                # Fix for the bytearray error
                pdf_output = pdf.output(dest='S')
                if isinstance(pdf_output, str):
                    return pdf_output.encode('latin1')
                else:
                    return bytes(pdf_output)
            try:
                st.download_button(
                    label="📥 Export PDF Briefing",
                    data=create_pdf(),
                    file_name=f"Vector_Audit_{imo_input}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"PDF Error: {e}")
