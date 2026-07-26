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
# MAIN UI CONTROLS 
# ==========================================
st.title("⚓ VECTOR OS: PSC INTELLIGENCE")
st.markdown("Data-centric target deficiency prediction.")

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

# --- SESSION STATE FIX: Prevents preloading ---
if "audit_started" not in st.session_state:
    st.session_state.audit_started = False

if run_audit:
    st.session_state.audit_started = True

# ==========================================
# DUAL-STREAM AI ENGINE
# ==========================================
def generate_audit_intelligence(vessel_type, vessel_age, target_port, regime, groq_key):
    ui_table = ""
    detailed_pdf_text = ""
    
    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            
            # Stream 1: UI Minimal Table
            prompt_ui = f"Vessel: {vessel_age}-year-old {vessel_type}. Port: {target_port} ({regime}). Identify TOP 3 PSC detention risks. NO conversational filler. Format EXACTLY as a Markdown table: \n| Code | Vulnerability | Immediate Action |\n|---|---|---|"
            response_ui = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_ui}],
                temperature=0.1,
                max_tokens=300
            )
            ui_table = response_ui.choices[0].message.content
            
            # Stream 2: PDF Comprehensive Dossier
            prompt_pdf = f"""
            Act as a Senior Marine Superintendent writing a highly detailed pre-arrival audit briefing for the Master of a {vessel_age}-year-old {vessel_type} bound for {target_port} ({regime}).
            Write 3 detailed paragraphs identifying the top 3 specific vulnerabilities. 
            For each vulnerability, include:
            1. The specific SOLAS or MARPOL regulation reference.
            2. The root cause of why this fails on {vessel_type}s of this age.
            3. A detailed, step-by-step corrective action plan for the Chief Engineer or Chief Officer.
            Do not use markdown tables. Write in professional, formal maritime paragraphs.
            """
            response_pdf = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_pdf}],
                temperature=0.2,
                max_tokens=1000
            )
            detailed_pdf_text = response_pdf.choices[0].message.content
            
            return ui_table, detailed_pdf_text
        except Exception:
            pass 

    # Fallback Data
    ui_table = "| Code | High-Risk Vulnerability | Corrective Action |\n|---|---|---|\n| **LSA 01** | Lifeboat on-load release mechanism failure | Drop test mechanism & verify overhaul certs |\n| **MAR 02** | 15 PPM OWS 3-way valve malfunction | Flush optical sensor, test auto-recirculation |"
    detailed_pdf_text = f"Regulatory Analysis for {vessel_type}:\n\n1. Life Saving Appliances (SOLAS Ch. III):\nHistorical data shows vessels of {vessel_age} years experience severe corrosion in lifeboat limit switches and release gear cables. The Chief Officer must conduct a physical drop test and verify all greasing points are free of hardened debris.\n\n2. MARPOL Annex I Compliance:\nThe USCG and Paris MOU heavily target the Oily Water Separator on older tonnage. The Chief Engineer must physically flush the 15 PPM optical sensor with freshwater and manually test the automatic 3-way recirculation valve prior to entering territorial waters. Ensure the Oil Record Book Part 1 entries perfectly match sounding logs."
    
    return ui_table, detailed_pdf_text

# ==========================================
# ENTERPRISE PDF GENERATOR CLASS
# ==========================================
class PDFReport(FPDF):
    def header(self):
        # 1. Background Watermark (Rendered first so it sits behind text)
        self.set_font('Arial', 'B', 65)
        self.set_text_color(240, 245, 250)  # Very faint, professional blue-grey
        self.set_y(130)
        self.cell(0, 10, 'VECTOR OS', 0, 0, 'C')
        
        # 2. Reset Y position back to top for the header text
        self.set_y(10)
        
        # 3. Proper Header Text
        self.set_font('Arial', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'VECTOR OS - CONFIDENTIAL FLEET INTELLIGENCE', 0, 1, 'R')
        
        # 4. Fixed Header Underline (The typo was here!)
        self.line(10, 20, 200, 20) 

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()} | Generated by Vector OS Predictive Engine', 0, 0, 'C')

# ==========================================
# DASHBOARD LOGIC (Gated by Session State)
# ==========================================
if st.session_state.audit_started and imo_input:
    if df.empty:
        st.error("Database unavailable.")
    else:
        vessel_match = df[df["IMO_Number"] == imo_input]
        
        if not vessel_match.empty:
            vessel = vessel_match.iloc[0]
            vessel_name = str(vessel.get("Vessel_Name", "Unknown Vessel"))
            vessel_type = str(vessel.get("Vessel_Type", "Bulk Carrier"))
            raw_year = vessel.get("Year_Built", 2014)
            year_built = int(float(raw_year)) if pd.notna(raw_year) and str(raw_year).upper() != "N/A" else 2014
            vessel_age = datetime.now().year - year_built
            base_risk = float(vessel.get("Risk_Score", 68.5))
            adjusted_risk = min(round(base_risk * port_info["multiplier"], 1), 98.0)
            
            st.divider()

            # --- 1. METRICS ---
            st.subheader(f"📊 Fleet Data: {vessel_name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Type", vessel_type)
            c2.metric("Age", f"{vessel_age} Years")
            c3.metric("Regime", port_info["regime"])
            c4.metric("Risk Index", f"{adjusted_risk}%")

            # --- 2. DUAL-STREAM GENERATION ---
            st.subheader("🚨 Priority Target Matrix")
            with st.spinner("Processing Marine Intelligence Dossier..."):
                ui_table, pdf_dossier = generate_audit_intelligence(vessel_type, vessel_age, selected_port, port_info["regime"], groq_api_key)
            st.markdown(ui_table)

            st.divider()

            # --- 3. Q&A SECTION ---
            st.subheader("Terminal: Regulation Query")
            with st.form("qna_form"):
                st.caption("Enter deficiency code or regulation focus:")
                query_input = st.text_input("Query:", label_visibility="collapsed")
                submit_query = st.form_submit_button("Query Vector AI")
                
                if submit_query and query_input:
                    if groq_api_key:
                        try:
                            client = Groq(api_key=groq_api_key)
                            response = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[
                                    {"role": "system", "content": "You are a raw data terminal. Output strict, technical maritime compliance data. Use bullets. NO conversational filler."},
                                    {"role": "user", "content": query_input}
                                ],
                                temperature=0.1,
                                max_tokens=400
                            )
                            st.info(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"API Error: {e}")
                    else:
                        st.warning("Enter API Key to run live queries.")

            st.divider()

            # --- 4. FINANCIAL ENGINE ---
            expected_exposure = 50000 * (adjusted_risk / 100.0)
            st.error(f"**Calculated Detention Exposure:** ${expected_exposure:,.2f} / day")
            
            # --- 5. DETAILED ENTERPRISE PDF GENERATION ---
            def create_detailed_pdf():
                pdf = PDFReport()
                pdf.add_page()
                
                # Cover Page Elements
                pdf.set_font("Arial", 'B', 20)
                pdf.ln(10)
                pdf.cell(0, 15, "PRE-ARRIVAL PSC AUDIT DOSSIER", ln=True, align='C')
                pdf.set_font("Arial", '', 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%d %B %Y')}", ln=True, align='C')
                pdf.ln(20)
                
                # Section 1: Vessel Particulars
                pdf.set_fill_color(240, 240, 240)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, " 1. VESSEL & VOYAGE PARTICULARS", ln=True, fill=True)
                pdf.set_font("Arial", '', 10)
                pdf.ln(2)
                pdf.cell(0, 6, f"  Vessel Name: {vessel_name}", ln=True)
                pdf.cell(0, 6, f"  IMO Number: {imo_input}", ln=True)
                pdf.cell(0, 6, f"  Vessel Profile: {vessel_age}-year-old {vessel_type}", ln=True)
                pdf.cell(0, 6, f"  Target Port & Regime: {selected_port}", ln=True)
                pdf.cell(0, 6, f"  Predicted Risk Index: {adjusted_risk}%", ln=True)
                pdf.ln(10)

                # Section 2: Financial Exposure
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, " 2. COMMERCIAL EXPOSURE CALCULATION", ln=True, fill=True)
                pdf.set_font("Arial", '', 10)
                pdf.ln(2)
                pdf.cell(0, 6, f"  Baseline Detention Cost: $50,000 USD / day", ln=True)
                pdf.set_text_color(200, 0, 0)
                pdf.cell(0, 6, f"  Adjusted Target Risk Exposure: ${expected_exposure:,.2f} USD / day", ln=True)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(10)
                
                # Page Break for Deep Dive
                pdf.add_page()
                
                # Section 3: AI Detailed Dossier
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f" 3. {port_info['regime'].upper()} TARGETED REGULATORY ANALYSIS", ln=True, fill=True)
                pdf.set_font("Arial", '', 10)
                pdf.ln(5)
                
                # Encode and format the detailed AI text properly for FPDF
                clean_text = pdf_dossier.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 6, clean_text)
                pdf.ln(10)

                # Section 4: Master's Verification
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, " 4. MASTER's ACKNOWLEDGEMENT", ln=True, fill=True)
                pdf.set_font("Arial", '', 10)
                pdf.ln(5)
                pdf.cell(0, 6, "I confirm that the above targeted vulnerabilities have been inspected and rectified.", ln=True)
                pdf.ln(15)
                pdf.cell(80, 6, "_______________________", ln=False)
                pdf.cell(80, 6, "_______________________", ln=True)
                pdf.cell(80, 6, "Master's Signature", ln=False)
                pdf.cell(80, 6, "Date / Ship's Stamp", ln=True)

                pdf_output = pdf.output(dest='S')
                return pdf_output.encode('latin1') if isinstance(pdf_output, str) else bytes(pdf_output)

            try:
                st.download_button(
                    label="📥 Export Comprehensive DPA Dossier (PDF)",
                    data=create_detailed_pdf(),
                    file_name=f"Vector_OS_Dossier_{imo_input}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            except Exception as e:
                st.error(f"PDF Error: {e}")
                
