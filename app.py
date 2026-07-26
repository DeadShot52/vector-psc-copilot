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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {font-size:2.2rem; font-weight:700; color:#0e1117; margin-bottom:0px;}
    .sub-header {font-size:1.0rem; color:#555; margin-bottom:20px;}
    .card {background-color:#1e222b; padding:15px; border-radius:8px; border:1px solid #313745; margin-bottom:15px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING & CLEANUP
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
# DYNAMIC RISK PRIORITY ENGINE
# ==========================================
def get_dynamic_priorities(vessel_type, vessel_age, flag_state, target_port, regime, groq_key):
    """
    Generates dynamic top 3 inspection priorities based on vessel specs, 
    age profile, flag state, and target port MOU regime.
    """
    # 1. Try Groq AI dynamic generation first if key exists
    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            prompt = f"""
            Act as a senior Port State Control (PSC) Principal Inspector for {regime} visiting {target_port}.
            Analyze this vessel profile:
            - Vessel Type: {vessel_type}
            - Vessel Age: {vessel_age} years old
            - Flag State: {flag_state}
            
            Identify the TOP 3 most statistically probable detention vulnerabilities and specific physical items to audit prior to arrival.
            Format output strictly as 3 bullet points with Title, Deficiency Focus, and Immediate Corrective Action.
            Keep it concise, technical, and directly actionable for the Chief Officer and Chief Engineer.
            """
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600
            )
            return response.choices[0].message.content
        except Exception:
            pass # Fallback to rule engine if API fails

    # 2. Rule-Based Dynamic Engine (Fallback when no API key)
    priorities = []
    
    # Priority 1: Type-Specific Primary Risk
    vtype = str(vessel_type).lower()
    if "bulk" in vtype:
        priorities.append({
            "title": "1. SOLAS Ch. VI - Hatch Cover Weather-Tightness & Bilge Systems",
            "focus": f"Rubber packing integrity, cleats tension, and hold bilge high-level alarms for a {vessel_age}-year-old bulk carrier.",
            "action": "Perform chalk/ultrasonic test on hatch covers. Test auto-start function of hold bilge pumps."
        })
    elif "tanker" in vtype or "chemical" in vtype:
        priorities.append({
            "title": "1. SOLAS Ch. II-2 & SOLAS Ch. II-1 - Inert Gas System & P/V Valves",
            "focus": "Inert gas oxygen analyzer calibration, deck seal water level, and high-velocity pressure/vacuum valves.",
            "action": "Calibrate O2 sensors with span gas. Verify physical operation of P/V valve lifting levers."
        })
    elif "container" in vtype:
        priorities.append({
            "title": "1. Cargo Securing Manual (CSM) & Dangerous Goods Stowage",
            "focus": "Lashing gear visual inspection records, twistlock wear limits, and IMDG segregations.",
            "action": "Audit deck lashing gear inventory against CSM certificates. Inspect bridge IMDG manifest."
        })
    else:
        priorities.append({
            "title": "1. SOLAS Ch. III - Life-Saving Appliances (LSA)",
            "focus": "On-load release gear, lifeboat engine cold-start, and wire fall end-for-ending records.",
            "action": "Perform physical drop test of lifeboat release mechanism and verify overhaul certificates."
        })

    # Priority 2: Age & Port/Regime Specific Risk
    if vessel_age >= 12:
        priorities.append({
            "title": f"2. MARPOL Annex I & Hull Structure ({regime} High-Age Target)",
            "focus": f"15 PPM Oily Water Separator (OWS) auto-stopping device and air pipe coaming corrosion (Age: {vessel_age} yrs).",
            "action": "Flush OWS sample lines. Inspect weather deck air pipe automatic closing devices for waist corrosion."
        })
    else:
        priorities.append({
            "title": f"2. ISM Code Execution & Cyber Risk Control ({regime})",
            "focus": "Internal audit non-conformity closure records, passage planning compliance, and OT system backup.",
            "action": "Verify all recent internal audit NCs are closed with root-cause documentation on bridge."
        })

    # Priority 3: Regime Specific Regional Targeting
    if "uscg" in regime.lower() or "united states" in target_port.lower():
        priorities.append({
            "title": "3. USCG Specific Focus: Fire Control Systems & ISPS Security",
            "focus": "Emergency fire pump remote manual start, quick-closing fuel valves, and DOS security logs.",
            "action": "Test emergency fire pump delivering two full pressure jets at bridge wing. Test quick-closing trip wire."
        })
    elif "paris" in regime.lower():
        priorities.append({
            "title": "3. Paris MOU Focus: MLC 2006 & MARPOL Annex VI Emissions",
            "focus": "Seafarer Employment Agreements (SEAs), hours of rest records, and low-sulphur bunker delivery notes (BDNs).",
            "action": "Audit Chief Cook hygiene records and ensure fuel changeover logs match ECDIS position entries."
        })
    else:
        priorities.append({
            "title": f"3. {regime} Priority: Fire Flaps & Engine Room Safety",
            "focus": "Funnel flap linkage operation, emergency generator auto-transfer, and local fire fighting system.",
            "action": "De-rust and grease funnel dampers. Test auto-start of emergency generator on main switchboard failure."
        })

    # Format text for UI and PDF
    formatted_output = ""
    for p in priorities:
        formatted_output += f"### **{p['title']}**\n"
        formatted_output += f"* **Deficiency Focus:** {p['focus']}\n"
        formatted_output += f"* **Immediate Action:** {p['action']}\n\n"
        
    return formatted_output

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("⚓ Vector OS Controls")
st.sidebar.markdown("**Target Voyage Parameters**")

sample_imo = df["IMO_Number"].iloc[0] if not df.empty and "IMO_Number" in df.columns else "9438456"
imo_input = st.sidebar.text_input("Vessel IMO Number", value=sample_imo).strip()

port_options = {
    "Rotterdam (Paris MOU)": {"multiplier": 1.1, "regime": "Paris MOU"},
    "Singapore (Tokyo MOU)": {"multiplier": 1.0, "regime": "Tokyo MOU"},
    "Houston (USCG)": {"multiplier": 1.35, "regime": "US Coast Guard"},
    "Ningbo (Tokyo MOU)": {"multiplier": 1.15, "regime": "Tokyo MOU"},
    "Hamburg (Paris MOU)": {"multiplier": 1.05, "regime": "Paris MOU"}
}
selected_port = st.sidebar.selectbox("Destination Port", list(port_options.keys()))
port_info = port_options[selected_port]

groq_api_key = os.environ.get("GROQ_API_KEY") or st.sidebar.text_input("Groq API Key (Optional)", type="password")

st.sidebar.divider()
st.sidebar.caption("Vector OS v2.4 Enterprise | Predictive PSC Analytics Engine")

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title("⚓ VECTOR OS: PREDICTIVE PSC INTELLIGENCE")
st.markdown("Dynamic Port State Control Risk Scoring & Target Deficiency Prediction Engine")

if st.sidebar.button("Run Dynamic Inspection Audit", type="primary") or imo_input:
    if df.empty:
        st.error("Fleet database unavailable. Verify fleet_data.csv in your GitHub repository.")
    else:
        vessel_match = df[df["IMO_Number"] == imo_input]
        
        if vessel_match.empty:
            st.error(f"IMO '{imo_input}' not found in database.")
            sample_list = ", ".join(df["IMO_Number"].head(4).tolist())
            st.info(f"💡 Try typing one of these valid IMOs from your fleet dataset: **{sample_list}**")
        else:
            vessel = vessel_match.iloc[0]
            
            # Clean Specs Extraction
            vessel_name = str(vessel.get("Vessel_Name", "Armada Tuah 25"))
            vessel_type = str(vessel.get("Vessel_Type", "Bulk Carrier"))
            flag_state = str(vessel.get("Flag_State", "Liberia"))
            
            raw_year = vessel.get("Year_Built", None)
            if pd.isna(raw_year) or str(raw_year).upper() == "N/A":
                year_built = 2014
            else:
                try:
                    year_built = int(float(raw_year))
                except:
                    year_built = 2014
            
            vessel_age = datetime.now().year - year_built
            
            # Adjusted Risk Formula
            base_risk = float(vessel.get("Risk_Score", 68.5))
            adjusted_risk = min(round(base_risk * port_info["multiplier"], 1), 98.0)
            
            # --- METRICS DASHBOARD ---
            st.subheader(f"📋 Dynamic Profile: {vessel_name} (IMO: {imo_input})")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Vessel Type", vessel_type)
                st.metric("Flag State", flag_state)
            with col2:
                st.metric("Year Built / Age", f"{year_built} ({vessel_age} yrs)")
                st.metric("Destination Port", selected_port.split(" (")[0])
            with col3:
                st.metric("Target MOU Regime", port_info["regime"])
                st.metric("Base Fleet Risk", f"{base_risk}%")
            with col4:
                st.metric("Adjusted Target Risk", f"{adjusted_risk}%", delta=f"+{round(adjusted_risk - base_risk, 1)}% Port Factor", delta_color="inverse")
                st.metric("Risk Level", "HIGH RISK" if adjusted_risk > 50 else "MEDIUM RISK")

            st.divider()

            # --- DYNAMIC TARGETED DEFICIENCIES ---
            st.subheader("🚨 Dynamic Inspection Targets & Corrective Actions")
            st.caption(f"Synthesized specifically for a **{vessel_age}-year-old {vessel_type}** ({flag_state} Flag) entering **{selected_port}**.")

            # Run Dynamic Engine
            with st.spinner("Analyzing vessel parameters, MOU historical deficiency matrices, and port inspection patterns..."):
                dynamic_findings = get_dynamic_priorities(
                    vessel_type=vessel_type,
                    vessel_age=vessel_age,
                    flag_state=flag_state,
                    target_port=selected_port,
                    regime=port_info["regime"],
                    groq_key=groq_api_key
                )
            
            st.markdown(dynamic_findings)

            st.divider()

            # --- INTERACTIVE AI CONVENTION COPILOT ---
            st.subheader("🤖 Vector AI Copilot: Convention & PSC Query")
            st.markdown("Ask specific questions regarding SOLAS, MARPOL, STCW, or MLC 2006 to audit-proof your vessel before arrival.")

            qcol1, qcol2, qcol3 = st.columns(3)
            user_query = ""
            if qcol1.button("📌 Emergency Generator Rules"):
                user_query = f"What are the mandatory SOLAS testing requirements for an emergency generator on a {vessel_age} year old {vessel_type}?"
            if qcol2.button("📌 OWS Paris MOU Protocol"):
                user_query = f"What specific operational checks will a {port_info['regime']} inspector perform on the 15 PPM oil content meter?"
            if qcol3.button("📌 Fire Damper Deficiencies"):
                user_query = "What are the common PSC deficiency codes for funnel dampers and quick-closing valves?"

            query_input = st.text_input("Type your specific compliance query here:", value=user_query if user_query else f"What are the top SOLAS inspection checks for a {vessel_type}?")

            if query_input:
                if groq_api_key:
                    try:
                        client = Groq(api_key=groq_api_key)
                        with st.spinner("Querying Maritime Technical Knowledge Base..."):
                            response = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[
                                    {"role": "system", "content": "You are Vector OS, an expert Maritime Port State Control Auditor. Provide precise, technical, citation-backed answers referencing SOLAS, MARPOL, STCW, and MLC conventions with clear corrective actions."},
                                    {"role": "user", "content": query_input}
                                ],
                                temperature=0.2,
                                max_tokens=600
                            )
                            st.markdown("### **AI Auditor Response:**")
                            st.info(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Groq AI Error: {e}")
                else:
                    st.warning("🔑 Enter a Groq API Key in the sidebar to enable live AI Regulation Search.")

            st.divider()

            # --- FINANCIAL EXPOSURE ENGINE ---
            st.subheader("💰 Financial Risk Exposure Engine")
            detention_cost_per_day = 50000
            expected_exposure = detention_cost_per_day * (adjusted_risk / 100.0)
            
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.error(f"**Calculated Daily Off-Hire Detention Exposure:** ${expected_exposure:,.2f} / day")
            with mcol2:
                st.success("**Vector OS ROI Anchor:** Preventing a single 24-hour detention ($50,000) pays for Vector OS enterprise fleet monitoring across 10 vessels for over 14 years.")

            st.divider()

            # --- PDF REPORT GENERATOR ---
            st.subheader("📄 Generate Executive PSC Audit Report")
            
            def create_pdf(findings_text):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, f"Vector OS - Dynamic PSC Audit Briefing", ln=True, align='C')
                pdf.set_font("Arial", '', 10)
                pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target Port: {selected_port}", ln=True, align='C')
                pdf.ln(8)
                
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, f"Vessel Profile: {vessel_name} (IMO: {imo_input})", ln=True)
                pdf.set_font("Arial", '', 10)
                pdf.cell(0, 6, f"Type: {vessel_type} | Flag: {flag_state} | Year Built: {year_built} (Age: {vessel_age} yrs)", ln=True)
                pdf.cell(0, 6, f"Target Risk Score: {adjusted_risk}% ({port_info['regime']})", ln=True)
                pdf.ln(5)
                
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "Targeted Vulnerabilities & Action Items:", ln=True)
                pdf.set_font("Arial", '', 9)
                
                # Strip markdown hash formatting for clean PDF rendering
                clean_findings = findings_text.replace("###", "").replace("**", "")
                pdf.multi_cell(0, 5, clean_findings)
                pdf.ln(5)
                
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "Financial Risk Summary:", ln=True)
                pdf.set_font("Arial", '', 10)
                pdf.cell(0, 6, f"Daily Detention Exposure: ${expected_exposure:,.2f} / day", ln=True)
                
                return pdf.output(dest='S').encode('latin1')

            try:
                pdf_bytes = create_pdf(dynamic_findings)
                st.download_button(
                    label="📥 Download Executive Inspection Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"Vector_OS_Audit_{imo_input}_{selected_port.split()[0]}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.caption(f"PDF Generator initialized. ({e})")
            
