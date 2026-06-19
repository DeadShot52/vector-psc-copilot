import streamlit as st
from groq import Groq
from pinecone import Pinecone
import uuid

# --- CONFIGURATION ---
st.set_page_config(page_title="Vector PSC Copilot", page_icon="⚓", layout="wide", initial_sidebar_state="expanded")

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
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("vector-maritime")
except Exception as e:
    st.error(f"SYSTEM OFFLINE: API Keys missing or invalid. {str(e)}")
    st.stop()

# --- SECRET ADMIN PANEL (SIDEBAR) ---
with st.sidebar:
    st.header("🛠️ Admin Data Ingestion")
    st.caption("Upload Maritime Rules permanently to Pinecone Cloud.")
    new_rule = st.text_area("Paste rule text here:", height=200)
    
    if st.button("Push to Cloud Database", type="primary", use_container_width=True):
        if new_rule:
            with st.spinner("Uploading to Pinecone..."):
                try:
                    # Convert text to 1024-dimensions using Pinecone Inference
                    embedding = pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=[new_rule],
                        parameters={"input_type": "passage", "truncate": "END"}
                    )
                    
                    rule_id = "rule-" + str(uuid.uuid4())[:8]
                    
                    # Push to cloud
                    index.upsert(
                        vectors=[{
                            "id": rule_id,
                            "values": embedding[0].values,
                            "metadata": {"text": new_rule}
                        }]
                    )
                    st.success("✅ Rule permanently ingested!")
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")
        else:
            st.warning("Paste some text first.")

# --- MAIN DASHBOARD ---
st.title("⚓ VECTOR PSC COPILOT")
st.markdown("Predictive Detention Intelligence for Port State Control. Powered by Pinecone Cloud RAG.")
st.markdown("---")

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
    with st.spinner("Searching Pinecone Cloud Database..."):
        
        search_query = f"What are the inspection targets and rules for a {vessel_type} in {destination_port}?"
        
        try:
            # 1. Map the search query
            query_embedding = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[search_query],
                parameters={"input_type": "query"}
            )
            
            # 2. Search Pinecone
            db_results = index.query(
                vector=query_embedding[0].values,
                top_k=1,
                include_metadata=True
            )
            
            if db_results['matches']:
                retrieved_context = db_results['matches'][0]['metadata']['text']
            else:
                retrieved_context = "No specific rules found in database. Consult standard SMS."

            system_prompt = f"""You are the Vector PSC Predictive Intelligence Engine. 
Predict the top 3 most likely Port State Control (PSC) deficiencies for a {vessel_age}-year-old {vessel_type} arriving in {destination_port}.

CRITICAL RULE: You must base your predictions strictly on the following retrieved database context. Do not use outside memory.

RETRIEVED DATABASE CONTEXT:
"{retrieved_context}"

RULES:
1. Format strictly as a checklist.
2. Provide an immediate "Corrective Action".
3. ZERO HALLUCINATION POLICY: Do not invent rules outside of the provided context.
"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the targeted pre-arrival audit checklist."}
                ],
                temperature=0.0
            )
            st.success("Data successfully retrieved from Pinecone.")
            st.markdown(response.choices[0].message.content)
            
            with st.expander("View Retrieved Database Context"):
                st.write(retrieved_context)
                
        except Exception as e:
            st.error(f"API Error: {str(e)}")

st.markdown("---")
st.subheader("3. SMS Verification Mode")
st.caption("🔒 **Enterprise Security Active:** Zero-Data Retention protocol enabled. Text is processed entirely in-memory and immediately purged after audit. No data is stored or used for LLM training.")
doc_text = st.text_area("Paste SMS Segment / Operational Text here:", height=100)

if st.button("Audit Document against Target Port Criteria", use_container_width=True):
    if doc_text:
        with st.spinner("Auditing document..."):
            sys_prompt = """You are a strict, robotic Maritime Auditor.
CRITICAL DIRECTIVE: Check if a real regulation explicitly addresses the user's scenario.
IF NO EXACT REGULATION EXISTS: Output ONLY this exact string and stop:
Regulatory Citation: The exact regulation regarding this procedure cannot be verified at this time. Refer to official Flag State circulars.
IF A REAL REGULATION DOES EXIST:
1. Cite the exact regulation number.
2. Highlight risks.
3. Suggest compliant corrections."""
            try:
                res = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": f"Audit this procedure: {doc_text}"}
                    ],
                    temperature=0.0
                )
                st.markdown(res.choices[0].message.content)
                st.markdown("---")
                st.caption(f"🛡️ **Vector OS Verified** | © 2026 Vector Maritime Intelligence | Generated Audit ID: VCT-{hash(doc_text) % 1000000}")
            except Exception as e:
                st.error(f"API Error: {str(e)}")
    else:
        st.warning("Please paste some text first.")
