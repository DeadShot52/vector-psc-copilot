import streamlit as st
from groq import Groq
from pinecone import Pinecone
import uuid
import PyPDF2

# --- CONFIGURATION & 3D UI ---
st.set_page_config(page_title="Vector OS | Intelligence", page_icon="⚓", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0A0E17; color: #E0E6ED; }
    h1, h2, h3 { color: #00F2FE !important; font-family: 'Courier New', Courier, monospace; letter-spacing: 1.5px; text-shadow: 0 0 10px rgba(0, 242, 254, 0.4); }
    
    div[data-testid="stExpander"], div.stTextArea>div>div {
        background: linear-gradient(145deg, #111827, #0d121c) !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.9), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #E63946, #9b2226);
        color: white;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(230, 57, 70, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(230, 57, 70, 0.8);
        border: 1px solid #ff4d4d;
    }
    
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #111827 !important;
        color: #00F2FE !important;
        border: 1px solid #374151 !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.5) !important;
    }
    .stAlert { background-color: rgba(230, 57, 70, 0.1); border-left: 4px solid #E63946; }
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

# --- SIDEBAR: PIPELINE CONTEXT DATA ---
with st.sidebar:
    st.header("🛠️ Data Pipeline")
    st.caption("Upload regulatory PDFs. Select the Authority tag to prevent cross-contamination.")
    
    doc_source = st.selectbox("Document Authority Tag", ["USCG", "ABS", "Paris MoU", "General Compliance"])
    uploaded_file = st.file_uploader("Upload Rulebook (PDF)", type="pdf")
    
    if st.button("Process & Upload PDF", type="primary", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Executing secure chunking pipeline..."):
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    full_text = ""
                    for page in pdf_reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            full_text += extracted + "\n\n"
                    
                    chunk_size = 1000
                    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
                    
                    progress_bar = st.progress(0)
                    total_chunks = len(chunks)
                    
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) < 50: 
                            continue
                            
                        embedding = pc.inference.embed(
                            model="multilingual-e5-large",
                            inputs=[chunk],
                            parameters={"input_type": "passage", "truncate": "END"}
                        )
                        
                        rule_id = f"pdf-{doc_source.lower()}-{str(uuid.uuid4())[:8]}"
                        
                        index.upsert(
                            vectors=[{
                                "id": rule_id,
                                "values": embedding[0].values,
                                "metadata": {"text": chunk, "source": doc_source}
                            }]
                        )
                        progress_bar.progress((i + 1) / total_chunks)
                        
                    st.success(f"✅ Securely ingested {total_chunks} blocks tagged under [{doc_source}].")
                except Exception as e:
                    st.error(f"Pipeline Failure: {str(e)}")
        else:
            st.warning("Please upload a PDF file first.")

# --- MAIN DASHBOARD ---
st.title("⚓ VECTOR OS: MARITIME INTELLIGENCE")
st.markdown("Predictive Detention Intelligence for Port State Control. Powered by Cloud Vector RAG.")
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
    with st.spinner("Scanning Target Vector Space..."):
        # Explicit target calculation based on selection
        target_tag = "USCG" if "USCG" in destination_port else "Paris MoU" if "Paris" in destination_port else "ABS"
        search_query = f"What are the inspection targets, deficiencies, and rules for {target_tag}?"
        
        try:
            query_embedding = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[search_query],
                parameters={"input_type": "query"}
            )
            
            db_results = index.query(
                vector=query_embedding[0].values,
                top_k=4,
                include_metadata=True
            )
            
            if db_results['matches']:
                contexts = [match['metadata']['text'] for match in db_results['matches']]
                retrieved_context = "\n\n---\n\n".join(contexts)
            else:
                retrieved_context = "Standard core compliance frameworks active."

            system_prompt = f"""You are the Vector OS Predictive Intelligence Engine. 
Predict the top 3 high-risk Port State Control (PSC) deficiencies for a {vessel_age}-year-old {vessel_type} arriving under {destination_port} jurisdiction.

CRITICAL RULE: Rely heavily on the provided database context to formulate real, concrete targeting trends.

RETRIEVED DATABASE CONTEXT:
"{retrieved_context}"
"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate targeted pre-arrival audit matrix."}
                ],
                temperature=0.2
            )
            st.success("Target context locked.")
            st.markdown(response.choices[0].message.content)
                
        except Exception as e:
            st.error(f"API Interface Error: {str(e)}")

st.markdown("---")
st.subheader("3. SMS Verification Mode (Fortified)")
st.caption("🔒 **Zero-Retention Contextual Audit:** Input text is mapped directly to uploaded database vectors to cross-examine legality.")
doc_text = st.text_area("Paste SMS Segment / Operational Text here:", height=120)

if st.button("Audit Document against Target Port Criteria", use_container_width=True):
    if doc_text:
        with st.spinner("Executing real-time vector cross-examination..."):
            try:
                # FIXED: Section 3 now converts user input into a vector query
                audit_embedding = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=[doc_text],
                    parameters={"input_type": "query"}
                )
                
                audit_matches = index.query(
                    vector=audit_embedding[0].values,
                    top_k=3,
                    include_metadata=True
                )
                
                audit_context = "\n\n---\n\n".join([m['metadata']['text'] for m in audit_matches['matches']]) if audit_matches['matches'] else ""

                sys_prompt = f"""You are a strict, non-hallucinating Maritime Safety Auditor.
Evaluate the operational procedure submitted by the user. Cross-reference it directly against the verified regulatory knowledge base attached below.

VERIFIED REGULATORY KNOWLEDGE BASE:
\"\"\"{audit_context}\"\"\"

OUTPUT REQUIREMENT:
1. State clearly if the action is COMPLIANT or NON-COMPLIANT based on the text.
2. Cite specific regulation structures or reference rules mentioned in the context.
3. Provide the exact professional correction required to secure the ship from detention."""

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
                st.caption(f"🛡️ **Vector OS Secured** | Generation ID: VCT-{hash(doc_text) % 1000000}")
            except Exception as e:
                st.error(f"Audit Pipeline Error: {str(e)}")
    else:
        st.warning("Please paste some text first.")
