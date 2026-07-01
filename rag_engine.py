import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

# Initialize embeddings (runs on Streamlit cloud for free)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_pinecone_client():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc

def ingest_data_to_pinecone(text, source_name):
    """Chunks dynamic PDF text and upserts to Pinecone."""
    pc = get_pinecone_client()
    index_name = "maritime-regulations"
    
    # Check if index exists, if not create it
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    index = pc.Index(index_name)
        
    # Chunk the text (Expanded for heavy PDFs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        return "No text chunks to ingest!"
        
    # Create embeddings and upsert
    vectors = []
    for i, chunk in enumerate(chunks):
        emb = embeddings.embed_query(chunk)
        unique_id = f"{source_name}_chunk_{i}_{str(uuid.uuid4())[:8]}"
        vectors.append({
            "id": unique_id,
            "values": emb,
            "metadata": {"text": chunk, "source": source_name}
        })
        
    index.upsert(vectors=vectors)
    return f"Successfully ingested {len(chunks)} tactical data chunks from {source_name}!"

def query_pinecone(query_text, n_results=3):
    """Searches Pinecone for relevant context."""
    pc = get_pinecone_client()
    index = pc.Index("maritime-regulations")
    
    # Create embedding for the query
    query_emb = embeddings.embed_query(query_text)
    
    # Search Pinecone
    results = index.query(
        vector=query_emb,
        top_k=n_results,
        include_metadata=True
    )
    
    # Extract the text chunks
    if results and 'matches' in results:
        context = [match['metadata']['text'] for match in results['matches']]
        return "\n\n".join(context)
    return "No context found."
