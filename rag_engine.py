import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize embeddings (runs on Streamlit cloud for free)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_pinecone_client():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc

def ingest_data_to_pinecone():
    """Reads knowledge_base.txt, chunks it, and upserts to Pinecone."""
    pc = get_pinecone_client()
    index_name = "maritime-regulations"
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    index = pc.Index(index_name)
    
    # Check if already populated
    stats = index.describe_index_stats()
    if stats['total_vector_count'] > 0:
        return "Knowledge base already ingested!"
        
    # Read the text file
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        text = f.read()
        
    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    # Create embeddings and upsert
    vectors = []
    for i, chunk in enumerate(chunks):
        emb = embeddings.embed_query(chunk)
        vectors.append({
            "id": f"chunk_{i}",
            "values": emb,
            "metadata": {"text": chunk}
        })
        
    index.upsert(vectors=vectors)
    return f"Successfully ingested {len(chunks)} chunks into Pinecone!"

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
    context = [match['metadata']['text'] for match in results['matches']]
    return "\n\n".join(context)
