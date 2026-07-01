import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid


def get_pinecone_client():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc

def ingest_data_to_pinecone(text, source_name):
    """Chunks dynamic PDF text and upserts to Pinecone via Cloud Inference."""
    pc = get_pinecone_client()
    index_name = "maritime-regulations"
    
    # Check if index exists, if not create it with 1024 dimensions for e5-large
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1024, 
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    index = pc.Index(index_name)
        
    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        return "No text chunks to ingest!"
        
    # Embed and Upsert in batches to prevent network timeouts
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        
        # Using Pinecone's Serverless Cloud Embedding
        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=batch_chunks,
            parameters={"input_type": "passage"}
        )
        
        vectors = []
        for j, chunk in enumerate(batch_chunks):
            unique_id = f"{source_name}_chunk_{i+j}_{str(uuid.uuid4())[:8]}"
            vectors.append({
                "id": unique_id,
                "values": embeddings[j].values,
                "metadata": {"text": chunk, "source": source_name}
            })
            
        index.upsert(vectors=vectors)
        
    return f"Successfully ingested {len(chunks)} tactical data chunks from {source_name}!"

def query_pinecone(query_text, n_results=3):
    """Searches Pinecone for relevant context using Cloud Inference."""
    pc = get_pinecone_client()
    index = pc.Index("maritime-regulations")
    
    # Create embedding for the query
    query_emb = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query_text],
        parameters={"input_type": "query"}
    )
    
    # Search Pinecone
    results = index.query(
        vector=query_emb[0].values,
        top_k=n_results,
        include_metadata=True
    )
    
    # Extract the text chunks
    if results and 'matches' in results:
        context = [match['metadata']['text'] for match in results['matches']]
        return "\n\n".join(context)
    return "No context found."
