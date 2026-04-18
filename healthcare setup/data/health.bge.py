import time
import json
import numpy as np
import pandas as pd
import chromadb
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances

# Configuration
MODEL_NAME = 'BAAI/bge-small-en-v1.5'
FILE_PATH = 'healthcare setup/data/chunked_patient_data.jsonl'
K_NEIGHBORS = 5
MAX_CHROMA_BATCH = 5000 

print(f"🚀 Initializing BGE Research Environment")
# BGE-small is very memory efficient compared to BERT
model = SentenceTransformer(MODEL_NAME)

# Using PersistentClient to save RAM by writing to disk
client = chromadb.PersistentClient(path="./bge_healthcare_indices")

def run_healthcare_analysis():
    print(f"Analyzing Dataset: chunked_patient_data")
    
    # 1. Load clinical JSONL data
    records = []
    with open(FILE_PATH, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    documents = df['rag_document'].tolist()
    ids = df['chunk_id'].tolist()
    total_records = len(documents)
    print(f"Loaded {total_records} clinical chunks.")

    # 2. Generate BGE Embeddings (768-dim)
    print("Generating BGE embeddings...")
    start_enc = time.time()
    # normalize_embeddings is recommended for BGE models
    embeddings = model.encode(
        documents, 
        normalize_embeddings=True, 
        show_progress_bar=True, 
        batch_size=32  # Smaller batch size helps keep RAM usage stable
    ).astype('float32')
    encoding_time = time.time() - start_enc
    print(f"✅ Encoding completed in {encoding_time:.2f}s")

    # 3. Establish Ground Truth (Euclidean)
    query_vec = embeddings[0].reshape(1, -1)
    dist_matrix = pairwise_distances(query_vec, embeddings, metric='euclidean')
    true_indices = np.argsort(dist_matrix[0])[:K_NEIGHBORS].tolist()

    # 4. ChromaDB Performance with Batching
    coll_name = f"health_bge_{int(time.time())}"
    collection = client.get_or_create_collection(name=coll_name)
    
    print(f"Indexing into ChromaDB...")
    start_idx_chroma = time.time()
    for i in range(0, total_records, MAX_CHROMA_BATCH):
        batch_end = min(i + MAX_CHROMA_BATCH, total_records)
        collection.add(
            embeddings=embeddings[i:batch_end].tolist(),
            ids=ids[i:batch_end]
        )
    chroma_idx_time = time.time() - start_idx_chroma

    # Query ChromaDB
    start_q_chroma = time.time()
    results = collection.query(query_embeddings=query_vec.tolist(), n_results=K_NEIGHBORS)
    chroma_query_ms = (time.time() - start_q_chroma) * 1000

    # Calculate Recall
    chroma_retrieved_ids = results['ids'][0]
    chroma_indices = [ids.index(cid) for cid in chroma_retrieved_ids]
    chroma_recall = len(set(chroma_indices).intersection(set(true_indices))) / K_NEIGHBORS

    # 5. FAISS Performance
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    
    print(f"Indexing into FAISS...")
    start_idx_faiss = time.time()
    index.add(embeddings)
    faiss_idx_time = time.time() - start_idx_faiss

    # Query FAISS
    start_q_faiss = time.time()
    D, I = index.search(query_vec, K_NEIGHBORS)
    faiss_query_ms = (time.time() - start_q_faiss) * 1000
    faiss_recall = len(set(I[0].tolist()).intersection(set(true_indices))) / K_NEIGHBORS

    return pd.DataFrame({
        "Dataset": ["Healthcare (BGE)", "Healthcare (BGE)"],
        "Engine": ["ChromaDB", "FAISS"],
        "Recall@K": [f"{chroma_recall*100}%", f"{faiss_recall*100}%"],
        "Query Latency (ms)": [f"{chroma_query_ms:.3f}", f"{faiss_query_ms:.3f}"],
        "Indexing Time (s)": [f"{chroma_idx_time:.3f}", f"{faiss_idx_time:.3f}"]
    })

if __name__ == "__main__":
    final_report = run_healthcare_analysis()
    print("\n" + "="*60 + "\nBGE HEALTHCARE PERFORMANCE REPORT\n" + "="*60)
    print(final_report.to_string(index=False))