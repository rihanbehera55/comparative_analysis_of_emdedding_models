import time
import json
import torch
import numpy as np
import pandas as pd
import chromadb
import faiss
from transformers import BertTokenizer, BertModel
from sklearn.metrics import pairwise_distances

# Configuration
MODEL_NAME = 'bert-base-uncased'
FILE_PATH = 'healthcare setup/data/chunked_patient_data.jsonl'
K_NEIGHBORS = 5
MAX_CHROMA_BATCH = 5000 

print(f"Initializing BERT Research Environment for Healthcare")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
client = chromadb.Client()

def get_bert_embeddings(text_list):
    # Process in smaller internal batches to avoid OOM on CPU/GPU
    all_embeddings = []
    batch_size = 32 
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean Pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

def run_healthcare_analysis():
    print(f"Analyzing Dataset: chunked_patient_data with BERT")
    records = []
    with open(FILE_PATH, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    documents = df['rag_document'].tolist()
    ids = df['chunk_id'].tolist()
    total_records = len(documents)

    # 2. Generate BERT Embeddings (768-dim)
    print("Generating BERT embeddings...")
    start_enc = time.time()
    embeddings = get_bert_embeddings(documents).astype('float32')
    encoding_time = time.time() - start_enc
    print(f"Encoding completed in {encoding_time:.2f}s")

    # 3. Ground Truth
    query_vec = embeddings[0].reshape(1, -1)
    dist_matrix = pairwise_distances(query_vec, embeddings, metric='euclidean')
    true_indices = np.argsort(dist_matrix[0])[:K_NEIGHBORS].tolist()

    # 4. ChromaDB Indexing
    coll_name = f"health_bert_{int(time.time())}"
    collection = client.create_collection(name=coll_name)
    start_idx_chroma = time.time()
    for i in range(0, total_records, MAX_CHROMA_BATCH):
        batch_end = min(i + MAX_CHROMA_BATCH, total_records)
        collection.add(embeddings=embeddings[i:batch_end].tolist(), ids=ids[i:batch_end])
    chroma_idx_time = time.time() - start_idx_chroma

    # Query ChromaDB
    start_q_chroma = time.time()
    results = collection.query(query_embeddings=query_vec.tolist(), n_results=K_NEIGHBORS)
    chroma_query_ms = (time.time() - start_q_chroma) * 1000
    chroma_indices = [ids.index(cid) for cid in results['ids'][0]]
    chroma_recall = len(set(chroma_indices).intersection(set(true_indices))) / K_NEIGHBORS

    # 5. FAISS Indexing
    index = faiss.IndexFlatL2(embeddings.shape[1])
    start_idx_faiss = time.time()
    index.add(embeddings)
    faiss_idx_time = time.time() - start_idx_faiss

    # Query FAISS
    start_q_faiss = time.time()
    D, I = index.search(query_vec, K_NEIGHBORS)
    faiss_query_ms = (time.time() - start_q_faiss) * 1000
    faiss_recall = len(set(I[0].tolist()).intersection(set(true_indices))) / K_NEIGHBORS

    return pd.DataFrame({
        "Dataset": ["Healthcare (BERT)", "Healthcare (BERT)"],
        "Engine": ["ChromaDB", "FAISS"],
        "Recall@K": [f"{chroma_recall*100}%", f"{faiss_recall*100}%"],
        "Query Latency (ms)": [f"{chroma_query_ms:.3f}", f"{faiss_query_ms:.3f}"],
        "Indexing Time (s)": [f"{chroma_idx_time:.3f}", f"{faiss_idx_time:.3f}"]
    })

if __name__ == "__main__":
    report = run_healthcare_analysis()
    print("\n" + "="*60 + "\nBERT HEALTHCARE REPORT\n" + "="*60)
    print(report.to_string(index=False))