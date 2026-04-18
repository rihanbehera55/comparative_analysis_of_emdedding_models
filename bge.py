import time
import numpy as np
import pandas as pd
import chromadb
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances

# BGE-small-en-v1.5 Setup as per research paper
MODEL_NAME = 'BAAI/bge-small-en-v1.5'
SAMPLE_SIZE = 1000
K_NEIGHBORS = 5
DATASETS = ['ag_news', 'dbpedia_14']

print(f"Initializing Research Environment for BGE")
# BGE is compatible with SentenceTransformer library
model = SentenceTransformer(MODEL_NAME)
client = chromadb.Client()

def run_performance_analysis(dataset_name):
    print(f" Analyzing Dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split='train')
    df = pd.DataFrame(ds.select(range(SAMPLE_SIZE)))
    text_col = 'text' if 'text' in df.columns else 'content'
    documents = df[text_col].tolist()
    
    # BGE specific encoding
    start_enc = time.time()
    embeddings = model.encode(documents, normalize_embeddings=True).astype('float32')
    encoding_time = time.time() - start_enc
    
    query_vec = embeddings[0].reshape(1, -1)
    dist_matrix = pairwise_distances(query_vec, embeddings, metric='euclidean')
    true_indices = np.argsort(dist_matrix[0])[:K_NEIGHBORS].tolist()
    
    coll_name = f"bge_bench_{dataset_name.replace('_', '')}_{int(time.time())}"
    try:
        client.delete_collection(coll_name)
    except:
        pass
    
    collection = client.create_collection(name=coll_name)
    start_idx = time.time()
    collection.add(
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(SAMPLE_SIZE)]
    )
    chroma_idx_time = time.time() - start_idx
    
    start_q = time.time()
    results = collection.query(query_embeddings=query_vec.tolist(), n_results=K_NEIGHBORS)
    chroma_query_ms = (time.time() - start_q) * 1000
    chroma_ids = [int(i) for i in results['ids'][0]]
    chroma_recall = len(set(chroma_ids).intersection(set(true_indices))) / K_NEIGHBORS
    
    dim = embeddings.shape[1] # Will be 768
    index = faiss.IndexFlatL2(dim)
    start_idx = time.time()
    index.add(embeddings)
    faiss_idx_time = time.time() - start_idx
    
    start_q = time.time()
    D, I = index.search(query_vec, K_NEIGHBORS)
    faiss_query_ms = (time.time() - start_q) * 1000
    faiss_ids = I[0].tolist()
    faiss_recall = len(set(faiss_ids).intersection(set(true_indices))) / K_NEIGHBORS
    
    return pd.DataFrame({
        "Dataset": [dataset_name, dataset_name],
        "Engine": ["ChromaDB", "FAISS"],
        "Recall@K": [f"{chroma_recall*100}%", f"{faiss_recall*100}%"],
        "Query Latency (ms)": [f"{chroma_query_ms:.3f}", f"{faiss_query_ms:.3f}"],
        "Indexing Time (s)": [f"{chroma_idx_time:.3f}", f"{faiss_idx_time:.3f}"]
    })

final_report = pd.concat([run_performance_analysis(d) for d in DATASETS])
print("\n" + "="*60)
print("BGE PERFORMANCE ANALYSIS REPORT")
print("="*60)
print(final_report.to_string(index=False))