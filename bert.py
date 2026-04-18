import time
import torch
import numpy as np
import pandas as pd
import chromadb
import faiss
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import pairwise_distances

MODEL_NAME = 'bert-base-uncased'
SAMPLE_SIZE = 1000
K_NEIGHBORS = 5
DATASETS = ['ag_news', 'dbpedia_14']

print(f"Initializing Research Environment for BERT")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)

# Helper function to generate BERT embeddings as per the report's "deep" context logic
def get_bert_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Using the mean of the last hidden state to represent the document vector
    return outputs.last_hidden_state.mean(dim=1).numpy()

client = chromadb.Client()

def run_performance_analysis(dataset_name):
    print(f" Analyzing Dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split='train')
    df = pd.DataFrame(ds.select(range(SAMPLE_SIZE)))
    text_col = 'text' if 'text' in df.columns else 'content'
    documents = df[text_col].tolist()
    
    start_enc = time.time()
    embeddings = get_bert_embeddings(documents).astype('float32')
    encoding_time = time.time() - start_enc
    
    query_vec = embeddings[0].reshape(1, -1)
    dist_matrix = pairwise_distances(query_vec, embeddings, metric='euclidean')
    true_indices = np.argsort(dist_matrix[0])[:K_NEIGHBORS].tolist()
    
    coll_name = f"bench_{dataset_name.replace('_', '')}_{int(time.time())}"
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
    
    dim = embeddings.shape[1]
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
print("BERT PERFORMANCE ANALYSIS REPORT")
print("="*60)
print(final_report.to_string(index=False))