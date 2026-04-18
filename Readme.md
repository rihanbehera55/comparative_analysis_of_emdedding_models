### INSTALLING THE DEPENDENCIES

pip install -r requirements.txt

### Run Benchmarks:

python all-MiniLM-L6-v2.py
python bert_analysis.py
python bge_analysis.py


### 📁 Repository Structure

1. all-MiniLM-L6-v2.py: Benchmark script for Sentence-Transformers.
2. bert_analysis.py: Implementation for BERT embeddings.
3. bge_analysis.py: Implementation for BGE models.
4. requirements.txt: Full list of verified library versions.
5. .gitignore: Configured to exclude heavy local caches (chroma_data, __pycache__).

### Future Work
1. Integration of the Healthcare JSONL dataset.

Author: Rihan Behera

Institution: Kalinga Institute of Industrial Technology (KIIT)