import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from multiprocessing import Pool

# Load the dataset
file_path = 'Resume.csv'
df = pd.read_csv(file_path)

# Use a pre-trained model from sentence-transformers, with GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=device)  # Using a faster model
num_resumes = len(df)

# Print the number of resumes
print(f"Number of resumes in the CSV file: {num_resumes}")

# Step 1: Preprocess the Resumes - Convert Resume_str to embeddings in batches
def batch_encode(texts, model, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Use multiprocessing to speed up encoding
def parallel_encode(batch):
    return model.encode(batch)

def encode_resumes_multiprocessing(texts, model, batch_size=64, num_processes=4):
    with Pool(processes=num_processes) as pool:
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        embeddings = pool.map(parallel_encode, batches)
    return np.vstack(embeddings)

# Encode the resumes
resume_embeddings = encode_resumes_multiprocessing(df['Resume_str'].tolist(), model, batch_size=64, num_processes=4)

# Step 2: Store Resumes in a Vector Database (using FAISS with CPU or GPU)
d = resume_embeddings.shape[1]
nlist = 50  # Reduced number of clusters
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train the index
index.train(resume_embeddings)
index.add(resume_embeddings)

# Optional: Store metadata like IDs if needed
id_map = {i: df.iloc[i]['ID'] for i in range(len(df))}

# Step 3: Build a Custom GPT-like Search System
def search_resume(query, top_k=5):
    # Convert query to embedding
    query_embedding = model.encode([query])
    
    # Search the index
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the best matches
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        result = {
            'ID': id_map[idx],
            'Resume_str': df.iloc[idx]['Resume_str'],
            'Category': df.iloc[idx]['Category'],
            'Distance': distances[0][i]
        }
        results.append(result)
    
    return results

# Example query
query = "machine learning with web  skill  should be in Python and javascript with 5 years of experience category engineering"
results = search_resume(query) 

# Print the results
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"ID: {result['ID']}")
    print(f"Resume_str: {result['Resume_str']}")
    print(f"Category: {result['Category']}")
    print(f"Distance: {result['Distance']}\n")
