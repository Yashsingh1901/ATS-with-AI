import os
import numpy as np

EMBEDDING_DIR = "backend/embeddings"
COMPANIES = ["Google", "Atlassian", "Microsoft", "Amazon"]

def check_embeddings():
    for company in COMPANIES:
        file_path = os.path.join(EMBEDDING_DIR, f"{company.lower()}_embeddings.npy")
        
        if os.path.exists(file_path):
            embeddings = np.load(file_path)
            print(f"✅ {company}: Loaded {embeddings.shape[0]} embeddings, Shape: {embeddings.shape}")
            print(f"   Sample: {embeddings[0][:5]}...\n")  # Print first 5 values of the first embedding
        else:
            print(f"❌ {company}: Embeddings file not found!")

if __name__ == "__main__":
    check_embeddings()
