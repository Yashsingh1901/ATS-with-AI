import os
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import json
from scipy.spatial.distance import cosine

# Define paths
DATA_DIR = os.path.join("Resume-Ranking-System", "data", "kaggle", "data", "data")
PROCESSED_DIR = os.path.join("Resume-Ranking-System", "data", "kaggle", "processed")
MODEL_DIR = os.path.join("Resume-Ranking-System", "models")

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def create_job_description(category):
    """Create a job description from the category name."""
    # More detailed job descriptions for different categories
    descriptions = {
        "ACCOUNTANT": "We are looking for an experienced Accountant to join our finance team. The ideal candidate should have strong knowledge of accounting principles, financial reporting, tax preparation, and budget management. Experience with accounting software and Excel is essential.",
        
        "ENGINEER": "Seeking a skilled Engineer with proven experience in designing, developing and implementing solutions. Strong problem-solving abilities and technical expertise required. Must be able to work in collaborative environments and communicate complex concepts effectively.",
        
        "INFORMATION-TECHNOLOGY": "Looking for an IT professional with experience in system administration, network security, and troubleshooting. Knowledge of cloud platforms, programming languages, and database management is highly desirable.",
        
        "HEALTHCARE": "Seeking a dedicated healthcare professional with strong patient care experience. Must have relevant certification, compassionate approach to patient care, and ability to work in fast-paced environments.",
        
        "SALES": "We need a results-driven Sales professional with proven track record in meeting and exceeding targets. Strong communication, negotiation and relationship building skills are essential. Experience in CRM systems preferred."
    }
    
    # Return specific description if available, otherwise a generic one
    return descriptions.get(category, f"Looking for a {category.lower().replace('-', ' ')} professional with relevant experience and skills in this field. The ideal candidate should have strong problem-solving abilities and excellent communication skills.")

def calculate_realistic_score(resume_embedding, jd_embedding, random_factor=0.15):
    """Calculate a more realistic match score based on cosine similarity with some randomness."""
    # Base score on cosine similarity
    similarity = 1 - cosine(resume_embedding, jd_embedding)
    
    # Add some randomness to create variety
    # The random factor determines how much variation we want
    import random
    random_adjustment = random.uniform(-random_factor, 0)
    
    # Ensure score stays between 0 and 1
    score = max(0.0, min(1.0, similarity + random_adjustment))
    return score

def process_kaggle_dataset():
    """Process the Kaggle resume dataset."""
    print("Processing Kaggle dataset...")
    
    # Initialize lists to store data
    resume_texts = []
    job_descriptions = []
    match_scores = []
    
    # Process each job category
    for category in os.listdir(DATA_DIR):
        category_path = os.path.join(DATA_DIR, category)
        if not os.path.isdir(category_path):
            continue
            
        print(f"Processing category: {category}")
        
        # Create job description for this category
        jd_text = create_job_description(category)
        jd_embedding = model.encode(jd_text)
        
        # Process resumes in this category
        for resume_file in os.listdir(category_path):
            if resume_file.endswith('.pdf'):
                resume_path = os.path.join(category_path, resume_file)
                resume_text = extract_text_from_pdf(resume_path)
                
                if resume_text:
                    # Get resume embedding
                    resume_embedding = model.encode(resume_text)
                    
                    # Calculate a realistic score instead of perfect 1.0
                    score = calculate_realistic_score(
                        resume_embedding.flatten(), 
                        jd_embedding.flatten()
                    )
                    
                    resume_texts.append(resume_text)
                    job_descriptions.append(jd_text)
                    match_scores.append(score)
                    
                    print(f"  - Processed resume: {resume_file} (Score: {score:.2f})")
    
    if not resume_texts:
        print("No resumes found in the dataset!")
        return
    
    # Create embeddings
    print("Creating embeddings for resumes...")
    resume_embeddings = model.encode(resume_texts)
    
    print("Creating embeddings for job descriptions...")
    jd_embeddings = model.encode(job_descriptions)
    
    # Combine features
    X = np.concatenate([resume_embeddings, jd_embeddings], axis=1)
    y = np.array(match_scores)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save processed data
    print("Saving processed data...")
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)
    
    # Save metadata
    score_distribution = {
        "min": float(min(match_scores)),
        "max": float(max(match_scores)),
        "mean": float(np.mean(match_scores)),
        "median": float(np.median(match_scores))
    }
    
    metadata = {
        'num_samples': len(resume_texts),
        'num_categories': len(os.listdir(DATA_DIR)),
        'embedding_dim': resume_embeddings.shape[1],
        'score_distribution': score_distribution
    }
    with open(os.path.join(PROCESSED_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Data preprocessing completed!")
    print(f"Total samples: {len(resume_texts)}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Score distribution: Min={score_distribution['min']:.2f}, Max={score_distribution['max']:.2f}, Mean={score_distribution['mean']:.2f}")

if __name__ == "__main__":
    process_kaggle_dataset()
