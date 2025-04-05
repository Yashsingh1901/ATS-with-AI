from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
import os
import joblib
from typing import List, Optional
import uvicorn
import sys
import traceback

app = FastAPI(title="AI Resume Ranking System")

# Add CORS middleware with explicit origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "null", "file://", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
model_path = os.path.join(project_root, "models", "resume_ranker.joblib")

# Print debug information
print(f"Current directory: {current_dir}")
print(f"Project root: {project_root}")
print(f"Model path: {model_path}")

try:
    # Load models
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SBERT model loaded successfully")
    
    if os.path.exists(model_path):
        ranker = joblib.load(model_path)
        print("ML model loaded successfully")
    else:
        print(f"Warning: Model file not found at {model_path}. Using fallback scoring.")
        ranker = None
except Exception as e:
    print(f"Error loading models: {e}")
    ranker = None

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            return text if text else "No text could be extracted from this PDF."
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return "Error extracting text from PDF."

def get_resume_embedding(resume_text: str) -> np.ndarray:
    """Get SBERT embedding for resume text."""
    return model.encode([resume_text])[0]

def get_jd_embedding(jd_text: str) -> np.ndarray:
    """Get SBERT embedding for job description text."""
    return model.encode([jd_text])[0]

def fallback_score(resume_embedding, jd_embedding):
    """Fallback scoring using cosine similarity."""
    from scipy.spatial.distance import cosine
    return 1 - cosine(resume_embedding, jd_embedding)

@app.get("/")
async def root():
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")

@app.post("/rank-resume")
async def rank_resume(
    resume: UploadFile = File(...),
    job_description: Optional[str] = Form(None)
):
    """Rank a resume against a job description."""
    try:
        print(f"Request received: file={resume.filename}, job_desc length={len(job_description) if job_description else 0}")
        
        if job_description is None or job_description.strip() == "":
            job_description = "Looking for a qualified candidate with relevant skills and experience."
            print("Using default job description")
        
        # Save uploaded resume temporarily
        temp_path = f"temp_{resume.filename}"
        print(f"Saving file to {temp_path}")
        
        with open(temp_path, "wb") as buffer:
            content = await resume.read()
            buffer.write(content)
            print(f"File saved, size: {len(content)} bytes")
        
        # Extract text from resume
        print("Extracting text from PDF...")
        resume_text = extract_text_from_pdf(temp_path)
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        print(f"Text extracted, length: {len(resume_text)} characters")
        
        # Get embeddings
        print("Generating embeddings...")
        resume_embedding = get_resume_embedding(resume_text)
        jd_embedding = get_jd_embedding(job_description)
        
        # Combine features
        features = np.concatenate([resume_embedding, jd_embedding])
        print("Features shape:", features.shape)
        
        # Get prediction
        print("Making prediction...")
        if ranker is not None:
            score = float(ranker.predict([features])[0])
            print(f"ML model prediction: {score}")
        else:
            # Fallback to cosine similarity if model not loaded
            score = float(fallback_score(resume_embedding, jd_embedding))
            print(f"Fallback score: {score}")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Temporary file {temp_path} removed")
        
        # Ensure score is between 0 and 1
        score = max(0, min(1, score))
        print(f"Final score: {score}")
        
        result = {
            "score": score,
            "resume_text": resume_text[:500] + "..."  # Return first 500 chars for preview
        }
        print("Returning result")
        return result
    
    except Exception as e:
        print(f"Error in rank_resume: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
