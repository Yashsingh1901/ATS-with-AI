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
import re
from sklearn.feature_extraction.text import CountVectorizer

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

def extract_resume_sections(resume_text: str) -> dict:
    """Extract different sections from a resume."""
    sections = {}
    
    # Simple pattern-based section extraction - you can make this more sophisticated
    education_pattern = r'(education|academic|degree|university|college|school)'
    experience_pattern = r'(experience|work|employment|job|career|professional)'
    skills_pattern = r'(skills|expertise|proficiency|competenc(y|ies)|technical|technologies)'
    
    # Find sections - making the entire pattern case-insensitive with (?i) at the start
    sections['education'] = re.findall(r'(?i)' + education_pattern + r'[^.]*(?:\.\s*[^.]*){0,3}\.', resume_text)
    sections['experience'] = re.findall(r'(?i)' + experience_pattern + r'[^.]*(?:\.\s*[^.]*){0,3}\.', resume_text)
    sections['skills'] = re.findall(r'(?i)' + skills_pattern + r'[^.]*(?:\.\s*[^.]*){0,3}\.', resume_text)
    
    # Extract sections if not found by the patterns above (backup)
    if not sections['education'] and 'education' in resume_text.lower():
        sections['education'] = ['Education section detected but not fully extracted']
    if not sections['experience'] and 'experience' in resume_text.lower():
        sections['experience'] = ['Experience section detected but not fully extracted']
    if not sections['skills'] and 'skills' in resume_text.lower():
        sections['skills'] = ['Skills section detected but not fully extracted']
    
    return sections

def extract_keywords(job_description: str) -> List[str]:
    """Extract keywords from job description."""
    # Create and configure a CountVectorizer to extract key terms
    vectorizer = CountVectorizer(
        stop_words='english',
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 2)
    )
    
    # Fit and transform the job description
    X = vectorizer.fit_transform([job_description])
    
    # Get feature names and their frequencies
    feature_names = vectorizer.get_feature_names_out()
    frequencies = X.toarray()[0]
    
    # Sort keywords by frequency and return top 10
    sorted_indices = frequencies.argsort()[::-1]
    top_keywords = [feature_names[idx] for idx in sorted_indices[:15]]
    
    return top_keywords

def check_keyword_presence(resume_text: str, keywords: List[str]) -> dict:
    """Check which keywords from the job description are present in the resume."""
    resume_lower = resume_text.lower()
    keyword_matches = {}
    
    for keyword in keywords:
        # Check if keyword is present in resume
        if keyword.lower() in resume_lower:
            keyword_matches[keyword] = True
        else:
            keyword_matches[keyword] = False
    
    return keyword_matches

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

def rescale_score(score: float, min_score: float = 0.25, max_score: float = 0.45) -> float:
    """
    Rescales a score from the observed range (min_score, max_score) to (0, 1)
    """
    # If score is outside the observed range, clip it
    score = max(min_score, min(max_score, score))
    
    # Rescale to 0-1 range
    rescaled = (score - min_score) / (max_score - min_score)
    
    # Ensure result is between 0 and 1
    return max(0.0, min(1.0, rescaled))

@app.get("/")
async def root():
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")

@app.post("/rank-resume")
async def rank_resume(
    resume: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    use_default_job_desc: Optional[bool] = Form(True)
):
    """Rank a resume against a job description."""
    try:
        print(f"Request received: file={resume.filename}, job_desc length={len(job_description) if job_description else 0}")
        
        # Default job description handling
        used_default_jd = False
        if (job_description is None or job_description.strip() == ""):
            if use_default_job_desc:
                job_description = "Looking for a qualified candidate with relevant skills and experience in software development, data analysis, and project management. The ideal candidate will have strong communication skills, problem-solving abilities, and experience with programming languages."
                used_default_jd = True
                print("Using default job description")
            else:
                # If no job description and not using default, return a zero score
                print("No job description provided and default is disabled - returning zero score")
                return {
                    "score": 0.0,
                    "resume_text": extract_text_from_pdf(await save_temp_file(resume)),
                    "used_default_jd": False,
                    "match_details": {
                        "sections": extract_resume_sections(extract_text_from_pdf(await save_temp_file(resume))),
                        "keywords": {},
                        "missing_keywords": []
                    }
                }
        
        # Save uploaded resume temporarily
        temp_path = await save_temp_file(resume)
        
        # Extract text from resume
        print("Extracting text from PDF...")
        resume_text = extract_text_from_pdf(temp_path)
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        print(f"Text extracted, length: {len(resume_text)} characters")
        
        # Extract resume sections
        resume_sections = extract_resume_sections(resume_text)
        
        # Extract keywords from job description
        jd_keywords = extract_keywords(job_description)
        keyword_presence = check_keyword_presence(resume_text, jd_keywords)
        missing_keywords = [k for k, v in keyword_presence.items() if not v]
        
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
            raw_score = float(ranker.predict([features])[0])
            print(f"ML model prediction (raw): {raw_score}")
        else:
            # Fallback to cosine similarity if model not loaded
            raw_score = float(fallback_score(resume_embedding, jd_embedding))
            print(f"Fallback score (raw): {raw_score}")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Temporary file {temp_path} removed")
        
        # Rescale the score to use the full range - adjust min_score and max_score 
        # based on observed range in your logs (approx 0.29 to 0.40)
        score = rescale_score(raw_score, 0.29, 0.40)
        print(f"Rescaled score: {score} (from raw score: {raw_score})")
        
        result = {
            "score": score,
            "resume_text": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
            "used_default_jd": used_default_jd,
            "match_details": {
                "sections": resume_sections,
                "keywords": keyword_presence,
                "missing_keywords": missing_keywords
            }
        }
        print("Returning result")
        return result
    
    except Exception as e:
        print(f"Error in rank_resume: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

async def save_temp_file(uploaded_file: UploadFile) -> str:
    """Save uploaded file to a temporary location and return the path."""
    temp_path = f"temp_{uploaded_file.filename}"
    print(f"Saving file to {temp_path}")
    
    with open(temp_path, "wb") as buffer:
        content = await uploaded_file.read()
        buffer.write(content)
        print(f"File saved, size: {len(content)} bytes")
    
    return temp_path

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
