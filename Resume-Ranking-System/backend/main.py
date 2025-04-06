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
    """Extract different sections from a resume with detailed information."""
    sections = {
        'education': [],
        'experience': [],
        'skills': []
    }
    
    # Split text into lines for better processing
    lines = resume_text.split('\n')
    current_section = None
    section_text = []
    
    # Regular expressions for section headers
    education_header = re.compile(r'(?i)^\s*(education|academic|qualification)', re.IGNORECASE)
    experience_header = re.compile(r'(?i)^\s*(experience|work|employment)', re.IGNORECASE)
    skills_header = re.compile(r'(?i)^\s*(skills|expertise|technologies|technical)', re.IGNORECASE)
    
    # Regular expressions for content
    education_content = re.compile(r'(?i)(university|college|school|institute|academy|degree|bachelor|master|phd|b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?)')
    skills_content = re.compile(r'(?i)(python|java|c\+\+|javascript|html|css|sql|database|machine learning|ai|data science|web|cloud|aws|azure|gcp|docker|kubernetes|git|linux|windows|macos|android|ios)')
    date_pattern = re.compile(r'(?i)(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if education_header.search(line):
            if current_section and section_text:
                sections[current_section].extend(section_text)
            current_section = 'education'
            section_text = []
        elif experience_header.search(line):
            if current_section and section_text:
                sections[current_section].extend(section_text)
            current_section = 'experience'
            section_text = []
        elif skills_header.search(line):
            if current_section and section_text:
                sections[current_section].extend(section_text)
            current_section = 'skills'
            section_text = []
        elif current_section:
            # Process content based on current section
            if current_section == 'education' and education_content.search(line):
                section_text.append(line)
            elif current_section == 'experience' and (date_pattern.search(line) or len(line) > 30):
                section_text.append(line)
            elif current_section == 'skills':
                # Split skills by common separators and filter
                skills = [s.strip() for s in re.split(r'[,;|]', line)]
                skills = [s for s in skills if s and len(s) > 2 and not s.lower().startswith(('skill', 'technology'))]
                if skills:
                    section_text.extend(skills)
    
    # Add any remaining section text
    if current_section and section_text:
        sections[current_section].extend(section_text)
    
    # Fallback: If sections are empty, try to find content using patterns
    if not any(sections.values()):
        # Look for education entries
        education_matches = re.findall(r'(?i)(?:(?:bachelor|master|phd|b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?)[^.]*(?:university|college|institute)[^.]*\.)', resume_text)
        if education_matches:
            sections['education'] = education_matches
            
        # Look for experience entries
        experience_matches = re.findall(r'(?i)(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})[^.]*(?:present|current|worked|developed|managed)[^.]*\.)', resume_text)
        if experience_matches:
            sections['experience'] = experience_matches
            
        # Look for skills
        skills_text = re.findall(r'(?i)(?:skills|technologies)[^.]*\.', resume_text)
        if skills_text:
            skills = []
            for text in skills_text:
                skills.extend([s.strip() for s in re.split(r'[,;|]', text) if s.strip() and len(s.strip()) > 2])
            sections['skills'] = skills
    
    # Clean up the sections
    for section in sections:
        # Remove duplicates while preserving order
        seen = set()
        sections[section] = [x for x in sections[section] if not (x.lower() in seen or seen.add(x.lower()))]
        # Remove very short entries and common headers
        sections[section] = [x for x in sections[section] if len(x) > 5 and not re.match(r'(?i)^\s*(education|experience|skills|expertise|technologies)\s*$', x)]
    
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
