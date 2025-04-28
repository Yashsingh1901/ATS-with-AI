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
from scipy.spatial.distance import cosine

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

def calculate_section_scores(sections: dict, job_description: str) -> dict:
    """Calculate weighted scores for each section with enhanced industry-specific scoring."""
    section_weights = {
        'experience': 0.5,
        'skills': 0.3,
        'education': 0.2
    }
    
    # Industry-specific keywords and their weights
    industry_keywords = {
        'software_development': {
            'keywords': ['software', 'development', 'programming', 'coding', 'application', 'web', 'mobile', 'frontend', 'backend', 'fullstack'],
            'weight': 0.3
        },
        'data_science': {
            'keywords': ['data', 'analytics', 'machine learning', 'ai', 'statistics', 'python', 'r', 'sql', 'big data', 'visualization'],
            'weight': 0.3
        },
        'cloud_computing': {
            'keywords': ['cloud', 'aws', 'azure', 'gcp', 'devops', 'kubernetes', 'docker', 'ci/cd', 'infrastructure', 'microservices'],
            'weight': 0.2
        },
        'project_management': {
            'keywords': ['project', 'management', 'agile', 'scrum', 'kanban', 'leadership', 'team', 'planning', 'budget', 'stakeholder'],
            'weight': 0.2
        }
    }
    
    section_scores = {}
    for section, content in sections.items():
        if not content:
            section_scores[section] = 0.0
            continue
            
        # Get section embedding
        section_text = " ".join(content)
        section_embedding = get_resume_embedding(section_text)
        jd_embedding = get_jd_embedding(job_description)
        
        # Calculate semantic similarity
        similarity = 1 - cosine(section_embedding, jd_embedding)
        
        # Apply section-specific scoring
        if section == 'experience':
            # Check for years of experience with more granular scoring
            years_pattern = r'(\d+)\s*(?:years?|yrs?)'
            years_matches = re.findall(years_pattern, section_text.lower())
            total_years = sum(map(int, years_matches))
            
            # More granular years scoring
            if total_years >= 10:
                years_score = 1.0
            elif total_years >= 7:
                years_score = 0.9
            elif total_years >= 5:
                years_score = 0.8
            elif total_years >= 3:
                years_score = 0.7
            elif total_years >= 2:
                years_score = 0.6
            elif total_years >= 1:
                years_score = 0.5
            else:
                years_score = 0.3
            
            # Enhanced leadership role detection
            leadership_roles = {
                'senior': ['senior', 'lead', 'principal', 'architect'],
                'management': ['manager', 'head', 'director', 'supervisor', 'team lead'],
                'executive': ['cto', 'ceo', 'vp', 'chief', 'founder']
            }
            
            leadership_score = 0
            for role_type, keywords in leadership_roles.items():
                if any(keyword in section_text.lower() for keyword in keywords):
                    if role_type == 'senior':
                        leadership_score = max(leadership_score, 0.3)
                    elif role_type == 'management':
                        leadership_score = max(leadership_score, 0.6)
                    elif role_type == 'executive':
                        leadership_score = max(leadership_score, 0.9)
            
            # Industry-specific experience scoring
            industry_score = 0
            for industry, details in industry_keywords.items():
                keyword_matches = sum(1 for keyword in details['keywords'] if keyword in section_text.lower())
                if keyword_matches > 0:
                    industry_score += (keyword_matches / len(details['keywords'])) * details['weight']
            
            section_scores[section] = (
                similarity * 0.4 +
                years_score * 0.2 +
                leadership_score * 0.2 +
                industry_score * 0.2
            ) * section_weights[section]
            
        elif section == 'skills':
            # Enhanced skill level detection
            skill_levels = {
                'expert': ['expert', 'master', 'advanced', 'proficient', 'extensive'],
                'intermediate': ['intermediate', 'experienced', 'skilled', 'competent'],
                'beginner': ['beginner', 'basic', 'familiar', 'knowledge']
            }
            
            # Industry-specific skills
            industry_skills = {
                'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust'],
                'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express'],
                'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
                'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
                'ai_ml': ['tensorflow', 'pytorch', 'scikit-learn', 'numpy', 'pandas']
            }
            
            skills_score = 0
            total_skills = 0
            
            for skill in content:
                # Check skill level
                level_score = 0.5  # Default level
                for level, indicators in skill_levels.items():
                    if any(indicator in skill.lower() for indicator in indicators):
                        if level == 'expert':
                            level_score = 1.0
                        elif level == 'intermediate':
                            level_score = 0.7
                        elif level == 'beginner':
                            level_score = 0.4
                
                # Check industry relevance
                industry_relevance = 0
                for category, skills in industry_skills.items():
                    if any(s in skill.lower() for s in skills):
                        industry_relevance = 1.0
                        break
                
                # Combine scores
                skill_score = (level_score * 0.7 + industry_relevance * 0.3)
                skills_score += skill_score
                total_skills += 1
            
            skills_score = skills_score / total_skills if total_skills > 0 else 0
            section_scores[section] = (similarity * 0.6 + skills_score * 0.4) * section_weights[section]
            
        else:  # education
            # Enhanced degree relevance
            degree_relevance = {
                'computer_science': ['computer science', 'cs', 'software engineering', 'se'],
                'data_science': ['data science', 'machine learning', 'ai', 'statistics'],
                'information_technology': ['it', 'information technology', 'information systems'],
                'engineering': ['computer engineering', 'electrical engineering', 'mechanical engineering']
            }
            
            degree_score = 0
            for field, keywords in degree_relevance.items():
                if any(keyword in section_text.lower() for keyword in keywords):
                    degree_score = 1.0
                    break
            
            # Enhanced GPA scoring
            gpa_pattern = r'GPA:\s*(\d+\.\d+)'
            gpa_match = re.search(gpa_pattern, section_text)
            if gpa_match:
                gpa = float(gpa_match.group(1))
                if gpa >= 3.8:
                    gpa_score = 1.0
                elif gpa >= 3.5:
                    gpa_score = 0.9
                elif gpa >= 3.2:
                    gpa_score = 0.8
                elif gpa >= 3.0:
                    gpa_score = 0.7
                else:
                    gpa_score = 0.5
            else:
                gpa_score = 0.5
            
            # Check for relevant certifications
            cert_keywords = ['certified', 'certification', 'professional', 'expert', 'specialist']
            cert_score = 0.3 if any(keyword in section_text.lower() for keyword in cert_keywords) else 0
            
            section_scores[section] = (
                similarity * 0.4 +
                degree_score * 0.3 +
                gpa_score * 0.2 +
                cert_score * 0.1
            ) * section_weights[section]
    
    return section_scores

def extract_keywords(job_description: str) -> List[str]:
    """Extract keywords from job description with improved accuracy."""
    # Create and configure a CountVectorizer with better parameters for single document
    vectorizer = CountVectorizer(
        stop_words='english',
        min_df=1,  # Minimum document frequency
        max_df=1.0,  # Maximum document frequency
        ngram_range=(1, 3),  # Include up to 3-word phrases
        max_features=50  # Limit to top 50 keywords
    )
    
    try:
        # Fit and transform the job description
        X = vectorizer.fit_transform([job_description])
        
        # Get feature names and their frequencies
        feature_names = vectorizer.get_feature_names_out()
        frequencies = X.toarray()[0]
        
        # Calculate TF-IDF scores
        from sklearn.feature_extraction.text import TfidfTransformer
        tfidf = TfidfTransformer()
        tfidf_scores = tfidf.fit_transform(X).toarray()[0]
        
        # Combine frequency and TF-IDF scores
        combined_scores = frequencies * tfidf_scores
        
        # Sort keywords by combined score
        sorted_indices = combined_scores.argsort()[::-1]
        top_keywords = [feature_names[idx] for idx in sorted_indices[:20]]
        
        return top_keywords
    except Exception as e:
        print(f"Error in keyword extraction: {e}")
        # Fallback to simple keyword extraction
        words = job_description.lower().split()
        # Remove common words and short terms
        stop_words = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of'])
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        # Return top 20 unique keywords
        return list(set(keywords))[:20]

def check_keyword_presence(resume_text: str, keywords: List[str]) -> dict:
    """Check which keywords from the job description are present in the resume with context."""
    resume_lower = resume_text.lower()
    keyword_matches = {}
    
    for keyword in keywords:
        # Check for exact matches
        if keyword.lower() in resume_lower:
            # Look for context around the keyword
            context_start = max(0, resume_lower.find(keyword.lower()) - 50)
            context_end = min(len(resume_lower), resume_lower.find(keyword.lower()) + len(keyword) + 50)
            context = resume_lower[context_start:context_end]
            
            # Check for experience indicators
            experience_indicators = ['years', 'experience', 'worked', 'developed', 'implemented']
            has_experience = any(indicator in context for indicator in experience_indicators)
            
            keyword_matches[keyword] = {
                'present': True,
                'has_experience': has_experience,
                'context': context.strip()
            }
        else:
            keyword_matches[keyword] = {
                'present': False,
                'has_experience': False,
                'context': None
            }
    
    return keyword_matches

def get_resume_embedding(resume_text: str) -> np.ndarray:
    """Get SBERT embedding for resume text."""
    return model.encode([resume_text])[0]

def get_jd_embedding(jd_text: str) -> np.ndarray:
    """Get SBERT embedding for job description text."""
    return model.encode([jd_text])[0]

def fallback_score(resume_embedding, jd_embedding):
    """Fallback scoring using cosine similarity."""
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
    """Rank a resume against a job description with enhanced scoring."""
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
        sections = extract_resume_sections(resume_text)
        
        # Calculate section scores
        section_scores = calculate_section_scores(sections, job_description)
        
        # Extract and check keywords
        keywords = extract_keywords(job_description)
        keyword_matches = check_keyword_presence(resume_text, keywords)
        
        # Calculate final score
        if ranker is not None:
            # Get embeddings
            resume_embedding = get_resume_embedding(resume_text)
            jd_embedding = get_jd_embedding(job_description)
            
            # Get base score from model
            base_score = ranker.predict([resume_embedding])[0]
        else:
            # Fallback to cosine similarity
            base_score = fallback_score(
                get_resume_embedding(resume_text),
                get_jd_embedding(job_description)
            )
        
        # Calculate keyword match score
        keyword_score = sum(1 for match in keyword_matches.values() if match['present']) / len(keywords)
        
        # Calculate section match score
        section_score = sum(section_scores.values())
        
        # Combine scores with weights
        final_score = (
            base_score * 0.4 +  # Base semantic similarity
            keyword_score * 0.3 +  # Keyword matching
            section_score * 0.3  # Section-specific scoring
        )
        
        # Rescale score to 0-1 range
        final_score = rescale_score(final_score)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Temporary file {temp_path} removed")
        
        result = {
            "score": final_score,
            "resume_text": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
            "used_default_jd": used_default_jd,
            "match_details": {
                "sections": sections,
                "section_scores": section_scores,
                "keywords": keyword_matches,
                "missing_keywords": [k for k, v in keyword_matches.items() if not v['present']]
            }
        }
        print("Returning result")
        return result
    
    except Exception as e:
        print(f"Error processing resume: {e}")
        traceback.print_exc()
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
