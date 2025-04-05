# AI Resume Ranking System

An intelligent system that ranks resumes against job descriptions using machine learning and natural language processing.

## Features

- **ML-Powered Resume Scoring**: Uses an ensemble machine learning model (Random Forest + XGBoost) trained on a Kaggle dataset to provide realistic scores for resumes.
- **PDF Resume Parsing**: Automatically extracts text from PDF resumes.
- **Embedding-Based Matching**: Uses SBERT embeddings to represent both resumes and job descriptions.
- **FastAPI Backend**: Modern, high-performance API with automatic documentation.
- **Responsive Frontend**: Clean, black-themed UI for uploading resumes and displaying scores.
- **Cross-Origin Support**: Properly configured for local development.
- **Detailed Resume Analysis**: Extracts key sections and matches keywords from job descriptions.
- **Keyword Extraction**: Identifies important keywords from job descriptions and checks for matches.
- **Full-Range Scoring**: Uses a rescaled scoring mechanism for intuitive 0-100% scoring.
- **Transparent Job Description Handling**: Options for default job description behavior.

## Project Structure

```
Resume-Ranking-System/
├── backend/           # FastAPI server implementation
├── data/              # Dataset storage
│   ├── kaggle/        # Kaggle resume dataset
│   └── microsoft/     # Test resumes
├── frontend/          # Web UI implementation
├── models/            # Trained ML models
└── scripts/           # Utility scripts for preprocessing and training
```

## Technologies Used

- **Backend**: FastAPI, pdfplumber, SentenceTransformers, scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **ML/NLP**: Ensemble of RandomForest and XGBoost, SBERT (Sentence-BERT)
- **Data Processing**: NumPy, Pandas, regex

## Installation & Setup

1. Clone this repository:

   ```
   git clone https://github.com/Yashsingh1901/ATS-with-AI.git
   cd ATS-with-AI
   ```

2. Create a virtual environment:

   ```
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Process the dataset and train the model:

   ```
   python Resume-Ranking-System/scripts/preprocess_data.py
   python Resume-Ranking-System/scripts/train_ensemble_model.py  # Uses ensemble model (RF + XGBoost)
   ```

5. Start the application:
   ```
   python Resume-Ranking-System/launch_app.py
   ```

## How to Use

1. Start the application using the launcher script.
2. Open your browser and go to `http://localhost:5500`.
3. Upload a PDF resume and enter a job description.
4. Choose whether to use a default job description if none is provided.
5. Click "Calculate Score" to see the resume ranking.
6. Review the detailed analysis showing keyword matches and resume sections.

## Model Details

The ranking model uses an ensemble of:

1. **Random Forest Regressor**: Handles non-linear relationships in the data
2. **XGBoost Regressor**: Provides boosting capabilities for better predictions

The ensemble model is trained on resume-job description pairs with realistic scoring based on:

- Text similarity between resume and job description
- Domain-specific keyword matching
- Contextual understanding of content

Scores are rescaled to provide intuitive 0-100% values rather than raw model outputs.

## Advanced Analytics

The system now provides:

- **Keyword Matching**: Identifies which key terms from the job description appear in the resume
- **Missing Keywords**: Highlights important terms that are absent from the resume
- **Section Extraction**: Recognizes education, experience, and skills sections in the resume

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for providing the resume dataset
- SentenceTransformers library for text embeddings
- FastAPI for the backend framework
