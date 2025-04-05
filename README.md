# AI Resume Ranking System

An intelligent system that ranks resumes against job descriptions using machine learning and natural language processing.

## Features

- **ML-Powered Resume Scoring**: Uses a machine learning model trained on a Kaggle dataset to provide realistic scores for resumes.
- **PDF Resume Parsing**: Automatically extracts text from PDF resumes.
- **Embedding-Based Matching**: Uses SBERT embeddings to represent both resumes and job descriptions.
- **FastAPI Backend**: Modern, high-performance API with automatic documentation.
- **Responsive Frontend**: Clean, black-themed UI for uploading resumes and displaying scores.
- **Cross-Origin Support**: Properly configured for local development.

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
- **ML/NLP**: RandomForest, SBERT (Sentence-BERT)
- **Data Processing**: NumPy, Pandas

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
   python Resume-Ranking-System/scripts/train_model_simple.py
   ```

5. Start the application:
   ```
   python Resume-Ranking-System/launch_app.py
   ```

## How to Use

1. Start the application using the launcher script.
2. Open your browser and go to `http://localhost:5500`.
3. Upload a PDF resume and optionally enter a job description.
4. Click "Calculate Score" to see the resume ranking.

## Model Details

The ranking model uses a Random Forest regressor trained on resume-job description pairs with realistic scoring between 0-1 based on:

- Text similarity between resume and job description
- Domain-specific keyword matching
- Contextual understanding of content

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for providing the resume dataset
- SentenceTransformers library for text embeddings
- FastAPI for the backend framework
