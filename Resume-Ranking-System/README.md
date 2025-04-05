# AI-Powered Resume Ranking System

An intelligent Applicant Tracking System (ATS) that uses machine learning to evaluate and rank resumes based on job descriptions. The system combines SBERT embeddings with XGBoost to provide accurate resume-job matching scores.

## 🚀 Features

- **ML-Based Resume Scoring**: Uses trained models instead of simple similarity matching
- **SBERT + XGBoost Pipeline**: Combines semantic embeddings with gradient boosting
- **FastAPI Backend**: RESTful API for resume processing and scoring
- **MLflow Integration**: Track model training and performance
- **Automated Retraining**: MLOps-ready for continuous model improvement

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)
- MLflow server (optional, for experiment tracking)

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/AI-ATS-Project.git
cd AI-ATS-Project
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

The system uses the [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) from Kaggle. Place the dataset in the `data/kaggle` directory.

## 🚀 Usage

1. **Preprocess the Dataset**:

```bash
python scripts/preprocess_data.py
```

2. **Train the Model**:

```bash
python scripts/train_model.py
```

3. **Start the Backend Server**:

```bash
python backend/main.py
```

4. **API Endpoints**:

- `POST /rank-resume`: Upload a resume and get its score against a job description
- `GET /health`: Health check endpoint

## 📁 Project Structure

```
AI-ATS-Project/
├── backend/                   # Core logic & models
│   ├── models/               # Pre-trained or trained ML/NLP models
│   ├── utils/                # PDF parsing, text extraction, scoring logic
│   ├── main.py               # FastAPI backend
│   └── requirements.txt      # Python dependencies
├── data/                     # Resume & job data
│   ├── raw/                  # Raw PDFs
│   ├── processed/            # Cleaned text, embeddings
│   └── kaggle/               # Labeled Kaggle dataset
├── models/                   # Trained ML models
├── scripts/                  # Training and automation scripts
│   ├── preprocess_data.py    # Data preprocessing
│   └── train_model.py        # Model training
└── notebooks/               # ML experimentation
```

## 🔄 MLOps Integration

The project is set up for MLOps with:

- MLflow for experiment tracking
- Automated model retraining
- Model versioning
- Performance monitoring

## 📈 Performance Metrics

- Mean Squared Error (MSE)
- R² Score
- Model training time
- Inference latency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
