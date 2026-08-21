# 🤖 AI Resume Screening and Candidate Ranking Tool

An industry-style resume screening dashboard built with **Python, Streamlit and Sentence Transformers**.

## 🚀 Key Idea

Traditional keyword matching can miss resumes that use different wording. This project uses **sentence embeddings** to compare the meaning of a job description with the meaning of each resume.

The system combines:

**AI Semantic Match (75%) + Skill Match (25%) = Final Candidate Score**

## ✨ Features

- Multiple PDF resume upload
- AI semantic similarity using `all-MiniLM-L6-v2`
- Skill matching and skill-gap analysis
- Candidate ranking
- Experience extraction
- Education detection
- Email and phone extraction
- Recruiter dashboard
- Candidate comparison
- Ranking chart
- CSV export
- Configurable minimum score
- Professional Streamlit UI

## 🧠 How AI Matching Works

```text
Job Description
       ↓
Text Cleaning
       ↓
Sentence Transformer
       ↓
Job Embedding
       ↓
Cosine Similarity
       ↓
Semantic Match Score
       ↓
Skill Match Score
       ↓
Weighted Final Score
       ↓
Candidate Ranking
```

### Why Sentence Transformers?

A sentence-transformer model represents text as numerical vectors called **embeddings**. Similar meanings tend to have similar vector representations.

For example, a resume saying:

> "Built predictive models using Python"

can be semantically related to a job requirement saying:

> "Experience developing machine learning solutions in Python"

even when the exact wording is different.

## 🛠️ Tech Stack

- Python
- Streamlit
- Sentence Transformers
- PyTorch
- Pandas
- PyPDF

## 📂 Project Structure

```text
AI-ResumeRanker/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── sample_resumes/
```

## 💻 Installation

```bash
git clone https://github.com/YOUR-USERNAME/AI-Resume-Screening-and-Candidate-Ranking-Tool.git
cd AI-Resume-Screening-and-Candidate-Ranking-Tool
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Linux/macOS**
```bash
source venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

The first AI run may download the Sentence Transformer model.

## 📊 Scoring

| Component | Weight |
|---|---:|
| AI Semantic Similarity | 75% |
| Required Skill Match | 25% |
| **Final Score** | **100%** |

## 🔮 Future Improvements

- Fine-tuned BERT model
- Named Entity Recognition for organizations and degrees
- Experience timeline extraction
- Database integration
- Recruiter authentication
- Explainable AI recommendations
- Bias/fairness testing
- Cloud deployment
- Job-specific skill weighting
- Resume section classification

## ⚠️ Responsible AI

This project is an educational **decision-support** application. It should not automatically reject or hire candidates. Real recruitment systems should use job-relevant criteria, human review, privacy protections and fairness testing.

## 👩‍💻 Author

**Khushi Kumari**  
Data Science Student

## 📄 License

Educational project.
