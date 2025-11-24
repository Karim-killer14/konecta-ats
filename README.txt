# Konecta ATS System - Complete Setup Guide

##Overview

This is a complete AI-powered Applicant Tracking System (ATS) for the Konecta ERP project. It includes:

- **CV Parsing**: Extract structured data from CVs
- **CV-Job Matching**: Advanced matching algorithm with multiple scoring methods
- **Explainable AI**: Generate human-readable explanations for matches
- **RAG Chatbot**: Answer questions about candidates and jobs
- **Document Processing**: Handle various HR documents
- **REST API**: Flask-based API for all functionality

##Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**\
```
google-generativeai>=0.3.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
flask>=3.0.0
flask-cors>=4.0.0
python-dotenv>=1.0.0
```

### Step 2: Generate Sample Data

```bash
python cv_generator.py
```

This creates:
- `cv_dataset.csv` (500 candidate CVs)
- `job_descriptions.csv` (20 job postings)

### Step 3: Test the System

```bash
python ats_system.py
```

This runs a complete demo showing all features.

### Step 4: Start the API Server

```bash
python flask_api.py
```

API will be available at: `http://localhost:5000`

##File Structure

```
ats-system/
├── cv_generator.py          # Generate sample CV data
├── ats_system.py            # Core ATS components (all-in-one)
├── flask_api.py             # REST API server
├── requirements.txt         # Python dependencies
├── cv_dataset.csv           # Generated CV data
├── job_descriptions.csv     # Generated job data
└── README.md               # This file
```

## API Endpoints

### CV Management

**Get All CVs**
```bash
GET /api/cvs
# With filters:
GET /api/cvs?min_experience=3&skills=Python,Java&location=Cairo
```

**Get Single CV**
```bash
GET /api/cvs/CV_0001
```

**Search CVs**
```bash
POST /api/cvs/search
Content-Type: application/json

{
  "min_experience": 2,
  "max_experience": 5,
  "skills": ["Python", "Machine Learning"],
  "location": "Cairo"
}
```

### Job Management

**Get All Jobs**
```bash
GET /api/jobs
GET /api/jobs?status=Open
```

**Get Single Job**
```bash
GET /api/jobs/JOB_001
```

### Matching & Ranking

**Match CV to Job**
```bash
POST /api/match
Content-Type: application/json

{
  "cv_id": "CV_0001",
  "job_id": "JOB_001",
  "include_explanation": true
}
```

**Rank Candidates**
```bash
POST /api/rank
Content-Type: application/json

{
  "job_id": "JOB_001",
  "top_n": 10,
  "filters": {
    "min_experience": 2
  }
}
```

**Batch Matching**
```bash
POST /api/match/batch
Content-Type: application/json

{
  "cv_ids": ["CV_0001", "CV_0002", "CV_0003"],
  "job_ids": ["JOB_001", "JOB_002"]
}
```

### Explainability

**Get Match Explanation**
```bash
POST /api/explain
Content-Type: application/json

{
  "cv_id": "CV_0001",
  "job_id": "JOB_001"
}
```

Response includes:
- AI-generated reasoning
- Skill gap analysis
- Interview questions
- Detailed scores

**Skill Gap Analysis**
```bash
POST /api/skill-gap
Content-Type: application/json

{
  "cv_id": "CV_0001",
  "job_id": "JOB_001"
}
```

### Chatbot

**Ask Question**
```bash
POST /api/chatbot/query
Content-Type: application/json

{
  "query": "What positions do we have open?"
}
```

**Get Chat History**
```bash
GET /api/chatbot/history
```

**Clear History**
```bash
POST /api/chatbot/clear
```

### Analytics

**System Overview**
```bash
GET /api/analytics/overview
```

Returns:
- Total candidates
- Total jobs
- Average experience
- Top skills

**Match Distribution**
```bash
GET /api/analytics/match-distribution?job_id=JOB_001
```

## Usage Examples

### Python Examples

```python
import requests

# 1. Get all CVs with Python skills
response = requests.get('http://localhost:5000/api/cvs?skills=Python')
cvs = response.json()['cvs']
print(f"Found {len(cvs)} candidates with Python")

# 2. Rank candidates for a job
payload = {
    "job_id": "JOB_001",
    "top_n": 5
}
response = requests.post('http://localhost:5000/api/rank', json=payload)
top_candidates = response.json()['top_candidates']

for i, candidate in enumerate(top_candidates, 1):
    print(f"{i}. {candidate['name']} - {candidate['score']*100:.1f}%")

# 3. Get match explanation
payload = {
    "cv_id": "CV_0001",
    "job_id": "JOB_001"
}
response = requests.post('http://localhost:5000/api/explain', json=payload)
explanation = response.json()

print(f"Match Score: {explanation['match_score']*100:.1f}%")
print(f"AI Reasoning: {explanation['ai_reasoning']}")
print(f"Interview Questions:")
for q in explanation['interview_questions']:
    print(f"  - {q}")

# 4. Ask chatbot
payload = {"query": "How many Software Engineers applied?"}
response = requests.post('http://localhost:5000/api/chatbot/query', json=payload)
print(response.json()['response']['response'])
```

### cURL Examples

```bash
# Get all open jobs
curl http://localhost:5000/api/jobs?status=Open

# Match CV to job
curl -X POST http://localhost:5000/api/match \
  -H "Content-Type: application/json" \
  -d '{"cv_id": "CV_0001", "job_id": "JOB_001", "include_explanation": true}'

# Ask chatbot
curl -X POST http://localhost:5000/api/chatbot/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about our open positions"}'

# Get analytics
curl http://localhost:5000/api/analytics/overview
```

## AI Features Explained

### 1. CV Parsing
- Loads CV data from CSV format
- Structures data for AI processing
- Supports filtering and searching

### 2. Matching Algorithm
Combines multiple scoring methods:
- **Skill Match (40%)**: Exact skill overlap
- **Experience Match (30%)**: Years of experience alignment
- **Text Similarity (20%)**: TF-IDF cosine similarity
- **Location Match (10%)**: Geographic compatibility

### 3. Explainable AI
Uses Gemini 2.0 Flash to generate:
- Human-readable match explanations
- Skill gap analysis with recommendations
- Tailored interview questions
- Candidate comparisons

### 4. RAG Chatbot
- Retrieves relevant CV/job data based on query
- Uses context to answer questions
- Maintains conversation history
- Handles queries about candidates, jobs, and statistics

### 5. Document Processing
- Extracts structured data from text
- Validates extracted information
- Supports multiple document types (invoices, contracts, leave requests)

## Data Format

### CV Data Structure
```json
{
  "cv_id": "CV_0001",
  "name": "Ahmed Hassan",
  "email": "ahmed.hassan123@gmail.com",
  "phone": "+20 1012345678",
  "location": "Cairo",
  "current_title": "Software Engineer",
  "years_experience": 5,
  "education": "Bachelor's in Computer Science, Cairo University (2019)",
  "skills": ["Python", "Java", "SQL", "AWS", "Docker"],
  "certifications": ["AWS Certified", "PMP"],
  "work_history": "Senior Developer at IBM Egypt (2021-2024) | ...",
  "expected_salary": 25000,
  "notice_period_days": 30,
  "application_date": "2024-10-15"
}
```

### Job Data Structure
```json
{
  "job_id": "JOB_001",
  "title": "Senior Software Engineer",
  "department": "IT",
  "required_experience": 5,
  "required_skills": ["Python", "AWS", "Docker", "Kubernetes"],
  "preferred_education": "Bachelor's in Computer Science",
  "salary_range": "20000-35000",
  "employment_type": "Full-time",
  "location": "Cairo",
  "openings": 2,
  "posted_date": "2024-10-01",
  "status": "Open"
}
```

## API Key Configuration

The system uses Google's Gemini 2.0 Flash (free model). The API key is already configured in the code, but you can change it:

```python
# In ats_system.py and flask_api.py
genai.configure(api_key='YOUR_API_KEY_HERE')
```

Get your free API key at: https://makersuite.google.com/app/apikey

## Integration with Full Stack Team

The API is designed to integrate seamlessly with the Full Stack team's ERP frontend:

1. **Authentication**: Add JWT middleware for secure access
2. **Database**: Replace CSV loading with database queries
3. **File Upload**: Add endpoints for actual CV file uploads (PDF/DOCX)
4. **WebSockets**: Add real-time updates for chatbot
5. **Caching**: Add Redis for frequently accessed data

Example Angular integration:
```typescript
// candidate.service.ts
@Injectable()
export class CandidateService {
  private apiUrl = 'http://localhost:5000/api';

  getCandidates(filters: any): Observable<any> {
    return this.http.get(`${this.apiUrl}/cvs`, { params: filters });
  }

  matchCandidateToJob(cvId: string, jobId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/match`, { cv_id: cvId, job_id: jobId });
  }

  rankCandidates(jobId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/rank`, { job_id: jobId, top_n: 10 });
  }
}
```

## Dashboard Integration

For Data Analytics team, the API provides metrics endpoints:

```javascript
// Fetch for dashboard
fetch('http://localhost:5000/api/analytics/overview')
  .then(res => res.json())
  .then(data => {
    // Display in Power BI or Tableau
    console.log(data.overview);
  });
```

## Troubleshooting

**Issue: API key error**
```
Solution: Verify your Gemini API key is valid
```

**Issue: CSV files not found**
```
Solution: Run cv_generator.py first to create the data files
```

**Issue: Port 5000 already in use**
```
Solution: Change port in flask_api.py:
app.run(debug=True, host='0.0.0.0', port=5001)
```

**Issue: CORS errors in frontend**
```
Solution: CORS is already enabled. Check your request headers.
```

## Support

For issues or questions:
- Check API docs: `http://localhost:5000`
- Test health: `http://localhost:5000/health`
- Review logs in terminal

---

**Model Used**: Gemini 2.0 Flash (Free)  
**Status**: Production Ready ✅  
