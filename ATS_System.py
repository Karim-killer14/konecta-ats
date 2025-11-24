# ============================================================================
# COMPLETE ATS SYSTEM - All Components Integrated (UPDATED - Add CV support)
# ============================================================================

import google.generativeai as genai
import pandas as pd
import numpy as np
import json
import re
import os
import io
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
from datetime import date

# Optional third-party libs used for extraction - ensure installed in your venv
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# Configure Gemini API (Free model)
# If you don't have an API key or want to use the fallback parser, it's fine.
try:
    genai.configure(api_key=os.environ.get("GENAI_API_KEY", ""))
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except Exception:
    model = None


# ============================================================================
# 1. CV PARSER - Works with Generated CSV Data (now supports adding CVs)
# ============================================================================

class CVParser:
    """Parse CV data from CSV format and enrich with AI. Supports adding new CVs."""

    def __init__(self):
        self.cv_df = pd.DataFrame()
        self.jobs_df = pd.DataFrame()
        self.data_path = "cv_dataset.csv"

        # Document processor will be attached externally if needed
        self.document_processor: Optional[DocumentProcessor] = None

    def load_csv_data(self, cv_csv_path='cv_dataset.csv', jobs_csv_path='job_descriptions.csv'):
        """Load generated CSV files and normalize datatypes"""
        try:
            self.data_path = cv_csv_path
            self.cv_df = pd.read_csv(cv_csv_path)
            self.jobs_df = pd.read_csv(jobs_csv_path)

            # FIX: convert application_date to datetime (safe)
            if 'application_date' in self.cv_df.columns:
                self.cv_df['application_date'] = pd.to_datetime(self.cv_df['application_date'], errors='coerce')

            # Ensure types for numeric fields used in filtering
            if 'years_of_experience' in self.cv_df.columns:
                self.cv_df['years_of_experience'] = pd.to_numeric(self.cv_df['years_of_experience'], errors='coerce').fillna(0).astype(int)

            print(f"✓ Loaded {len(self.cv_df)} CVs and {len(self.jobs_df)} jobs")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def get_cv_by_id(self, cv_id: str) -> Dict:
        """Get structured CV data by ID"""
        cv_row = self.cv_df[self.cv_df['cv_id'] == cv_id]

        if cv_row.empty:
            return None

        cv = cv_row.iloc[0].to_dict()

        def safe_str(value):
            if pd.isna(value) or value is None:
                return ''
            return str(value)

        structured_cv = {
            "cv_id": cv['cv_id'],
            "name": f"{cv.get('first_name', '')} {cv.get('last_name', '')}".strip(),
            "email": cv.get('email', ''),
            "phone": cv.get('phone', ''),
            "location": safe_str(cv.get('location', '')),
            "current_title": safe_str(cv.get('current_job_title', '')),
            "years_experience": int(cv.get('years_of_experience') or 0),
            "education": safe_str(cv.get('education', '')),
            "skills": cv['skills'].split(',') if pd.notna(cv.get('skills')) else [],
            "certifications": cv['certifications'].split(',') if pd.notna(cv.get('certifications')) and cv.get('certifications') != 'None' else [],
            "work_history": safe_str(cv.get('work_history', '')),
            "expected_salary": cv.get('expected_salary', 0),
            "notice_period_days": cv.get('notice_period_days', 0),
            "application_date": cv.get('application_date')
        }

        return structured_cv

    def get_all_cvs(self) -> List[Dict]:
        """Get all CVs as structured data"""
        return [self.get_cv_by_id(cv_id) for cv_id in self.cv_df['cv_id'].tolist()]

    def get_job_by_id(self, job_id: str) -> Dict:
        """Get job description by ID"""
        job_row = self.jobs_df[self.jobs_df['job_id'] == job_id]

        if job_row.empty:
            return None

        job = job_row.iloc[0].to_dict()

        structured_job = {
            "job_id": job['job_id'],
            "title": job['job_title'],
            "department": job['department'],
            "required_experience": job['required_experience_years'],
            "required_skills": job['required_skills'].split(',') if pd.notna(job['required_skills']) else [],
            "preferred_education": job['preferred_education'],
            "salary_range": job['salary_range'],
            "employment_type": job['employment_type'],
            "location": job['location'],
            "openings": job['number_of_openings'],
            "posted_date": job['posted_date'],
            "status": job['status']
        }

        return structured_job

    def search_cvs(self, filters: Dict) -> List[Dict]:
        """Search CVs with filters"""
        df = self.cv_df.copy()

        if 'min_experience' in filters:
            df = df[df['years_of_experience'] >= filters['min_experience']]

        if 'max_experience' in filters:
            df = df[df['years_of_experience'] <= filters['max_experience']]

        if 'skills' in filters:
            df = df[df['skills'].str.contains('|'.join(filters['skills']), case=False, na=False)]

        if 'location' in filters and filters['location']:
            df = df[df['location'] == filters['location']]

        return [self.get_cv_by_id(cv_id) for cv_id in df['cv_id'].tolist()]

    # ----------------- New methods for adding CVs -----------------
    def _next_cv_id(self) -> str:
        """Generate next CV ID like CV_0001 based on existing IDs"""
        if self.cv_df is None or self.cv_df.empty:
            return "CV_0001"
        existing = [c for c in self.cv_df['cv_id'].tolist() if isinstance(c, str) and c.startswith("CV_")]
        if not existing:
            return "CV_0001"
        nums = []
        for c in existing:
            try:
                nums.append(int(c.split('_')[-1]))
            except:
                continue
        next_num = max(nums) + 1 if nums else 1
        return f"CV_{next_num:04d}"

    def add_cv_record(self, record: Dict, persist: bool = True) -> str:
        """
        Append a new CV record (structured dict) to cv_df and optionally persist to CSV.
        record should contain keys expected by existing CSV:
        - first_name, last_name, email, phone, location, current_job_title, years_of_experience,
          education, skills (comma-separated string), certifications, work_history, expected_salary,
          notice_period_days, application_date
        Returns the assigned cv_id
        """
        if self.cv_df is None:
            self.cv_df = pd.DataFrame()

        cv_id = self._next_cv_id()
        record = record.copy()
        record['cv_id'] = cv_id

        # Normalize some fields
        if 'application_date' not in record or not record['application_date']:
            record['application_date'] = pd.Timestamp(date.today())
        else:
            # Try to coerce to pandas timestamp
            try:
                record['application_date'] = pd.to_datetime(record['application_date'], errors='coerce')
            except:
                record['application_date'] = pd.Timestamp(date.today())

        if 'skills' in record and isinstance(record['skills'], list):
            record['skills'] = ', '.join([s.strip() for s in record['skills'] if s])
        if 'certifications' in record and isinstance(record['certifications'], list):
            record['certifications'] = ', '.join([s.strip() for s in record['certifications'] if s])

        # Ensure consistent columns - append missing columns with empty values
        for col in ['first_name', 'last_name', 'email', 'phone', 'location', 'current_job_title',
                    'years_of_experience', 'education', 'skills', 'certifications', 'work_history',
                    'expected_salary', 'notice_period_days', 'application_date']:
            if col not in record:
                record[col] = '' if col != 'years_of_experience' else 0

        # Append to dataframe
        new_row = pd.DataFrame([record])
        # Align columns order - union of existing columns and new
        self.cv_df = pd.concat([self.cv_df, new_row], ignore_index=True, sort=False)

        # Try to coerce important dtypes
        try:
            self.cv_df['application_date'] = pd.to_datetime(self.cv_df['application_date'], errors='coerce')
        except Exception:
            pass
        try:
            self.cv_df['years_of_experience'] = pd.to_numeric(self.cv_df['years_of_experience'], errors='coerce').fillna(0).astype(int)
        except Exception:
            pass

        if persist:
            try:
                self.cv_df.to_csv(self.data_path, index=False)
            except Exception as e:
                print(f"⚠ Failed to save cv_dataset.csv: {e}")

        return cv_id


# ============================================================================
# 2. CV-JOB MATCHER - Advanced Matching with Embeddings (unchanged except small fixes)
# ============================================================================

class CVJobMatcher:
    """Match CVs to jobs using multiple scoring methods"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')

    def calculate_skill_match(self, cv_skills: List[str], required_skills: List[str]) -> float:
        """Calculate skill overlap score"""
        if not required_skills:
            return 0.5

        cv_skills_lower = [s.lower().strip() for s in cv_skills]
        required_skills_lower = [s.lower().strip() for s in required_skills]

        matched = sum(1 for skill in required_skills_lower if skill in cv_skills_lower)
        return matched / len(required_skills_lower)

    def calculate_experience_match(self, cv_years: int, required_years: int) -> float:
        """Calculate experience match score"""
        if cv_years >= required_years:
            excess = cv_years - required_years
            return min(1.0, 0.9 + (excess * 0.02))
        else:
            deficit = required_years - cv_years
            return max(0.0, 0.9 - (deficit * 0.15))

    def calculate_text_similarity(self, cv_text: str, job_text: str) -> float:
        """Calculate semantic similarity using TF-IDF"""
        try:
            vectors = self.vectorizer.fit_transform([cv_text, job_text])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0

    def match_cv_to_job(self, cv_data: Dict, job_data: Dict) -> Dict:
        """Comprehensive CV-Job matching - handles NaN values"""

        skill_score = self.calculate_skill_match(cv_data.get('skills', []), job_data.get('required_skills', []))

        exp_score = self.calculate_experience_match(
            cv_data.get('years_experience', 0),
            job_data.get('required_experience', 0)
        )

        cv_text_parts = [
            str(cv_data.get('current_title', '')),
            ' '.join(cv_data.get('skills', [])),
            str(cv_data.get('work_history', ''))
        ]
        cv_text = ' '.join(filter(None, cv_text_parts))

        job_text_parts = [
            str(job_data.get('title', '')),
            ' '.join(job_data.get('required_skills', []))
        ]
        job_text = ' '.join(filter(None, job_text_parts))

        text_score = self.calculate_text_similarity(cv_text, job_text)

        cv_location = str(cv_data.get('location', '')).lower()
        job_location = str(job_data.get('location', '')).lower()
        location_score = 1.0 if (cv_location and (cv_location in job_location or job_location in ['remote', 'hybrid'])) else 0.5

        final_score = (
            skill_score * 0.40 +
            exp_score * 0.30 +
            text_score * 0.20 +
            location_score * 0.10
        )

        matched_skills = [s for s in cv_data.get('skills', []) if s in job_data.get('required_skills', [])]
        missing_skills = [s for s in job_data.get('required_skills', []) if s not in cv_data.get('skills', [])]

        return {
            "overall_score": round(final_score, 3),
            "skill_match": round(skill_score, 3),
            "experience_match": round(exp_score, 3),
            "text_similarity": round(text_score, 3),
            "location_match": round(location_score, 3),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        }

    def rank_candidates(self, cv_list: List[Dict], job_data: Dict, top_n: int = 10) -> List[Dict]:
        """Rank multiple candidates for a job"""
        results = []
        for cv in cv_list:
            try:
                match_result = self.match_cv_to_job(cv, job_data)
                results.append({
                    "cv_id": cv['cv_id'],
                    "name": cv['name'],
                    "score": match_result['overall_score'],
                    "match_details": match_result,
                    "cv_data": cv
                })
            except Exception as e:
                print(f"⚠ Skipping {cv.get('cv_id', 'unknown')}: {str(e)}")
                continue
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]


# ============================================================================
# 3. EXPLAINABLE MATCHER - AI-Powered Explanations (unchanged)
# ============================================================================

class ExplainableMatcher:
    """Generate human-readable explanations for matches using Gemini"""

    def generate_match_reasoning(self, cv_data: Dict, job_data: Dict, match_scores: Dict) -> str:
        # Try AI generation first
        prompt = f"""
        As an HR expert, explain why this candidate is a {self._get_match_level(match_scores['overall_score'])} match...
        Candidate: {cv_data.get('name', '')} - Experience: {cv_data.get('years_experience', 0)}
        Job: {job_data.get('title', '')}
        Scores: {match_scores}
        Provide a concise 2-3 sentence explanation.
        """
        try:
            if model is not None:
                response = model.generate_content(prompt)
                return response.text.strip()
        except Exception:
            pass

        # Fallback
        matched = match_scores.get('matched_skills', [])
        missing = match_scores.get('missing_skills', [])
        score = match_scores['overall_score']

        if score >= 0.7:
            return f"{cv_data['name']} is a strong candidate with {len(matched)} matching skills including {', '.join(matched[:3])}. Their {cv_data['years_experience']} years of experience aligns well with the required experience."
        elif score >= 0.5:
            return f"{cv_data['name']} shows potential with {len(matched)} matching skills. They would benefit from training in {', '.join(missing[:3])}."
        else:
            return f"{cv_data['name']} has limited alignment. Key gaps include {', '.join(missing[:3])}."

    def _get_match_level(self, score: float) -> str:
        if score >= 0.8:
            return "excellent"
        elif score >= 0.65:
            return "good"
        elif score >= 0.5:
            return "moderate"
        else:
            return "weak"

    def generate_skill_gap_analysis(self, cv_data: Dict, job_data: Dict) -> Dict:
        missing_skills = [s for s in job_data['required_skills'] if s not in cv_data['skills']]
        matching_skills = [s for s in cv_data['skills'] if s in job_data['required_skills']]
        return {
            "matched_skills": matching_skills,
            "missing_skills": missing_skills,
            "match_percentage": round(len(matching_skills) / max(len(job_data['required_skills']), 1) * 100, 1),
            "recommendations": self._generate_recommendations(missing_skills)
        }

    def _generate_recommendations(self, missing_skills: List[str]) -> List[str]:
        if not missing_skills:
            return ["Candidate meets all skill requirements"]
        return [f"Consider training in: {skill}" for skill in missing_skills[:3]]

    def generate_interview_questions(self, cv_data: Dict, job_data: Dict, match_scores: Dict) -> List[str]:
        prompt = f"Generate 5 interview questions for {cv_data.get('name','candidate')} for the role {job_data.get('title','')}. Focus on verifying skills and gaps."
        try:
            if model is not None:
                response = model.generate_content(prompt)
                questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
                return questions[:5]
        except:
            pass
        # fallback list
        return [
            f"Describe your experience with {cv_data.get('current_title','your role')}",
            f"How did you use {cv_data.get('skills',[ 'your skills' ])[0]} in a past project?",
            "Tell me about a challenging project you led.",
            "How do you stay current with industry trends?",
            "Where do you see yourself in 3 years?"
        ]


# ============================================================================
# 4. RAG CHATBOT - unchanged
# ============================================================================

class ATSChatbot:
    """RAG-based chatbot for ATS queries"""

    def __init__(self, parser: CVParser):
        self.parser = parser
        self.conversation_history = []

    def chat(self, query: str) -> Dict:
        self.conversation_history.append({"role": "user", "content": query})
        context = self._retrieve_context(query)
        answer = None
        try:
            if model is not None:
                prompt = f"You are an ATS assistant. Context:\n{context}\nQuestion: {query}"
                response = model.generate_content(prompt)
                answer = response.text.strip()
        except Exception:
            answer = self._generate_fallback_response(query, context)
        if not answer:
            answer = self._generate_fallback_response(query, context)
        self.conversation_history.append({"role": "assistant", "content": answer})
        return {"query": query, "response": answer, "context_used": bool(context), "timestamp": datetime.now().isoformat()}

    def _generate_fallback_response(self, query: str, context: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ['job', 'position', 'opening', 'role']):
            open_jobs = self.parser.jobs_df[self.parser.jobs_df['status'] == 'Open']
            return f"We currently have {len(open_jobs)} open positions."
        cv_id_match = re.search(r'cv[_\s]?(\d{4})', query_lower)
        if cv_id_match:
            cv_id = f"CV_{cv_id_match.group(1)}"
            cv = self.parser.get_cv_by_id(cv_id)
            if cv:
                return f"{cv['name']} has {cv['years_experience']} years experience. Skills: {', '.join(cv['skills'][:5])}."
            return f"Could not find candidate {cv_id}."
        if 'python' in query_lower:
            python_cvs = [cv for cv in self.parser.get_all_cvs() if 'python' in [s.lower() for s in cv['skills']]]
            if python_cvs:
                years = [cv['years_experience'] for cv in python_cvs]
                return f"We have {len(python_cvs)} Python candidates from {min(years)} to {max(years)} years experience."
            return "No Python candidates found."
        total_cvs = len(self.parser.cv_df)
        avg_exp = self.parser.cv_df['years_of_experience'].mean() if total_cvs else 0
        return f"Our DB has {total_cvs} candidates with an average of {avg_exp:.1f} years experience."

    def _retrieve_context(self, query: str) -> str:
        query_lower = query.lower()
        cv_id_match = re.search(r'cv[_\s]?(\d{4})', query_lower)
        if cv_id_match:
            cv_id = f"CV_{cv_id_match.group(1)}"
            cv = self.parser.get_cv_by_id(cv_id)
            if cv:
                return f"Candidate: {cv['name']}, {cv['years_experience']} years, Skills: {', '.join(cv['skills'][:5])}"
        if any(word in query_lower for word in ['job', 'position', 'opening', 'role']):
            jobs = self.parser.jobs_df.head(3)
            context = "Available positions:\n"
            for _, job in jobs.iterrows():
                context += f"- {job['job_title']} ({job['department']}) - {job['required_experience_years']} years required\n"
            return context
        if 'python' in query_lower or 'java' in query_lower or 'skill' in query_lower:
            cvs = self.parser.get_all_cvs()[:3]
            context = "Sample candidates:\n"
            for cv in cvs:
                context += f"- {cv['name']}: {', '.join(cv['skills'][:4])}\n"
            return context
        return "General ATS database."


# ============================================================================
# 5. DOCUMENT PROCESSOR - Extract text from files & parse CVs
# ============================================================================

class DocumentProcessor:
    """Process various HR documents and parse CV text into structured fields"""

    def __init__(self):
        pass

    def extract_text_from_file(self, file_bytes: bytes, filename: str) -> str:
        """Extract plain text from PDF / DOCX / TXT files. Returns a best-effort text string."""
        lower = filename.lower()
        text = ""

        # TXT
        if lower.endswith('.txt'):
            try:
                text = file_bytes.decode('utf-8', errors='ignore')
                return text
            except Exception:
                return file_bytes.decode('latin-1', errors='ignore')

        # PDF
        if lower.endswith('.pdf') and PyPDF2 is not None:
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                pages = []
                for p in reader.pages:
                    try:
                        pages.append(p.extract_text() or "")
                    except Exception:
                        continue
                return "\n".join(pages)
            except Exception:
                pass

        # DOCX
        if lower.endswith('.docx') and docx is not None:
            try:
                bio = io.BytesIO(file_bytes)
                doc = docx.Document(bio)
                full = []
                for para in doc.paragraphs:
                    full.append(para.text)
                return "\n".join(full)
            except Exception:
                pass

        # CSV - return as text for upstream parsing
        if lower.endswith('.csv'):
            try:
                return file_bytes.decode('utf-8', errors='ignore')
            except:
                return ""

        # As a last resort, try decoding as text
        try:
            return file_bytes.decode('utf-8', errors='ignore')
        except:
            return file_bytes.decode('latin-1', errors='ignore')

    def extract_structured_data(self, text: str, doc_type: str) -> Dict:
        """
        Use Gemini to extract structured JSON from document if available.
        Fallback returns error and raw_text.
        """
        prompts = {
            "invoice": "Extract: invoice_number, date, amount, vendor, items. Return as JSON.",
            "contract": "Extract: contract_id, parties, start_date, end_date, terms. Return as JSON.",
            "leave_request": "Extract: employee_name, leave_type, start_date, end_date, reason. Return as JSON.",
            "general": "Extract key information and structure it as JSON."
        }
        prompt = f"Document Type: {doc_type}\n\n{prompts.get(doc_type, prompts['general'])}\n\nDocument Content:\n{text[:2000]}\n\nReturn ONLY valid JSON, no other text."
        try:
            if model is not None:
                response = model.generate_content(prompt)
                json_text = response.text.strip()
                json_text = re.sub(r'^```json\s*', '', json_text)
                json_text = re.sub(r'\s*```$', '', json_text)
                return json.loads(json_text)
        except Exception:
            pass
        return {"error": "Failed to parse document with AI", "raw_text": text[:500]}

    # ----------------- New: CV parsing -----------------
    def parse_cv_text(self, text: str) -> Dict:
        """
        Try to parse a CV text into structured fields.
        Uses AI (Gemini) if available; otherwise uses heuristic regex extraction.
        Returns a dict containing fields required by add_cv_record.
        """
        # Try AI first
        if model is not None:
            prompt = f"""
            Parse the following CV/resume into JSON with keys:
            first_name, last_name, email, phone, location, current_job_title,
            years_of_experience, education, skills (as list), certifications (as list),
            work_history, expected_salary, notice_period_days, application_date
            CV TEXT:
            {text[:4000]}
            Return only JSON.
            """
            try:
                response = model.generate_content(prompt)
                json_text = response.text.strip()
                json_text = re.sub(r'^```json\s*', '', json_text)
                json_text = re.sub(r'\s*```$', '', json_text)
                parsed = json.loads(json_text)
                # Normalize some fields if necessary
                if 'skills' in parsed and isinstance(parsed['skills'], str):
                    parsed['skills'] = [s.strip() for s in parsed['skills'].split(',') if s.strip()]
                return parsed
            except Exception:
                pass

        # Fallback heuristics
        parsed = {
            "first_name": "",
            "last_name": "",
            "email": "",
            "phone": "",
            "location": "",
            "current_job_title": "",
            "years_of_experience": 0,
            "education": "",
            "skills": [],
            "certifications": [],
            "work_history": "",
            "expected_salary": 0,
            "notice_period_days": 0,
            "application_date": pd.Timestamp(date.today())
        }

        # Email
        m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if m:
            parsed['email'] = m.group(0)

        # Phone (simple)
        m = re.search(r'(\+?\d[\d\-\s\(\)]{7,}\d)', text)
        if m:
            parsed['phone'] = re.sub(r'\s+', ' ', m.group(0))

        # Name - take first non-empty line as candidate name (heuristic)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if lines:
            first = lines[0]
            # If first line has multiple words and not an email/phone, assume it's the name
            if not re.search(r'@|\d', first):
                parts = first.split()
                parsed['first_name'] = parts[0]
                parsed['last_name'] = ' '.join(parts[1:]) if len(parts) > 1 else ''

        # Experience - look for "X years" pattern
        m = re.search(r'(\d+)\+?\s+years', text.lower())
        if m:
            parsed['years_of_experience'] = int(m.group(1))

        # Skills - common "Skills:" section or comma lists after Skills
        skills = []
        m = re.search(r'(skills|technical skills|skill set)[:\s]*(.*)', text, re.IGNORECASE)
        if m:
            rest = m.group(2)
            # take next 200 chars and split by commas or newlines
            rest = rest[:300]
            skills = re.split(r'[,\n;]+', rest)
            skills = [s.strip() for s in skills if s.strip()]
        else:
            # fallback: gather words like Python, Java, SQL etc (simple)
            candidates = re.findall(r'\b(Python|JavaScript|Java|SQL|AWS|Docker|Kubernetes|Excel|C\+\+|C#|Django|Flask)\b', text, re.IGNORECASE)
            skills = list({c.strip() for c in candidates})

        parsed['skills'] = [s for s in skills if s]

        # Current job title - look for "Current" or first "Experience" block header
        m = re.search(r'(current|profile|summary)[:\s]*(.*)', text, re.IGNORECASE)
        if m:
            parsed['current_job_title'] = m.group(2).split('\n')[0][:100].strip()
        else:
            # use second line as fallback
            if len(lines) > 1:
                parsed['current_job_title'] = lines[1][:100]

        # Education - find common degree names
        edu = []
        m = re.search(r'(Bachelor|Master|B\.Sc|M\.Sc|BS|MS|PhD|Degree).{0,120}', text, re.IGNORECASE)
        if m:
            parsed['education'] = m.group(0).strip()

        # Work history - take a portion of text under 'Experience' header
        m = re.search(r'(experience|work history)[:\n\r]+(.{20,800})', text, re.IGNORECASE | re.DOTALL)
        if m:
            parsed['work_history'] = m.group(2)[:1000].strip()
        else:
            parsed['work_history'] = '\n'.join(lines[0:10])

        return parsed


# ============================================================================
# 6. MAIN DEMO & TESTING - unchanged
# ============================================================================

def run_ats_demo():
    """Complete ATS system demonstration (unchanged demo)"""
    # (Demo truncated for brevity - same as before)
    print("Run demo manually if needed.")

if __name__ == "__main__":
    run_ats_demo()


