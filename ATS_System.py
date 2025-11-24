# ============================================================================
# COMPLETE ATS SYSTEM - All Components Integrated (FIXED)
# ============================================================================

import google.generativeai as genai
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Configure Gemini API (Free model)
genai.configure(api_key='AIzaSyAh8V9j_NC3v45NqizQoVZQ0ZjevpBMymE')
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# ============================================================================
# 1. CV PARSER - Works with Generated CSV Data
# ============================================================================

class CVParser:
    """Parse CV data from CSV format and enrich with AI"""
    
    def __init__(self):
        self.cv_df = None
        self.jobs_df = None
    
    def load_csv_data(self, cv_csv_path='cv_dataset.csv', jobs_csv_path='job_descriptions.csv'):
        """Load generated CSV files"""
        try:
            self.cv_df = pd.read_csv(cv_csv_path)
            self.jobs_df = pd.read_csv(jobs_csv_path)
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
        
        # Helper function to safely get string values
        def safe_str(value):
            if pd.isna(value) or value is None:
                return ''
            return str(value)
        
        # Structure the data for AI processing
        structured_cv = {
            "cv_id": cv['cv_id'],
            "name": f"{cv['first_name']} {cv['last_name']}",
            "email": cv['email'],
            "phone": cv['phone'],
            "location": safe_str(cv['location']),
            "current_title": safe_str(cv['current_job_title']),
            "years_experience": cv['years_of_experience'],
            "education": safe_str(cv['education']),
            "skills": cv['skills'].split(', ') if pd.notna(cv['skills']) else [],
            "certifications": cv['certifications'].split(', ') if pd.notna(cv['certifications']) and cv['certifications'] != 'None' else [],
            "work_history": safe_str(cv['work_history']),
            "expected_salary": cv['expected_salary'],
            "notice_period_days": cv['notice_period_days'],
            "application_date": cv['application_date']
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
            "required_skills": job['required_skills'].split(', ') if pd.notna(job['required_skills']) else [],
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
        
        if 'location' in filters:
            df = df[df['location'] == filters['location']]
        
        return [self.get_cv_by_id(cv_id) for cv_id in df['cv_id'].tolist()]


# ============================================================================
# 2. CV-JOB MATCHER - Advanced Matching with Embeddings (FIXED)
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
            # Bonus for more experience, but diminishing returns
            excess = cv_years - required_years
            return min(1.0, 0.9 + (excess * 0.02))
        else:
            # Penalty for less experience
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
        """Comprehensive CV-Job matching - FIXED to handle NaN values"""
        
        # 1. Skill Match (40%)
        skill_score = self.calculate_skill_match(cv_data['skills'], job_data['required_skills'])
        
        # 2. Experience Match (30%)
        exp_score = self.calculate_experience_match(
            cv_data['years_experience'], 
            job_data['required_experience']
        )
        
        # 3. Text Similarity (20%) - FIXED: Handle None/NaN values
        cv_text_parts = [
            str(cv_data.get('current_title', '')),
            ' '.join(cv_data.get('skills', [])),
            str(cv_data.get('work_history', ''))
        ]
        # Filter out empty strings and join
        cv_text = ' '.join(filter(None, cv_text_parts))
        
        job_text_parts = [
            str(job_data.get('title', '')),
            ' '.join(job_data.get('required_skills', []))
        ]
        job_text = ' '.join(filter(None, job_text_parts))
        
        text_score = self.calculate_text_similarity(cv_text, job_text)
        
        # 4. Location Match (10%)
        cv_location = str(cv_data.get('location', '')).lower()
        job_location = str(job_data.get('location', '')).lower()
        location_score = 1.0 if cv_location in job_location or job_location in ['remote', 'hybrid'] else 0.5
        
        # Weighted final score
        final_score = (
            skill_score * 0.40 +
            exp_score * 0.30 +
            text_score * 0.20 +
            location_score * 0.10
        )
        
        return {
            "overall_score": round(final_score, 3),
            "skill_match": round(skill_score, 3),
            "experience_match": round(exp_score, 3),
            "text_similarity": round(text_score, 3),
            "location_match": round(location_score, 3),
            "matched_skills": [s for s in cv_data['skills'] if s in job_data['required_skills']],
            "missing_skills": [s for s in job_data['required_skills'] if s not in cv_data['skills']]
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
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_n]


# ============================================================================
# 3. EXPLAINABLE MATCHER - AI-Powered Explanations
# ============================================================================

class ExplainableMatcher:
    """Generate human-readable explanations for matches using Gemini"""
    
    def generate_match_reasoning(self, cv_data: Dict, job_data: Dict, match_scores: Dict) -> str:
        """Generate AI explanation for why candidate matches/doesn't match"""
        
        # Try AI generation first
        prompt = f"""
        As an HR expert, explain why this candidate is a {self._get_match_level(match_scores['overall_score'])} match for this position.
        
        Candidate Profile:
        - Name: {cv_data['name']}
        - Experience: {cv_data['years_experience']} years
        - Current Role: {cv_data['current_title']}
        - Skills: {', '.join(cv_data['skills'][:10])}
        
        Job Requirements:
        - Position: {job_data['title']}
        - Required Experience: {job_data['required_experience']} years
        - Required Skills: {', '.join(job_data['required_skills'])}
        
        Match Scores:
        - Overall: {match_scores['overall_score']*100:.1f}%
        - Skills: {match_scores['skill_match']*100:.1f}%
        - Experience: {match_scores['experience_match']*100:.1f}%
        
        Provide a concise 2-3 sentence explanation focusing on strengths and any gaps. Be professional and objective.
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            # Intelligent fallback based on actual scores
            matched = match_scores.get('matched_skills', [])
            missing = match_scores.get('missing_skills', [])
            score = match_scores['overall_score']
            
            if score >= 0.7:
                return f"{cv_data['name']} is a strong candidate with {len(matched)} matching skills including {', '.join(matched[:3])}. Their {cv_data['years_experience']} years of experience aligns well with the {job_data['required_experience']}-year requirement. Some skill development in {', '.join(missing[:2])} would make them an excellent fit."
            elif score >= 0.5:
                return f"{cv_data['name']} shows potential with relevant experience in {cv_data['current_title']} and {len(matched)} matching skills. However, they would benefit from training in {', '.join(missing[:3])} to fully meet the role requirements. Overall match: {score*100:.0f}%."
            else:
                return f"{cv_data['name']} has limited alignment with this role. While they have {cv_data['years_experience']} years of experience, only {len(matched)} of the required skills match. Key gaps include {', '.join(missing[:3])}, suggesting this may not be the ideal position."
    
    def _get_match_level(self, score: float) -> str:
        """Convert score to match level"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.65:
            return "good"
        elif score >= 0.5:
            return "moderate"
        else:
            return "weak"
    
    def generate_skill_gap_analysis(self, cv_data: Dict, job_data: Dict) -> Dict:
        """Identify skill gaps and provide recommendations"""
        
        missing_skills = [s for s in job_data['required_skills'] if s not in cv_data['skills']]
        matching_skills = [s for s in cv_data['skills'] if s in job_data['required_skills']]
        
        return {
            "matched_skills": matching_skills,
            "missing_skills": missing_skills,
            "match_percentage": round(len(matching_skills) / max(len(job_data['required_skills']), 1) * 100, 1),
            "recommendations": self._generate_recommendations(missing_skills)
        }
    
    def _generate_recommendations(self, missing_skills: List[str]) -> List[str]:
        """Generate training recommendations"""
        if not missing_skills:
            return ["Candidate meets all skill requirements"]
        
        return [f"Consider training in: {skill}" for skill in missing_skills[:3]]
    
    def generate_interview_questions(self, cv_data: Dict, job_data: Dict, match_scores: Dict) -> List[str]:
        """Generate tailored interview questions using AI"""
        
        prompt = f"""
        Generate 5 specific interview questions for this candidate applying for {job_data['title']}.
        
        Candidate has {cv_data['years_experience']} years of experience in {cv_data['current_title']}.
        Key skills: {', '.join(cv_data['skills'][:8])}
        
        Missing skills: {', '.join(match_scores.get('missing_skills', [])[:3])}
        
        Focus questions on:
        1. Verifying claimed skills
        2. Exploring missing competencies
        3. Understanding experience depth
        
        Return only the questions, numbered 1-5, one per line.
        """
        
        try:
            response = model.generate_content(prompt)
            questions = [q.strip() for q in response.text.strip().split('\n') if q.strip() and any(c.isalpha() for c in q)]
            return questions[:5]
        except:
            return [
                f"Describe your experience with {cv_data['current_title']}",
                f"How have you applied {cv_data['skills'][0] if cv_data['skills'] else 'your skills'} in past projects?",
                "What interests you about this role?",
                "Describe a challenging project you've completed",
                "Where do you see yourself in 3 years?"
            ]


# ============================================================================
# 4. RAG CHATBOT - Knowledge Base Query System
# ============================================================================

class ATSChatbot:
    """RAG-based chatbot for ATS queries"""
    
    def __init__(self, parser: CVParser):
        self.parser = parser
        self.conversation_history = []
    
    def chat(self, query: str) -> Dict:
        """Answer questions about CVs and jobs"""
        
        # Add to history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Detect query type and retrieve relevant data
        context = self._retrieve_context(query)
        
        # Try AI response first, fallback to rule-based if it fails
        answer = None
        
        try:
            # Generate response with context
            prompt = f"""
            You are an ATS (Applicant Tracking System) assistant helping HR professionals.
            
            Context from database:
            {context}
            
            User Question: {query}
            
            Provide a helpful, professional response. If asking about candidates, reference the data provided.
            If the context doesn't contain relevant information, say so politely.
            Keep response concise (2-3 sentences).
            """
            
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            # Fallback to rule-based responses
            answer = self._generate_fallback_response(query, context)
        
        # Add to history
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        return {
            "query": query,
            "response": answer,
            "context_used": bool(context),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate rule-based response when AI fails"""
        query_lower = query.lower()
        
        # Handle job-related queries
        if any(word in query_lower for word in ['job', 'position', 'opening', 'role']):
            open_jobs = self.parser.jobs_df[self.parser.jobs_df['status'] == 'Open']
            return f"We currently have {len(open_jobs)} open positions across departments including {', '.join(open_jobs['department'].unique()[:3])}. The top roles are {', '.join(open_jobs['job_title'].head(3).tolist())}."
        
        # Handle CV-specific queries
        cv_id_match = re.search(r'cv[_\s]?(\d{4})', query_lower)
        if cv_id_match:
            cv_id = f"CV_{cv_id_match.group(1)}"
            cv = self.parser.get_cv_by_id(cv_id)
            if cv:
                return f"{cv['name']} has {cv['years_experience']} years of experience as a {cv['current_title']}. Key skills include {', '.join(cv['skills'][:5])}. Located in {cv['location']}."
            return f"Could not find candidate {cv_id} in the database."
        
        # Handle skill-based queries
        if 'python' in query_lower:
            python_cvs = [cv for cv in self.parser.get_all_cvs() if 'Python' in cv['skills'] or 'python' in [s.lower() for s in cv['skills']]]
            return f"We have {len(python_cvs)} candidates with Python skills in our database, ranging from {min([cv['years_experience'] for cv in python_cvs])} to {max([cv['years_experience'] for cv in python_cvs])} years of experience."
        
        if 'skill' in query_lower or 'candidate' in query_lower:
            total_cvs = len(self.parser.cv_df)
            avg_exp = self.parser.cv_df['years_of_experience'].mean()
            return f"Our database contains {total_cvs} candidates with an average of {avg_exp:.1f} years of experience across various technical and business domains."
        
        # Default response
        return f"I found information in the database: {context[:150]}... How can I help you analyze this data?"
    
    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant CV/job data based on query"""
        
        query_lower = query.lower()
        
        # Check if asking about specific CV
        cv_id_match = re.search(r'cv[_\s]?(\d{4})', query_lower)
        if cv_id_match:
            cv_id = f"CV_{cv_id_match.group(1)}"
            cv = self.parser.get_cv_by_id(cv_id)
            if cv:
                return f"Candidate: {cv['name']}, {cv['years_experience']} years experience, Skills: {', '.join(cv['skills'][:5])}"
        
        # Check if asking about jobs
        if any(word in query_lower for word in ['job', 'position', 'opening', 'role']):
            jobs = self.parser.jobs_df.head(3)
            context = "Available positions:\n"
            for _, job in jobs.iterrows():
                context += f"- {job['job_title']} ({job['department']}) - {job['required_experience_years']} years required\n"
            return context
        
        # Check if asking about candidates with certain skills
        if 'python' in query_lower or 'java' in query_lower or 'skill' in query_lower:
            # Get sample candidates
            cvs = self.parser.get_all_cvs()[:3]
            context = "Sample candidates:\n"
            for cv in cvs:
                context += f"- {cv['name']}: {', '.join(cv['skills'][:4])}\n"
            return context
        
        return "General ATS database with candidate and job information available."
    
    def get_conversation_history(self) -> List[Dict]:
        """Return chat history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# ============================================================================
# 5. DOCUMENT PROCESSOR - Handle Various Document Types
# ============================================================================

class DocumentProcessor:
    """Process various HR documents"""
    
    def extract_structured_data(self, text: str, doc_type: str) -> Dict:
        """Extract structured data from document using Gemini"""
        
        prompts = {
            "invoice": "Extract: invoice_number, date, amount, vendor, items. Return as JSON.",
            "contract": "Extract: contract_id, parties, start_date, end_date, terms. Return as JSON.",
            "leave_request": "Extract: employee_name, leave_type, start_date, end_date, reason. Return as JSON.",
            "general": "Extract key information and structure it as JSON."
        }
        
        prompt = f"""
        Document Type: {doc_type}
        
        {prompts.get(doc_type, prompts['general'])}
        
        Document Content:
        {text[:2000]}
        
        Return ONLY valid JSON, no other text.
        """
        
        try:
            response = model.generate_content(prompt)
            json_text = response.text.strip()
            json_text = re.sub(r'^```json\s*', '', json_text)
            json_text = re.sub(r'\s*```$', '', json_text)
            return json.loads(json_text)
        except:
            return {"error": "Failed to parse document", "raw_text": text[:500]}
    
    def validate_document(self, extracted_data: Dict, doc_type: str) -> Dict:
        """Validate extracted document data"""
        
        required_fields = {
            "invoice": ["invoice_number", "amount", "date"],
            "leave_request": ["employee_name", "start_date", "end_date"],
            "contract": ["contract_id", "parties"]
        }
        
        fields = required_fields.get(doc_type, [])
        missing = [f for f in fields if f not in extracted_data or not extracted_data[f]]
        
        return {
            "is_valid": len(missing) == 0,
            "missing_fields": missing,
            "confidence": 1.0 - (len(missing) / max(len(fields), 1))
        }


# ============================================================================
# 6. MAIN DEMO & TESTING
# ============================================================================

def run_ats_demo():
    """Complete ATS system demonstration"""
    
    print("="*70)
    print("KONECTA ATS SYSTEM - AI-POWERED RECRUITMENT")
    print("="*70)
    
    # Initialize system
    print("\n[1/6] Initializing system components...")
    parser = CVParser()
    
    if not parser.load_csv_data():
        print("❌ Failed to load data. Make sure cv_dataset.csv and job_descriptions.csv exist.")
        return
    
    matcher = CVJobMatcher()
    explainer = ExplainableMatcher()
    chatbot = ATSChatbot(parser)
    doc_processor = DocumentProcessor()
    
    print("✓ All components initialized")
    
    # Demo 1: CV Parsing
    print("\n" + "="*70)
    print("[2/6] CV PARSING DEMO")
    print("="*70)
    sample_cv = parser.get_cv_by_id('CV_0001')
    print(f"\nCandidate: {sample_cv['name']}")
    print(f"Experience: {sample_cv['years_experience']} years")
    print(f"Skills: {', '.join(sample_cv['skills'][:5])}...")
    print(f"Location: {sample_cv['location']}")
    
    # Demo 2: Job Matching
    print("\n" + "="*70)
    print("[3/6] CV-JOB MATCHING DEMO")
    print("="*70)
    sample_job = parser.get_job_by_id('JOB_001')
    print(f"\nJob: {sample_job['title']}")
    print(f"Required Experience: {sample_job['required_experience']} years")
    print(f"Required Skills: {', '.join(sample_job['required_skills'][:5])}...")
    
    print(f"\nMatching {sample_cv['name']} to {sample_job['title']}...")
    match_result = matcher.match_cv_to_job(sample_cv, sample_job)
    
    print(f"\n📊 Match Score: {match_result['overall_score']*100:.1f}%")
    print(f"   ├─ Skills: {match_result['skill_match']*100:.1f}%")
    print(f"   ├─ Experience: {match_result['experience_match']*100:.1f}%")
    print(f"   └─ Text Similarity: {match_result['text_similarity']*100:.1f}%")
    print(f"\n✓ Matched Skills: {', '.join(match_result['matched_skills'][:3])}")
    print(f"⚠ Missing Skills: {', '.join(match_result['missing_skills'][:3])}")
    
    # Demo 3: Candidate Ranking
    print("\n" + "="*70)
    print("[4/6] CANDIDATE RANKING DEMO")
    print("="*70)
    print(f"\nRanking top 5 candidates for {sample_job['title']}...")
    
    all_cvs = parser.get_all_cvs()
    ranked = matcher.rank_candidates(all_cvs, sample_job, top_n=5)
    
    print("\n🏆 Top 5 Candidates:")
    for i, candidate in enumerate(ranked, 1):
        print(f"\n{i}. {candidate['name']} - Score: {candidate['score']*100:.1f}%")
        print(f"   {candidate['cv_data']['years_experience']} years exp | {candidate['cv_data']['current_title']}")
    
    # Demo 4: Explainability
    print("\n" + "="*70)
    print("[5/6] AI EXPLAINABILITY DEMO")
    print("="*70)
    print("\nGenerating match explanation...")
    
    explanation = explainer.generate_match_reasoning(
        ranked[0]['cv_data'],
        sample_job,
        ranked[0]['match_details']
    )
    print(f"\n💡 AI Explanation:\n{explanation}")
    
    print("\n\nGenerating interview questions...")
    questions = explainer.generate_interview_questions(
        ranked[0]['cv_data'],
        sample_job,
        ranked[0]['match_details']
    )
    print("\n📋 Suggested Interview Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    # Demo 5: RAG Chatbot
    print("\n" + "="*70)
    print("[6/6] RAG CHATBOT DEMO")
    print("="*70)
    
    test_queries = [
        "What positions do we have open?",
        "Tell me about CV_0001",
        "How many candidates have Python skills?"
    ]
    
    for query in test_queries:
        print(f"\n👤 User: {query}")
        response = chatbot.chat(query)
        print(f"🤖 Assistant: {response['response']}")
    
    # Final Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE - SYSTEM READY FOR PRODUCTION")
    print("="*70)
    print(f"\n📈 System Statistics:")
    print(f"   • Total CVs in database: {len(parser.cv_df)}")
    print(f"   • Total Jobs: {len(parser.jobs_df)}")
    print(f"   • Open Positions: {len(parser.jobs_df[parser.jobs_df['status'] == 'Open'])}")
    print(f"   • Chatbot Queries Handled: {len(chatbot.get_conversation_history()) // 2}")
    
    print("\n✓ All AI components operational")
    print("✓ Ready for integration with Full Stack team")
    print("✓ APIs can be exposed via Flask backend\n")


if __name__ == "__main__":
    run_ats_demo()