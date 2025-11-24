"""
Flask API for ATS System
Exposes all AI functionality via REST endpoints
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from collections import Counter

# Import the complete ATS system
# Make sure to save the fixed ATS system as 'ats_system.py'
from ATS_System import (
    CVParser, CVJobMatcher, ExplainableMatcher, 
    ATSChatbot, DocumentProcessor
)

app = Flask(__name__)
CORS(app)

# Initialize components
print("Initializing ATS components...")
parser = CVParser()
if not parser.load_csv_data('cv_dataset.csv', 'job_descriptions.csv'):
    print("❌ ERROR: Could not load data files!")
    print("Make sure cv_dataset.csv and job_descriptions.csv exist in the same directory")
    exit(1)

matcher = CVJobMatcher()
explainer = ExplainableMatcher()
chatbot = ATSChatbot(parser)
doc_processor = DocumentProcessor()

print("✓ ATS API ready\n")

# ============================================================================
# CV ENDPOINTS
# ============================================================================

@app.route('/api/cvs', methods=['GET'])
def get_all_cvs():
    """Get all CVs with optional filters"""
    try:
        # Get query parameters
        min_exp = request.args.get('min_experience', type=int)
        max_exp = request.args.get('max_experience', type=int)
        skills = request.args.get('skills', '').split(',') if request.args.get('skills') else None
        location = request.args.get('location')
        
        filters = {}
        if min_exp is not None:
            filters['min_experience'] = min_exp
        if max_exp is not None:
            filters['max_experience'] = max_exp
        if skills:
            filters['skills'] = [s.strip() for s in skills if s.strip()]
        if location:
            filters['location'] = location
        
        if filters:
            cvs = parser.search_cvs(filters)
        else:
            cvs = parser.get_all_cvs()
        
        return jsonify({
            "success": True,
            "total": len(cvs),
            "cvs": cvs[:50]  # Limit to 50 for performance
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/cvs/<cv_id>', methods=['GET'])
def get_cv(cv_id):
    """Get single CV by ID"""
    try:
        cv = parser.get_cv_by_id(cv_id)
        
        if not cv:
            return jsonify({"success": False, "error": "CV not found"}), 404
        
        return jsonify({
            "success": True,
            "cv": cv
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/cvs/search', methods=['POST'])
def search_cvs():
    """Advanced CV search with multiple criteria"""
    try:
        filters = request.get_json()
        cvs = parser.search_cvs(filters)
        
        return jsonify({
            "success": True,
            "total": len(cvs),
            "filters_applied": filters,
            "cvs": cvs
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# JOB ENDPOINTS
# ============================================================================

@app.route('/api/jobs', methods=['GET'])
def get_all_jobs():
    """Get all job postings"""
    try:
        jobs = [parser.get_job_by_id(job_id) for job_id in parser.jobs_df['job_id'].tolist()]
        
        # Filter by status if provided
        status = request.args.get('status')
        if status:
            jobs = [j for j in jobs if j['status'].lower() == status.lower()]
        
        return jsonify({
            "success": True,
            "total": len(jobs),
            "jobs": jobs
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get single job by ID"""
    try:
        job = parser.get_job_by_id(job_id)
        
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        return jsonify({
            "success": True,
            "job": job
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# MATCHING ENDPOINTS
# ============================================================================

@app.route('/api/match', methods=['POST'])
def match_cv_to_job():
    """Match a single CV to a job"""
    try:
        data = request.get_json()
        cv_id = data.get('cv_id')
        job_id = data.get('job_id')
        
        if not cv_id or not job_id:
            return jsonify({"success": False, "error": "Both cv_id and job_id required"}), 400
        
        cv = parser.get_cv_by_id(cv_id)
        job = parser.get_job_by_id(job_id)
        
        if not cv or not job:
            return jsonify({"success": False, "error": "CV or Job not found"}), 404
        
        match_result = matcher.match_cv_to_job(cv, job)
        
        # Add explanation if requested
        include_explanation = data.get('include_explanation', True)
        if include_explanation:
            explanation = explainer.generate_match_reasoning(cv, job, match_result)
            match_result['ai_explanation'] = explanation
        
        return jsonify({
            "success": True,
            "cv_id": cv_id,
            "job_id": job_id,
            "candidate_name": cv['name'],
            "job_title": job['title'],
            "match_result": match_result,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/rank', methods=['POST'])
def rank_candidates():
    """Rank all candidates for a specific job"""
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        top_n = data.get('top_n', 10)
        
        if not job_id:
            return jsonify({"success": False, "error": "job_id required"}), 400
        
        job = parser.get_job_by_id(job_id)
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        # Get all CVs or filtered list
        filters = data.get('filters', {})
        if filters:
            cvs = parser.search_cvs(filters)
        else:
            cvs = parser.get_all_cvs()
        
        print(f"Ranking {len(cvs)} candidates for {job['title']}...")
        
        # Rank candidates
        ranked = matcher.rank_candidates(cvs, job, top_n=top_n)
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "job_title": job['title'],
            "total_candidates_evaluated": len(cvs),
            "top_candidates": ranked,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/match/batch', methods=['POST'])
def batch_match():
    """Match multiple CVs to multiple jobs"""
    try:
        data = request.get_json()
        cv_ids = data.get('cv_ids', [])
        job_ids = data.get('job_ids', [])
        
        if not cv_ids or not job_ids:
            return jsonify({"success": False, "error": "cv_ids and job_ids required"}), 400
        
        results = []
        
        for cv_id in cv_ids:
            cv = parser.get_cv_by_id(cv_id)
            if not cv:
                continue
            
            cv_results = []
            for job_id in job_ids:
                job = parser.get_job_by_id(job_id)
                if not job:
                    continue
                
                match = matcher.match_cv_to_job(cv, job)
                cv_results.append({
                    "job_id": job_id,
                    "job_title": job['title'],
                    "score": match['overall_score'],
                    "skill_match": match['skill_match']
                })
            
            # Sort by score
            cv_results.sort(key=lambda x: x['score'], reverse=True)
            
            results.append({
                "cv_id": cv_id,
                "name": cv['name'],
                "best_matches": cv_results[:3]
            })
        
        return jsonify({
            "success": True,
            "results": results
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# EXPLAINABILITY ENDPOINTS
# ============================================================================

@app.route('/api/explain', methods=['POST'])
def explain_match():
    """Get detailed explanation for a match"""
    try:
        data = request.get_json()
        cv_id = data.get('cv_id')
        job_id = data.get('job_id')
        
        cv = parser.get_cv_by_id(cv_id)
        job = parser.get_job_by_id(job_id)
        
        if not cv or not job:
            return jsonify({"success": False, "error": "CV or Job not found"}), 404
        
        match_result = matcher.match_cv_to_job(cv, job)
        
        # Generate comprehensive explanation
        reasoning = explainer.generate_match_reasoning(cv, job, match_result)
        skill_gap = explainer.generate_skill_gap_analysis(cv, job)
        questions = explainer.generate_interview_questions(cv, job, match_result)
        
        return jsonify({
            "success": True,
            "cv_id": cv_id,
            "job_id": job_id,
            "candidate_name": cv['name'],
            "job_title": job['title'],
            "match_score": match_result['overall_score'],
            "ai_reasoning": reasoning,
            "skill_analysis": skill_gap,
            "interview_questions": questions,
            "detailed_scores": match_result
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/skill-gap', methods=['POST'])
def analyze_skill_gap():
    """Analyze skill gaps for a candidate"""
    try:
        data = request.get_json()
        cv_id = data.get('cv_id')
        job_id = data.get('job_id')
        
        cv = parser.get_cv_by_id(cv_id)
        job = parser.get_job_by_id(job_id)
        
        if not cv or not job:
            return jsonify({"success": False, "error": "CV or Job not found"}), 404
        
        analysis = explainer.generate_skill_gap_analysis(cv, job)
        
        return jsonify({
            "success": True,
            "analysis": analysis
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# CHATBOT ENDPOINTS
# ============================================================================

@app.route('/api/chatbot/query', methods=['POST'])
def chatbot_query():
    """Ask chatbot a question"""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({"success": False, "error": "query required"}), 400
        
        response = chatbot.chat(query)
        
        return jsonify({
            "success": True,
            "response": response
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chatbot/history', methods=['GET'])
def get_chat_history():
    """Get conversation history"""
    try:
        history = chatbot.get_conversation_history()
        
        return jsonify({
            "success": True,
            "total_messages": len(history),
            "history": history
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chatbot/clear', methods=['POST'])
def clear_chat_history():
    """Clear conversation history"""
    try:
        chatbot.clear_history()
        
        return jsonify({
            "success": True,
            "message": "Chat history cleared"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.route('/api/analytics/overview', methods=['GET'])
def get_analytics_overview():
    """Get system analytics overview"""
    try:
        total_cvs = len(parser.cv_df)
        total_jobs = len(parser.jobs_df)
        open_jobs = len(parser.jobs_df[parser.jobs_df['status'] == 'Open'])
        
        # Calculate average experience
        avg_experience = parser.cv_df['years_of_experience'].mean()
        
        # Top skills
        all_skills = []
        for skills_str in parser.cv_df['skills'].dropna():
            all_skills.extend([s.strip() for s in skills_str.split(',')])
        
        skill_counts = Counter(all_skills)
        top_skills = [{"skill": k, "count": v} for k, v in skill_counts.most_common(10)]
        
        # Location distribution
        locations = parser.cv_df['location'].value_counts().head(5).to_dict()
        
        return jsonify({
            "success": True,
            "overview": {
                "total_candidates": total_cvs,
                "total_jobs": total_jobs,
                "open_positions": open_jobs,
                "avg_candidate_experience": round(avg_experience, 1),
                "top_skills": top_skills,
                "top_locations": locations
            },
            "chatbot": {
                "total_queries": len(chatbot.get_conversation_history()) // 2
            }
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/analytics/match-distribution', methods=['GET'])
def get_match_distribution():
    """Get distribution of match scores for a job"""
    try:
        job_id = request.args.get('job_id')
        
        if not job_id:
            return jsonify({"success": False, "error": "job_id required"}), 400
        
        job = parser.get_job_by_id(job_id)
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        # Sample 100 CVs for performance
        cvs = parser.get_all_cvs()[:100]
        scores = []
        
        for cv in cvs:
            match = matcher.match_cv_to_job(cv, job)
            scores.append(match['overall_score'])
        
        # Calculate distribution
        excellent = sum(1 for s in scores if s >= 0.8)
        good = sum(1 for s in scores if 0.65 <= s < 0.8)
        moderate = sum(1 for s in scores if 0.5 <= s < 0.65)
        weak = sum(1 for s in scores if s < 0.5)
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "job_title": job['title'],
            "sample_size": len(scores),
            "distribution": {
                "excellent": excellent,
                "good": good,
                "moderate": moderate,
                "weak": weak
            },
            "avg_score": round(sum(scores) / len(scores), 3) if scores else 0
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# HEALTH & INFO
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        "status": "healthy",
        "service": "Konecta ATS AI Service",
        "components": {
            "parser": "operational",
            "matcher": "operational",
            "explainer": "operational",
            "chatbot": "operational"
        },
        "data": {
            "cvs_loaded": len(parser.cv_df),
            "jobs_loaded": len(parser.jobs_df)
        },
        "timestamp": datetime.now().isoformat()
    })


@app.route('/', methods=['GET'])
def api_info():
    """API documentation"""
    return jsonify({
        "service": "Konecta ATS AI Service",
        "version": "1.0.0",
        "description": "AI-powered Applicant Tracking System for ERP",
        "endpoints": {
            "CVs": {
                "GET /api/cvs": "Get all CVs (supports filters: min_experience, max_experience, skills, location)",
                "GET /api/cvs/<cv_id>": "Get specific CV",
                "POST /api/cvs/search": "Advanced CV search with filters"
            },
            "Jobs": {
                "GET /api/jobs": "Get all jobs (supports filter: status)",
                "GET /api/jobs/<job_id>": "Get specific job"
            },
            "Matching": {
                "POST /api/match": "Match CV to job (body: {cv_id, job_id, include_explanation})",
                "POST /api/rank": "Rank candidates for job (body: {job_id, top_n, filters})",
                "POST /api/match/batch": "Batch matching (body: {cv_ids[], job_ids[]})"
            },
            "Explainability": {
                "POST /api/explain": "Get match explanation (body: {cv_id, job_id})",
                "POST /api/skill-gap": "Analyze skill gaps (body: {cv_id, job_id})"
            },
            "Chatbot": {
                "POST /api/chatbot/query": "Ask chatbot (body: {query})",
                "GET /api/chatbot/history": "Get chat history",
                "POST /api/chatbot/clear": "Clear history"
            },
            "Analytics": {
                "GET /api/analytics/overview": "System overview statistics",
                "GET /api/analytics/match-distribution?job_id=X": "Match score distribution for a job"
            }
        },
        "model": "Gemini 2.0 Flash (Free)",
        "data_source": "CSV-based candidate & job database",
        "docs": "Visit http://localhost:5000 for this documentation"
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 KONECTA ATS API SERVER")
    print("="*70)
    print(f"📍 Running on: http://localhost:5000")
    print(f"📖 API Docs: http://localhost:5000")
    print(f"💚 Health Check: http://localhost:5000/health")
    print("="*70)
    print(f"✓ {len(parser.cv_df)} CVs loaded")
    print(f"✓ {len(parser.jobs_df)} Jobs loaded")
    print(f"✓ All AI components operational")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)