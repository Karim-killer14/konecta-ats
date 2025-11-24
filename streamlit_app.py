"""
Streamlit ATS System - Web Application
Complete AI-powered recruitment system accessible via web browser
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from collections import Counter

# Import ATS components
from ATS_System import (
    CVParser, CVJobMatcher, ExplainableMatcher,
    ATSChatbot, DocumentProcessor
)

# Page configuration
st.set_page_config(
    page_title="Konecta ATS System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def init_ats_system():
    """Initialize ATS components (cached)"""
    parser = CVParser()
    if not parser.load_csv_data('cv_dataset.csv', 'job_descriptions.csv'):
        st.error("❌ Failed to load data files!")
        st.stop()
    
    matcher = CVJobMatcher()
    explainer = ExplainableMatcher()
    chatbot = ATSChatbot(parser)
    
    return parser, matcher, explainer, chatbot

# Load components
parser, matcher, explainer, chatbot = init_ats_system()

# Sidebar Navigation
st.sidebar.title("🎯 Konecta ATS")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "👥 Candidates", "💼 Jobs", "🤝 Matching", 
     "💡 Explainability", "💬 Chatbot", "📊 Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**System Stats**
- 📄 CVs: {len(parser.cv_df)}
- 💼 Jobs: {len(parser.jobs_df)}
- ✅ Open: {len(parser.jobs_df[parser.jobs_df['status'] == 'Open'])}
""")

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">🎯 Konecta ATS Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Recruitment System")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Candidates", len(parser.cv_df))
    with col2:
        st.metric("Total Jobs", len(parser.jobs_df))
    with col3:
        open_jobs = len(parser.jobs_df[parser.jobs_df['status'] == 'Open'])
        st.metric("Open Positions", open_jobs)
    with col4:
        avg_exp = parser.cv_df['years_of_experience'].mean()
        st.metric("Avg Experience", f"{avg_exp:.1f} yrs")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Experience Distribution")
        exp_dist = parser.cv_df['years_of_experience'].value_counts().sort_index()
        fig = px.bar(x=exp_dist.index, y=exp_dist.values,
                     labels={'x': 'Years of Experience', 'y': 'Number of Candidates'},
                     color=exp_dist.values, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📍 Location Distribution")
        loc_dist = parser.cv_df['location'].value_counts()
        fig = px.pie(values=loc_dist.values, names=loc_dist.index,
                     title="Candidate Locations")
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Skills
    st.subheader("🔥 Top Skills in Database")
    all_skills = []
    for skills_str in parser.cv_df['skills'].dropna():
        all_skills.extend([s.strip() for s in skills_str.split(',')])
    
    skill_counts = Counter(all_skills)
    top_skills = pd.DataFrame(skill_counts.most_common(10), columns=['Skill', 'Count'])
    
    fig = px.bar(top_skills, x='Count', y='Skill', orientation='h',
                 color='Count', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent Applications
    st.subheader("📅 Recent Applications")
    recent = parser.cv_df.nlargest(5, 'application_date')[
        ['cv_id', 'first_name', 'last_name', 'current_job_title', 
         'years_of_experience', 'application_date']
    ]
    st.dataframe(recent, use_container_width=True)

# ============================================================================
# PAGE 2: CANDIDATES
# ============================================================================
elif page == "👥 Candidates":
    st.title("👥 Candidate Management")
    
    # Filters
    st.sidebar.subheader("🔍 Filters")
    
    min_exp = st.sidebar.slider("Min Experience (years)", 0, 10, 0)
    max_exp = st.sidebar.slider("Max Experience (years)", 0, 10, 10)
    
    location = st.sidebar.multiselect(
        "Location",
        options=parser.cv_df['location'].unique(),
        default=[]
    )
    
    skill_search = st.sidebar.text_input("Search by Skill (comma-separated)")
    
    # Apply filters
    filters = {
        'min_experience': min_exp,
        'max_experience': max_exp
    }
    
    if location:
        # Filter by multiple locations
        filtered_df = parser.cv_df[
            (parser.cv_df['years_of_experience'] >= min_exp) &
            (parser.cv_df['years_of_experience'] <= max_exp) &
            (parser.cv_df['location'].isin(location))
        ]
        cvs = [parser.get_cv_by_id(cv_id) for cv_id in filtered_df['cv_id'].tolist()]
    elif skill_search:
        skills = [s.strip() for s in skill_search.split(',')]
        filters['skills'] = skills
        cvs = parser.search_cvs(filters)
    else:
        cvs = parser.search_cvs(filters)
    
    st.success(f"Found {len(cvs)} candidates matching criteria")
    
    # Display candidates
    for cv in cvs[:20]:  # Limit to 20 for performance
        with st.expander(f"📄 {cv['name']} - {cv['current_title']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Email:** {cv['email']}")
                st.write(f"**Phone:** {cv['phone']}")
                st.write(f"**Location:** {cv['location']}")
            
            with col2:
                st.write(f"**Experience:** {cv['years_experience']} years")
                st.write(f"**Education:** {cv['education']}")
                st.write(f"**Expected Salary:** ${cv['expected_salary']:,}")
            
            with col3:
                st.write(f"**Notice Period:** {cv['notice_period_days']} days")
                st.write(f"**Applied:** {cv['application_date']}")
            
            st.write("**Skills:**")
            st.write(", ".join(cv['skills'][:10]))
            
            if cv['certifications']:
                st.write("**Certifications:**")
                st.write(", ".join(cv['certifications']))

# ============================================================================
# PAGE 3: JOBS
# ============================================================================
elif page == "💼 Jobs":
    st.title("💼 Job Postings")
    
    # Filter by status
    status_filter = st.sidebar.radio("Status", ["All", "Open", "Closed"])
    
    jobs = [parser.get_job_by_id(job_id) for job_id in parser.jobs_df['job_id'].tolist()]
    
    if status_filter != "All":
        jobs = [j for j in jobs if j['status'] == status_filter]
    
    st.success(f"Showing {len(jobs)} job postings")
    
    # Display jobs
    for job in jobs:
        status_emoji = "✅" if job['status'] == "Open" else "⏸️"
        
        with st.expander(f"{status_emoji} {job['title']} - {job['department']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Job ID:** {job['job_id']}")
                st.write(f"**Department:** {job['department']}")
                st.write(f"**Required Experience:** {job['required_experience']} years")
                st.write(f"**Location:** {job['location']}")
                st.write(f"**Employment Type:** {job['employment_type']}")
            
            with col2:
                st.write(f"**Salary Range:** ${job['salary_range']}")
                st.write(f"**Openings:** {job['openings']}")
                st.write(f"**Posted:** {job['posted_date']}")
                st.write(f"**Status:** {job['status']}")
            
            st.write("**Required Skills:**")
            st.write(", ".join(job['required_skills']))

# ============================================================================
# PAGE 4: MATCHING
# ============================================================================
elif page == "🤝 Matching":
    st.title("🤝 CV-Job Matching")
    
    tab1, tab2 = st.tabs(["Single Match", "Rank Candidates"])
    
    # TAB 1: Single Match
    with tab1:
        st.subheader("Match CV to Job")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cv_id = st.selectbox("Select Candidate", parser.cv_df['cv_id'].tolist())
            cv = parser.get_cv_by_id(cv_id)
            
            if cv:
                st.info(f"**{cv['name']}**\n\n{cv['current_title']}\n\n{cv['years_experience']} years experience")
        
        with col2:
            job_id = st.selectbox("Select Job", parser.jobs_df['job_id'].tolist())
            job = parser.get_job_by_id(job_id)
            
            if job:
                st.info(f"**{job['title']}**\n\n{job['department']}\n\n{job['required_experience']} years required")
        
        if st.button("🎯 Match", type="primary"):
            with st.spinner("Analyzing match..."):
                match_result = matcher.match_cv_to_job(cv, job)
                
                # Display results
                st.markdown("### Match Results")
                
                score = match_result['overall_score']
                color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
                
                st.markdown(f"### Overall Match: <span style='color:{color}'>{score*100:.1f}%</span>", 
                           unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Skill Match", f"{match_result['skill_match']*100:.1f}%")
                col2.metric("Experience Match", f"{match_result['experience_match']*100:.1f}%")
                col3.metric("Text Similarity", f"{match_result['text_similarity']*100:.1f}%")
                
                # Matched skills
                if match_result['matched_skills']:
                    st.success("✅ Matched Skills: " + ", ".join(match_result['matched_skills']))
                
                # Missing skills
                if match_result['missing_skills']:
                    st.warning("⚠️ Missing Skills: " + ", ".join(match_result['missing_skills']))
    
    # TAB 2: Rank Candidates
    with tab2:
        st.subheader("Rank Candidates for Job")
        
        job_id = st.selectbox("Select Job Position", parser.jobs_df['job_id'].tolist(), key="rank_job")
        top_n = st.slider("Number of candidates to show", 5, 20, 10)
        
        if st.button("🏆 Rank Candidates", type="primary"):
            with st.spinner("Ranking candidates..."):
                job = parser.get_job_by_id(job_id)
                cvs = parser.get_all_cvs()
                
                ranked = matcher.rank_candidates(cvs, job, top_n=top_n)
                
                st.success(f"Top {len(ranked)} candidates for {job['title']}")
                
                # Display ranked candidates
                for i, candidate in enumerate(ranked, 1):
                    score = candidate['score']
                    color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
                    
                    with st.expander(f"#{i} - {candidate['name']} - {score*100:.1f}%"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Current Role:** {candidate['cv_data']['current_title']}")
                            st.write(f"**Experience:** {candidate['cv_data']['years_experience']} years")
                            st.write(f"**Location:** {candidate['cv_data']['location']}")
                        
                        with col2:
                            details = candidate['match_details']
                            st.metric("Skill Match", f"{details['skill_match']*100:.0f}%")
                            st.metric("Experience Match", f"{details['experience_match']*100:.0f}%")
                        
                        st.write("**Matched Skills:**", ", ".join(details['matched_skills'][:5]))
                        if details['missing_skills']:
                            st.write("**Gaps:**", ", ".join(details['missing_skills'][:3]))

# ============================================================================
# PAGE 5: EXPLAINABILITY
# ============================================================================
elif page == "💡 Explainability":
    st.title("💡 AI Explainability")
    
    st.markdown("Get detailed AI-powered explanations for candidate-job matches")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cv_id = st.selectbox("Candidate", parser.cv_df['cv_id'].tolist(), key="explain_cv")
    
    with col2:
        job_id = st.selectbox("Job", parser.jobs_df['job_id'].tolist(), key="explain_job")
    
    if st.button("🔍 Generate Explanation", type="primary"):
        with st.spinner("Generating AI explanation..."):
            cv = parser.get_cv_by_id(cv_id)
            job = parser.get_job_by_id(job_id)
            
            match_result = matcher.match_cv_to_job(cv, job)
            
            # AI Reasoning
            st.markdown("### 🤖 AI Analysis")
            reasoning = explainer.generate_match_reasoning(cv, job, match_result)
            st.info(reasoning)
            
            # Skill Gap Analysis
            st.markdown("### 📊 Skill Gap Analysis")
            skill_gap = explainer.generate_skill_gap_analysis(cv, job)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Matched Skills ({len(skill_gap['matched_skills'])})**")
                st.write(", ".join(skill_gap['matched_skills']) if skill_gap['matched_skills'] else "None")
            
            with col2:
                st.warning(f"**Missing Skills ({len(skill_gap['missing_skills'])})**")
                st.write(", ".join(skill_gap['missing_skills']) if skill_gap['missing_skills'] else "None")
            
            # Match Percentage
            st.metric("Skill Match Percentage", f"{skill_gap['match_percentage']}%")
            
            # Interview Questions
            st.markdown("### ❓ Suggested Interview Questions")
            questions = explainer.generate_interview_questions(cv, job, match_result)
            
            for i, question in enumerate(questions, 1):
                st.write(f"{i}. {question}")

# ============================================================================
# PAGE 6: CHATBOT
# ============================================================================
elif page == "💬 Chatbot":
    st.title("💬 ATS Assistant Chatbot")
    
    st.markdown("Ask questions about candidates, jobs, or the recruitment process")
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the ATS..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.chat(prompt)
                answer = response['response']
                st.markdown(answer)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        chatbot.clear_history()
        st.rerun()

# ============================================================================
# PAGE 7: ANALYTICS
# ============================================================================
elif page == "📊 Analytics":
    st.title("📊 System Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Overview", "Skills Analysis", "Job Analytics"])
    
    with tab1:
        st.subheader("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total CVs", len(parser.cv_df))
        with col2:
            st.metric("Total Jobs", len(parser.jobs_df))
        with col3:
            st.metric("Open Positions", len(parser.jobs_df[parser.jobs_df['status'] == 'Open']))
        with col4:
            st.metric("Avg Experience", f"{parser.cv_df['years_of_experience'].mean():.1f} yrs")
        
        # Experience distribution over time
        st.subheader("📈 Applications Over Time")
        parser.cv_df['application_date'] = pd.to_datetime(parser.cv_df['application_date'])
        apps_by_date = parser.cv_df.groupby('application_date').size().reset_index(name='count')
        
        fig = px.line(apps_by_date, x='application_date', y='count',
                     title="Daily Applications",
                     labels={'application_date': 'Date', 'count': 'Applications'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Skills Demand Analysis")
        
        # All skills
        all_skills = []
        for skills_str in parser.cv_df['skills'].dropna():
            all_skills.extend([s.strip() for s in skills_str.split(',')])
        
        skill_counts = Counter(all_skills)
        top_20 = pd.DataFrame(skill_counts.most_common(20), columns=['Skill', 'Count'])
        
        fig = px.bar(top_20, x='Skill', y='Count',
                    title="Top 20 Skills in Candidate Pool",
                    color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Job Market Analysis")
        
        # Jobs by department
        dept_dist = parser.jobs_df['department'].value_counts()
        fig = px.pie(values=dept_dist.values, names=dept_dist.index,
                    title="Jobs by Department")
        st.plotly_chart(fig, use_container_width=True)
        
        # Experience requirements
        exp_req = parser.jobs_df.groupby('required_experience_years').size().reset_index(name='count')
        fig = px.bar(exp_req, x='required_experience_years', y='count',
                    title="Experience Requirements Distribution",
                    labels={'required_experience_years': 'Years Required', 'count': 'Number of Jobs'})
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎯 Konecta ATS System v1.0 | Powered by Gemini 2.0 Flash | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)