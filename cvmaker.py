import pandas as pd
import random
from datetime import datetime, timedelta
import csv

# Set random seed for reproducibility
random.seed(42)

# Sample data pools
first_names = ["Ahmed", "Fatma", "Mohamed", "Sara", "Omar", "Nour", "Ali", "Layla", "Hassan", "Mona",
               "Youssef", "Heba", "Khaled", "Dina", "Mahmoud", "Rana", "Karim", "Salma", "Tarek", "Yasmin",
               "Amr", "Mariam", "Sherif", "Noha", "Hossam", "Dalia", "Mostafa", "Aya", "Adel", "Eman"]

last_names = ["Hassan", "Ibrahim", "Mahmoud", "Ali", "Ahmed", "Mostafa", "Abdel Rahman", "Sayed", 
              "Khalil", "Farouk", "Nasser", "Salem", "Kamal", "Rashed", "Fouad", "Samir", "Gamal",
              "Mansour", "Amin", "Zaki", "Shafik", "El Sayed", "Hamdi", "Taha", "Morsi", "El Din"]

job_titles = [
    "Software Engineer", "Data Analyst", "Project Manager", "HR Specialist", "Financial Analyst",
    "Marketing Specialist", "DevOps Engineer", "Full Stack Developer", "UI/UX Designer",
    "Business Analyst", "System Administrator", "Customer Success Manager", "Sales Representative",
    "Account Manager", "Product Manager", "Cloud Architect", "Cybersecurity Specialist",
    "Machine Learning Engineer", "Quality Assurance Engineer", "Technical Support Specialist"
]

skills_pool = {
    "technical": ["Python", "Java", "JavaScript", "C++", "SQL", "React", "Angular", "Node.js", 
                  "Django", "Flask", "Spring Boot", "AWS", "Azure", "Docker", "Kubernetes",
                  "Git", "Jenkins", "Machine Learning", "Data Analysis", "Power BI", "Tableau",
                  "Excel", "MongoDB", "PostgreSQL", "Redis", "Microservices", "REST APIs"],
    "soft": ["Communication", "Leadership", "Problem Solving", "Team Collaboration", "Time Management",
             "Critical Thinking", "Adaptability", "Attention to Detail", "Project Management",
             "Conflict Resolution", "Negotiation", "Presentation Skills", "Analytical Thinking"]
}

education_degrees = ["Bachelor's in Computer Science", "Bachelor's in Business Administration",
                     "Bachelor's in Engineering", "Master's in Data Science", "Master's in Business Administration",
                     "Bachelor's in Information Technology", "Bachelor's in Finance", "Bachelor's in Marketing",
                     "Master's in Computer Science", "Bachelor's in Economics"]

universities = ["Cairo University", "American University in Cairo", "Ain Shams University", 
                "Alexandria University", "German University in Cairo", "British University in Egypt",
                "Helwan University", "Mansoura University", "Nile University", "Suez Canal University"]

companies = ["IBM Egypt", "Microsoft Egypt", "Vodafone", "Orange Egypt", "Etisalat Misr",
             "Banque Misr", "Commercial International Bank", "Fawry", "Swvl", "Vezeeta",
             "Jumia Egypt", "Careem", "Uber Egypt", "Telecom Egypt", "E-Finance", "Raya IT"]

certifications = ["PMP Certification", "AWS Certified Solutions Architect", "Google Analytics Certified",
                  "Certified ScrumMaster", "CISSP", "CompTIA Security+", "Azure Fundamentals",
                  "Salesforce Administrator", "HubSpot Marketing", "SHRM-CP", "CFA Level 1",
                  "Six Sigma Green Belt", "ITIL Foundation", "Certified Kubernetes Administrator"]


def generate_phone():
    """Generate Egyptian phone number"""
    return f"+20 1{random.choice([0, 1, 2, 5])}{random.randint(10000000, 99999999)}"


def generate_email(first_name, last_name):
    """Generate email address"""
    domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]
    separators = [".", "_", ""]
    sep = random.choice(separators)
    return f"{first_name.lower()}{sep}{last_name.lower()}{random.randint(1, 999)}@{random.choice(domains)}"


def generate_experience_years():
    """Generate years of experience with weighted distribution"""
    weights = [0.15, 0.25, 0.25, 0.20, 0.10, 0.05]  # More junior to mid-level candidates
    experience_ranges = [0, 1, 2, 3, 5, 8]
    return random.choices(experience_ranges, weights=weights)[0] + random.randint(0, 2)


def generate_work_history(years_exp, job_title):
    """Generate work history based on years of experience"""
    history = []
    current_year = 2024
    years_covered = 0
    
    num_jobs = min(max(1, years_exp // 2), 4)  # 1-4 previous jobs
    
    for i in range(num_jobs):
        if years_covered >= years_exp:
            break
            
        duration = random.randint(1, min(4, years_exp - years_covered))
        end_year = current_year - years_covered
        start_year = end_year - duration
        
        # Adjust job title based on experience level
        if i == 0:  # Most recent job
            title = job_title
        else:
            title = random.choice(job_titles)
        
        company = random.choice(companies)
        history.append(f"{title} at {company} ({start_year}-{end_year})")
        years_covered += duration
    
    return " | ".join(history)


def generate_skills(num_technical, num_soft):
    """Generate random skill set"""
    tech_skills = random.sample(skills_pool["technical"], min(num_technical, len(skills_pool["technical"])))
    soft_skills = random.sample(skills_pool["soft"], min(num_soft, len(skills_pool["soft"])))
    return tech_skills + soft_skills


def generate_cv_data(num_records=500):
    """Generate synthetic CV dataset"""
    
    cv_data = []
    
    for i in range(num_records):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        
        # Generate experience and related fields
        years_exp = generate_experience_years()
        job_title = random.choice(job_titles)
        
        # Generate skills (more skills for experienced candidates)
        num_tech_skills = random.randint(3, min(8, 4 + years_exp))
        num_soft_skills = random.randint(2, 5)
        skills = generate_skills(num_tech_skills, num_soft_skills)
        
        # Generate education
        education = random.choice(education_degrees)
        university = random.choice(universities)
        grad_year = 2024 - years_exp - random.randint(0, 2)
        
        # Generate certifications (more likely for experienced candidates)
        num_certs = 0 if years_exp < 2 else random.randint(0, min(3, years_exp // 2))
        certs = random.sample(certifications, num_certs) if num_certs > 0 else []
        
        # Calculate match score (will be used for ranking)
        # This is a placeholder that could be based on job requirements
        match_score = random.uniform(0.4, 0.95)
        
        cv_record = {
            "cv_id": f"CV_{i+1:04d}",
            "first_name": first_name,
            "last_name": last_name,
            "email": generate_email(first_name, last_name),
            "phone": generate_phone(),
            "current_job_title": job_title,
            "years_of_experience": years_exp,
            "education": f"{education}, {university} ({grad_year})",
            "skills": ", ".join(skills),
            "certifications": ", ".join(certs) if certs else "None",
            "work_history": generate_work_history(years_exp, job_title),
            "location": random.choice(["Cairo", "Giza", "Alexandria", "New Cairo", "6th of October"]),
            "expected_salary": random.randint(8000, 50000) if years_exp > 0 else random.randint(5000, 12000),
            "notice_period_days": random.choice([0, 15, 30, 60, 90]),
            "match_score": round(match_score, 2),
            "application_date": (datetime.now() - timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d")
        }
        
        cv_data.append(cv_record)
    
    return cv_data


def generate_job_descriptions(num_jobs=20):
    """Generate job description dataset"""
    
    job_data = []
    
    for i in range(num_jobs):
        job_title = random.choice(job_titles)
        required_years = random.choice([0, 1, 2, 3, 5, 7, 10])
        
        # Select relevant skills for the job
        num_required_skills = random.randint(5, 10)
        required_skills = random.sample(skills_pool["technical"] + skills_pool["soft"], num_required_skills)
        
        # Salary range
        base_salary = 10000 + (required_years * 5000)
        salary_range = f"{base_salary}-{base_salary + 15000}"
        
        job_record = {
            "job_id": f"JOB_{i+1:03d}",
            "job_title": job_title,
            "department": random.choice(["IT", "Finance", "HR", "Operations", "Marketing", "Sales"]),
            "required_experience_years": required_years,
            "required_skills": ", ".join(required_skills),
            "preferred_education": random.choice(education_degrees),
            "salary_range": salary_range,
            "employment_type": random.choice(["Full-time", "Full-time", "Full-time", "Contract", "Part-time"]),
            "location": random.choice(["Cairo", "Giza", "Alexandria", "Remote", "Hybrid"]),
            "number_of_openings": random.randint(1, 5),
            "posted_date": (datetime.now() - timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d"),
            "status": random.choice(["Open", "Open", "Open", "Closed"])
        }
        
        job_data.append(job_record)
    
    return job_data


def main():
    """Main function to generate all datasets"""
    
    print("Generating CV dataset...")
    cv_data = generate_cv_data(num_records=500)
    cv_df = pd.DataFrame(cv_data)
    cv_df.to_csv("cv_dataset.csv", index=False, encoding='utf-8-sig')
    print(f"✓ Generated cv_dataset.csv with {len(cv_data)} records")
    
    print("\nGenerating job descriptions dataset...")
    job_data = generate_job_descriptions(num_jobs=20)
    job_df = pd.DataFrame(job_data)
    job_df.to_csv("job_descriptions.csv", index=False, encoding='utf-8-sig')
    print(f"✓ Generated job_descriptions.csv with {len(job_data)} records")
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"\nCV Dataset Statistics:")
    print(f"  Total CVs: {len(cv_df)}")
    print(f"  Average Experience: {cv_df['years_of_experience'].mean():.1f} years")
    print(f"  Experience Range: {cv_df['years_of_experience'].min()}-{cv_df['years_of_experience'].max()} years")
    print(f"\n  Top 5 Job Titles:")
    for title, count in cv_df['current_job_title'].value_counts().head().items():
        print(f"    - {title}: {count}")
    
    print(f"\nJob Descriptions Statistics:")
    print(f"  Total Job Postings: {len(job_df)}")
    print(f"  Open Positions: {len(job_df[job_df['status'] == 'Open'])}")
    print(f"  Total Openings: {job_df['number_of_openings'].sum()}")
    
    print("\n" + "="*60)
    print("Files generated successfully!")
    print("="*60)
    print("\nNext steps for AI team:")
    print("1. Use cv_dataset.csv for CV parsing and field extraction")
    print("2. Use job_descriptions.csv for job-CV matching")
    print("3. Implement embedding-based similarity scoring")
    print("4. Build ranking algorithm with match_score as baseline")
    print("5. Add explainability layer (SHAP/LIME) for rankings")


if __name__ == "__main__":
    main()