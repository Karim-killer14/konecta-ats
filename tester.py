"""
ATS System Test Script
Run this to verify all components are working correctly
"""

import requests
import json
import time
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

BASE_URL = "http://localhost:5000"

def print_header(text):
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}{text}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")

def print_success(text):
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

def print_error(text):
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

def print_info(text):
    print(f"{Fore.YELLOW}ℹ {text}{Style.RESET_ALL}")

def test_health_check():
    """Test 1: Health Check"""
    print_header("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"API is healthy: {data['status']}")
            print_info(f"Service: {data['service']}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Make sure flask_api.py is running!")
        print_info("Run: python flask_api.py")
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_get_cvs():
    """Test 2: Get CVs"""
    print_header("TEST 2: Get All CVs")
    
    try:
        response = requests.get(f"{BASE_URL}/api/cvs")
        data = response.json()
        
        if data.get('success'):
            print_success(f"Retrieved {data['total']} CVs")
            
            # Display first CV
            if data['cvs']:
                cv = data['cvs'][0]
                print_info(f"Sample CV: {cv['name']}")
                print(f"  - Experience: {cv['years_experience']} years")
                print(f"  - Skills: {', '.join(cv['skills'][:3])}...")
            return True
        else:
            print_error("Failed to retrieve CVs")
            return False
    
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_get_jobs():
    """Test 3: Get Jobs"""
    print_header("TEST 3: Get All Jobs")
    
    try:
        response = requests.get(f"{BASE_URL}/api/jobs")
        data = response.json()
        
        if data.get('success'):
            print_success(f"Retrieved {data['total']} jobs")
            
            # Display first job
            if data['jobs']:
                job = data['jobs'][0]
                print_info(f"Sample Job: {job['title']}")
                print(f"  - Department: {job['department']}")
                print(f"  - Required Experience: {job['required_experience']} years")
                print(f"  - Status: {job['status']}")
            return True
        else:
            print_error("Failed to retrieve jobs")
            return False
    
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_cv_matching():
    """Test 4: CV-Job Matching"""
    print_header("TEST 4: CV-Job Matching")
    
    try:
        payload = {
            "cv_id": "CV_0001",
            "job_id": "JOB_001",
            "include_explanation": True
        }
        
        print_info("Matching CV_0001 to JOB_001...")
        response = requests.post(f"{BASE_URL}/api/match", json=payload)
        data = response.json()
        
        if data.get('success'):
            match = data['match_result']
            print_success(f"Match Score: {match['overall_score']*100:.1f}%")
            print(f"  - Skill Match: {match['skill_match']*100:.1f}%")
            print(f"  - Experience Match: {match['experience_match']*100:.1f}%")
            print(f"  - Text Similarity: {match['text_similarity']*100:.1f}%")
            
            if 'ai_explanation' in match:
                print_info("AI Explanation:")
                print(f"  {match['ai_explanation'][:200]}...")
            
            return True
        else:
            print_error("Matching failed")
            return False
    
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_candidate_ranking():
    """Test 5: Candidate Ranking"""
    print_header("TEST 5: Candidate Ranking")
    
    try:
        payload = {
            "job_id": "JOB_001",
            "top_n": 5
        }
        
        print_info("Ranking top 5 candidates for JOB_001...")
        response = requests.post(f"{BASE_URL}/api/rank", json=payload)
        data = response.json()
        
        if data.get('success'):
            print_success(f"Evaluated {data['total_candidates_evaluated']} candidates")
            print_info(f"Top 5 for {data['job_title']}:")
            
            for i, candidate in enumerate(data['top_candidates'][:5], 1):
                print(f"  {i}. {candidate['name']:<20} Score: {candidate['score']*100:>5.1f}%")
            
            return True
        else:
            print_error("Ranking failed")
            return False
    
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_explainability():
    """Test 6: Explainability"""
    print_header("TEST 6: AI Explainability")
    
    try:
        payload = {
            "cv_id": "CV_0001",
            "job_id": "JOB_001"
        }
        
        print_info("Generating explanation for CV_0001 and JOB_001...")
        response = requests.post(f"{BASE_URL}/api/explain", json=payload)
        data = response.json()
        
        if data.get('success'):
            print_success(f"Match Score: {data['match_score']*100:.1f}%")
            
            print_info("AI Reasoning:")
            print(f"  {data['ai_reasoning'][:150]}...")
            
            print_info("Skill Analysis:")
            skill_analysis = data['skill_analysis']
            print(f"  Matched: {len(skill_analysis['matched_skills'])} skills")
            print(f"  Missing: {len(skill_analysis['missing_skills'])} skills")
            
            print_info("Interview Questions Generated:")
            for i, q in enumerate(data['interview_questions'][:3], 1):
                print(f"  {i}. {q[:80]}...")
            
            return True
        else:
            print_error("Explainability test failed")
            return False
    
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_chatbot():
    """Test 7: RAG Chatbot"""
    print_header("TEST 7: RAG Chatbot")
    
    try:
        test_queries = [
            "What positions do we have open?",
            "Tell me about CV_0001",
            "How many candidates have Python skills?"
        ]
        
        for query in test_queries:
            print_info(f"Query: {query}")
            
            payload = {"query": query}
            response = requests.post(f"{BASE_URL}/api/chatbot/query", json=payload)
            data = response.json()
            
            if data.get('success'):
                print(f"  Response: {data['response']['response'][:150]}...")
                print_success("Query successful")
            else:
                print_error("Query failed")
                return False
            
            time.sleep(1)  # Brief pause between queries
        
        return True
    
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_analytics():
    """Test 8: Analytics"""
    print_header("TEST 8: Analytics")
    
    try:
        print_info("Fetching system analytics...")
        response = requests.get(f"{BASE_URL}/api/analytics/overview")
        data = response.json()
        
        if data.get('success'):
            overview = data['overview']
            print_success("Analytics retrieved")
            print(f"  - Total Candidates: {overview['total_candidates']}")
            print(f"  - Total Jobs: {overview['total_jobs']}")
            print(f"  - Open Positions: {overview['open_positions']}")
            print(f"  - Avg Experience: {overview['avg_candidate_experience']} years")
            
            print_info("Top 5 Skills:")
            for skill in overview['top_skills'][:5]:
                print(f"  - {skill['skill']}: {skill['count']} candidates")
            
            return True
        else:
            print_error("Analytics test failed")
            return False
    
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_search_functionality():
    """Test 9: Search Functionality"""
    print_header("TEST 9: CV Search")
    
    try:
        print_info("Searching for candidates with 3+ years experience and Python skills...")
        
        params = {
            "min_experience": 3,
            "skills": "Python"
        }
        
        response = requests.get(f"{BASE_URL}/api/cvs", params=params)
        data = response.json()
        
        if data.get('success'):
            print_success(f"Found {data['total']} matching candidates")
            
            if data['cvs']:
                print_info("Sample results:")
                for cv in data['cvs'][:3]:
                    print(f"  - {cv['name']}: {cv['years_experience']} years")
            
            return True
        else:
            print_error("Search failed")
            return False
    
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def run_all_tests():
    """Run all tests and generate report"""
    
    print(f"\n{Fore.MAGENTA}{'='*70}")
    print(f"{Fore.MAGENTA}KONECTA ATS SYSTEM - INTEGRATION TEST SUITE")
    print(f"{Fore.MAGENTA}{'='*70}{Style.RESET_ALL}\n")
    
    tests = [
        ("Health Check", test_health_check),
        ("Get CVs", test_get_cvs),
        ("Get Jobs", test_get_jobs),
        ("CV-Job Matching", test_cv_matching),
        ("Candidate Ranking", test_candidate_ranking),
        ("AI Explainability", test_explainability),
        ("RAG Chatbot", test_chatbot),
        ("Analytics", test_analytics),
        ("Search Functionality", test_search_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            time.sleep(0.5)  # Brief pause between tests
        except Exception as e:
            print_error(f"Test crashed: {e}")
            results.append((test_name, False))
    
    # Generate report
    print_header("TEST RESULTS SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{Fore.GREEN}✓ PASS" if result else f"{Fore.RED}✗ FAIL"
        print(f"{status:<20} {test_name}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}{'='*70}")
    percentage = (passed / total) * 100
    
    if percentage == 100:
        print(f"{Fore.GREEN}ALL TESTS PASSED! ({passed}/{total}){Style.RESET_ALL}")
        print(f"{Fore.GREEN}🎉 System is fully operational and ready for integration!{Style.RESET_ALL}")
    elif percentage >= 70:
        print(f"{Fore.YELLOW}MOSTLY WORKING ({passed}/{total} - {percentage:.1f}%){Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Some issues detected. Review failed tests.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}TESTS FAILED ({passed}/{total} - {percentage:.1f}%){Style.RESET_ALL}")
        print(f"{Fore.RED}Major issues detected. Check API server and data files.{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    # Recommendations
    if percentage < 100:
        print_header("RECOMMENDATIONS")
        
        if not results[0][1]:  # Health check failed
            print_error("API server is not running")
            print_info("Start it with: python flask_api.py")
        
        if not results[1][1] or not results[2][1]:  # Data issues
            print_error("Data files may be missing or corrupted")
            print_info("Regenerate data with: python cv_generator.py")
        
        print()

if __name__ == "__main__":
    # Check if colorama is installed
    try:
        from colorama import init, Fore, Style
        init(autoreset=True)
    except ImportError:
        print("Installing colorama for colored output...")
        import subprocess
        subprocess.check_call(["pip", "install", "colorama"])
        from colorama import init, Fore, Style
        init(autoreset=True)
    
    run_all_tests()