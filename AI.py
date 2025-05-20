# app.py
import os
from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import re
from difflib import SequenceMatcher
import io
from fpdf import FPDF

# Load environment variables from .env file
load_dotenv()

import PyPDF2
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------------------------------------------------
# 1.  FLASK APP CONFIG
# ----------------------------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Add this after the Flask app configuration
# Store job matching results in memory (in production, use a proper database)
job_matching_results = {}

# ----------------------------------------------------------------------
# 2.  LLM + EMBEDDINGS + TEXT-SPLITTER
# ----------------------------------------------------------------------
# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2_000,
    chunk_overlap=200,
    length_function=len,
)

# ----------------------------------------------------------------------
# 3.  PROMPT FOR RESUME SUMMARY
# ----------------------------------------------------------------------
resume_summary_template = """
Role: You are an AI Career Coach.

Task: Given the candidate's resume, provide a comprehensive summary that covers:
• Career Objective
• Skills and Expertise
• Professional Experience
• Educational Background
• Notable Achievements

Write a concise, well-structured summary that highlights strengths relevant to industry standards.

Resume:
{resume}
"""

resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template=resume_summary_template,
)

# Use modern chain syntax instead of deprecated LLMChain
resume_analysis_chain = resume_prompt | llm

# ----------------------------------------------------------------------
# 4.  HELPER FUNCTIONS
# ----------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Pull plain text from every page of a PDF."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def build_vector_index(text: str, index_dir: str = "vector_index") -> None:
    """Split text, build FAISS index, and save locally."""
    chunks = text_splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local(index_dir)


def process_resume_analysis(text: str) -> str:
    """Format the resume analysis output to be more structured and clean."""
    if not text:
        return "No analysis available."
    
    # Replace bullet points with HTML entities if needed
    text = text.replace('•', '&bull;')
    
    # Add section formatting - look for common section headers
    sections = [
        "Career Objective",
        "Skills and Expertise", 
        "Professional Experience",
        "Educational Background",
        "Notable Achievements"
    ]
    
    for section in sections:
        # Style the section headers
        pattern = r'(' + re.escape(section) + r')(:)?'
        replacement = r'<h3 style="margin-top: 20px; color: #2C3E50; border-bottom: 1px solid #eaecef;">\1</h3>'
        text = re.sub(pattern, replacement, text)
    
    # Format lists - look for patterns that might be lists
    # Convert lines starting with dash or asterisk to HTML lists
    text = re.sub(r'(?m)^[ \t]*[-*][ \t]*(.*?)$', r'<li>\1</li>', text)
    text = re.sub(r'(?s)(<li>.*?</li>)', r'<ul style="padding-left: 20px;">\1</ul>', text)
    
    # Fix any doubled-up lists
    text = text.replace('</ul><ul style="padding-left: 20px;">', '')
    
    # Style key skills - identify and highlight skills
    skills_pattern = r'\b(Python|Java|JavaScript|HTML|CSS|SQL|React|Angular|Node\.js|AWS|Excel|Word|PowerPoint|Leadership|Management|Communication|Teamwork|Problem[ -]Solving|Data Analysis|Project Management|Marketing|Sales|Design|Customer Service|Research)\b'
    text = re.sub(skills_pattern, r'<span style="background-color: #f0f7ff; padding: 2px 4px; border-radius: 3px; font-weight: 500;">\1</span>', text)
    
    # Add general styling
    text = f'<div style="line-height: 1.6;">{text}</div>'
    
    return text


def perform_qa(query: str, index_dir: str = "vector_index") -> str:
    """Retrieve context from FAISS and answer with Gemini."""
    if not isinstance(query, str):
        query = str(query)
    
    try:
        # Check if the query is about job matching or skills
        job_match_keywords = ['match', 'matching', 'percentage', 'skills', 'requirements', 'missing', 'improve', 'better']
        is_job_match_query = any(keyword in query.lower() for keyword in job_match_keywords)
        
        if is_job_match_query and 'current' in job_matching_results:
            # Get the stored job matching results
            match_data = job_matching_results['current']
            
            # Create a specialized prompt for job matching questions
            prompt_template = f"""
            You are an AI Career Coach helping with job matching and skill development.
            
            Current Job Match Analysis:
            - Match Percentage: {match_data['match_percentage']:.1f}%
            - Matching Skills: {', '.join(match_data['matches'])}
            - Missing Skills: {', '.join(match_data['missing'])}
            
            Based on this specific analysis and the question, provide targeted advice:
            
            Question: {query}
            
            Please provide:
            1. Specific skills to develop (focus on the missing skills: {', '.join(match_data['missing'])})
            2. How to improve the match percentage from {match_data['match_percentage']:.1f}%
            3. Actionable steps to enhance the resume
            4. Resources or learning paths for the missing skills
            
            Format the response in a clear, structured way with bullet points and sections.
            """
            
            response = llm.invoke(prompt_template)
            
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and "text" in response:
                return response["text"]
            else:
                return str(response)
        else:
            # Original QA logic for general resume questions
            db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(query, k=4)
            context = "\n\n".join(doc.page_content for doc in docs)
            
            prompt_template = f"""
            Answer the following question based on the provided context:
            
            Context:
            {context}
            
            Question: {query}
            """
            
            response = llm.invoke(prompt_template)
            
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and "text" in response:
                return response["text"]
            else:
                return str(response)
            
    except Exception as e:
        print(f"Error in perform_qa: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Sorry, I encountered an error: {str(e)}"

def extract_skills(text: str) -> list:
    """Extract skills from text using common skill patterns."""
    # Common technical skills pattern
    skills_pattern = r'\b(Python|Java|JavaScript|HTML|CSS|SQL|React|Angular|Node\.js|AWS|Azure|GCP|Docker|Kubernetes|Git|Linux|Agile|Scrum|Machine Learning|Data Analysis|Project Management|Communication|Leadership|Teamwork|Problem[ -]Solving)\b'
    skills = re.findall(skills_pattern, text, re.IGNORECASE)
    return list(set(skills))  # Remove duplicates

def calculate_skill_match(resume_skills: list, job_requirements: list) -> dict:
    """Calculate how well resume skills match job requirements."""
    resume_skills = [skill.lower() for skill in resume_skills]
    job_requirements = [req.lower() for req in job_requirements]
    
    matches = []
    missing = []
    
    for req in job_requirements:
        found = False
        for skill in resume_skills:
            # Check for exact match or high similarity
            if req == skill or SequenceMatcher(None, req, skill).ratio() > 0.8:
                matches.append(req)
                found = True
                break
        if not found:
            missing.append(req)
    
    match_percentage = (len(matches) / len(job_requirements)) * 100 if job_requirements else 0
    
    return {
        'matches': matches,
        'missing': missing,
        'match_percentage': match_percentage
    }

def analyze_job_match(resume_text: str, job_description: str) -> str:
    """Analyze how well the resume matches the job requirements."""
    resume_skills = extract_skills(resume_text)
    job_requirements = extract_skills(job_description)
    
    match_result = calculate_skill_match(resume_skills, job_requirements)
    
    # Store the results for QA system
    job_matching_results['current'] = {
        'resume_skills': resume_skills,
        'job_requirements': job_requirements,
        'matches': match_result['matches'],
        'missing': match_result['missing'],
        'match_percentage': match_result['match_percentage'],
        'resume_text': resume_text,
        'job_description': job_description
    }
    
    # Format the results as HTML
    result_html = f"""
    <div class="job-match-analysis">
        <h3 style="margin-top: 20px; color: #2C3E50; border-bottom: 1px solid #eaecef;">Job Match Analysis</h3>
        
        <div class="match-percentage" style="margin: 20px 0; padding: 15px; background-color: {'#d4edda' if match_result['match_percentage'] >= 70 else '#fff3cd' if match_result['match_percentage'] >= 40 else '#f8d7da'}; border-radius: 5px;">
            <h4 style="margin: 0;">Overall Match: {match_result['match_percentage']:.1f}%</h4>
        </div>
        
        <div class="matching-skills" style="margin-bottom: 20px;">
            <h4>Matching Skills:</h4>
            <ul style="padding-left: 20px;">
                {''.join(f'<li>{skill}</li>' for skill in match_result['matches'])}
            </ul>
        </div>
        
        <div class="missing-skills" style="margin-bottom: 20px;">
            <h4>Missing Skills:</h4>
            <ul style="padding-left: 20px;">
                {''.join(f'<li>{skill}</li>' for skill in match_result['missing'])}
            </ul>
        </div>
        
        <div class="recommendations" style="margin-top: 20px; padding: 15px; background-color: #e9ecef; border-radius: 5px;">
            <h4>Recommendations:</h4>
            <ul style="padding-left: 20px;">
                {f'<li>Consider highlighting your experience with {", ".join(match_result["matches"][:3])} in your resume</li>' if match_result['matches'] else ''}
                {f'<li>Consider developing skills in {", ".join(match_result["missing"][:3])}</li>' if match_result['missing'] else ''}
            </ul>
        </div>
    </div>
    """
    
    return result_html

# ----------------------------------------------------------------------
# 5.  ROUTES
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")  # simple upload form


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Extract résumé text
    resume_text = extract_text_from_pdf(filepath)
    
    # Get job description from form if provided
    job_description = request.form.get("job_description", "")
    
    # Run résumé summary prompt
    resume_analysis = resume_analysis_chain.invoke({"resume": resume_text})
    if isinstance(resume_analysis, dict) and "text" in resume_analysis:
        resume_analysis = resume_analysis["text"]
    elif hasattr(resume_analysis, 'content'):
        resume_analysis = resume_analysis.content
    
    # Process the analysis to make it more structured and clean
    formatted_analysis = process_resume_analysis(resume_analysis)
    
    # Add job matching analysis if job description is provided
    if job_description:
        job_match_analysis = analyze_job_match(resume_text, job_description)
        formatted_analysis += job_match_analysis
    
    return render_template("results.html", resume_analysis=formatted_analysis)


@app.route("/ask_query", methods=["GET", "POST"])
def ask_query():
    if request.method == "POST":
        query = request.form["query"]
        answer = perform_qa(query)
        # Pass job matching context if available
        missing_skills = job_matching_results['current']['missing'] if 'current' in job_matching_results else []
        resume_text = job_matching_results['current']['resume_text'] if 'current' in job_matching_results and 'resume_text' in job_matching_results['current'] else ''
        job_description = job_matching_results['current']['job_description'] if 'current' in job_matching_results and 'job_description' in job_matching_results['current'] else ''
        return render_template("qa_results.html", query=query, result=answer, missing_skills=missing_skills, resume_text=resume_text, job_description=job_description)
    return render_template("ask.html")  # simple form with <textarea name="query">


@app.route('/improve_resume', methods=['POST'])
def improve_resume():
    data = request.json
    resume_text = data.get('resume_text', '')
    missing_skills = data.get('missing_skills', [])
    job_description = data.get('job_description', '')
    
    # Prompt to LLM to improve resume
    prompt = f"""
    You are an expert resume writer. Here is the candidate's original resume:
    {resume_text}
    
    The job description is:
    {job_description}
    
    The following skills are missing and should be incorporated naturally into the resume (in skills, experience, or summary): {', '.join(missing_skills)}
    
    Rewrite the resume to better match the job description and include the missing skills. Make the changes realistic and professional.
    """
    improved_resume = llm.invoke(prompt)
    if hasattr(improved_resume, 'content'):
        improved_resume = improved_resume.content
    elif isinstance(improved_resume, dict) and 'text' in improved_resume:
        improved_resume = improved_resume['text']
    else:
        improved_resume = str(improved_resume)
    return jsonify({'improved_resume': improved_resume})

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    data = request.json
    resume_text = data.get('resume_text', '')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Arial', '', 12)
    for line in resume_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return send_file(pdf_output, as_attachment=True, download_name='improved_resume.pdf', mimetype='application/pdf')

if __name__ == "__main__":
    # Use `flask run` in production; debug=True only for local dev
    app.run(debug=True)
#resume_chatbot