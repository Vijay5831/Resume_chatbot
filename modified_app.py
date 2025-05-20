import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import PyPDF2

# Load environment variables from .env file
load_dotenv()

# ----------------------------------------------------------------------
# 1.  FLASK APP CONFIG
# ----------------------------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------------------------------------------------
# 2.  HELPER FUNCTIONS
# ----------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Pull plain text from every page of a PDF."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# ----------------------------------------------------------------------
# 3.  MOCK FUNCTIONS (TO REPLACE LLM FUNCTIONALITY) 
# ----------------------------------------------------------------------
def mock_resume_analysis(text: str) -> str:
    """A mock function that returns a placeholder for resume analysis."""
    # In a real implementation, this would use LLM to analyze the resume
    return """
    <h3 style="margin-top: 20px; color: #2C3E50; border-bottom: 1px solid #eaecef;">Career Objective</h3>
    <div>This section would contain the extracted career objective from the resume.</div>
    
    <h3 style="margin-top: 20px; color: #2C3E50; border-bottom: 1px solid #eaecef;">Skills and Expertise</h3>
    <ul style="padding-left: 20px;">
        <li>This is where your skills would be listed</li>
        <li>Based on analysis of your resume</li>
    </ul>
    
    <h3 style="margin-top: 20px; color: #2C3E50; border-bottom: 1px solid #eaecef;">Professional Experience</h3>
    <div>Your work history would be summarized here.</div>
    
    <h3 style="margin-top: 20px; color: #2C3E50; border-bottom: 1px solid #eaecef;">Educational Background</h3>
    <div>Your education would be summarized here.</div>
    
    <h3 style="margin-top: 20px; color: #2C3E50; border-bottom: 1px solid #eaecef;">Notable Achievements</h3>
    <div>Your key achievements would be highlighted here.</div>
    
    <p style="margin-top: 30px; color: #721c24; background-color: #f8d7da; padding: 10px; border-radius: 5px;">
        <strong>Note:</strong> This is simulated output. To get actual AI-powered analysis, you need to install the 
        Microsoft Visual C++ Redistributable package and set up a valid Google API key.
    </p>
    """

def mock_qa(query: str) -> str:
    """A mock function that returns a placeholder answer to a query."""
    # In a real implementation, this would use LLM to answer questions
    return f"""
    <div class="answer">
        <p>Your question was: <strong>{query}</strong></p>
        <p>To get AI-powered answers about your resume, you need to:</p>
        <ol>
            <li>Install Microsoft Visual C++ Redistributable (https://aka.ms/vs/16/release/vc_redist.x64.exe)</li>
            <li>Set up a valid Google API key for Gemini in a .env file</li>
            <li>Run the original AI.py file</li>
        </ol>
    </div>
    """

# ----------------------------------------------------------------------
# 4.  ROUTES
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
    try:
        resume_text = extract_text_from_pdf(filepath)
        # Use mock analysis instead of real LLM
        formatted_analysis = mock_resume_analysis(resume_text)
    except Exception as e:
        formatted_analysis = f"<p>Error processing PDF: {str(e)}</p>"
    
    return render_template("results.html", resume_analysis=formatted_analysis)


@app.route("/ask_query", methods=["GET", "POST"])
def ask_query():
    if request.method == "POST":
        query = request.form["query"]
        # Use mock QA instead of real LLM
        answer = mock_qa(query)
        return render_template("qa_results.html", query=query, result=answer)
    return render_template("ask.html")  # simple form with <textarea name="query">


if __name__ == "__main__":
    app.run(debug=True) 