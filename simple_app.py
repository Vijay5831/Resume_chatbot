from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    return render_template("results.html", resume_analysis="This is a placeholder for resume analysis. The full functionality requires setting up the Google API key and installing all dependencies correctly.")

@app.route("/ask_query", methods=["GET", "POST"])
def ask_query():
    return render_template("ask.html")

if __name__ == "__main__":
    app.run(debug=True) 