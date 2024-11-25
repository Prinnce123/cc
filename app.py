from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from PyPDF2 import PdfReader

app = Flask(__name__)

# Initialize the HuggingFace summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        text = request.form.get("text")
        uploaded_file = request.files.get("file")

        # Extract text from PDF if a file is uploaded
        if uploaded_file and uploaded_file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)

        if not text:
            return jsonify({"error": "No text provided for summarization"}), 400

        # Summarize the text
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
