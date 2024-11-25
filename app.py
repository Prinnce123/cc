from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from PyPDF2 import PdfReader
import os
import math

app = Flask(__name__)

# Initialize the summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to summarize long text
def summarize_large_text(text, max_chunk_size=1024):
    # Split the text into chunks if it's too long
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk + sentence) <= max_chunk_size:
            current_chunk += sentence + '.'
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + '.'
    if current_chunk:
        chunks.append(current_chunk)

    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    # Join all the summaries together
    return ' '.join(summaries)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    # Get text input or uploaded file
    text = request.form.get("text")
    uploaded_file = request.files.get("file")

    if uploaded_file and uploaded_file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)

    # Validate input
    if not text:
        return jsonify({"error": "No text provided for summarization"}), 400

    # Perform summarization
    try:
        summary_text = summarize_large_text(text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Add "Made by Prince Sharma" to the summary
    summary_text += "\n\nMade by Prince Sharma"

    return jsonify({"summary": summary_text})

if __name__ == "__main__":
    app.run(debug=True)
