import os
import pandas as pd
import pickle
import re
import streamlit as st
from pypdf import PdfReader
from docx import Document

# Load models
word_vector = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Function to clean resume text
def clean_resume(txt):
    clean_text = re.sub('http\S+\s', ' ', txt)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7F]+', ' ', clean_text)  # Removing non-ASCII characters
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()

# Category Mapping
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and Fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text.strip()

# Function to extract text from Word (.docx)
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

# Function to categorize resumes
def categorize_resumes(uploaded_files, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    results = []

    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        text = ""

        if file_extension == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == "docx":
            text = extract_text_from_docx(uploaded_file)

        if text:
            cleaned_resume = clean_resume(text)
            input_features = word_vector.transform([cleaned_resume])
            prediction_id = model.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")

            # Create category folder
            category_folder = os.path.join(output_directory, category_name)
            os.makedirs(category_folder, exist_ok=True)

            # Save file in respective category folder
            target_path = os.path.join(category_folder, uploaded_file.name)
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            results.append({'Filename': uploaded_file.name, 'Category': category_name})

    return results

# Streamlit UI
st.title("Resume Categorizer Application")
st.subheader("Supports PDF & Word Files")

uploaded_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
output_directory = st.text_input("Output Directory", "categorized_resumes")

if st.button("Categorize Resumes"):
    if uploaded_files and output_directory:
        results_list = categorize_resumes(uploaded_files, output_directory)

        if results_list:
            st.write("### Categorized Resumes")
            for result in results_list:
                st.write(f"**Filename:** {result['Filename']}")  # Display filename
                st.write(f"**Category:** {result['Category']}")  # Display category
                st.write("---")  # Separator for better readability

            # Enable CSV download
            results_df = pd.DataFrame(results_list)
            results_csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=results_csv,
                file_name='categorized_resumes.csv',
                mime='text/csv',
            )
            st.success("Resumes categorized successfully.")
        else:
            st.warning("No valid text extracted from resumes.")
    else:
        st.error("Please upload files and specify the output directory.")
