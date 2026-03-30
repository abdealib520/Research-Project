import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from PyPDF2 import PdfReader

# -------------------------------
# LOAD MODELS (CACHED)
# -------------------------------
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return nlp, tokenizer, model

nlp, tokenizer, model = load_models()

# -------------------------------
# CLEANING FUNCTIONS
# -------------------------------
def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\+?\d[\d\s\-]{8,}\d", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_entities(text):
    doc = nlp(text)
    return " ".join([
        token.text for token in doc
        if token.ent_type_ not in ["PERSON", "ORG", "GPE", "LOC"]
    ])

def preprocess_resume(text):
    return remove_entities(basic_clean(text))

# -------------------------------
# EMBEDDING FUNCTION
# -------------------------------
def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).numpy()

# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "

    return text

# -------------------------------
# DEGREE FILTER
# -------------------------------
def filter_by_degree(df, degree_type):

    if degree_type == "All":
        return df

    if degree_type == "Bachelors":
        keywords = ["bachelor", "b.tech", "b.e", "bsc", "b.sc"]

    elif degree_type == "Masters":
        keywords = ["master", "m.tech", "m.e", "msc", "m.sc", "mba"]

    pattern = "|".join(keywords)

    return df[df["Resume_str"].str.contains(pattern, case=False, na=False)]

# -------------------------------
# LOAD DATA + EMBEDDINGS (CACHED)
# -------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv(
        r"Resume.csv"
    )

    df = df[df["Resume_str"].str.contains("Python|Java", case=False, na=False)]

    df["clean_resume"] = df["Resume_str"].apply(preprocess_resume)

    resume_embeddings = np.load(
        r"resume_embeddings.npy"
    )

    return df, resume_embeddings

df, resume_embeddings = load_data()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("AI Resume Similarity Matcher")

uploaded_files = st.file_uploader(
    "Upload High Performer Resume(s) (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

degree_filter = st.selectbox(
    "Filter by Degree",
    ["All", "Bachelors", "Masters"]
)

mode = st.radio(
    "Matching Mode",
    ["Combined Matching (Recommended)", "Separate Matching"]
)

# -------------------------------
# PROCESS FILES
# -------------------------------
if uploaded_files:

    filtered_df = filter_by_degree(df, degree_filter)

    # Fix index issue
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_embeddings = resume_embeddings[:len(filtered_df)]

    # =============================
    # 🔥 COMBINED MATCHING
    # =============================
    if mode == "Combined Matching (Recommended)":

        st.info(f"{len(uploaded_files)} resumes uploaded. Creating combined profile...")

        all_embeddings = []

        for file in uploaded_files:

            hp_text = extract_text_from_pdf(file)
            hp_clean = preprocess_resume(hp_text)

            emb = get_embedding(hp_clean)
            all_embeddings.append(emb)

        # Combine embeddings
        combined_embedding = np.mean(np.vstack(all_embeddings), axis=0).reshape(1, -1)

        similarities = cosine_similarity(combined_embedding, filtered_embeddings)[0]
        filtered_df["similarity"] = similarities

        top_5 = filtered_df.sort_values("similarity", ascending=False).head(5)

        st.subheader("Top 5 Matches (Combined Profile)")

        for i, row in top_5.iterrows():

            preview = row["Resume_str"][:400]

            with st.expander(f"Similarity Score: {row['similarity']:.4f}"):

                st.write(preview + "...")

                if st.button(f"Read Full Resume"):
                    st.write(row["Resume_str"])

    # =============================
    # 🔥 SEPARATE MATCHING
    # =============================
    else:

        for file in uploaded_files:

            st.success(f"Processing: {file.name}")

            hp_text = extract_text_from_pdf(file)
            hp_clean = preprocess_resume(hp_text)

            hp_embedding = get_embedding(hp_clean)

            similarities = cosine_similarity(hp_embedding, filtered_embeddings)[0]
            filtered_df["similarity"] = similarities

            top_5 = filtered_df.sort_values("similarity", ascending=False).head(5)

            st.subheader(f"Top Matches for {file.name}")

            for i, row in top_5.iterrows():

                preview = row["Resume_str"][:400]

                with st.expander(f"Similarity Score: {row['similarity']:.4f}"):

                    st.write(preview + "...")

                    if st.button(f"Read Full Resume"):
                        st.write(row["Resume_str"])

            st.divider()
