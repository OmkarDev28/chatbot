from zmq.decorators import context
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import re
from datetime import datetime, timedelta
from flask import Flask, request, render_template

Key = "AIzaSyDQ0S8anqLotDjxKcdLpnDeDAMNTbyEHT0"
from google import genai
def geminiGen(Command):
        client = genai.Client(api_key= Key)
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=Command
        )
        n = response
        return n

def geminiEmbedding(Task):
        client = genai.Client(api_key=Key)

        result = client.models.embed_content(
                model="gemini-embedding-exp-03-07",
                contents=Task)
        r = result.embeddings
        return r
def gemtextgenforstraightapicall(Task):
        client = genai.Client(api_key=Key)
        response = client.models.generate_content(
            model = "gemini-2.0-flash", contents = Task
        )
        n = response
        return n.text


path1 = "C:\\Python\\GenAiGemini\\accidentReportduetoPotholes.csv"
path2  = "C:\\Python\\GenAiGemini\\SampleDatasetPotholes.csv"

import pandas as pd
ACC = pd.read_csv(path1)
ACC

Potholes = pd.read_csv(path2)

ACC['text_representation'] = ACC['State/UT']  + " " + ACC['2020 - Number of Accidents'].astype(str) + " " + ACC['2021 - Number of Accidents'].astype(str) + " " + ACC['2022 - Number of Accidents'].astype(str)
# Combine columns into a single text representation
Potholes['text_representation'] = Potholes['Location'] + " " + Potholes['Severity'] + " " + Potholes['Description']

# # Check the result
# print(Potholes['text_representation'][0])  # This will print the combined text for the first complaint


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(Potholes['text_representation'])   #embedding for potholes dataset
embeddings1 = model.encode(ACC['text_representation'])  #embeddings for Accidents dataset

import faiss
import numpy as np

# Convert to float32 if needed
embedding_dim = embeddings.shape[1]
embeddings_np = np.array(embeddings).astype("float32")

# Create FAISS index for complain dataset
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings_np)

embeddings1_dim = embeddings1.shape[1]
embeddings1_np = np.array(embeddings1).astype("float32")

# Create FAISS index for ACC dataset
index = faiss.IndexFlatL2(embeddings1_dim)
index.add(embeddings1_np)

texts = Potholes['text_representation'].tolist()
texts1 = ACC['text_representation'].tolist()

import re
from datetime import timedelta, datetime

# Add date column and monthly stats
Potholes['Reported Date'] = pd.to_datetime(Potholes['Reported Date'])
monthly_complaints = Potholes.groupby(Potholes['Reported Date'].dt.to_period("M")).size()
average_complaints = monthly_complaints.mean()

# Helper to extract year-month from question
def extract_month_year(query):
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12"
    }
    query = query.lower()

    if "last month" in query:
        last_month = (datetime.today().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
        return last_month
    elif "this month" in query:
        return datetime.today().strftime("%Y-%m")
    elif "this year" in query:
        return datetime.today().strftime("%Y")

    for m in months:
        if m in query:
            match = re.search(r"(20\d{2})", query)
            year = match.group(1) if match else "2025"
            return f"{year}-{months[m]}"
    return None



def ask_question(query, k=3):
    query_embedding = model.encode(query, normalize_embeddings=True).astype("float32")
    _, I = index.search(np.array([query_embedding]), k)
    context_chunks = [texts[i] for i in I[0]]
    context = "\n\n".join(context_chunks)

    month_str = extract_month_year(query)
    if month_str and len(month_str) == 7:
        complaints = monthly_complaints.get(month_str, 0)
        context += f"\n\nIn {month_str}, there were {complaints} pothole complaints."
        context += f" The average number of complaints per month is {average_complaints:.1f}."
    elif month_str and len(month_str) == 4:
        year = int(month_str)
        year_complaints = Potholes[Potholes['Reported Date'].dt.year == year].shape[0]
        context += f"\n\nIn {year}, there were {year_complaints} pothole complaints."



Potholes['Reported Date'] = pd.to_datetime(Potholes['Reported Date'])

# Group by month & count complaints
monthly_complaints = Potholes.groupby(Potholes['Reported Date'].dt.to_period("M")).size()

# Optionally, get average per month
average_complaints = monthly_complaints.mean()


# Add date column and monthly stats
Potholes['Reported Date'] = pd.to_datetime(Potholes['Reported Date'])
monthly_complaints = Potholes.groupby(Potholes['Reported Date'].dt.to_period("M")).size()
average_complaints = monthly_complaints.mean()

# Helper to extract year-month from question
def extract_month_year(query):
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12"
    }
    query = query.lower()

    if "last month" in query:
        last_month = (datetime.today().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
        return last_month
    elif "this month" in query:
        return datetime.today().strftime("%Y-%m")
    elif "this year" in query:
        return datetime.today().strftime("%Y")

    for m in months:
        if m in query:
            match = re.search(r"(20\d{2})", query)
            year = match.group(1) if match else "2025"
            return f"{year}-{months[m]}"
    return None

# ENHANCED: Handle monthly stats naturally in queries
def ask_question(query, k=3):
    query_embedding = model.encode(query, normalize_embeddings=True).astype("float32")
    _, I = index.search(np.array([query_embedding]), k)
    context_chunks = [texts[i] for i in I[0]]
    context = "\n\n".join(context_chunks)

    month_str = extract_month_year(query)
    if month_str and len(month_str) == 7:
        complaints = monthly_complaints.get(month_str, 0)
        context += f"\n\nIn {month_str}, there were {complaints} pothole complaints."
        context += f" The average number of complaints per month is {average_complaints:.1f}."
    elif month_str and len(month_str) == 4:
        year = int(month_str)
        year_complaints = Potholes[Potholes['Reported Date'].dt.year == year].shape[0]
        context += f"\n\nIn {year}, there were {year_complaints} pothole complaints."

    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = geminiGen(prompt).text
    print(f"Answer: {answer}")
    return answer


index1 = faiss.IndexFlatL2(embeddings1.shape[1])
index1.add(np.array(embeddings1).astype("float32"))

def ask_question_accidents(query, k=3):
    query_embedding = model.encode(query, normalize_embeddings=True).astype("float32")
    _, I = index1.search(np.array([query_embedding]), k)
    context_chunks = [texts1[i] for i in I[0]]
    context = "\n\n".join(context_chunks)

    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = geminiGen(prompt).text
    print(f"Answer: {answer}")
    return answer
app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    query = ""
    if request.method == "POST":
        query = request.form.get("query", "")
        method = request.form.get("method", "")

        print(f"\n📥 Query Received: {query}")
        print(f"📌 Selected Method: {method}")

        try:
            if method == "potholes":
                print("🔍 Calling ask_question()...")
                answer = ask_question(query)
            elif method == "accidents":
                print("🚨 Calling ask_question_accidents()...")
                answer = ask_question_accidents(query)
            elif method == "direct":
                print("🧠 Calling gemtextgenforstraightapicall()...")
                answer = gemtextgenforstraightapicall(query)
            else:
                answer = "❌ Invalid method selected."
        except Exception as e:
            print(f"⚠️ Error while handling query: {e}")
            answer = f"Something went wrong: {e}"

    return render_template("index.html", answer=answer, query=query)


if __name__ == "__main__":
    app.run(debug=True)