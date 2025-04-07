from dotenv import load_dotenv
load_dotenv()
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import joblib
# === Configure Gemini API ===
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


rag_data = joblib.load("data.pkl")

nn = rag_data["nn"]
texts = rag_data["texts"]
data = rag_data["data"]

# === Load Sentence Embedding Model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === RAG Retrieval (Scikit-learn version) ===
def retrieve_response(query, k=1):
    query_vec = model.encode([query])
    distances, indices = nn.kneighbors(query_vec, n_neighbors=k)
    return [data[i] for i in indices[0]]

# === Gemini-Powered Enhancement ===
def generate_answer(user_query, context, base_response):
    prompt = f"""
You are a kind and thoughtful mental health assistant.

Here is the user's message:
"{user_query}"

You also have extra information from our system:
Context: "{context}"
Base Response: "{base_response}"

Your job is to support the user by:
- Responding **directly** to their message.
- Using parts of the context and base response **only if relevant** to the user's concern.
- Keeping the language warm, caring, and easy to read.
- Avoiding references to things the user **has not mentioned**, unless they appear in both the query and context.
- Removing anything confusing, off-topic, or assumptive.
- Optionally using bullet points to improve clarity and emotional support.

Your reply should feel personal and grounded. Give your best, most helpful answer below.
"""

    response = gemini_model.generate_content(prompt)
    return response.text
