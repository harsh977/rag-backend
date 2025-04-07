import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import joblib

# Load dataset
with open("mental.json") as f:
    data = json.load(f)

# Combine "Context ||| Response"
texts = [f"{item['Context']} ||| {item['Response']}" for item in data]

# Embed with SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# Initialize and train NearestNeighbors
nn = NearestNeighbors(n_neighbors=5, metric="cosine")
nn.fit(embeddings)

# Save NearestNeighbors model and data
joblib.dump({
    "nn": nn,
    "embeddings": embeddings,
    "texts": texts,
    "data": data
}, "sklearn_rag_index.pkl")

print("✅ scikit-learn index and original data saved!")
