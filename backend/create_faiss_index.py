import json
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load your instruction-response dataset
with open("mental.json") as f:
      data = json.load(f)

# Create combined text for each item: "instruction ||| response"
texts = [f"{item['Context']} ||| {item['Response']}" for item in data]

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# Build FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index and original data
faiss.write_index(index, "faiss_index.idx")
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ FAISS index and data saved!")
