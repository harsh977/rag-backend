from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import retrieve_response, generate_answer

app = FastAPI()

# Allow CORS from any origin (for development or testing purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be cautious using "*" in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/rag")
def get_rag_response(payload: QueryRequest):
    query = payload.query

    # Retrieve context and base response
    results = retrieve_response(query)
    context = results[0]['Context']
    base_response = results[0]['Response']

    # Generate final answer with Gemini
    final_answer = generate_answer(query, context, base_response)

    return {
        "query": query,
        "context": context,
        "base_response": base_response,
        "final_answer": final_answer
    }

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 10000))  # Render injects PORT, fallback for local
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
