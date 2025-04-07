from fastapi import FastAPI
from pydantic import BaseModel
from rag import retrieve_response, generate_answer

app = FastAPI()

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
