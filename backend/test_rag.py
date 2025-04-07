from rag import retrieve_response, generate_answer

query = "my girlfriend left me , now i feel very lonely and sad"


results = retrieve_response(query)
context = results[0]['Context']
base_response = results[0]['Response']

print("🔍 Retrieved Context and Response:")
print(f"Context: {context}")
print(f"Base Response: {base_response}")

# Gemini-enhanced answer
final_answer = generate_answer(query, context, base_response)

print("\n💬 Gemini Final Answer:")
print(final_answer)


