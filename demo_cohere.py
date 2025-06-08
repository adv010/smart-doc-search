import cohere
co = cohere.ClientV2("4dkVC224oH8uSqKsYCAypJ6fOwxdsfUqQux2Rf1B")
# response = co.chat(
#     model="command-a-03-2025", 
#     messages=[{"role": "user", "content": "hello world!"}]
# )

# print(response)

# ''' Test Cohere SDK Chat '''

# response = co.chat(
#     model="command",  # Use "command" or another supported model
#     messages=[{"role": "user", "content": "You are a helpful AI assistant. Answer concisely. What year was Isaac Newton born?"}],  # Directly ask the question
#     temperature=0.3  # Optional: Controls randomness (0-1)
# )

# print("Answer:", response)

''' Test Cohere  SDK RAG'''
documents = [
    {
        "data": {
            "text": "Reimbursing Travel Expenses: Easily manage your travel expenses by submitting them through our finance tool. Approvals are prompt and straightforward."
        }
    },
    {
        "data": {
            "text": "Working from Abroad: Working remotely from another country is possible. Simply coordinate with your manager and ensure your availability during core hours."
        }
    },
    {
        "data": {
            "text": "Health and Wellness Benefits: We care about your well-being and offer gym memberships, on-site yoga classes, and comprehensive health insurance."
        }
    },
]



# Add the user query
query = "Are there health benefits?"

# Generate the response
response = co.chat(
    model="command-a-03-2025",
    messages=[{"role": "user", "content": query}],
    documents=documents,
)

# Display the response
print(response.message.content[0].text)

# Display the citations and source documents
if response.message.citations:
    print("\nCITATIONS:")
    for citation in response.message.citations:
        print(citation, "\n")

