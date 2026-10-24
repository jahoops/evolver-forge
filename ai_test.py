import ollama

response = ollama.chat(
    model='llama3',
    messages=[{'role': 'user', 'content': 'Summarize the purpose of The Forge project in one sentence.'}]
)
print(response['message']['content'])
