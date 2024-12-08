from server_client_sdk import ChatClient

# Initialize the client
client = ChatClient(server_url="http://localhost:8000")

response = client.chat_completions_create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ],
    temperature=0.5,
    top_p=0.8
)

output = response["response"]
print(output)
