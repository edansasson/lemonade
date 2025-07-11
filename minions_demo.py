from openai import OpenAI

print("Starting Minions demo...")
print("Ensure the server is running with `lemonade-server-dev serve`")

try:
    # Initialize the client to use Lemonade Server
    client = OpenAI(
        base_url="http://localhost:8000/api/v1",
        api_key="lemonade",  # Please set environment variable instead for now (OPENAI_API_KEY)
    )

    # Create a chat completion without streaming
    completion = client.chat.completions.create(
        model="Qwen3-0.6B-GGUF|o4-mini",  # or any other available model
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    # Print the response
    print(completion.choices[0].message.content)

except Exception as e:
    raise Exception("Please take a look at the server logs for details") from e
