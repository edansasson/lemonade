from openai import OpenAI

print("Starting Minions demo...")
print("Ensure the server is running with `lemonade-server-dev serve`")

try:
    # Initialize the client to use Lemonade Server
    client = OpenAI(
        base_url="http://localhost:8000/api/v1",
        api_key="lemonade",  # Please set environment variable instead for now (OPENAI_API_KEY)
    )

    # Example 1: Minions Protocol
    # Note: Minions Plural must ave a context sextion
    

    # Example 2: Basic Minion Protocol
    print("\n=== Example 2: Basic Minion ===")
    completion2 = client.chat.completions.create(
        model="Qwen3-4B-Instruct-2507-GGUF|gpt-4o",
        messages=[{
            "role": "user", 
            "content": """##TASK## Make space invaders, but I can fly around the whole screen instead of being stuck on the bottom.
            
            ##CONTEXT## write me code in python to solve the task shown."""
        }],
        extra_body={
            "protocol": "minion",
            "max_rounds": 2
        }
    )
    
    print("Basic Minion Response:")
    print(completion2.choices[0].message.content)


    # Print the response
    #print(completion.choices[0].message.content)

except Exception as e:
    raise Exception("Please take a look at the server logs for details") from e
