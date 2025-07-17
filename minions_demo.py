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
    print("\n=== Example 1: Minions ===")
    completion = client.chat.completions.create(
        model="Qwen3-0.6B-GGUF|gpt-4o",
        messages=[{
            "role": "user", 
            "content": """##TASK## Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications.
            
            ##CONTEXT## Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.

            Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable."""
        }],
        extra_body={
            "protocol": "minions",
            "max_rounds": 2
        }
    )
    
    print("Minions Response:")
    print(completion.choices[0].message.content)

    # Example 2: Basic Minion Protocol
    print("\n=== Example 2: Basic Minion ===")
    completion2 = client.chat.completions.create(
        model="Qwen3-0.6B-GGUF|gpt-4o",
        messages=[{
            "role": "user", 
            "content": """##TASK## Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications.
            
            ##CONTEXT## Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.

            Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable."""
        }],
        extra_body={
            "protocol": "minion",
            "max_rounds": 2
        }
    )
    
    print("Basic Minion Response:")
    print(completion2.choices[0].message.content)


    # Print the response
    print(completion.choices[0].message.content)

except Exception as e:
    raise Exception("Please take a look at the server logs for details") from e
