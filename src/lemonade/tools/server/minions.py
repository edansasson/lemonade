import os
import logging
import requests
import openai
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from openai import OpenAI
from lemonade_server.pydantic_models import ChatCompletionRequest
import lemonade.tools.server.llamacpp as llamacpp

# Import Minions necessary code
try:
    from minions.minion import Minion
    from minions.minions import Minions
    from minions.clients.openai import OpenAIClient
    from minions.clients.lemonade import LemonadeClient
except ImportError:
    logging.debug("Minions library not found. Please install it first.")
    logging.debug("Visit the Minions repository: https://github.com/HazyResearch/minions")
    


from pydantic import BaseModel
import re

def get_model_name_from_llamacpp_server(port: int) -> str:
    """
    Fetches the actual model name from the llama.cpp server's /v1/models endpoint.
    Returns the model ID that the server recognizes.
    """
    try:
        url = f"http://localhost:{port}/v1/models"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and len(data['data']) > 0:
            model_id = data['data'][0]['id']
            return model_id
        else:
            logging.debug("No models found in server response")
            return None
    except Exception as e:
        logging.debug(f"Error fetching model name from llama.cpp server: {e}")
        return None

def chat_completion(
    chat_completion_request: ChatCompletionRequest, 
    telemetry: llamacpp.LlamaTelemetry
):
    # Extract the local model name from the checkpoint
    local_model, remote_model = chat_completion_request.model.split("|")
    
    logging.debug(f"Using a combined model: {local_model} | {remote_model}")

    # extract all extra feature parameters
    protocol = chat_completion_request.protocol if chat_completion_request.protocol else "minions"  # default to minions
    max_rounds = chat_completion_request.max_rounds if chat_completion_request.max_rounds else 2 if protocol == "minion" else 5  # default to 2 if minions, 5 if minions
    multi_turn_mode = chat_completion_request.multi_turn_mode if chat_completion_request.multi_turn_mode else False # default to False
    max_history_turns = chat_completion_request.max_history_turns if chat_completion_request.max_history_turns else 0 # default to 0
    use_responses_api = chat_completion_request.use_responses_api if chat_completion_request.use_responses_api else False # default to False
    reasoning_effort = chat_completion_request.reasoning_effort if chat_completion_request.reasoning_effort else False # default to False
    local_temperature=chat_completion_request.temperature if chat_completion_request.temperature else 0 # default 0
    remote_temperature=chat_completion_request.remote_temperature if chat_completion_request.remote_temperature else 0 # default 0
    remote_max_tokens=chat_completion_request.remote_max_tokens if chat_completion_request.remote_max_tokens else 4096 # default 4096
    local_max_tokens=chat_completion_request.max_tokens if chat_completion_request.max_tokens else 4096 # default 4096

    # Configure remote client
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")

    # USE TELEMETRY PORT DIRECTLY
    if not hasattr(telemetry, 'port') or telemetry.port is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Telemetry port not available. Make sure Lemonade Server is running with a loaded model."
        )
    
    port = telemetry.port
    
    # GET THE ACTUAL MODEL NAME FROM THE SERVER
    actual_model_name = get_model_name_from_llamacpp_server(port)
    if actual_model_name is None:
        logging.debug(f"Falling back to local_model: {local_model}")
        actual_model_name = local_model
    
    # Create LemonadeClient with the actual model name from the server
    base_url = f"http://localhost:{port}/v1"
    
    # Configure remote client using OpenAI
    remote_client = OpenAIClient(
        model_name=remote_model,
        api_key=os.getenv("OPENAI_API_KEY"),
        use_responses_api=use_responses_api,
        reasoning_effort=reasoning_effort,
        tools=chat_completion_request.tools,
        temperature=remote_temperature,
        max_tokens=remote_max_tokens
    )
    
    # Check if streaming is requested
    if chat_completion_request.stream:
        raise NotImplementedError(
            "Streaming is not supported for basic Minion protocol with LemonadeClient at this time"
        )
    else:
        # Non-streaming response using basic Minion protocol
        try:
            # Convert messages to context format expected by Minion
            context = ""
            task = ""

            # Extract the last user message as the task
            for message in chat_completion_request.messages:
                if message["role"] == "user":
                    content = message["content"]

                    # Try to extract task and context using structured delimiters
                    task_match = re.search(r'##TASK##\s*(.*?)(?=##CONTEXT##|$)', content, re.IGNORECASE | re.DOTALL)
                    context_match = re.search(r'##CONTEXT##\s*(.*?)(?=##TASK##|$)', content, re.IGNORECASE | re.DOTALL)
                    
                    if task_match:
                        task_from_content = task_match.group(1).strip()
                    if context_match:
                        context_from_content = context_match.group(1).strip()
                    
                    # Use parsed values or fall back to entire content
                    if task_from_content:
                        task = task_from_content
                    if context_from_content:
                        context = context_from_content
                    elif not task_from_content:
                        task = content  # Use entire content as task if no special format
                elif message["role"] == "system":
                    context += message["content"] + "\n"
                elif message["role"] == "assistant":
                    context += f"Assistant: {message['content']}\n"
            

            if protocol.lower() == "minions":
                # Define the structured output schema for local client
                class StructuredLocalOutput(BaseModel):
                    explanation: str
                    citation: str | None
                    answer: str | None

                # Initialize the basic Minion protocol with LemonadeClient
                lemonade_client = LemonadeClient(
                    base_url=base_url,
                    model_name=actual_model_name,  # Use the actual model name from server
                    temperature=local_temperature,
                    max_tokens=local_max_tokens,
                    structured_output_schema=StructuredLocalOutput,
                    use_async=True
                )

                # Initialize the basic Minion protocol with LemonadeClient
                minions = Minions(lemonade_client, remote_client)

                # Use the basic Minion protocol with LemonadeClient
                response = minions(
                    task=task,
                    doc_metadata="Chat Context",
                    context=[context] if context else ["""Use the task as context."""],
                    max_rounds=max_rounds
                )
            else:
                lemonade_client = LemonadeClient(
                    base_url=base_url,
                    model_name=actual_model_name,  # Use the actual model name from server
                    temperature=local_temperature,
                    max_tokens=local_max_tokens,
                    structured_output_schema=None,
                    use_async=False
                )

                # Initialize the basic Minion protocol with LemonadeClient
                minion = Minion(lemonade_client, remote_client, is_multi_turn=multi_turn_mode, max_history_turns=max_history_turns)

                # Use the basic Minion protocol with LemonadeClient
                response = minion(
                    task=task,
                    doc_metadata="Chat Context",
                    context=[context] if context else ["""Use the task as context."""],
                    max_rounds=max_rounds
                )
            
            logging.debug(f"Basic Minion protocol with LemonadeClient response: {response}")
            
            # Convert response back to OpenAI format
            openai_response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response.get("final_answer", str(response))
                    },
                    "finish_reason": "stop"
                }],
                "model": f"{local_model}|{remote_model}",
                "usage": {
                    "prompt_tokens": 0,  # Would need to calculate actual usage
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            # Show telemetry after completion
            telemetry.show_telemetry()
            
            return openai_response
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Basic Minion protocol with LemonadeClient error: {str(e)}",
            )
