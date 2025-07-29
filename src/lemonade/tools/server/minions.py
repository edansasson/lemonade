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

    # Add minions code imports
    from minions.minion_code import DevMinion
    from minions.utils.workspace import WorkspaceManager
except ImportError:
    logging.debug("Minions library not found. Please install it first.")
    logging.debug("Visit the Minions repository: https://github.com/HazyResearch/minions")
    


from pydantic import BaseModel
import re
import asyncio
import json
from typing import Dict, List

async def run_devminion_in_background(chat_completion_request, task, context, 
                                     actual_model_name, base_url, local_temperature, 
                                     local_max_tokens, remote_client):
    """
    Runs DevMinion protocol and formats output for Continue.dev streaming
    """
    try:
        # Create temporary workspace that we won't actually use
        temp_workspace = "temp_devminion_workspace"

        class StructuredLocalOutput(BaseModel):
            files: Dict[str, str]  # filename
            documentation: str
            setup_instructions: List[str]
            completion_notes: str
        
        # Initialize DevMinion with your clients
        lemonade_client = LemonadeClient(
            base_url=base_url,
            model_name=actual_model_name,
            temperature=local_temperature,
            max_tokens=local_max_tokens,
            structured_output_schema=StructuredLocalOutput
        )
        
        dev = DevMinion(
            local_client=lemonade_client,
            remote_client=remote_client,
            workspace_dir=temp_workspace,
            max_edit_rounds=3
        )
        
        # Execute DevMinion task
        result = dev(
            task=task,
            requirements=context if context else "Use the task as context."
        )
        
        # Format result for Continue.dev
        formatted_response = format_devminion_for_continue(result)
        
        # Clean up temporary workspace
        import shutil
        if os.path.exists(temp_workspace):
            shutil.rmtree(temp_workspace)
        
        return {"final_answer": formatted_response}
        
    except Exception as e:
        return {"final_answer": f"Error: {str(e)}"}

def format_devminion_for_continue(devminion_result) -> str:
    """
    Converts DevMinion output dict to a markdown-formatted string compatible with Continue.dev.
    """
    # Header with project overview or fallback title
    project_name = devminion_result.get('runbook', {}).get('project_overview', 'DevMinion Generated Project')
    output = f"# {project_name}\n\n"

    # Extract all generated files and format as markdown code blocks
    workspace_summary = devminion_result.get('workspace_summary', {})
    files = workspace_summary.get('files', {})

    if files:
        output += "## Generated Files\n\n"
        for filepath, content in files.items():
            # Determine language for syntax highlighting from extension
            language = get_language_from_extension(filepath)
            # Format filename with language in code block header (supports Continue.dev)
            output += f"``````\n\n"

    # Optionally add final assessment section
    assessment = devminion_result.get('final_assessment')
    if assessment:
        output += "## Final Assessment\n\n"
        # If assessment is dict, parse keys nicely
        if isinstance(assessment, dict):
            status = assessment.get('project_status')
            if status:
                output += f"**Status:** {status}\n\n"
            completion = assessment.get('completion_percentage')
            if completion:
                output += f"**Completion:** {completion}\n\n"
            strengths = assessment.get('final_assessment', {}).get('strengths')
            if strengths:
                output += "**Strengths:**\n"
                for strength in strengths:
                    output += f"- {strength}\n"
                output += "\n"
            # Add any other fields you want here
        else:
            output += f"{assessment}\n\n"

    # Add setup instructions if present
    session_log = devminion_result.get('session_log', {})
    runbook = session_log.get('runbook', {})
    final_testing = runbook.get('final_testing') if runbook else None
    if final_testing:
        output += "## Setup Instructions\n\n"
        output += f"``````\n\n"

    return output

def get_language_from_extension(filepath: str) -> str:
    """Return the code language identifier based on file extension for syntax highlighting."""
    extension = filepath.split('.')[-1].lower()
    language_map = {
        'py': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'jsx': 'jsx',
        'tsx': 'tsx',
        'java': 'java',
        'cpp': 'cpp',
        'c': 'c',
        'cs': 'csharp',
        'php': 'php',
        'rb': 'ruby',
        'go': 'go',
        'rs': 'rust',
        'kt': 'kotlin',
        'swift': 'swift',
        'sh': 'bash',
        'md': 'markdown',
        'html': 'html',
        'css': 'css',
        'json': 'json',
        'yaml': 'yaml',
        'yml': 'yaml',
        'xml': 'xml',
        'sql': 'sql',
        'dockerfile': 'dockerfile',
        'txt': 'text'
    }
    return language_map.get(extension, 'text')


# CODE TO ALLOW REGULAR MINIONS PROTOCOL WITHIN CONTINUE.DEV

async def fake_streaming_minions(chat_completion_request, telemetry, task, context, 
                                local_model, remote_model, protocol, max_rounds, 
                                actual_model_name, base_url, local_temperature, 
                                local_max_tokens, remote_client):
    """
    Provides fake streaming while running minions protocol in background
    """
    
    # Start the actual minions processing in background
    minions_task = asyncio.create_task(
        run_minions_in_background(
            chat_completion_request, task, context, protocol, max_rounds,
            actual_model_name, base_url, local_temperature, local_max_tokens, remote_client
        )
    )
    
    # Fake streaming messages based on protocol
    if protocol.lower() == "minions":
        progress_messages = [
            "🔍 Local model analyzing task structure...",
            "📡 Remote model providing initial insights...",
            "🔄 Iterating between local and remote models...",
            "🎯 Refining solution through multiple rounds...",
            "✨ Finalizing comprehensive response..."
        ]
    else:  # minion protocol
        progress_messages = [
            "🧠 Local model examining the request...",
            "🌐 Consulting remote model for expertise...",
            "🔄 Synthesizing perspectives...",
            "✅ Preparing final answer..."
        ]
    
    # Stream progress updates
    for i, message in enumerate(progress_messages):
        if not minions_task.done():
            # Format as SSE (Server-Sent Events) for streaming
            chunk = {
                "choices": [{
                    "delta": {"content": f"{message}\n"},
                    "index": 0,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(1.5)
            
            # Add thinking dots
            for _ in range(3):
                if not minions_task.done():
                    dot_chunk = {
                        "choices": [{
                            "delta": {"content": "."},
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(dot_chunk)}\n\n"
                    await asyncio.sleep(0.5)
            
            if not minions_task.done():
                newline_chunk = {
                    "choices": [{
                        "delta": {"content": "\n\n"},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(newline_chunk)}\n\n"
    
    # Wait for actual result
    result = await minions_task
    
    # Stream the final response
    final_content = f"**Final Response:**\n\n{result.get('final_answer', str(result))}"
    final_chunk = {
        "choices": [{
            "delta": {"content": final_content},
            "index": 0,
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    
    # End stream
    yield "data: [DONE]\n\n"

async def run_minions_in_background(chat_completion_request, task, context, protocol, 
                                   max_rounds, actual_model_name, base_url, 
                                   local_temperature, local_max_tokens, remote_client):
    """
    Runs the actual minions protocol in the background (your existing logic)
    """
    try:
        # Force sync execution for LemonadeClient
        import asyncio
        loop = asyncio.get_event_loop()

        class StructuredLocalOutput(BaseModel):
                explanation: str
                citation: str | None
                answer: str | None
        
        if protocol.lower() == "minions":
            # Try using use_async=False in the background task
            lemonade_client = LemonadeClient(
                base_url=base_url,
                model_name=actual_model_name,
                temperature=local_temperature,
                max_tokens=local_max_tokens,
                structured_output_schema=StructuredLocalOutput,
                use_async=False  # ← Try False instead of True
            )
            
            # Run in thread pool to avoid async conflicts
            minions = Minions(lemonade_client, remote_client)
            response = await loop.run_in_executor(
                None, 
                lambda: minions(
                    task=task,
                    doc_metadata="Chat Context",
                    context=[context] if context else ["Use the task as context."],
                    max_rounds=max_rounds
                )
            )
        else:
            # Your existing minion code
            lemonade_client = LemonadeClient(
                base_url=base_url,
                model_name=actual_model_name,
                temperature=local_temperature,
                max_tokens=local_max_tokens,
                structured_output_schema=None,
                use_async=False
            )
            
            minion = Minion(lemonade_client, remote_client, 
                          is_multi_turn=chat_completion_request.multi_turn_mode, 
                          max_history_turns=chat_completion_request.max_history_turns)
            response = minion(
                task=task,
                doc_metadata="Chat Context",
                context=[context] if context else ["Use the task as context."],
                max_rounds=max_rounds
            )
        
        return response
        
    except Exception as e:
        return {"final_answer": f"Error: {str(e)}"}


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
    
def should_use_devminion(task, context):
    """
    Determine if DevMinion should be used based on task content
    """
    devminion_keywords = [
        "create project", "build application", "develop software",
        "generate files", "create files", "build function",
        "implement", "develop", "create code"
    ]
    
    task_lower = task.lower()
    return any(keyword in task_lower for keyword in devminion_keywords)

async def fake_streaming_devminion(chat_completion_request, telemetry, task, context,
                                  actual_model_name, base_url, local_temperature,
                                  local_max_tokens, remote_client):
    """
    DevMinion-specific fake streaming with appropriate progress messages
    """
    
    # Start the actual DevMinion processing in background
    devminion_task = asyncio.create_task(
        run_devminion_in_background(
            chat_completion_request, task, context,
            actual_model_name, base_url, local_temperature, local_max_tokens, remote_client
        )
    )
    
    # DevMinion-specific progress messages
    progress_messages = [
        "🔍 Analyzing development requirements...",
        "📋 Generating project runbook...",
        "🛠️ Setting up development workspace...",
        "💻 Implementing code solutions...",
        "🧪 Running tests and validation...",
        "📝 Generating documentation...",
        "✅ Finalizing project deliverables..."
    ]
    
    # Stream progress updates (same format as your existing implementation)
    for i, message in enumerate(progress_messages):
        if not devminion_task.done():
            chunk = {
                "choices": [{
                    "delta": {"content": f"{message}\n"},
                    "index": 0,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(2.0)  # Longer delays for DevMinion steps
            
            # Add thinking dots
            for _ in range(3):
                if not devminion_task.done():
                    dot_chunk = {
                        "choices": [{
                            "delta": {"content": "."},
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(dot_chunk)}\n\n"
                    await asyncio.sleep(0.5)
            
            if not devminion_task.done():
                newline_chunk = {
                    "choices": [{
                        "delta": {"content": "\n\n"},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(newline_chunk)}\n\n"
    
    # Wait for actual result
    result = await devminion_task
    
    # Stream the final response
    final_content = result.get('final_answer', str(result))
    final_chunk = {
        "choices": [{
            "delta": {"content": final_content},
            "index": 0,
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    
    # End stream
    yield "data: [DONE]\n\n"

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

    # Convert messages to context format expected by Minion
    context = ""
    task = ""

    # Extract the last user message as the task
    for message in chat_completion_request.messages:
        if message["role"] == "user" or message["role"] == "system" or message["role"] == "assistant":
            content = message["content"]

            task_from_content = None
            context_from_content = None

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

    # Check if streaming is requested
    if chat_completion_request.stream:
        # Check if this is a DevMinion request (you can add a flag or detect from content)
        #use_devminion = should_use_devminion(task, context)
        use_devminion = True
        
        if use_devminion:
            return StreamingResponse(
                fake_streaming_devminion(
                    chat_completion_request, telemetry, task, context,
                    actual_model_name, base_url, local_temperature,
                    local_max_tokens, remote_client
                ),
                media_type="text/plain"
            )
        else:
            # Use fake streaming
            return StreamingResponse(
                fake_streaming_minions(
                    chat_completion_request, telemetry, task, context,
                    local_model, remote_model, protocol, max_rounds,
                    actual_model_name, base_url, local_temperature,
                    local_max_tokens, remote_client
                ),
                media_type="text/plain"
            )
    else:
        # Non-streaming response using basic Minion protocol
        try:
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
