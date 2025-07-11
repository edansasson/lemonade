import os
import logging
import openai

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from openai import OpenAI

from lemonade_server.pydantic_models import (
    ChatCompletionRequest,
)
import lemonade.tools.server.llamacpp as llamacpp


def chat_completion(
    chat_completion_request: ChatCompletionRequest, telemetry: llamacpp.LlamaTelemetry
):
    # Extract the local model name from the checkpoint
    local_model, remote_model = chat_completion_request.model.split("|")
    logging.debug(f"Using a combined model: {local_model} | {remote_model}")

    # Configure remote client
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    remote_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Configure local client
    base_url = llamacpp.llamacpp_address(telemetry.port)
    local_client = OpenAI(
        base_url=base_url,
        api_key="lemonade",
    )
    # Prepare local request
    local_request_dict = chat_completion_request.model_dump(
        exclude_unset=True, exclude_none=True
    )

    # Prepare remote request
    remote_request_dict = {
        "model": remote_model,
        "messages": local_request_dict["messages"],
    }

    # Check if streaming is requested
    if chat_completion_request.stream:
        raise NotImplementedError(
            "Streaming is not supported for Minions models at this time"
        )
    else:
        # Non-streaming response
        try:
            # Call local model
            local_response = local_client.chat.completions.create(**local_request_dict)
            logging.debug(f"Local model response: {local_response}")

            # Call remote model
            remote_response = remote_client.chat.completions.create(
                **remote_request_dict
            )
            logging.debug(f"Remote model response: {remote_response}")

            # Combine the responses by concatenating their content
            local_content = local_response.choices[0].message.content
            remote_content = remote_response.choices[0].message.content
            combined_response = local_response
            combined_response.choices[0].message.content = (
                f"{local_content}\n\n{remote_content}"
            )

            # Show telemetry after completion
            telemetry.show_telemetry()

            return combined_response

        except Exception as e:  # pylint: disable=broad-exception-caught
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chat completion error: {str(e)}",
            )
