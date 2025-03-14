"""
Copyright (c) 2024, UChicago Argonne, LLC. All rights reserved.

Copyright 2024. UChicago Argonne, LLC. This software was produced
under U.S. Government contract DE-AC02-06CH11357 for Argonne National
Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
U.S. Department of Energy. The U.S. Government has rights to use,
reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
modified to produce derivative works, such modified software should
be clearly marked, so as not to confuse it with the version available
from ANL.

Additionally, redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.

    * Neither the name of UChicago Argonne, LLC, Argonne National
      Laboratory, ANL, the U.S. Government, nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

### Initial Author <2024>: Xiangyu Yin

import os, base64, json, getpass
from abc import ABC, abstractmethod
import litellm
from datetime import datetime


def get_client(model_name, **kwargs):
    """
    Factory function to create appropriate client based on model name.

    Handles three scenarios:
    1. Just model name (e.g., "gpt-4o") - Let LiteLLM figure out the provider
    2. Model name with provider keyword (e.g., model="gpt-4o", provider="openai")
    3. Provider/name convention (e.g., "openai/gpt-4o")

    Args:
        model_name (str): Name of the model to use
        **kwargs: Additional arguments including optional 'provider'

    Returns:
        LLM_Client or VLM_Client: Appropriate client instance for the requested model
    """
    # Handle special clients first
    if model_name == "mock":
        return Mock_LLM_Client(**kwargs)
    elif model_name == "mock_vlm":
        return Mock_VLM_Client(**kwargs)

    # Get provider from kwargs if specified (Scenario 2)
    provider = kwargs.pop("provider", None)

    # Get tracing_enabled from kwargs
    tracing_enabled = kwargs.pop("tracing_enabled", False)

    # Handle provider/model format (Scenario 3)
    if "/" in model_name and provider is None:
        provider, model_name = model_name.split("/", 1)

    # Create LiteLLM client - for Scenario 1, provider will be None
    try:
        return LiteLLM_Client(
            model_name, provider=provider, tracing_enabled=tracing_enabled, **kwargs
        )
    except Exception as e:
        raise ValueError(f"Error creating client for model {model_name}: {e}")


def configure_langfuse(public_key=None, secret_key=None, host=None, enabled=True):
    """
    Configure Langfuse for observability.

    Args:
        public_key (str, optional): Langfuse public key. Defaults to LANGFUSE_PUBLIC_KEY env var.
        secret_key (str, optional): Langfuse secret key. Defaults to LANGFUSE_SECRET_KEY env var.
        host (str, optional): Langfuse host URL. Defaults to LANGFUSE_HOST env var or https://cloud.langfuse.com.
        enabled (bool, optional): Whether to enable Langfuse tracing. Defaults to True.
    """
    if not enabled:
        litellm.success_callback = []
        litellm.failure_callback = []
        return

    # Set environment variables if provided
    if public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    if secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    if host:
        os.environ["LANGFUSE_HOST"] = host

    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


class LLM_Client(ABC):
    """Base abstract class for Language Model clients."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, messages, **kwargs) -> str:
        """
        Process messages and return model response.

        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            **kwargs: Additional model-specific parameters

        Returns:
            str: Model's response text
        """
        pass


class VLM_Client(LLM_Client):
    """Base abstract class for Vision Language Model clients."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def process_images(self, messages, images, **kwargs) -> list:
        """
        Process and format images for the model.

        Args:
            messages (list): List of message dictionaries
            images (list): List of image file paths
            **kwargs: Additional processing parameters

        Returns:
            list: Updated messages with processed images
        """
        pass


class Mock_LLM_Client(LLM_Client):
    def __init__(self, response=None, **kwargs) -> None:
        super().__init__()
        self.response = response
        self.model_name = "mock"

    def __call__(self, messages, **kwargs) -> str:
        response = (
            "\n".join([msg["role"] + ": " + msg["content"] for msg in messages])
            if self.response is None
            else self.response
        )
        return response


class Mock_VLM_Client(VLM_Client):
    def __init__(self, response=None, **kwargs) -> None:
        super().__init__()
        self.response = response
        self.model_name = "mock_vlm"

    def __call__(self, messages, images=None, **kwargs) -> str:
        if images is not None:
            messages = self.process_images(messages, images)
        if self.response is None:
            message_parts = []
            for msg in messages:
                content = msg["content"]
                if isinstance(content, str):
                    message_parts.append(f"{msg['role']}: {content}")
                else:  # content is already a list of text/image objects
                    parts = []
                    for item in content:
                        if item["type"] == "text":
                            parts.append(item["text"])
                        elif item["type"] == "image":
                            parts.append(f"[Image: {item['image_url']['url']}]")
                    message_parts.append(f"{msg['role']}: {' '.join(parts)}")
            return "\n".join(message_parts)
        return self.response

    def process_images(self, messages, images, **kwargs) -> list:
        # Make a copy to avoid modifying the original
        messages = messages.copy()

        # Simply append a placeholder for each image
        for img in images:
            if isinstance(messages[-1]["content"], str):
                messages[-1]["content"] = [
                    {"type": "text", "text": messages[-1]["content"]},
                    {"type": "image", "image_url": {"url": f"mock_processed_{img}"}},
                ]
            elif isinstance(messages[-1]["content"], list):
                messages[-1]["content"].append(
                    {"type": "image", "image_url": {"url": f"mock_processed_{img}"}}
                )
        return messages


class LiteLLM_Client(VLM_Client):
    """
    Unified client for all LLM/VLM providers using LiteLLM.
    Supports both text and image inputs across multiple providers.
    """

    def __init__(
        self,
        model_name,
        provider=None,
        model_options=None,
        api_key=None,
        tracing_enabled=False,
    ) -> None:
        """
        Initialize LiteLLM client.

        Args:
            model_name (str): Name of the model to use
            provider (str, optional): Provider name (openai, anthropic, etc.)
            model_options (dict): Model parameters like temperature and top_p
            api_key (str, optional): API key for the specified provider
            tracing_enabled (bool, optional): Whether to enable Langfuse tracing. Defaults to False.
        """
        super().__init__()
        self.model_options = model_options if model_options else {}
        self.tracing_enabled = tracing_enabled

        # Set API key if provided
        if api_key and provider:
            os.environ[f"{provider.upper()}_API_KEY"] = api_key

        # Construct the model name for LiteLLM based on whether provider is specified
        # If provider is None, LiteLLM will infer the provider from the model name
        self.model_name = f"{provider}/{model_name}" if provider else model_name

    def collect_langfuse_metadata(
        self,
        workflow=None,
        node=None,
        **kwargs,
    ):
        """
        Collect metadata for Langfuse tracing from workflow and node information.

        Args:
            workflow: The workflow instance (optional)
            node: The node instance (optional)
            **kwargs: Additional metadata to include

        Returns:
            dict: Metadata dictionary formatted for Langfuse
        """
        metadata = {}

        timestamp = datetime.now().strftime("%Y%m%d")
        user_id = getpass.getuser()
        session_id_str = f"{user_id}-{timestamp}"

        metadata["trace_metadata"] = {}

        # Extract workflow metadata if available
        if workflow:
            # Use workflow class name as generation name
            metadata["generation_name"] = workflow.__class__.__name__
            session_id_str += f"-{workflow.__class__.__name__}"

            # Create a generation ID based on workflow name and timestamp
            metadata["generation_id"] = f"gen-{workflow.name}-{timestamp}"

            # Add user ID if available
            if hasattr(workflow, "user_name"):
                metadata["trace_user_id"] = workflow.user_name
            else:
                metadata["trace_user_id"] = user_id

        # Extract node metadata if available
        if node:
            # Use node type as trace name
            metadata["trace_name"] = node.node_type
            session_id_str += f"-{node.node_type}"

            # Add node metadata to trace metadata
            metadata["trace_metadata"].update(
                {
                    "required_keys": node.required_keys,
                    "sink": node.sink,
                    "sink_format": node.sink_format,
                    "image_keys": node.image_keys,
                    "use_conversation": node.use_conversation,
                    "prompt_template": node.prompt_template,
                }
            )

        # Add session ID based on timestamp
        metadata["session_id"] = f"session-{session_id_str}"

        # Add any additional metadata from kwargs
        metadata["trace_metadata"].update(kwargs)

        return metadata

    def process_images(self, messages, images):
        """
        Process and format images for the model using LiteLLM's format.

        Args:
            messages (list): List of message dictionaries
            images (list): List of image file paths

        Returns:
            list: Updated messages with processed images
        """
        # Make a copy to avoid modifying the original
        messages = messages.copy()

        # Convert images to base64
        image_contents = []
        for img in images:
            with open(img, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                image_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )

        # Add images to the last message
        if isinstance(messages[-1]["content"], str):
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]}
            ] + image_contents
        elif isinstance(messages[-1]["content"], list):
            messages[-1]["content"] += image_contents

        return messages

    def __call__(
        self, messages, images=None, format=None, workflow=None, node=None, **kwargs
    ) -> str:
        """
        Process messages and return model response using LiteLLM.

        Args:
            messages (list): List of message dictionaries
            images (list, optional): List of image file paths
            format (str, optional): Response format (e.g., 'json')
            workflow (optional): The workflow instance for metadata extraction
            node (optional): The node instance for metadata extraction
            **kwargs: Additional parameters including metadata for Langfuse

        Returns:
            str: Model's response text
        """
        # Process images if provided
        if images is not None:
            messages = self.process_images(messages, images)

        # Set up response format if needed
        response_format = {"type": "json_object"} if format == "json" else None

        # Extract Langfuse metadata only if tracing is enabled
        langfuse_metadata = {}
        if self.tracing_enabled:
            langfuse_metadata = self.collect_langfuse_metadata(
                workflow=workflow,
                node=node,
                **kwargs,
            )

        try:
            # Use LiteLLM's built-in retry mechanism with Langfuse metadata
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                response_format=response_format,
                num_retries=3,
                metadata=langfuse_metadata if self.tracing_enabled else {},
                **self.model_options,
            )

            content = response.choices[0].message.content

            # Validate JSON if requested
            if format == "json":
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON response from {self.model_name}")

            return content

        except Exception as e:
            raise ValueError(
                f"Failed to generate response from {self.model_name}. Error: {str(e)}"
            )
