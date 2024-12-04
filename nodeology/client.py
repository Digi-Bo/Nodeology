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

import os, base64, json, subprocess
from abc import ABC, abstractmethod
from openai import OpenAI, NOT_GIVEN
import requests

OAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
R2R_URL = "http://localhost:7272"
PPLX_URL = "https://api.perplexity.ai/chat/completions"

OAI_RECOMMENDED_LLMS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
    "gpt-4-turbo",
    "o1-mini",
]
OAI_RECOMMENDED_VLMS = [
    "gpt-4" "gpt-4-turbo",
    "gpt-4o",
]
OLLAMA_RECOMMENDED_LLMS = [
    "llama3.1:70b",
    "llama3.1:405b",
    "qwen2.5:14b",
    "qwen2.5:32b",
    "qwen2.5:72b",
    "qwen2.5-coder:32b",
]
OLLAMA_RECOMMENDED_VLMS = [
    "llama3.2-vision:90b",
    "llava:34b",
]

try:
    terminal_output = subprocess.check_output(["ollama", "list"], text=True)
    OLLAMA_AVAILABLE_MODELS = [
        line.split()[0] for line in terminal_output.strip().split("\n")[1:]
    ]
except Exception:
    OLLAMA_AVAILABLE_MODELS = []

OLLAMA_RECOMMENDED_LLMS = [
    model for model in OLLAMA_AVAILABLE_MODELS if model in OLLAMA_RECOMMENDED_LLMS
]
OLLAMA_RECOMMENDED_VLMS = [
    model for model in OLLAMA_AVAILABLE_MODELS if model in OLLAMA_RECOMMENDED_VLMS
]

TGT_RECOMMENDED_LLMS = [
    "tgt:llama3.1:70b",
    "tgt:llama3.1:405b",
    "tgt:qwen2.5:72b",
    "tgt:qwen2.5-coder:32b",
]
TGT_RECOMMENDED_VLMS = [
    "tgt:llama3.2:90b",
]
TGT_MODEL_NAME_MAP = {
    "tgt:llama3.1:70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "tgt:llama3.1:405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "tgt:qwen2.5:72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "tgt:qwen2.5-coder:32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "tgt:llama3.2:90b": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
}
PPLX_RECOMMENDED_MODELS = [
    "llama-3.1-sonar-small-128k-online",
    "llama-3.1-sonar-large-128k-online",
    "llama-3.1-sonar-huge-128k-online",
]
ANTHROPIC_RECOMMENDED_LLMS = [
    "claude-3.5-haiku",
    "claude-3.5-sonnet",
]
ANTHROPIC_RECOMMENDED_VLMS = [
    "claude-3.5-sonnet",
    "claude-3-opus",
]

"""
This module provides a unified interface for interacting with various LLM and VLM services.
It supports OpenAI, Ollama, Together AI, Anthropic, and Perplexity AI models.
"""


def get_client(model_name, **kwargs):
    """
    Factory function to create appropriate client based on model name.

    Args:
        model_name (str): Name of the model to use
        **kwargs: Additional arguments passed to the client constructor

    Returns:
        LLM_Client or VLM_Client: Appropriate client instance for the requested model

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name in OAI_RECOMMENDED_LLMS + OAI_RECOMMENDED_VLMS:
        return OAI_Client(model_name, base_url=OAI_BASE_URL, **kwargs)
    elif model_name in OLLAMA_RECOMMENDED_LLMS + OLLAMA_RECOMMENDED_VLMS:
        return OAI_Client(model_name, base_url=OLLAMA_BASE_URL, **kwargs)
    elif model_name in TGT_RECOMMENDED_LLMS + TGT_RECOMMENDED_VLMS:
        return OAI_Client(
            TGT_MODEL_NAME_MAP[model_name],
            base_url=TOGETHER_BASE_URL,
            **kwargs,
        )
    elif model_name in ANTHROPIC_RECOMMENDED_LLMS + ANTHROPIC_RECOMMENDED_VLMS:
        return ANTHROPIC_Client(model_name, base_url=ANTHROPIC_BASE_URL, **kwargs)
    elif model_name == "perplexity":
        return PPLX_Client(**kwargs)
    elif model_name == "r2r":
        return R2R_Client(**kwargs)
    elif model_name == "mock":
        return Mock_LLM_Client(**kwargs)
    elif model_name == "mock_vlm":
        return Mock_VLM_Client(**kwargs)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


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
    def __init__(self, response=None) -> None:
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
    def __init__(self, response=None) -> None:
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


class OAI_Client(VLM_Client):
    """
    Client for OpenAI-compatible APIs (OpenAI, Ollama, Together AI).
    Supports both text and image inputs.
    """

    def __init__(
        self,
        model_name,
        model_options={"temperature": 0.1, "top_p": 0.9},
        base_url=None,
        api_key=None,
    ) -> None:
        """
        Initialize OpenAI-compatible client.

        Args:
            model_name (str): Name of the model to use
            model_options (dict): Model parameters like temperature and top_p
            base_url (str, optional): API base URL. Defaults to OpenAI's API
            api_key (str, optional): API key. Will check environment variables if not provided
        """
        super().__init__()
        base_url = base_url if base_url is not None else "https://api.openai.com/v1"
        if base_url == "https://api.openai.com/v1":
            assert (
                api_key is not None or os.environ.get("OPENAI_API_KEY") is not None
            ), "OPENAI_API_KEY is not set"
            api_key = (
                api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
            )
        elif base_url == "http://localhost:11434/v1":
            assert (
                os.system("ollama list") == 0
            ), "ollama command line tool is not installed"
            api_key = "ollama"
        elif base_url == "https://api.together.xyz/v1":
            assert (
                os.environ.get("TOGETHER_API_KEY") is not None
            ), "TOGETHER_API_KEY is not set"
            api_key = (
                api_key if api_key is not None else os.environ.get("TOGETHER_API_KEY")
            )
        else:
            print(
                f"Using custom base URL: {base_url}, you must set the API key if necessary."
            )

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.model_options = model_options

    def process_images(self, messages, images):
        image_messages = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(open(img, 'rb').read()).decode('utf-8')}"
                },
            }
            for img in images
        ]
        if isinstance(messages[-1]["content"], str):
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]}
            ] + image_messages
        elif isinstance(messages[-1]["content"], list):
            messages[-1]["content"] += image_messages
        else:
            raise ValueError(
                f"Unsupported message content type: {type(messages[-1]['content'])}"
            )
        return messages

    def __call__(self, messages, images=None, format=None) -> str:
        if images is not None:
            messages = self.process_images(messages, images)
        # try three times until success
        for _ in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=(
                        {"type": "json_object"} if format == "json" else NOT_GIVEN
                    ),
                    **self.model_options if images is None else {},
                )
                if format == "json":
                    # check if the json is valid
                    test_json = json.loads(completion.choices[0].message.content)
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Error: {e}")
                continue
        raise ValueError("Failed to generate response")


class R2R_Client(LLM_Client):
    """
    Client for R2R (Retrieval-to-Response) service that combines RAG with LLM responses.

    Attributes:
        model_name (str): Name of the LLM to use
        search_strategy (str): Either 'vanilla' or 'hybrid' search approach
        rag_strategy (str): RAG strategy - 'vanilla', 'hyde', or 'rag_fusion'
    """

    def __init__(
        self, model_name="gpt-4o", search_strategy="hybrid", rag_strategy="vanilla"
    ):
        """
        Initialize R2R client.

        Args:
            model_name (str): Name of the LLM to use (must be in OAI_RECOMMENDED_LLMS)
            search_strategy (str): Search strategy ('vanilla' or 'hybrid')
            rag_strategy (str): RAG strategy ('vanilla', 'hyde', or 'rag_fusion')

        Raises:
            AssertionError: If parameters are invalid or R2R server is unhealthy
        """
        from r2r import R2RClient

        super().__init__()
        assert model_name in OAI_RECOMMENDED_LLMS, "Invalid LLM name"
        assert search_strategy in ["vanilla", "hybrid"], "Invalid search strategy"
        assert rag_strategy in ["vanilla", "hyde", "rag_fusion"], "Invalid RAG strategy"
        self.client = R2RClient(R2R_URL)
        assert (
            self.client.health()["results"]["response"] == "ok"
        ), "R2R server is not healthy"
        self.model_name = model_name
        self.search_strategy = search_strategy
        self.rag_strategy = rag_strategy

    def __call__(self, messages):
        vector_search_settings = {
            "use_hybrid_search": self.search_strategy == "hybrid",
        }
        if self.rag_strategy != "vanilla":
            vector_search_settings["search_strategy"] = self.rag_strategy
        return self.client.agent(
            messages=messages,
            rag_generation_config={
                "model": self.model_name,
            },
            vector_search_settings=vector_search_settings,
        )["results"]["messages"][-1]["content"]


class PPLX_Client(LLM_Client):
    """
    Client for Perplexity AI's API service.

    Attributes:
        model_name (str): Name of the Perplexity model to use
        model_options (dict): Configuration options for the model
        api_key (str): Perplexity API key
    """

    def __init__(
        self,
        model_name="llama-3.1-sonar-large-128k-online",
        model_options={
            "temperature": 0.2,
            "top_p": 0.9,
            "return_citations": False,
            "return_images": False,
            "return_related_questions": False,
            "search_domain_filter": None,
        },
        api_key=None,
    ) -> None:
        """
        Initialize Perplexity client.

        Args:
            model_name (str): Name of the model (must be in PPLX_RECOMMENDED_MODELS)
            model_options (dict): Model configuration options
            api_key (str, optional): API key. Will check environment if not provided

        Raises:
            AssertionError: If model_name is invalid or API key is not set
        """
        super().__init__()
        assert model_name in PPLX_RECOMMENDED_MODELS, "Invalid model name"
        assert (
            api_key is not None or os.environ.get("PERPLEXITY_API_KEY") is not None
        ), "PERPLEXITY_API_KEY is not set"

        self.api_key = (
            api_key if api_key is not None else os.environ.get("PERPLEXITY_API_KEY")
        )
        self.model_name = model_name
        self.model_options = model_options

    def _make_request(self, payload):
        """
        Make HTTP request to Perplexity API.

        Args:
            payload (dict): Request payload

        Returns:
            dict: API response

        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(PPLX_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def __call__(self, messages, format=None) -> str:
        if format == "json":
            raise ValueError("Perplexity API does not support JSON response format")

        for _ in range(3):
            try:
                # Filter out None values and handle special parameters
                filtered_options = {
                    k: v
                    for k, v in self.model_options.items()
                    if v is not None and k not in ["search_domain_filter"]
                }

                # Add search_domain_filter separately if it exists
                if self.model_options.get("search_domain_filter"):
                    if not isinstance(self.model_options["search_domain_filter"], list):
                        raise ValueError("search_domain_filter must be a list")
                    if len(self.model_options["search_domain_filter"]) > 3:
                        raise ValueError("search_domain_filter is limited to 3 domains")
                    filtered_options["search_domain_filter"] = self.model_options[
                        "search_domain_filter"
                    ]

                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    **filtered_options,
                }

                response = self._make_request(payload)

                # Return full response content if any of the return_* options are enabled
                if any(
                    self.model_options.get(key, False)
                    for key in [
                        "return_citations",
                        "return_images",
                        "return_related_questions",
                    ]
                ):
                    return response["choices"][0]["message"]

                return response["choices"][0]["message"]["content"]

            except Exception as e:
                print(f"Error: {e}")
                continue
        raise ValueError("Failed to generate response")


class ANTHROPIC_Client(VLM_Client):
    """
    Client for Anthropic's Claude models, supporting both text and vision capabilities.

    Attributes:
        model_name (str): Name of the Claude model
        model_options (dict): Model configuration options
        api_key (str): Anthropic API key
        base_url (str): API endpoint URL
    """

    def __init__(
        self,
        model_name,
        base_url=None,
        model_options={"temperature": 0.1},
        api_key=None,
    ) -> None:
        """
        Initialize Anthropic client.

        Args:
            model_name (str): Name of the Claude model
            base_url (str, optional): API endpoint URL
            model_options (dict): Model configuration options
            api_key (str, optional): API key. Will check environment if not provided

        Raises:
            AssertionError: If API key is not set
        """
        super().__init__()
        assert (
            api_key is not None or os.environ.get("ANTHROPIC_API_KEY") is not None
        ), "ANTHROPIC_API_KEY is not set"

        self.api_key = (
            api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model_name = model_name
        self.model_options = model_options
        self.base_url = base_url if base_url is not None else ANTHROPIC_BASE_URL

    def process_images(self, messages, images):
        """
        Process and format images for Claude's vision capabilities.

        Args:
            messages (list): List of message dictionaries
            images (list): List of image file paths

        Returns:
            list: Updated messages with base64-encoded images
        """
        # Convert images to base64 and format for Anthropic's API
        image_contents = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(open(img, "rb").read()).decode("utf-8"),
                },
            }
            for img in images
        ]

        if isinstance(messages[-1]["content"], str):
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]}
            ] + image_contents
        elif isinstance(messages[-1]["content"], list):
            messages[-1]["content"] += image_contents
        return messages

    def __call__(self, messages, images=None, format=None) -> str:
        """
        Send request to Claude API and get response.

        Args:
            messages (list): List of message dictionaries
            images (list, optional): List of image file paths
            format (str, optional): Response format (e.g., 'json')

        Returns:
            str: Claude's response

        Raises:
            ValueError: If request fails after three attempts
        """
        if images is not None:
            messages = self.process_images(messages, images)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Format messages if they're simple strings
        formatted_messages = []
        for msg in messages:
            if isinstance(msg["content"], str):
                msg["content"] = [{"type": "text", "text": msg["content"]}]
            formatted_messages.append(msg)

        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            **self.model_options,
        }

        # Add system prompt for JSON format
        if format == "json":
            payload["system"] = "Respond using only valid JSON."

        # Try three times until success
        for _ in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}/messages", headers=headers, json=payload
                )
                response.raise_for_status()
                response_data = response.json()

                if format == "json":
                    # Validate JSON response
                    json.loads(response_data["content"][0]["text"])

                return response_data["content"][0]["text"]
            except Exception as e:
                print(f"Error: {e}")
                continue
        raise ValueError("Failed to generate response")
