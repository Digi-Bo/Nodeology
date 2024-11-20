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

import os
from string import Formatter
from inspect import signature
from typing import Optional, Annotated, List, Union, Dict, Callable, Any

from nodeology.state import State
from nodeology.log import log_print_color
from nodeology.client import LLM_Client, VLM_Client


class Node:
    """Template for creating node functions that process data using LLMs or custom functions.

    A Node represents a processing unit in a workflow that can:
    - Execute LLM/VLM queries or custom functions
    - Manage state before and after execution
    - Handle pre/post processing steps
    - Process both text and image inputs
    - Format and validate outputs

    Args:
        name (str): Unique identifier for the node
        prompt_template (str): Template string for the LLM prompt. Uses Python string formatting 
            syntax (e.g., "{variable}"). Empty if using custom_function.
        sink (Optional[Union[List[str], str]]): Where to store results in state. Can be:
            - Single string key
            - List of keys for multiple outputs
            - None (results won't be stored)
        sink_format (Optional[str]): Format specification for LLM output (e.g., "json", "list").
            Used to ensure consistent response structure.
        image_keys (Optional[List[str]]): List of keys for image file paths when using VLM.
            Must provide at least one image path in kwargs when these are specified.
        pre_process (Optional[Callable]): Function to run before main execution.
            Signature: (state: State, client: LLM_Client, **kwargs) -> Optional[State]
        post_process (Optional[Callable]): Function to run after main execution.
            Signature: (state: State, client: LLM_Client, **kwargs) -> Optional[State]
        custom_function (Optional[Callable]): Custom function to execute instead of LLM query.
            Function parameters become required keys for node execution.

    Attributes:
        required_keys (List[str]): Keys required from state/kwargs for node execution.
            Extracted from either prompt_template or custom_function signature.
        prompt_history (List[str]): History of prompt templates used by this node.

    Raises:
        ValueError: If required keys are missing or response format is invalid
        FileNotFoundError: If specified image files don't exist
        ValueError: If VLM operations are attempted without proper client

    Example:
        ```python
        # Create a simple text processing node
        node = Node(
            name="summarizer",
            prompt_template="Summarize this text: {text}",
            sink="summary"
        )

        # Create a node with custom function
        def process_data(x, y):
            return x + y

        node = Node(
            name="calculator",
            prompt_template="",
            sink="result",
            custom_function=process_data
        )
        ```
    """

    def __init__(
        self,
        name: str,
        prompt_template: str,
        sink: Optional[Union[List[str], str]] = None,
        sink_format: Optional[str] = None,
        image_keys: Optional[List[str]] = None,
        pre_process: Optional[
            Callable[[State, LLM_Client, Any], Optional[State]]
        ] = None,
        post_process: Optional[
            Callable[[State, LLM_Client, Any], Optional[State]]
        ] = None,
        custom_function: Optional[Callable[..., Any]] = None,
    ):
        self.name = name
        self.prompt_template = prompt_template
        self.sink = sink
        self.image_keys = image_keys
        self.sink_format = sink_format
        self.pre_process = pre_process
        self.post_process = post_process
        self.custom_function = custom_function

        # Extract required keys from template or custom function signature
        if self.custom_function:
            # Get only required keys (those without default values) from function signature
            sig = signature(self.custom_function)
            self.required_keys = [
                param.name
                for param in sig.parameters.values()
                if param.default is param.empty
                and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
                and param.name != "self"
            ]
        else:
            # Get keys from prompt template
            self.required_keys = [
                fname
                for _, fname, _, _ in Formatter().parse(prompt_template)
                if fname is not None
            ]

        self._prompt_history = [
            prompt_template
        ]  # Add prompt history as private attribute

    @property
    def func(self):
        """Returns the node function without executing it"""

        def node_function(
            state: Annotated[State, "The current state"],
            client: Annotated[LLM_Client, "The LLM client"],
            sink: Optional[Union[List[str], str]] = None,
            source: Optional[Dict[str, str]] = None,
            **kwargs,
        ) -> State:
            return self(state, client, sink, source, **kwargs)

        # Attach the attributes to the function
        node_function.name = self.name
        node_function.prompt_template = self.prompt_template
        node_function.sink = self.sink
        node_function.image_keys = self.image_keys
        node_function.sink_format = self.sink_format
        node_function.pre_process = self.pre_process
        node_function.post_process = self.post_process
        node_function.required_keys = self.required_keys
        return node_function

    def __call__(
        self,
        state: Annotated[State, "The current state"],
        client: Annotated[LLM_Client, "The LLM/VLM client"],
        sink: Optional[Union[List[str], str]] = None,
        source: Optional[Union[Dict[str, str], str]] = None,
        **kwargs,
    ) -> State:
        """Creates and executes a node function from this template.
        
        Args:
            state: Current state object containing variables
            client: LLM or VLM client for making API calls
            sink: Optional override for where to store results
            source: Optional mapping of template keys to state keys
            **kwargs: Additional keyword arguments passed to function
            
        Returns:
            Updated state object with results stored in sink keys
            
        Raises:
            ValueError: If required keys are missing or response format is invalid
            FileNotFoundError: If specified image files don't exist
        """
        # Update node type
        state["previous_node_type"] = state.get("current_node_type", "")
        state["current_node_type"] = self.name

        # Pre-processing if defined
        if self.pre_process:
            pre_process_result = self.pre_process(state, client, **kwargs)
            if pre_process_result is None:
                return state
            state = pre_process_result
        else:
            record_messages(state, [("assistant", self.name + " started.", "green")])

        # Get values from state or kwargs
        if isinstance(source, str):
            source = {"source": source}

        message_values = {}
        for key in self.required_keys:
            if source and key in source:
                source_key = source[key]
                if source_key not in state:
                    raise ValueError(
                        f"Source mapping key '{source_key}' not found in state"
                    )
                message_values[key] = state[source_key]
            elif key in state:
                message_values[key] = state[key]
            elif key in kwargs:
                message_values[key] = kwargs[key]
            else:
                raise ValueError(f"Required key '{key}' not found in state or kwargs")

        # Execute either custom function or LLM call
        if self.custom_function:
            # Get default values from function signature
            sig = signature(self.custom_function)
            default_values = {
                k: v.default
                for k, v in sig.parameters.items()
                if v.default is not v.empty
            }
            # Update message_values with defaults for missing parameters
            for key, default in default_values.items():
                if key not in message_values:
                    message_values[key] = default
            if "state" in sig.parameters and "state" not in message_values:
                message_values["state"] = state
            if "client" in sig.parameters and "client" not in message_values:
                message_values["client"] = client
            response = self.custom_function(**message_values)
        else:
            # Construct message from template
            message = self.prompt_template.format(**message_values)

            # Get LLM response - handle VLM clients differently
            if self.image_keys:
                if not isinstance(client, VLM_Client):
                    raise ValueError("VLM client required for image keys")
                if not any(key in kwargs for key in self.image_keys):
                    raise ValueError(
                        "At least one image key must be provided in kwargs"
                    )

                image_paths = []
                for key in self.image_keys:
                    if key in kwargs:
                        path = kwargs[key]
                        if not os.path.exists(path):
                            raise FileNotFoundError(f"Image file not found: {path}")
                        image_paths.append(path)
                response = client(
                    messages=[{"role": "user", "content": message}],
                    images=image_paths,
                    format=self.sink_format,
                )
            else:
                response = client(
                    messages=[{"role": "user", "content": message}],
                    format=self.sink_format,
                )

        # Update state with response
        if sink is None:
            sink = self.sink

        if sink is None:
            log_print_color(
                f"Warning: No sink specified for node {self.name}", "yellow"
            )
            return state

        if isinstance(sink, str):
            state[sink] = (
                remove_markdown_blocks_formatting(response)
                if not self.custom_function
                else response
            )
        elif isinstance(sink, list):
            if not sink:
                log_print_color(
                    f"Warning: Empty sink list for node {self.name}", "yellow"
                )
                return state

            if len(sink) == 1:
                state[sink[0]] = (
                    remove_markdown_blocks_formatting(response)
                    if not self.custom_function
                    else response
                )
            else:
                if not isinstance(response, (list, tuple)):
                    raise ValueError(
                        f"Expected multiple responses for multiple sink in node {self.name}, but got a single response"
                    )
                if len(response) != len(sink):
                    raise ValueError(
                        f"Number of responses ({len(response)}) doesn't match number of sink ({len(sink)}) in node {self.name}"
                    )

                for key, value in zip(sink, response):
                    state[key] = (
                        remove_markdown_blocks_formatting(value)
                        if not self.custom_function
                        else value
                    )

        # Post-processing if defined
        if self.post_process:
            post_process_result = self.post_process(state, client, **kwargs)
            if post_process_result is None:
                return state
            state = post_process_result

        return state

    def __str__(self):
        MAX_WIDTH = 80

        # Format prompt with highlighted keys
        prompt_lines = self.prompt_template.split("\n")
        # First make the whole prompt green
        prompt_lines = [f"\033[92m{line}\033[0m" for line in prompt_lines]  # Green
        # Then highlight the keys in red
        for key in self.required_keys:
            for i, line in enumerate(prompt_lines):
                prompt_lines[i] = line.replace(
                    f"{{{key}}}",
                    f"\033[91m{{{key}}}\033[0m\033[92m",  # Red keys, return to green after
                )

        # Calculate width for horizontal line (min of actual width and MAX_WIDTH)
        width = min(max(len(line) for line in prompt_lines), MAX_WIDTH)
        double_line = "═" * width
        horizontal_line = "─" * width

        # Color formatting for keys in info section
        required_keys_colored = [
            f"\033[91m{key}\033[0m" for key in self.required_keys
        ]  # Red
        if isinstance(self.sink, str):
            sink_colored = [f"\033[94m{self.sink}\033[0m"]  # Blue
        elif isinstance(self.sink, list):
            sink_colored = [f"\033[94m{key}\033[0m" for key in self.sink]  # Blue
        else:
            sink_colored = ["None"]

        # Build the string representation
        result = [
            double_line,
            f"{self.name}",
            horizontal_line,
            *prompt_lines,
            horizontal_line,
            f"Required keys: {', '.join(required_keys_colored)}",
            f"Sink keys: {', '.join(sink_colored)}",
            f"Format: {self.sink_format or 'None'}",
            f"Image keys: {', '.join(self.image_keys) or 'None'}",
            f"Pre-process: {self.pre_process.__name__ if self.pre_process else 'None'}",
            f"Post-process: {self.post_process.__name__ if self.post_process else 'None'}",
            f"Custom function: {self.custom_function.__name__ if self.custom_function else 'None'}",
        ]

        return "\n".join(result)

    @property
    def prompt_history(self) -> list[str]:
        """Returns the history of prompt templates.

        Returns:
            list[str]: List of prompt templates, oldest to newest
        """
        return self._prompt_history.copy()


def as_node(
    name: str,
    sink: List[str],
    pre_process: Optional[Callable[[State, LLM_Client, Any], Optional[State]]] = None,
    post_process: Optional[Callable[[State, LLM_Client, Any], Optional[State]]] = None,
    as_function: bool = False,
):
    """Decorator to transform a regular Python function into a Node function.

    This decorator allows you to convert standard Python functions into Node objects
    that can be integrated into a nodeology workflow. The decorated function becomes
    the custom_function of the Node, with its parameters becoming required keys.

    Args:
        name (str): Unique identifier for the node
        sink (List[str]): List of state keys where the function's results will be stored.
            The number of sink keys should match the number of return values from the function.
        pre_process (Optional[Callable]): Function to run before main execution.
            Signature: (state: State, client: LLM_Client, **kwargs) -> Optional[State]
        post_process (Optional[Callable]): Function to run after main execution.
            Signature: (state: State, client: LLM_Client, **kwargs) -> Optional[State]
        as_function (bool): If True, returns a callable node function. If False, returns
            the Node object itself. Default is False.

    Returns:
        Union[Node, Callable]: Either a Node object or a node function, depending on
        the as_function parameter.

    Example:
        ```python
        # Basic usage
        @as_node(name="multiply", sink=["result"])
        def multiply(x: int, y: int) -> int:
            return x * y

        # With pre and post processing
        def log_start(state, client, **kwargs):
            print("Starting calculation...")
            return state

        def log_result(state, client, **kwargs):
            print(f"Result: {state['result']}")
            return state

        @as_node(
            name="add",
            sink=["result"],
            pre_process=log_start,
            post_process=log_result
        )
        def add(x: int, y: int) -> int:
            return x + y

        # Multiple return values
        @as_node(name="stats", sink=["mean", "std"])
        def calculate_stats(numbers: List[float]) -> Tuple[float, float]:
            return np.mean(numbers), np.std(numbers)
        ```

    Notes:
        - The decorated function's parameters become required keys for node execution
        - The function can access the state and client objects by including them
          as optional parameters
        - The number of sink keys should match the number of return values
        - When as_function=True, the decorator returns a callable that can be used
          directly in workflows
    """

    def decorator(func):
        # Create a Node instance with the custom function
        node = Node(
            name=name,
            prompt_template="",  # Empty template since we're using custom function
            sink=sink,
            pre_process=pre_process,
            post_process=post_process,
            custom_function=func,  # Pass the function to Node
        )

        # Get only required parameters (those without default values)
        sig = signature(func)
        node.required_keys = [
            param.name
            for param in sig.parameters.values()
            if param.default is param.empty
        ]

        return node.func if as_function else node

    return decorator


def record_messages(state: State, messages: List[tuple[str, str, str]]):
    """Record messages to state and log them with color.
    
    Args:
        state: State object to store messages in
        messages: List of (role, message, color) tuples to record
    """

    for role, message, color in messages:
        # Add check for messages key
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append({"role": role, "content": message})
        log_print_color(f"{role}: {message}", color)


def remove_markdown_blocks_formatting(text: str) -> str:
    """Remove common markdown code block delimiters from text.

    Args:
        text: Input text containing markdown code blocks

    Returns:
        str: Text with code block delimiters removed
    """
    lines = text.split("\n")
    cleaned_lines = []
    in_code_block = False

    for line in lines:
        stripped_line = line.strip()
        # Check if line starts with backticks (more robust than exact matches)
        if stripped_line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
