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
import ast

from nodeology.state import State
from nodeology.log import log_print_color
from nodeology.client import LLM_Client, VLM_Client


def _process_state_with_transforms(
    state: State, transforms: Dict[str, Callable], client: LLM_Client, **kwargs
) -> State:
    """Helper function to apply transforms to state values.

    Args:
        state: Current state
        transforms: Dictionary mapping state keys to transformation functions
        client: LLM client (unused but kept for signature compatibility)
    """
    for key, transform in transforms.items():
        if key in state:
            try:
                state[key] = transform(state[key])
            except Exception as e:
                raise ValueError(f"Error applying transform to {key}: {str(e)}")
    return state


class Node:
    """Template for creating node functions that process data using LLMs or custom functions.

    A Node represents a processing unit in a workflow that can:
    - Execute LLM/VLM queries or custom functions
    - Manage state before and after execution
    - Handle pre/post processing steps
    - Process both text and image inputs
    - Format and validate outputs

    Args:
        prompt_template (str): Template string for the LLM prompt. Uses Python string formatting
            syntax (e.g., "{variable}"). Empty if using custom_function.
        node_type (Optional[str]): Unique identifier for the node.
        sink (Optional[Union[List[str], str]]): Where to store results in state. Can be:
            - Single string key
            - List of keys for multiple outputs
            - None (results won't be stored)
        sink_format (Optional[str]): Format specification for LLM output (e.g., "json", "list").
            Used to ensure consistent response structure.
        image_keys (Optional[List[str]]): List of keys for image file paths when using VLM.
            Must provide at least one image path in kwargs when these are specified.
        pre_process (Optional[Union[Callable, Dict[str, Callable]]]): Either a function to run
            before execution or a dictionary mapping state keys to transform functions.
        post_process (Optional[Union[Callable, Dict[str, Callable]]]): Either a function to run
            after execution or a dictionary mapping state keys to transform functions.
        sink_transform (Optional[Union[Callable, List[Callable]]]): Transform(s) to apply to
            sink value(s). If sink is a string, must be a single callable. If sink is a list,
            can be either a single callable (applied to all sinks) or a list of callables.
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
            node_type="summarizer",
            prompt_template="Summarize this text: {text}",
            sink="summary"
        )

        # Create a node with custom function
        def process_data(x, y):
            return x + y

        node = Node(
            node_type="calculator",
            prompt_template="",
            sink="result",
            custom_function=process_data
        )
        ```
    """

    # Simplified set of allowed functions that return values
    ALLOWED_FUNCTIONS = {
        "len": len,  # Length of sequences
        "str": str,  # String conversion
        "int": int,  # Integer conversion
        "float": float,  # Float conversion
        "sum": sum,  # Sum of numbers
        "max": max,  # Maximum value
        "min": min,  # Minimum value
        "abs": abs,  # Absolute value
    }

    DISALLOWED_FUNCTION_NAMES = [
        "eval",
        "exec",
        "compile",
        "open",
        "print",
        "execfile",
        "exit",
        "quit",
        "help",
        "dir",
        "globals",
        "locals",
        "dir",
        "type",
        "hash",
        "repr",
        "filter",
        "enumerate",
        "reversed",
        "sorted",
        "any",
        "all",
    ]

    # String methods that return values
    ALLOWED_STRING_METHODS = {
        "upper": str.upper,
        "lower": str.lower,
        "strip": str.strip,
        "capitalize": str.capitalize,
    }

    def __init__(
        self,
        prompt_template: str,
        node_type: Optional[str] = None,
        sink: Optional[Union[List[str], str]] = None,
        sink_format: Optional[str] = None,
        image_keys: Optional[List[str]] = None,
        pre_process: Optional[
            Union[
                Callable[[State, LLM_Client, Any], Optional[State]], Dict[str, Callable]
            ]
        ] = None,
        post_process: Optional[
            Union[
                Callable[[State, LLM_Client, Any], Optional[State]], Dict[str, Callable]
            ]
        ] = None,
        sink_transform: Optional[Union[Callable, List[Callable]]] = None,
        custom_function: Optional[Callable[..., Any]] = None,
        use_conversation: Optional[bool] = False,
    ):
        # Set default node_type based on whether it's prompt or function-based
        if node_type is None:
            if custom_function:
                self.node_type = custom_function.__name__
            else:
                self.node_type = "prompt"
        else:
            self.node_type = node_type

        self.prompt_template = prompt_template
        self._escaped_sections = []  # Store escaped sections at instance level
        self.sink = sink
        self.image_keys = image_keys
        self.sink_format = sink_format
        self.custom_function = custom_function
        self.use_conversation = use_conversation

        # Handle pre_process
        if isinstance(pre_process, dict):
            transforms = pre_process
            self.pre_process = (
                lambda state, client, **kwargs: _process_state_with_transforms(
                    state, transforms, client, **kwargs
                )
            )
        else:
            self.pre_process = pre_process

        # Handle post_process
        if isinstance(post_process, dict):
            transforms = post_process
            self.post_process = (
                lambda state, client, **kwargs: _process_state_with_transforms(
                    state, transforms, client, **kwargs
                )
            )
        else:
            self.post_process = post_process

        # Handle sink_transform
        if sink_transform is not None:
            if isinstance(sink, str):
                if not callable(sink_transform):
                    raise ValueError(
                        "sink_transform must be callable when sink is a string"
                    )
                self._sink_transform = sink_transform
            elif isinstance(sink, list):
                if callable(sink_transform):
                    # If single transform provided for multiple sinks, apply it to all
                    self._sink_transform = [sink_transform] * len(sink)
                elif len(sink_transform) != len(sink):
                    raise ValueError("Number of transforms must match number of sinks")
                else:
                    self._sink_transform = sink_transform
            else:
                raise ValueError("sink must be specified to use sink_transform")
        else:
            self._sink_transform = None

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
            # Extract base variable names from expressions, excluding function names and escaped content
            self.required_keys = []
            # First, temporarily replace escaped content
            template = prompt_template

            # Replace {{{ }}} sections first
            import re

            triple_brace_pattern = (
                r"\{{3}[\s\S]*?\}{3}"  # Non-greedy match, including newlines
            )
            for i, match in enumerate(re.finditer(triple_brace_pattern, template)):
                placeholder = f"___ESCAPED_TRIPLE_{i}___"
                self._escaped_sections.append((placeholder, match.group(0)))
                template = template.replace(match.group(0), placeholder)

            # Then replace {{ }} sections
            double_brace_pattern = (
                r"\{{2}[\s\S]*?\}{2}"  # Non-greedy match, including newlines
            )
            for i, match in enumerate(re.finditer(double_brace_pattern, template)):
                placeholder = f"___ESCAPED_DOUBLE_{i}___"
                self._escaped_sections.append((placeholder, match.group(0)))
                template = template.replace(match.group(0), placeholder)

            self._template_with_placeholders = template  # Store modified template

            # Now parse the template normally
            for _, expr, _, _ in Formatter().parse(template):
                if expr is not None:
                    # Parse the expression to identify actual variables
                    try:
                        tree = ast.parse(expr, mode="eval")
                        variables = set()
                        for node in ast.walk(tree):
                            if (
                                isinstance(node, ast.Name)
                                and node.id not in self.ALLOWED_FUNCTIONS
                                and node.id not in self.DISALLOWED_FUNCTION_NAMES
                            ):
                                variables.add(node.id)
                        self.required_keys.extend(variables)
                    except SyntaxError:
                        # If parsing fails, fall back to basic extraction
                        base_var = expr.split("[")[0].split(".")[0].split("(")[0]
                        if (
                            base_var not in self.ALLOWED_FUNCTIONS
                            and base_var not in self.DISALLOWED_FUNCTION_NAMES
                            and base_var not in self.required_keys
                        ):
                            self.required_keys.append(base_var)

            # Remove duplicates while preserving order
            self.required_keys = list(dict.fromkeys(self.required_keys))

        self._prompt_history = [
            prompt_template
        ]  # Add prompt history as private attribute

    def _eval_expr(self, expr: str, context: dict) -> Any:
        """Safely evaluate a Python expression with limited scope."""
        try:
            # Add allowed functions to the context
            eval_context = {
                **context,
                **self.ALLOWED_FUNCTIONS,  # Include built-in functions
            }
            tree = ast.parse(expr, mode="eval")
            return self._eval_node(tree.body, eval_context)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax in expression: {str(e)}")
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")

    def _eval_node(self, node: ast.AST, context: dict) -> Any:
        """Recursively evaluate an AST node with security constraints."""
        if isinstance(node, ast.Name):
            if node.id not in context:
                raise ValueError(f"Variable '{node.id}' not found in context")
            return context[node.id]

        elif isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.UnaryOp):  # Add support for unary operations
            if isinstance(node.op, ast.USub):  # Handle negative numbers
                operand = self._eval_node(node.operand, context)
                return -operand
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):  # Add support for method calls
                obj = self._eval_node(node.func.value, context)
                method_name = node.func.attr
                # List of allowed string methods
                allowed_string_methods = ["upper", "lower", "title", "strip"]
                if method_name in allowed_string_methods:
                    method = getattr(obj, method_name)
                    args = [self._eval_node(arg, context) for arg in node.args]
                    return method(*args)
                raise ValueError(f"String method not allowed: {method_name}")
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in self.ALLOWED_FUNCTIONS:
                    raise ValueError(f"Function not allowed: {func_name}")
                func = context[
                    func_name
                ]  # Get function from context instead of globals
                args = [self._eval_node(arg, context) for arg in node.args]
                return func(*args)
            raise ValueError("Only simple function calls are allowed")

        elif isinstance(node, ast.Attribute):
            # Handle string methods (e.g., text.upper())
            if not isinstance(node.value, ast.Name):
                raise ValueError("Only simple string methods are allowed")

            obj = self._eval_node(node.value, context)
            if not isinstance(obj, str):
                raise ValueError("Methods are only allowed on strings")

            method_name = node.attr
            if method_name not in self.ALLOWED_STRING_METHODS:
                raise ValueError(f"String method not allowed: {method_name}")

            return self.ALLOWED_STRING_METHODS[method_name](obj)

        elif isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(elt, context) for elt in node.elts]

        elif isinstance(node, ast.Subscript):
            value = self._eval_node(node.value, context)
            if isinstance(node.slice, ast.Slice):
                lower = (
                    self._eval_node(node.slice.lower, context)
                    if node.slice.lower
                    else None
                )
                upper = (
                    self._eval_node(node.slice.upper, context)
                    if node.slice.upper
                    else None
                )
                step = (
                    self._eval_node(node.slice.step, context)
                    if node.slice.step
                    else None
                )
                return value[slice(lower, upper, step)]
            else:
                # Handle both numeric indices and string keys
                idx = self._eval_node(node.slice, context)
                try:
                    return value[idx]
                except (TypeError, KeyError) as e:
                    raise ValueError(f"Invalid subscript access: {str(e)}")

        elif isinstance(node, ast.Str):  # For string literals in subscripts
            return node.s

        raise ValueError(f"Unsupported expression type: {type(node).__name__}")

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
        node_function.node_type = self.node_type
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
        state: State,
        client: Union[LLM_Client, VLM_Client],
        sink: Optional[Union[List[str], str]] = None,
        source: Optional[Union[Dict[str, str], str]] = None,
        debug: bool = False,
        use_conversation: Optional[bool] = None,
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
        state["current_node_type"] = self.node_type

        # Pre-processing if defined
        if self.pre_process:
            pre_process_result = self.pre_process(state, client, **kwargs)
            if pre_process_result is None:
                return state
            state = pre_process_result

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
            # Create a context with state variables for expression evaluation
            eval_context = {**message_values}

            # First fill the template with placeholders
            message = self._template_with_placeholders
            for _, expr, _, _ in Formatter().parse(self._template_with_placeholders):
                if expr is not None:
                    try:
                        result = self._eval_expr(expr, eval_context)
                        message = message.replace(f"{{{expr}}}", str(result))
                    except Exception as e:
                        raise ValueError(
                            f"Error evaluating expression '{expr}': {str(e)}"
                        )

            # Now restore the escaped sections
            for placeholder, original in self._escaped_sections:
                message = message.replace(placeholder, original)

            # Record the formatted message
            record_messages(state, [("user", message, "white")], False)

            # Determine if we should use conversation mode
            should_use_conversation = (
                use_conversation
                if use_conversation is not None
                else self.use_conversation
            )
            if should_use_conversation:
                assert "conversation" in state and isinstance(
                    state["conversation"], list
                ), "Conversation does not exist in state or is not a list of messages"

            # Prepare messages for client call
            if should_use_conversation:
                if len(state["conversation"]) == 0 or state["end_conversation"]:
                    state["conversation"].append({"role": "user", "content": message})
                messages = state["conversation"]
            else:
                messages = [{"role": "user", "content": message}]

            # Handle VLM specific requirements
            if self.image_keys:
                if not isinstance(client, VLM_Client):
                    raise ValueError("VLM client required for image keys")

                # Check both state and kwargs for image keys
                image_paths = []
                for key in self.image_keys:
                    if key in state:
                        path = state[key]
                    elif key in kwargs:
                        path = kwargs[key]
                    else:
                        continue

                    if path is None:
                        raise TypeError(
                            f"Image path for '{key}' should be string, got None"
                        )
                    if not isinstance(path, str):
                        raise TypeError(
                            f"Image path for '{key}' should be string, got {type(path)}"
                        )
                    image_paths.append(path)

                if not image_paths:
                    raise ValueError(
                        "At least one image key must be provided in state or kwargs"
                    )

                # Verify all paths exist
                for path in image_paths:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Image file not found: {path}")

                response = client(
                    messages=messages,
                    images=image_paths,
                    format=self.sink_format,
                    workflow=kwargs.get("workflow"),
                    node=self,
                    previous_node_type=state["previous_node_type"],
                )
            else:
                response = client(
                    messages=messages,
                    format=self.sink_format,
                    workflow=kwargs.get("workflow"),
                    node=self,
                    previous_node_type=state["previous_node_type"],
                )

        log_print_color(f"Response: {response}", "white", False)

        # Update state with response
        if sink is None:
            sink = self.sink

        if sink is None:
            log_print_color(
                f"Warning: No sink specified for {self.node_type} node", "yellow"
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
                    f"Warning: Empty sink list for {self.node_type} node", "yellow"
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
                        f"Expected multiple responses for multiple sink in {self.node_type} node, but got a single response"
                    )
                if len(response) != len(sink):
                    raise ValueError(
                        f"Number of responses ({len(response)}) doesn't match number of sink ({len(sink)}) in {self.node_type} node"
                    )

                for key, value in zip(sink, response):
                    state[key] = (
                        remove_markdown_blocks_formatting(value)
                        if not self.custom_function
                        else value
                    )

        # After storing results but before post_process, apply sink transforms
        if self._sink_transform is not None:
            current_sink = sink or self.sink
            if isinstance(current_sink, str):
                state[current_sink] = self._sink_transform(state[current_sink])
            else:
                for key, transform in zip(current_sink, self._sink_transform):
                    state[key] = transform(state[key])

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
            f"{self.node_type}",
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
        @as_node(sink=["result"])
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
            sink=["result"],
            pre_process=log_start,
            post_process=log_result
        )
        def add(x: int, y: int) -> int:
            return x + y

        # Multiple return values
        @as_node(sink=["mean", "std"])
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
            prompt_template="",  # Empty template since we're using custom function
            node_type=func.__name__,
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


def record_messages(
    state: State, messages: List[tuple[str, str, str]], print_to_console: bool = True
):
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
        log_print_color(f"{role}: {message}", color, print_to_console)


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
