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

import os, logging
import json, yaml, re
import inspect
import getpass
from datetime import datetime
from jsonschema import validate
from typing import Dict, Any, Optional, List, Union, Callable
import ast, operator, traceback
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np

# Ensure that TypedDict is imported correctly for all Python versions
try:
    from typing import TypedDict, get_type_hints, is_typeddict
except ImportError:
    from typing_extensions import TypedDict, get_type_hints, is_typeddict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from nodeology.log import (
    logger,
    log_print_color,
    add_logging_level,
    setup_logging,
)
from nodeology.client import get_client, LLM_Client, VLM_Client
from nodeology.state import (
    State,
    StateEncoder,
    NumpyJsonPlusSerializer,
    process_state_definitions,
    _resolve_state_type,
    _type_from_str,
)
from nodeology.node import Node, as_node, record_messages


class Workflow(ABC):
    """Abstract base class for workflow management.

    The Workflow class provides a framework for creating and managing stateful workflows
    that combine language models, vision models, and custom processing nodes. It handles
    state management, logging, error recovery, and workflow execution.

    Key Features:
        - State Management: Maintains workflow state with type validation and history
        - Error Recovery: Automatic state restoration on failures
        - Logging: Comprehensive logging with custom levels
        - Checkpointing: Automatic state checkpointing
        - Human Interaction: Handles user input and interrupts
        - Model Integration: Supports both LLM and VLM clients

    Attributes:
        name (str): Unique workflow identifier
        llm_client (LLM_Client): Language model client for text processing
        vlm_client (Optional[VLM_Client]): Vision model client for image processing
        exit_commands (List[str]): Commands that will trigger workflow termination
        save_artifacts (bool): Whether to save state artifacts to disk
        debug_mode (bool): Enable detailed debug logging
        max_history (int): Maximum number of states to keep in history
        state_schema (Type[State]): Type definition for workflow state
        state_history (List[StateSnapshot]): History of workflow states
        state_index (int): Current state index
        graph (CompiledStateGraph): Compiled workflow graph
        langgraph_config (dict): Configuration for langgraph execution

    Example:
        ```python
        class MyWorkflow(Workflow):
            def create_workflow(self):
                # Define workflow structure
                self.workflow = StateGraph(self.state_schema)
                self.workflow.add_node("start", start_node)
                self.workflow.add_node("process", process_node)
                self.workflow.add_edge("start", "process")
                self.workflow.set_entry_point("start")
                self.graph = self.workflow.compile()

        # Create and run workflow
        workflow = MyWorkflow(
            name="example",
            llm_name="gpt-4o",
            save_artifacts=True
        )
        result = workflow.run()
        ```
    """

    def __init__(
        self,
        name: Optional[str] = None,
        llm_name: Union[str, LLM_Client] = "gpt-4o",
        vlm_name: Optional[Union[str, VLM_Client]] = None,
        state_defs: Optional[Union[List, State]] = None,
        exit_commands: Optional[List[str]] = None,
        save_artifacts: bool = True,
        debug_mode: bool = False,
        max_history: int = 1000,
        checkpointer: Union[BaseCheckpointSaver, str] = "memory",
        tracing: bool = False,
        **kwargs,
    ) -> None:
        """Initialize workflow

        Args:
            name: Workflow name (defaults to class name + timestamp)
            llm_name: Name of LLM model to use
            vlm_name: Optional name of VLM model
            state_defs: State definitions (defaults to class state_schema or State)
            exit_commands: List of commands that will exit the workflow
            save_artifacts: Whether to save state artifacts
            debug_mode: Enable debug logging
            max_history: Maximum number of states to keep in history
            tracing: Whether to enable Langfuse tracing (defaults to False)
        """
        # Generate default name if none provided
        self.name = (
            name
            or f"{self.__class__.__name__}_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
        )
        if not self.name:
            raise ValueError("Workflow name cannot be empty")

        # Store tracing configuration
        self.tracing = tracing

        # Configure Langfuse if tracing is enabled
        if self.tracing:
            from nodeology.client import configure_langfuse

            configure_langfuse(enabled=True)

        # Create clients
        if isinstance(llm_name, str):
            self.llm_client = get_client(llm_name, tracing_enabled=self.tracing)
        elif isinstance(llm_name, LLM_Client):
            self.llm_client = llm_name
            # If it's a LiteLLM_Client, set tracing_enabled
            if hasattr(self.llm_client, "tracing_enabled"):
                self.llm_client.tracing_enabled = self.tracing
        else:
            raise ValueError("llm_name must be a string or LLM_Client instance")

        if vlm_name:
            if isinstance(vlm_name, str):
                self.vlm_client = get_client(vlm_name, tracing_enabled=self.tracing)
            elif isinstance(vlm_name, VLM_Client):
                self.vlm_client = vlm_name
                # If it's a LiteLLM_Client, set tracing_enabled
                if hasattr(self.vlm_client, "tracing_enabled"):
                    self.vlm_client.tracing_enabled = self.tracing
            else:
                raise ValueError("vlm_name must be a string or VLM_Client instance")
        else:
            self.vlm_client = None
            logger.warning(
                "VLM client not provided - vision features will be unavailable"
            )

        # Store configuration
        self.exit_commands = (
            exit_commands
            if exit_commands
            else [
                "stop workflow",
                "quit workflow",
                "terminate workflow",
            ]
        )
        self.save_artifacts = save_artifacts
        self.checkpointer = checkpointer
        self.debug_mode = debug_mode
        self.max_history = max_history
        self.kwargs = kwargs

        # Process state definitions
        if is_typeddict(state_defs):
            self.state_schema = state_defs
        elif isinstance(state_defs, list):
            self.state_schema = self._compile_state_definitions(state_defs)
        else:
            self.state_schema = getattr(self, "state_schema", State)

        # Setup logging and initialize workflow
        self._setup_logging()
        self._node_configs = {}
        self._entry_point = None
        self._interrupt_before = []
        self.create_workflow()
        self.initialize()

    def _compile_state_definitions(self, state_defs):
        """Compile state definitions into a State class"""
        annotations = {}

        for state_def in state_defs:
            if isinstance(state_def, tuple) and len(state_def) == 2:
                name, type_hint = state_def
                if isinstance(type_hint, str):
                    # Use _resolve_state_type to get the actual type
                    type_hint = _resolve_state_type(type_hint)
                annotations[name] = type_hint
            elif isinstance(state_def, type) and is_typeddict(state_def):
                # Use get_type_hints to retrieve annotations from the TypedDict
                annotations.update(get_type_hints(state_def))
            else:
                raise ValueError(f"Invalid state definition format: {state_def}")

        # Dynamically create a new TypedDict class with the collected annotations
        CompiledState = TypedDict("CompiledState", annotations)
        return CompiledState

    def _setup_logging(self, base_dir: Optional[str] = None) -> None:
        """Setup workflow-specific logging configuration.

        Configures logging with custom levels and file handlers.

        Args:
            base_dir: Optional base directory for log files
        """
        # Set up basic workflow information
        self.user_name = getpass.getuser()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_name = f"{self.name}_{timestamp}"
        self.log_path = os.path.join("logs", self.name)

        # Add custom logging levels if not already present
        if not hasattr(logging, "PRINTLOG"):
            add_logging_level("PRINTLOG", logging.INFO + 5)
        if not hasattr(logging, "LOGONLY"):
            add_logging_level("LOGONLY", logging.INFO + 1)

        # Setup logging using the log_utils configuration
        setup_logging(
            log_dir=self.log_path,
            log_name=self.log_name,
            debug_mode=self.debug_mode,
            base_dir=base_dir,
        )

        # Log initial workflow configuration
        logger.logonly("########## Settings ##########")
        logger.logonly(f"Workflow name: {self.name}")
        logger.logonly(f"User name: {self.user_name}")
        logger.logonly(f"Debug mode: {self.debug_mode}")
        logger.logonly("##############################")

    def save_state(self, current_state: Optional[StateSnapshot] = None) -> None:
        """Save the current workflow state to history and optionally to disk.

        Args:
            current_state: State snapshot to save (fetched if None)

        Maintains a rolling history window and saves state files if save_artifacts is enabled.
        """
        try:
            if current_state is None:
                current_state = self.graph.get_state(self.langgraph_config)

            # Add to history
            self.state_history.append(current_state)
            current_state_values = current_state.values

            # Maintain rolling history window
            if len(self.state_history) > self.max_history:
                self.state_history = self.state_history[-self.max_history :]

            # Save state file if enabled
            if self.save_artifacts and not self.debug_mode:
                state_file = os.path.join(
                    self.log_path, f"state_{self.state_index}.json"
                )
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(current_state_values, f, indent=2, cls=StateEncoder)

            self.state_index += 1

        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
            if self.debug_mode:
                raise

    def load_state(self, state_index: int) -> None:
        """Load a previous workflow state by index.

        Args:
            state_index: Index of state to load

        Raises:
            ValueError: If state file not found or schema mismatch
        """
        # Try loading from recent history first
        if state_index < self.state_index and state_index >= (
            self.state_index - self.max_history
        ):
            state = self.state_history[-self.state_index + state_index]
            state_values = state.values
        else:
            # Fall back to loading from file
            state_file = os.path.join(self.log_path, f"state_{state_index}.json")
            if os.path.exists(state_file):
                with open(state_file, "r", encoding="utf-8") as f:
                    state_values = json.load(f)
            else:
                raise ValueError(f"State file not found: {state_file}")

        # Validate loaded state against current schema
        annotations = get_type_hints(self.state_schema)
        for field in annotations:
            if field not in state_values:
                raise ValueError("Loaded state does not match current schema")

        # Update graph state and save
        self.graph.update_state(self.langgraph_config, state_values)
        self.save_state()

    def update_state(
        self,
        values: Optional[Dict[str, Any]] = None,
        current_state: Optional[StateSnapshot] = None,
        human_input: Optional[str] = None,
        as_node: Optional[Node] = None,
    ) -> None:
        """Update the workflow state with new values and/or human input.

        Handles nested updates, type validation, and error recovery.

        Args:
            values: Dictionary of state values to update
            current_state: Current state snapshot (fetched if None)
            human_input: Human input to add to messages/conversation
            as_node: Node to attribute the update to

        Raises:
            TypeError: If provided values don't match schema types
            ValueError: If invalid fields are provided in debug mode
        """
        try:
            current_state = (
                self.graph.get_state(self.langgraph_config)
                if current_state is None
                else current_state
            )
            new_state_values = current_state.values.copy()

            annotations = get_type_hints(self.state_schema)

            if values:
                # Validate fields before updating
                invalid_fields = [field for field in values if field not in annotations]
                if invalid_fields:
                    if self.debug_mode:  # Only raise in debug mode
                        raise ValueError(f"Invalid fields in update: {invalid_fields}")
                    else:
                        logger.warning(
                            f"Ignoring invalid fields in update: {invalid_fields}"
                        )
                        # Filter out invalid fields
                        values = {k: v for k, v in values.items() if k in annotations}

                def update_nested(current: dict, updates: dict):
                    """Recursively update nested dictionary with type validation."""
                    for k, v in updates.items():
                        if (
                            k in current
                            and isinstance(current[k], dict)
                            and isinstance(v, dict)
                        ):
                            update_nested(current[k], v)
                        else:
                            if k in annotations:
                                field_type = annotations[k]
                                if not self._validate_type(v, field_type):
                                    raise TypeError(
                                        f"Invalid type for {k}: expected {field_type}, got {type(v)}"
                                    )
                            current[k] = v

                update_nested(new_state_values, values)

            if human_input is not None:
                # Update message-related fields if they exist in schema
                for field, update in [
                    (
                        "messages",
                        lambda: new_state_values["messages"]
                        + [{"role": "user", "content": human_input}],
                    ),
                    (
                        "conversation",
                        lambda: new_state_values["conversation"]
                        + [{"role": "user", "content": human_input}],
                    ),
                    ("human_input", lambda: human_input),
                ]:
                    if field in annotations:
                        if field not in new_state_values:
                            new_state_values[field] = (
                                [] if field in ["messages", "conversation"] else ""
                            )
                        new_value = update()
                        field_type = annotations[field]
                        if not self._validate_type(new_value, field_type):
                            raise TypeError(
                                f"Invalid type for {field}: expected {field_type}, got {type(new_value)}"
                            )
                        new_state_values[field] = new_value

            self.graph.update_state(
                config=self.langgraph_config,
                values=new_state_values,
                as_node=as_node,
            )

        except Exception as e:
            logger.error(f"Error in update_state: {str(e)}\n{traceback.format_exc()}")
            if self.debug_mode:
                raise
            else:
                self._restore_last_valid_state()

    def _create_checkpoint(self) -> None:
        """Create a checkpoint of the current workflow state.

        Saves the current state to a checkpoint file if save_artifacts is enabled.
        """
        if self.save_artifacts and not self.debug_mode:
            state = self.graph.get_state(self.langgraph_config)
            checkpoint_file = os.path.join(self.log_path, "checkpoint.json")
            try:
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(state.values, f, indent=2, cls=StateEncoder)
                logger.debug("Created checkpoint")
            except Exception as e:
                logger.error(f"Failed to create checkpoint: {str(e)}")

    def _restore_last_valid_state(self):
        """Attempt to restore the workflow to the last valid state.

        First tries recent history states, then falls back to checkpoint.
        Raises RuntimeError if no valid state can be restored.
        """
        # First try recent history
        for i in range(self.state_index - 1, max(-1, self.state_index - 4), -1):
            try:
                self.load_state(i)
                logger.info(f"Successfully restored to state {i}")
                return
            except Exception as e:
                logger.warning(f"Failed to restore state {i}: {str(e)}")

        # If that fails, try loading checkpoint
        checkpoint_file = os.path.join(self.log_path, "checkpoint.json")
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint_state = json.load(f)
                self.graph.update_state(self.langgraph_config, checkpoint_state)
                self.save_state()
                logger.info("Successfully restored from checkpoint")
                return
            except Exception as e:
                logger.error(f"Failed to restore from checkpoint: {str(e)}")

        raise RuntimeError("Could not restore to any valid state")

    @abstractmethod
    def create_workflow(self):
        """Create the workflow graph structure"""
        pass

    def add_node(
        self, name: str, node: Optional[Node] = None, client_type: str = "llm", **kwargs
    ):
        """Add a node to the workflow with simplified syntax"""
        # If node has image_keys, automatically set client_type to vlm
        if node and hasattr(node, "image_keys") and node.image_keys:
            client_type = "vlm"
            kwargs["image_keys"] = node.image_keys

        # Initialize configurations if needed
        if not hasattr(self, "_workflow_configs"):
            self._workflow_configs = {
                "nodes": {},
                "edges": [],
                "conditionals": [],
                "entry": None,
            }
        if not hasattr(self, "_node_configs"):
            self._node_configs = {}

        # Store workflow configuration
        workflow_config = {
            "client_type": client_type,
            "node": node,
            "kwargs": kwargs.copy(),
        }
        self._workflow_configs["nodes"][name] = workflow_config

        # Keep original template configuration structure
        node_config = {
            "client_type": client_type,
        }

        if node is not None:
            if node.prompt_template is not None and len(node.prompt_template) > 0:
                node_config.update(
                    {
                        "type": "prompt",
                        "template": node.prompt_template,
                    }
                )
            else:
                node_config["type"] = name

            if node.sink:
                node_config["sink"] = node.sink
            if node.sink_format:
                node_config["sink_format"] = node.sink_format
            if node.image_keys:
                node_config["image_keys"] = node.image_keys
                node_config["client_type"] = "vlm"

        # Handle special kwargs as before
        processed_kwargs = {}
        if kwargs:
            for k, v in kwargs.items():
                if k in node_config:
                    pass
                elif callable(v):
                    processed_kwargs[k] = "${" + k + "}"
                else:
                    processed_kwargs[k] = v

            if processed_kwargs:
                node_config["kwargs"] = processed_kwargs

        self._node_configs[name] = node_config

    def add_flow(self, from_node: str, to_node: str):
        """Add a simple edge between nodes"""
        if not hasattr(self, "_workflow_configs"):
            self._workflow_configs = {
                "nodes": {},
                "edges": [],
                "conditionals": [],
                "entry": None,
            }

        # Store workflow edge configuration
        self._workflow_configs["edges"].append({"from": from_node, "to": to_node})

        # Keep original template configuration
        self._node_configs[from_node]["next"] = to_node if to_node != END else "END"

    def add_conditional_flow(
        self,
        from_node: str,
        condition: Union[str, Callable[[dict], bool]],
        then: str,
        otherwise: str,
    ):
        """Add a conditional edge with simplified syntax"""
        if not hasattr(self, "_workflow_configs"):
            self._workflow_configs = {
                "nodes": {},
                "edges": [],
                "conditionals": [],
                "entry": None,
            }

        # Process condition for tracking
        if isinstance(condition, str):
            condition_str = condition

            # Create a closure to avoid variable capture issues
            def make_condition(cond_str):
                return lambda state: state[cond_str]

            condition_func = make_condition(condition)
        elif callable(condition):
            condition_src = inspect.getsource(condition).strip()

            # Create a closure to avoid variable capture issues
            def make_condition(cond):
                return lambda state: cond(state)

            condition_func = make_condition(condition)

            if condition_src.startswith("lambda"):
                # Extract just the lambda function definition using regex
                lambda_pattern = r"lambda\s+[^:]+:\s*([^,\n]+)"
                match = re.search(lambda_pattern, condition_src)
                if match:
                    condition_str = match.group(1).strip()
                else:
                    raise ValueError(
                        f"Could not parse lambda condition: {condition_src}"
                    )
                # Replace state['var'], state["var"], state.get["var"] and state.get('var') with just var
                condition_str = re.sub(r"state\['([^']+)'\]", r"\1", condition_str)
                condition_str = re.sub(r'state\["([^"]+)"\]', r"\1", condition_str)
                condition_str = re.sub(r'state\.get\("([^"]+)"\)', r"\1", condition_str)
                condition_str = re.sub(r"state\.get\('([^']+)'\)", r"\1", condition_str)
            else:
                # For complex functions, use template variable for tracking
                condition_str = "${" + condition.__name__ + "}"

        # Store workflow conditional configuration
        self._workflow_configs["conditionals"].append(
            {
                "from": from_node,
                "condition": condition_func,
                "then": then,
                "otherwise": otherwise,
            }
        )

        # Keep original template configuration
        self._node_configs[from_node]["next"] = {
            "condition": condition_str,
            "then": "END" if then == END else then,
            "otherwise": "END" if otherwise == END else otherwise,
        }

    def set_entry(self, node: str):
        """Set the workflow entry point"""
        if not hasattr(self, "_workflow_configs"):
            self._workflow_configs = {
                "nodes": {},
                "edges": [],
                "conditionals": [],
                "entry": None,
            }

        self._workflow_configs["entry"] = node
        self._entry_point = node  # Keep original entry point storage

    def compile(
        self,
        interrupt_before: Optional[List[str]] = None,
        checkpointer: Optional[Union[str, BaseCheckpointSaver]] = None,
        auto_input_nodes: bool = True,
    ):
        """Compile the workflow with optional interrupt points and checkpointing"""
        if not hasattr(self, "workflow"):
            self.workflow = StateGraph(self.state_schema)

        # Setup checkpointer
        checkpointer = checkpointer if checkpointer else self.checkpointer
        if checkpointer == "memory":
            checkpointer = MemorySaver(serde=NumpyJsonPlusSerializer())
        elif not isinstance(checkpointer, BaseCheckpointSaver):
            raise ValueError(
                "checkpointer must be 'memory' or a BaseCheckpointSaver instance"
            )

        # Track input nodes if auto creation is enabled
        input_nodes = set()
        node_mapping = {}  # Maps original nodes to their input nodes
        added_nodes = set()  # Track nodes that have been added

        if auto_input_nodes and interrupt_before:
            # Create input nodes for interrupted nodes (avoiding duplicates)
            for node_name in interrupt_before:
                input_node_name = f"{node_name}_input"

                # Skip if this input node was already created
                if input_node_name not in added_nodes:
                    input_nodes.add(input_node_name)
                    node_mapping[node_name] = input_node_name
                    # Add input node to workflow
                    self.workflow.add_node(input_node_name, lambda state: state)
                    added_nodes.add(input_node_name)

        # First pass: Create all nodes from workflow configs
        for node_name, config in self._workflow_configs["nodes"].items():
            if node_name not in added_nodes:
                # Create the actual node
                node = config.get("node")
                if node is None:
                    self.workflow.add_node(node_name, lambda state: state)
                else:
                    # Select appropriate client
                    client_type = config["client_type"].lower()
                    if client_type == "llm":
                        client = self.llm_client
                    elif client_type == "vlm":
                        if self.vlm_client is None:
                            raise ValueError("VLM client not available")
                        client = self.vlm_client
                    else:
                        raise ValueError(f"Invalid client_type: {client_type}")

                    # Create wrapped function with client injection
                    def wrapped_func(
                        state,
                        n=node,
                        c=client,
                        debug=self.debug_mode,
                        k=config["kwargs"],
                    ):
                        # Pass workflow and node to client for metadata tracking
                        return n(state, c, debug=debug, workflow=self, node=n, **k)

                    self.workflow.add_node(node_name, wrapped_func)
                    added_nodes.add(node_name)

        # Second pass: Create all edges with proper routing
        for edge in self._workflow_configs["edges"]:
            from_node = edge["from"]
            to_node = edge["to"]

            if auto_input_nodes and interrupt_before:
                # If the target node needs input, route through its input node
                if to_node in interrupt_before and to_node != END:
                    # Add edge from original source to target's input node
                    self.workflow.add_edge(from_node, node_mapping[to_node])
                    # Add edge from input node to actual target
                    self.workflow.add_edge(node_mapping[to_node], to_node)
                else:
                    # Regular edge
                    self.workflow.add_edge(
                        from_node, END if to_node == END else to_node
                    )
            else:
                # Regular edge without input node routing
                self.workflow.add_edge(from_node, END if to_node == END else to_node)

        # Third pass: Add conditional edges with proper routing
        for config in self._workflow_configs["conditionals"]:
            from_node = config["from"]
            condition = config["condition"]
            then_target = config["then"]
            otherwise_target = config["otherwise"]

            if auto_input_nodes and interrupt_before:
                # Route targets through input nodes if needed
                if then_target in interrupt_before and then_target != END:
                    # Add edge from input node to actual target for 'then' branch
                    self.workflow.add_edge(node_mapping[then_target], then_target)
                    then_target = node_mapping[then_target]

                if otherwise_target in interrupt_before and otherwise_target != END:
                    # Add edge from input node to actual target for 'otherwise' branch
                    self.workflow.add_edge(
                        node_mapping[otherwise_target], otherwise_target
                    )
                    otherwise_target = node_mapping[otherwise_target]

            # Use a function factory to create wrapped_condition_func
            def make_wrapped_condition_func(condition):
                if isinstance(condition, Callable):
                    return lambda state: "then" if condition(state) else "otherwise"
                elif isinstance(condition, str):
                    return lambda state: (
                        "then" if _eval_condition(condition, state) else "otherwise"
                    )
                else:
                    raise ValueError(f"Invalid condition type: {type(condition)}")

            wrapped_condition_func = make_wrapped_condition_func(condition)

            self.workflow.add_conditional_edges(
                from_node,
                wrapped_condition_func,
                {
                    "then": END if then_target == END else then_target,
                    "otherwise": END if otherwise_target == END else otherwise_target,
                },
            )

        # Set entry point with proper routing
        entry_point = self._workflow_configs["entry"]
        if entry_point is None:
            raise ValueError("No entry point set. Call set_entry first.")

        if auto_input_nodes and interrupt_before and entry_point in interrupt_before:
            # If entry point needs input, route through its input node
            self.workflow.add_edge(node_mapping[entry_point], entry_point)
            self.workflow.set_entry_point(node_mapping[entry_point])
        else:
            self.workflow.set_entry_point(entry_point)

        # Store interrupt configuration and compile
        self._interrupt_before = (
            list(input_nodes) if auto_input_nodes else (interrupt_before or [])
        )
        self.graph = self.workflow.compile(
            checkpointer=checkpointer, interrupt_before=self._interrupt_before
        )

    def _validate_type(self, value: Any, expected_type: Any) -> bool:
        """Validate that a value matches the expected type.

        Handles complex types including:
        - Union types
        - List types with element validation
        - Dict types with key/value validation
        - Numpy arrays

        Args:
            value: Value to validate
            expected_type: Type to validate against

        Returns:
            bool: Whether the value matches the expected type
        """
        from typing import get_origin, get_args, Union

        # Add numpy array handling
        if expected_type is np.ndarray:
            return isinstance(value, np.ndarray)

        origin_type = get_origin(expected_type)

        if origin_type is Union:
            return any(self._validate_type(value, t) for t in get_args(expected_type))
        elif origin_type is list:
            if not isinstance(value, list):
                return False
            elem_type = get_args(expected_type)[0]
            return all(self._validate_type(v, elem_type) for v in value)
        elif origin_type is dict:
            if not isinstance(value, dict):
                return False
            key_type, val_type = get_args(expected_type)
            return all(
                self._validate_type(k, key_type) and self._validate_type(v, val_type)
                for k, v in value.items()
            )
        else:
            return isinstance(value, expected_type)

    def initialize(
        self, init_values: Optional[Dict[str, Any]] = None, recursion_limit=99999999
    ) -> None:
        """Initialize the workflow state with proper None handling and type checking"""
        assert hasattr(self, "graph"), "Workflow graph must be defined"
        assert isinstance(
            self.graph, CompiledStateGraph
        ), "Graph must be a CompiledStateGraph instance"

        # Use get_type_hints to resolve actual types
        annotations = get_type_hints(self.state_schema)

        # Initialize default values for all state fields
        default_state = {}
        for field, field_type in annotations.items():
            if field_type in (str, int, float, bool):
                default_values = {str: "", int: 0, float: 0.0, bool: False}
                default_state[field] = default_values[field_type]
            elif field_type == list or (
                hasattr(field_type, "__origin__") and field_type.__origin__ is list
            ):
                default_state[field] = []
            elif field_type == dict or (
                hasattr(field_type, "__origin__") and field_type.__origin__ is dict
            ):
                default_state[field] = {}
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                # For Union types, use the first type's default value
                first_type = field_type.__args__[0]
                if first_type in (str, int, float, bool):
                    default_values = {str: "", int: 0, float: 0.0, bool: False}
                    default_state[field] = default_values[first_type]
                else:
                    default_state[field] = None
            else:
                default_state[field] = None

        # Validate input fields before updating
        if init_values:
            invalid_fields = [
                field for field in init_values if field not in annotations
            ]
            if invalid_fields and self.debug_mode:
                raise ValueError(f"Invalid fields in initialization: {invalid_fields}")

        # Update defaults with provided values
        if init_values:
            for field, value in init_values.items():
                if field in annotations:
                    if value is None:
                        # Keep the default value if None is provided
                        continue
                    field_type = annotations[field]
                    if not self._validate_type(value, field_type):
                        raise TypeError(
                            f"Invalid type for {field}: expected {field_type}, got {type(value)}"
                        )
                    default_state[field] = value

        self.langgraph_config = {
            "configurable": {"thread_id": self.log_name},
            "recursion_limit": recursion_limit,
        }
        self.graph.update_state(
            config=self.langgraph_config,
            values=default_state,
        )

        self.state_index = 0
        self.state_history = []
        self.save_state()

    def run(self, init_values: Optional[Dict] = None) -> Dict:
        """Run the workflow until completion or interruption.

        Executes the workflow graph, handling human input and state management.

        Args:
            init_values: Optional initial state values

        Returns:
            Dict: Final workflow state values or error state
        """
        log_print_color(f"Starting {self.name} workflow for {self.user_name}", "green")

        # Initialize graph state
        graph_input = (
            self.graph.get_state(self.langgraph_config).values
            if init_values is None
            else init_values
        )
        error_state = None
        current_state = self.graph.get_state(self.langgraph_config)

        try:
            while True:
                # Run the graph until it needs input or reaches the end
                for _ in self.graph.stream(graph_input, self.langgraph_config):
                    current_state = self.graph.get_state(self.langgraph_config)
                    self.save_state()

                    # Check if we've reached the end
                    if (
                        current_state.next is None
                        or len(current_state.next) == 0
                        or END in current_state.next
                        or "END" in current_state.next
                    ):
                        return current_state.values if current_state else {}

                # Get human input when the graph needs it
                human_input = self._get_human_input()
                if self._should_exit(human_input):
                    return current_state.values if current_state else {}

                # Update state with human input and continue
                self.update_state(
                    human_input=human_input, as_node=current_state.next[0]
                )
                self.save_state()
                graph_input = None  # Reset input for next iteration

        except Exception as e:
            logger.error(f"Error during workflow execution: {str(e)}")
            error_state = {"error": str(e)}
            if self.debug_mode:
                raise
            return (
                error_state
                if error_state
                else current_state.values if current_state else {}
            )

    def _should_exit(self, cmd_input: str) -> bool:
        """Check if a command should exit the workflow.

        Args:
            cmd_input: Command string to check

        Returns:
            bool: True if command matches any exit command
        """
        return any(cmd in cmd_input.lower() for cmd in self.exit_commands)

    def _get_human_input(self, mode: str = "cmd") -> str:
        """Get and log input from the user.

        Args:
            mode: Input mode (currently only supports 'cmd')

        Returns:
            str: User input

        Raises:
            ValueError: If invalid input mode specified
        """
        if mode == "cmd":
            human_input = input(f"{self.user_name}: ")
        else:
            raise ValueError(f"Invalid input mode: {mode}")
        logger.logonly(f"{self.user_name}: {human_input}")
        return human_input

    def __enter__(self):
        """Context manager entry point.

        Returns:
            Workflow: Self for use in with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point with cleanup.

        Creates final checkpoint and cleans up logging handlers.
        """
        try:
            # Create checkpoint before cleanup
            self._create_checkpoint()

            # Store a reference to the root logger
            root_logger = logging.getLogger()

            # Clean up logging handlers safely
            handlers = root_logger.handlers[:]  # Create a copy of the list
            for handler in handlers:
                try:
                    # Flush any remaining logs
                    handler.flush()
                    # Remove handler from logger first
                    root_logger.removeHandler(handler)
                    # Then close the handler
                    handler.close()
                except Exception as e:
                    # Log error without using file handler
                    print(f"Warning during handler cleanup: {str(e)}")

            # Clean up any graph resources
            if hasattr(self, "graph"):
                # Add graph cleanup if needed
                pass

        except Exception as e:
            # Print error since logging may not be available
            print(f"Error during workflow cleanup: {str(e)}")
            if self.debug_mode:
                raise

    def to_yaml(self, output_path: Optional[str] = None) -> Dict:
        """Export workflow configuration to YAML format.

        Args:
            output_path: Optional path to save YAML file. If not provided,
                        returns dictionary representation only.

        Returns:
            Dict: Template configuration dictionary

        Raises:
            ValueError: If workflow configuration is invalid for export
        """
        return export_workflow_to_template(self, output_path)


def _validate_template_structure(template: Dict) -> None:
    """Validate the basic structure of a workflow template against schema.

    Required fields:
    - name: string
    - state_defs: array of state definitions
    - nodes: object containing node definitions
    - entry_point: string

    Optional fields:
    - llm: string (default: "gpt-4o")
    - vlm: string
    - exit_commands: array of strings
    - intervene_before: array of strings

    Args:
        template: Template dictionary to validate

    Raises:
        ValueError: If template structure is invalid
    """
    schema = {
        "type": "object",
        "required": ["name", "state_defs", "nodes", "entry_point"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "state_defs": {"type": "array", "minItems": 1},
            "nodes": {"type": "object", "minProperties": 1},
            "entry_point": {"type": "string"},
            "llm": {"type": "string"},
            "vlm": {"type": "string"},
            "exit_commands": {"type": "array", "items": {"type": "string"}},
            "intervene_before": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }

    try:
        validate(instance=template, schema=schema)
    except Exception as e:
        raise ValueError(f"Invalid template structure: {str(e)}")


def _validate_nodes(nodes: Dict, node_registry: Dict) -> None:
    """Validate node configurations in the template.

    Checks:
    - Required fields presence
    - Node type validity
    - Prompt node specific configuration
    - Next/conditional logic validity

    Args:
        nodes: Dictionary of node configurations
        node_registry: Dictionary of available node types

    Raises:
        ValueError: If node configuration is invalid
    """
    for node_name, node_config in nodes.items():
        # Check required fields
        if "type" not in node_config:
            raise ValueError(f"Node {node_name} missing 'type' field")

        node_type = node_config["type"]

        # Validate by node type
        if node_type == "prompt":
            _validate_prompt_node(node_name, node_config)
        elif node_type not in node_registry:
            raise ValueError(
                f"Unknown node type: {node_type} (available: {list(node_registry.keys())})"
            )

        # Validate next/conditional logic
        if "next" not in node_config:
            raise ValueError(f"Node '{node_name}' missing 'next' field")

        # Convert string "END" to END constant
        if isinstance(node_config["next"], str) and node_config["next"] == "END":
            node_config["next"] = END
        elif isinstance(node_config["next"], dict):
            if node_config["next"].get("then") == "END":
                node_config["next"]["then"] = END
            if node_config["next"].get("otherwise") == "END":
                node_config["next"]["otherwise"] = END

        _validate_node_transitions(node_name, node_config["next"])


def _validate_prompt_node(node_name: str, config: Dict) -> None:
    """Validate prompt node specific configuration.

    Args:
        node_name: Name of the node
        config: Node configuration dictionary

    Raises:
        ValueError: If prompt node configuration is invalid
    """
    # Check required template field
    if "template" not in config:
        raise ValueError(f"Prompt node '{node_name}' missing 'template' field")

    # Validate optional image_keys field
    if "image_keys" in config:
        # Accept both string (single item) and list formats
        if not isinstance(config["image_keys"], (str, list)):
            raise ValueError(
                f"Prompt node '{node_name}' image_keys must be a string or list"
            )
        # Convert string to list internally
        if isinstance(config["image_keys"], str):
            config["image_keys"] = [config["image_keys"]]

    # Validate sink field if present
    if "sink" in config:
        # Accept both string (single item) and list formats
        if isinstance(config["sink"], str):
            config["sink"] = [config["sink"]]  # Convert single string to list
        elif not isinstance(config["sink"], list):
            raise ValueError(f"Prompt node '{node_name}' sink must be a string or list")

        # Validate each sink field name
        for sink in config["sink"]:
            if not isinstance(sink, str):
                raise ValueError(
                    f"Prompt node '{node_name}' sink values must be strings"
                )


def _validate_condition_expr(expr: str) -> bool:
    """Validate a conditional expression for security and correctness.

    Checks:
    - Python syntax validity
    - Allowed operations and functions
    - Security constraints

    Args:
        expr: Condition expression string

    Returns:
        bool: True if expression is valid

    Raises:
        ValueError: If expression is invalid or contains forbidden operations
    """
    import ast

    try:
        tree = ast.parse(expr, mode="eval")
        allowed_ops = (
            # Core expression nodes
            ast.Expression,  # Root node for eval mode
            ast.Name,  # Variable names
            ast.Constant,  # Literal values
            ast.List,  # List literals
            ast.Dict,  # Dictionary literals
            # Array/Dict access
            ast.Subscript,  # For array[index] or dict[key]
            ast.Index,  # For simple indexing
            ast.Slice,  # For slice operations
            # Comparison operators
            ast.Compare,
            ast.Eq,  # ==
            ast.NotEq,  # !=
            ast.Lt,  # <
            ast.LtE,  # <=
            ast.Gt,  # >
            ast.GtE,  # >=
            # Unary operators
            ast.UnaryOp,
            ast.Is,  # is
            ast.IsNot,  # is not
            ast.In,  # in
            ast.NotIn,  # not in
            # Boolean operators
            ast.BoolOp,
            ast.And,  # and
            ast.Or,  # or
            ast.Not,  # not
            # Function calls
            ast.Call,
            # Additional nodes for comprehensions
            ast.ListComp,  # List comprehensions
            ast.SetComp,  # Set comprehensions
            ast.DictComp,  # Dict comprehensions
            ast.GeneratorExp,  # Generator expressions
            ast.comprehension,  # The 'for' part of comprehensions
        )
        allowed_funcs = {
            "len",
            "upper",
            "lower",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "all",
            "any",
            "filter",
            "map",
            "sum",
            "max",
            "min",
        }

        # Walk AST and validate each node
        for node in ast.walk(tree):
            # Skip context attributes
            if isinstance(node, (ast.Load, ast.Store)):
                continue

            if not isinstance(node, allowed_ops):
                raise ValueError(
                    f"Invalid operation in condition: {type(node).__name__}"
                )
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in allowed_funcs:
                        raise ValueError(
                            f"Function not allowed in condition: {node.func.id}"
                        )
                else:
                    raise ValueError("Only simple function calls are allowed")

        return True
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax in condition: {str(e)}")
    except Exception as e:
        raise ValueError(f"Invalid condition expression: {str(e)}")


def _validate_node_transitions(node_name: str, next_config: Union[str, Dict]) -> None:
    """Validate node transition configuration.

    Checks transition configuration for both simple and conditional transitions.

    Args:
        node_name: Name of the node being validated
        next_config: Transition configuration (string for simple, dict for conditional)

    Raises:
        ValueError: If transition configuration is invalid
    """
    if isinstance(next_config, dict):
        # Validate conditional transition structure
        if "condition" not in next_config:
            raise ValueError(
                f"Conditional transition in node '{node_name}' missing 'condition'"
            )
        if "then" not in next_config or "otherwise" not in next_config:
            raise ValueError(
                f"Conditional transition in node '{node_name}' missing then/otherwise paths"
            )

        # Validate condition expression
        try:
            _validate_condition_expr(next_config["condition"])
        except ValueError as e:
            raise ValueError(f"Invalid condition in node '{node_name}': {str(e)}")


def _validate_state_definitions(state_defs: List, state_registry: Dict) -> None:
    """Validate state definitions in template.

    Supports multiple formats:
    1. String references to registered states: ["HilpState", "CustomState"]
    2. Tuple definitions: [["name", "type"], ["other", "str"]]
    3. Dictionary definitions: [{"name": "type"}, {"other": "str"}]

    Args:
        state_defs: List of state definitions
        state_registry: Dictionary of available state types

    Raises:
        ValueError: If state definitions are invalid
    """
    for state_def in state_defs:
        if isinstance(state_def, str):
            # Format 1: Validate reference to registered state
            if state_def not in state_registry:
                raise ValueError(f"Unknown state type: {state_def}")
        elif isinstance(state_def, list) and len(state_def) == 2:
            # Format 2: Validate (name, type) tuple format
            name, type_str = state_def
            if not isinstance(name, str):
                raise ValueError(f"State name must be a string: {name}")
            try:
                _resolve_state_type(type_str)
            except ValueError:
                raise ValueError(f"Cannot resolve state type: {type_str}")
        elif isinstance(state_def, dict) and len(state_def) == 1:
            # Format 3: Validate {"name": "type"} dictionary format
            name, type_str = next(iter(state_def.items()))
            if not isinstance(name, str):
                raise ValueError(f"State name must be a string: {name}")
            try:
                _resolve_state_type(type_str)
            except ValueError:
                raise ValueError(f"Cannot resolve state type: {type_str}")
        else:
            raise ValueError(
                f"Invalid state definition format: {state_def}. Must be a string, "
                "[name, type] list, or {'name': 'type'} dictionary"
            )


def _safe_read_template(template_path: str, node_registry: Dict, state_registry: Dict):
    """Safely load and perform initial validation of workflow template.

    Performs basic structural validation before any variable interpolation.

    Args:
        template_path: Path to YAML template file
        node_registry: Dictionary of available node types
        state_registry: Dictionary of available state types

    Returns:
        Dict: Loaded template dictionary

    Raises:
        ValueError: If template cannot be loaded or basic validation fails
    """
    # Load template
    try:
        with open(template_path, "r") as f:
            template = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load template: {str(e)}")

    # Check for required fields without validating their content yet
    required_fields = ["name", "state_defs", "nodes", "entry_point"]
    missing_fields = [field for field in required_fields if field not in template]
    if missing_fields:
        raise ValueError(f"Template missing required fields: {missing_fields}")

    # Basic type checking that won't be affected by variable interpolation
    if not isinstance(template.get("state_defs", []), list):
        raise ValueError("Template 'state_defs' must be a list")
    if not isinstance(template.get("nodes", {}), dict):
        raise ValueError("Template 'nodes' must be a dictionary")
    if not isinstance(template.get("exit_commands", []), list):
        raise ValueError("Template 'exit_commands' must be a list")
    if not isinstance(template.get("intervene_before", []), list):
        raise ValueError("Template 'intervene_before' must be a list")

    # Validate node structure (but not content)
    for node_name, node_config in template["nodes"].items():
        if not isinstance(node_config, dict):
            raise ValueError(f"Node '{node_name}' configuration must be a dictionary")
        if "type" not in node_config:
            raise ValueError(f"Node '{node_name}' missing 'type' field")
        if "next" not in node_config:
            raise ValueError(f"Node '{node_name}' missing 'next' field")

    return template


def _eval_condition_expr(node, state, allowed_funcs, operators):
    """Evaluate a single AST node in a condition expression.

    Recursively evaluates AST nodes while enforcing security constraints.

    Args:
        node: AST node to evaluate
        state: Current workflow state
        allowed_funcs: Dictionary of allowed functions
        operators: Dictionary of allowed operators

    Returns:
        Any: Result of evaluating the node

    Raises:
        ValueError: If evaluation encounters forbidden operations
    """
    if isinstance(node, ast.BoolOp):
        values = [
            _eval_condition_expr(v, state, allowed_funcs, operators)
            for v in node.values
        ]
        return operators[type(node.op)](*values)
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_condition_expr(node.operand, state, allowed_funcs, operators)
        return operators[type(node.op)](operand)
    elif isinstance(node, ast.Compare):
        left = _eval_condition_expr(node.left, state, allowed_funcs, operators)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_condition_expr(comparator, state, allowed_funcs, operators)
            # Special handling for 'in' operator - swap operands
            if isinstance(op, (ast.In, ast.NotIn)):
                if not operators[type(op)](right, left):
                    return False
            else:
                if not operators[type(op)](left, right):
                    return False
            left = right
        return True
    elif isinstance(node, ast.Name):
        return state.get(node.id, False)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        if node.func.id not in allowed_funcs:
            raise ValueError(f"Function not allowed: {node.func.id}")
        args = [
            _eval_condition_expr(arg, state, allowed_funcs, operators)
            for arg in node.args
        ]
        return allowed_funcs[node.func.id](*args)
    elif isinstance(node, ast.List):
        return [
            _eval_condition_expr(elt, state, allowed_funcs, operators)
            for elt in node.elts
        ]
    elif isinstance(node, ast.Dict):
        return {
            _eval_condition_expr(
                k, state, allowed_funcs, operators
            ): _eval_condition_expr(v, state, allowed_funcs, operators)
            for k, v in zip(node.keys, node.values)
        }
    elif isinstance(node, ast.Subscript):
        value = _eval_condition_expr(node.value, state, allowed_funcs, operators)
        if isinstance(node.slice, ast.Slice):
            # Handle slice operations
            lower = (
                _eval_condition_expr(node.slice.lower, state, allowed_funcs, operators)
                if node.slice.lower is not None
                else None
            )
            upper = (
                _eval_condition_expr(node.slice.upper, state, allowed_funcs, operators)
                if node.slice.upper is not None
                else None
            )
            step = (
                _eval_condition_expr(node.slice.step, state, allowed_funcs, operators)
                if node.slice.step is not None
                else None
            )
            return value[slice(lower, upper, step)]
        elif isinstance(node.slice, ast.Index):
            idx = _eval_condition_expr(
                node.slice.value, state, allowed_funcs, operators
            )
        else:
            idx = _eval_condition_expr(node.slice, state, allowed_funcs, operators)
        return value[idx]
    elif isinstance(node, (ast.Load, ast.Store)):
        return None
    else:
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _eval_condition(condition_expr: str, state: Dict) -> bool:
    """Evaluate a condition expression against the current state

    Args:
        condition_expr: String containing the condition expression
        state: Current workflow state dictionary

    Returns:
        bool: Result of evaluating the condition

    Raises:
        ValueError: If condition evaluation fails or result cannot be converted to bool
    """
    operators = {
        ast.And: operator.and_,
        ast.Or: operator.or_,
        ast.Not: operator.not_,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: operator.contains,
        ast.NotIn: lambda x, y: not operator.contains(y, x),
    }

    allowed_funcs = {
        "len": len,
        "upper": str.upper,
        "lower": str.lower,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "all": all,
        "any": any,
        "filter": filter,
        "map": map,
        "sum": sum,
        "max": max,
        "min": min,
    }

    try:
        # First validate the expression
        _validate_condition_expr(condition_expr)

        # If validation passes, evaluate it
        expr_ast = ast.parse(condition_expr, mode="eval")
        result = _eval_condition_expr(expr_ast.body, state, allowed_funcs, operators)

        # Convert result to boolean
        try:
            return bool(result)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Condition result cannot be converted to boolean: {result}"
            )

    except Exception as e:
        # Log the error but also re-raise it
        logger.error(f"Error evaluating condition '{condition_expr}': {str(e)}")
        raise ValueError(f"Invalid condition expression: {str(e)}")


def _interpolate_variables(template: Dict, kwargs: Dict) -> Dict:
    """Recursively interpolate variables in template with values from kwargs

    Args:
        template: Template dictionary containing ${var} placeholders
        kwargs: Dictionary of variable values

    Returns:
        Dict with interpolated values
    """

    def _interpolate_value(value):
        if isinstance(value, str):
            # Handle string interpolation
            import re

            pattern = r"\${([^}]+)}"
            matches = re.finditer(pattern, value)
            result = value

            for match in matches:
                var_name = match.group(1)
                if var_name not in kwargs:
                    raise ValueError(
                        f"Required variable '{var_name}' not found in kwargs"
                    )
                # Convert kwargs[var_name] to string for replacement
                replacement = str(kwargs[var_name])
                result = result.replace(f"${{{var_name}}}", replacement)

            return result
        elif isinstance(value, dict):
            return {k: _interpolate_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_interpolate_value(v) for v in value]
        return value

    return _interpolate_value(template)


def load_workflow_from_template(
    template_path: str,
    node_registry: Optional[Dict[str, Node]] = {},
    state_registry: Optional[Dict[str, State]] = {},
    **kwargs,
) -> Workflow:
    """Create a Workflow instance from a YAML template file."""
    # Load and validate template
    template = _safe_read_template(template_path, node_registry, state_registry)
    template = _interpolate_variables(template, kwargs)
    _validate_template_structure(template)
    _validate_state_definitions(template["state_defs"], state_registry)
    _validate_nodes(template["nodes"], node_registry)

    class UserWorkflow(Workflow):
        def __init__(self_, **kwargs):
            # Store template
            self_.template = template

            # Process state definitions
            state_defs = process_state_definitions(
                template["state_defs"], state_registry
            )

            # Create kwargs dictionary prioritizing template values
            workflow_kwargs = {}
            param_mappings = {
                "name": "name",
                "llm": "llm_name",
                "vlm": "vlm_name",
                "exit_commands": "exit_commands",
                "save_artifacts": "save_artifacts",
                "debug_mode": "debug_mode",
                "max_history": "max_history",
            }

            # Process template values first, then kwargs
            for template_key, param_name in param_mappings.items():
                if template_key in template:
                    workflow_kwargs[param_name] = template[template_key]
                elif param_name in kwargs:
                    workflow_kwargs[param_name] = kwargs[param_name]

            # Set defaults for required parameters
            workflow_kwargs.setdefault("llm_name", "gpt-4o")
            workflow_kwargs.setdefault("vlm_name", None)
            workflow_kwargs.setdefault("tracing", False)  # Always default to False
            workflow_kwargs["state_defs"] = state_defs

            # Add remaining kwargs
            remaining_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in workflow_kwargs and k not in param_mappings.values()
            }
            workflow_kwargs.update(remaining_kwargs)

            # Initialize workflow
            super().__init__(**workflow_kwargs)

        def create_workflow(self_):
            """Create workflow structure from template configuration"""
            nodes_config = template["nodes"]
            interrupt_before = template.get("intervene_before", [])

            # Store original template configuration
            self_._node_configs = nodes_config.copy()

            # Initialize workflow configuration
            self_._workflow_configs = {
                "nodes": {},
                "edges": [],
                "conditionals": [],
                "entry": None,
            }

            # First pass: Create all nodes
            for node_name, node_config in nodes_config.items():
                node_type = node_config["type"]

                # Create node instance
                if node_type == "prompt":
                    node = Node(
                        prompt_template=node_config["template"],
                        node_type=node_name,
                        sink=node_config.get("sink", []),
                        sink_format=node_config.get("sink_format"),
                        image_keys=node_config.get("image_keys", []),
                    )
                else:
                    node = node_registry[node_type]

                # Determine client type
                client_type = (
                    "vlm"
                    if (
                        "image_keys" in node_config
                        or (node_type == "prompt" and node_config.get("image_keys"))
                    )
                    else "llm"
                )

                # Extract kwargs
                kwargs = {
                    k: v
                    for k, v in node_config.items()
                    if k
                    not in [
                        "type",
                        "template",
                        "next",
                        "sink",
                        "sink_format",
                        "image_keys",
                    ]
                }

                # Store workflow configuration
                self_._workflow_configs["nodes"][node_name] = {
                    "node": node,
                    "client_type": client_type,
                    "kwargs": kwargs,
                }

            # Second pass: Create edges and conditionals
            for node_name, node_config in nodes_config.items():
                next_config = node_config.get("next")
                if isinstance(next_config, str):
                    # Simple edge
                    self_._workflow_configs["edges"].append(
                        {
                            "from": node_name,
                            "to": END if next_config == "END" else next_config,
                        }
                    )
                elif isinstance(next_config, dict):
                    # Conditional edge
                    self_._workflow_configs["conditionals"].append(
                        {
                            "from": node_name,
                            "condition": next_config["condition"],
                            "then": next_config["then"],
                            "otherwise": next_config["otherwise"],
                        }
                    )

            # Set entry point
            self_._workflow_configs["entry"] = template["entry_point"]
            self_._entry_point = template["entry_point"]

            # Compile workflow
            self_.compile(
                interrupt_before=interrupt_before,
                auto_input_nodes=True,
                checkpointer=MemorySaver(serde=NumpyJsonPlusSerializer()),
            )

    return UserWorkflow(**kwargs)


def export_workflow_to_template(
    workflow: Workflow, output_path: Optional[str] = None
) -> Dict:
    """Export a workflow configuration to a YAML template."""
    assert isinstance(workflow, Workflow), "Workflow instance required"
    assert hasattr(workflow, "_node_configs"), "Workflow must have node configurations"

    # Start with required fields
    template = {
        "name": workflow.name or "unnamed_workflow",
    }

    # Export state definitions
    state_hints = get_type_hints(workflow.state_schema)
    if not state_hints:
        template["state_defs"] = [{"state": "Dict[str, Any]"}]
    else:
        template["state_defs"] = [
            {name: _type_from_str(type_hint)} for name, type_hint in state_hints.items()
        ]

    # Export nodes configuration using original node_configs
    template["nodes"] = workflow._node_configs

    # Add entry point
    template["entry_point"] = workflow._entry_point

    # Add optional fields
    if workflow.llm_client and workflow.llm_client.model_name:
        template["llm"] = workflow.llm_client.model_name

    if workflow.vlm_client and workflow.vlm_client.model_name:
        template["vlm"] = workflow.vlm_client.model_name

    if workflow.exit_commands:
        template["exit_commands"] = workflow.exit_commands

    if workflow._interrupt_before:
        template["intervene_before"] = [
            node[:-6] if node.endswith("_input") else node
            for node in workflow._interrupt_before
        ]

    # Validate template
    node_registry = {}
    for node_name, node_config in workflow._node_configs.items():
        if "type" in node_config:
            if (
                not node_config["type"] in node_registry
                and node_config["type"] != "prompt"
            ):
                node_registry[node_config["type"]] = node_config["type"]
    try:
        _validate_template_structure(template)
        _validate_nodes(template["nodes"], node_registry)
        _validate_state_definitions(template["state_defs"], {})
    except ValueError as e:
        raise ValueError(f"Generated template failed validation: {str(e)}")

    # Save to file if path provided
    if output_path:
        output_template = template.copy()
        output_template["nodes"] = {}
        for node_name, node_config in workflow._node_configs.items():
            # Create node configuration
            node_template = OrderedDict()

            # 1. Type (always first)
            node_template["type"] = node_config.get("type", "prompt")

            # 2. Template (if present)
            if "template" in node_config:
                template_text = node_config["template"]
                # Clean up template text
                template_text = template_text.replace("\\n", "\n")
                template_text = template_text.replace("\t", "\n")
                template_text = template_text.replace("\\t", "\t")
                template_text = " ".join(template_text.split())
                template_text = template_text.strip()
                node_template["template"] = template_text

            # 3. Image keys (if present)
            if "image_keys" in node_config:
                image_keys = node_config["image_keys"]
                # Use direct value for single item
                node_template["image_keys"] = (
                    image_keys[0] if len(image_keys) == 1 else image_keys
                )

            # 4. Sink (if present)
            if "sink" in node_config:
                sink = node_config["sink"]
                # Use direct value for single item
                node_template["sink"] = (
                    sink[0] if isinstance(sink, list) and len(sink) == 1 else sink
                )

            # 5. Additional kwargs (if any)
            if "kwargs" in node_config:
                for k, v in node_config["kwargs"].items():
                    if k not in ["type", "template", "next"] and v is not None:
                        if callable(v):
                            node_template[k] = f"${{{k}}}"
                        else:
                            node_template[k] = v

            # 6. Next (always last)
            next_config = node_config.get("next", "END")
            if isinstance(next_config, dict):
                node_template["next"] = {
                    "condition": next_config.get("condition", "True"),
                    "then": (
                        "END"
                        if next_config.get("then") == END
                        else next_config.get("then", "END")
                    ),
                    "otherwise": (
                        "END"
                        if next_config.get("otherwise") == END
                        else next_config.get("otherwise", "END")
                    ),
                }
            else:
                node_template["next"] = (
                    "END" if next_config == END else (next_config or "END")
                )

            # Ensure the keys are in the desired order
            ordered_node_template = OrderedDict()
            for key in ["type", "template", "sink", "image_keys", "next"]:
                if key in node_template:
                    ordered_node_template[key] = node_template[key]
            # Add any remaining keys
            for key in node_template:
                if key not in ordered_node_template:
                    ordered_node_template[key] = node_template[key]

            output_template["nodes"][node_name] = ordered_node_template

        with open(output_path, "w") as f:

            class CustomDumper(yaml.SafeDumper):
                pass

            def str_presenter(dumper, data):
                return dumper.represent_scalar("tag:yaml.org,2002:str", data)

            def list_presenter(dumper, data):
                if len(data) <= 3:
                    return dumper.represent_sequence(
                        "tag:yaml.org,2002:seq", data, flow_style=True
                    )
                return dumper.represent_sequence(
                    "tag:yaml.org,2002:seq", data, flow_style=False
                )

            def ordered_dict_presenter(dumper, data):
                return dumper.represent_mapping("tag:yaml.org,2002:map", data)

            def dict_presenter(dumper, data):
                return dumper.represent_mapping("tag:yaml.org,2002:map", data)

            CustomDumper.add_representer(str, str_presenter)
            CustomDumper.add_representer(list, list_presenter)
            CustomDumper.add_representer(OrderedDict, ordered_dict_presenter)
            CustomDumper.add_representer(dict, dict_presenter)

            yaml.dump(
                output_template,
                f,
                Dumper=CustomDumper,
                sort_keys=False,
                default_flow_style=False,
                width=80,
                indent=2,
                allow_unicode=True,
            )

    return template
