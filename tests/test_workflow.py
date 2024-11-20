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

import os, sys
import tempfile
import shutil
from pathlib import Path
import json
import yaml

try:
    from typing import get_type_hints
except ImportError:
    from typing_extensions import get_type_hints

import pytest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from nodeology.workflow import (
    Workflow,
    load_workflow_from_template,
    _validate_template_structure,
    _validate_nodes,
    _validate_condition_expr,
    _validate_state_definitions,
    _validate_node_transitions,
    _validate_prompt_node,
    _eval_condition,
    _interpolate_variables,
)
from nodeology.client import LLM_Client
from nodeology.state import State
from nodeology.prebuilt import HilpState


class TestWorkflowBase:
    """Base test class with common fixtures"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_llm_client(self, mocker):
        """Mock LLM client for testing"""
        mock_client = mocker.Mock()
        mock_client.generate.return_value = "mocked response"
        return mock_client

    @pytest.fixture
    def basic_workflow(self):
        """Basic workflow configuration fixture"""

        class TestWorkflowImpl(Workflow):
            def create_workflow(self):
                self.workflow = StateGraph(self.state_schema)
                self.workflow.add_node("start", lambda x: x)
                self.workflow.set_entry_point("start")
                self.graph = self.workflow.compile(checkpointer=MemorySaver())

        return TestWorkflowImpl(
            name="test_workflow",
            llm_name="mock",
            debug_mode=True,
        )

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        original_env = os.environ.copy()
        yield
        from nodeology.log import cleanup_logging

        cleanup_logging()  # Clean up logging handlers
        os.environ.clear()
        os.environ.update(original_env)


class TestWorkflowInitialization(TestWorkflowBase):
    """Tests for workflow initialization and basic functionality"""

    def test_basic_workflow_creation(self, basic_workflow):
        """Test basic workflow initialization"""
        assert basic_workflow.name == "test_workflow"
        assert isinstance(basic_workflow.graph, CompiledStateGraph)
        assert isinstance(basic_workflow.llm_client, LLM_Client)
        assert basic_workflow.vlm_client is None

    def test_workflow_with_custom_config(self):
        """Test workflow initialization with custom configuration"""

        class CustomState(State):
            field1: str
            field2: int

        class CustomWorkflow(Workflow):
            def create_workflow(self):
                self.workflow = StateGraph(self.state_schema)
                self.workflow.add_node("start", lambda x: x)
                self.workflow.set_entry_point("start")
                self.graph = self.workflow.compile(checkpointer=MemorySaver())

        workflow = CustomWorkflow(
            name="custom",
            state_defs=CustomState,
            llm_name="mock",
            vlm_name="mock_vlm",
            exit_commands=["quit"],
            debug_mode=True,
        )

        assert workflow.name == "custom"
        assert workflow.vlm_client is not None
        assert workflow.exit_commands == ["quit"]
        annotations = get_type_hints(workflow.state_schema)
        assert "field1" in annotations
        assert "field2" in annotations

    def test_workflow_name_generation(self):
        """Test automatic workflow name generation"""

        class UnnamedWorkflow(Workflow):
            def create_workflow(self):
                self.workflow = StateGraph(self.state_schema)
                self.workflow.add_node("start", lambda x: x)
                self.workflow.set_entry_point("start")
                self.graph = self.workflow.compile(checkpointer=MemorySaver())

        workflow = UnnamedWorkflow(debug_mode=True, save_artifacts=False)
        assert workflow.name.startswith("UnnamedWorkflow_")

    def test_invalid_workflow_creation(self):
        """Test workflow creation with invalid parameters"""
        with pytest.raises(AssertionError):

            class InvalidWorkflow(Workflow):
                def create_workflow(self):
                    pass  # No graph defined

            InvalidWorkflow(name="test", debug_mode=True)


class TestWorkflowStateManagement(TestWorkflowBase):
    """Tests for state management functionality"""

    def test_state_initialization(self, basic_workflow):
        """Test state initialization with valid values"""
        basic_workflow.initialize({"input": "test", "output": "initial"})
        assert basic_workflow.state_history[0].values["input"] == "test"
        assert basic_workflow.state_history[0].values["output"] == "initial"

    def test_state_updates_and_history(self, basic_workflow):
        """Test state updates and history management"""
        # Initialize state (creates first state)
        basic_workflow.initialize({"input": "initial"})
        basic_workflow.save_state()  # Second state

        # Update state multiple times
        basic_workflow.update_state({"input": "update1"})
        basic_workflow.save_state()  # Third state
        basic_workflow.update_state({"input": "update2"})
        basic_workflow.save_state()  # Fourth state

        # Verify history
        assert len(basic_workflow.state_history) == 4
        assert basic_workflow.state_history[0].values["input"] == "initial"
        assert basic_workflow.state_history[1].values["input"] == "initial"
        assert basic_workflow.state_history[2].values["input"] == "update1"
        assert basic_workflow.state_history[3].values["input"] == "update2"

    def test_state_history_limits(self, basic_workflow):
        """Test state history size limits"""
        basic_workflow.max_history = 2
        basic_workflow.initialize({"input": "initial"})  # First state

        for i in range(5):
            basic_workflow.update_state({"input": f"update{i}"})
            basic_workflow.save_state()  # Explicit save after each update

        assert len(basic_workflow.state_history) == 2  # Only keeps last 2 states
        assert basic_workflow.state_history[-1].values["input"] == "update4"

    def test_state_type_validation(self, basic_workflow):
        """Test state type validation"""
        with pytest.raises(TypeError):
            basic_workflow.initialize({"input": 123})  # Should be string

        with pytest.raises(TypeError):
            basic_workflow.update_state(
                {"output": ["invalid", "type"]}
            )  # Should be string

    def test_state_checkpointing(self, basic_workflow, temp_dir):
        """Test state checkpointing functionality"""
        basic_workflow.log_path = temp_dir
        basic_workflow.save_artifacts = True
        basic_workflow.debug_mode = False

        basic_workflow.initialize({"input": "test_checkpoint"})
        basic_workflow._create_checkpoint()

        checkpoint_file = Path(temp_dir) / "checkpoint.json"
        assert checkpoint_file.exists()

        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            assert checkpoint["input"] == "test_checkpoint"

    def test_state_restoration(self, basic_workflow):
        """Test state restoration functionality"""
        # Initialize and create history (creates first state)
        basic_workflow.initialize({"input": "initial"})
        basic_workflow.save_state()  # Second state
        basic_workflow.update_state({"input": "modified"})
        basic_workflow.save_state()  # Third state

        # Restore to initial state
        basic_workflow.load_state(0)
        assert basic_workflow.state_history[-1].values["input"] == "initial"

        # Test invalid state index
        with pytest.raises(ValueError, match="State file not found"):
            basic_workflow.load_state(999)


class TestWorkflowExecution(TestWorkflowBase):
    """Tests for workflow execution functionality"""

    @pytest.fixture
    def conditional_workflow(self):
        """Workflow with conditional branching"""

        class ConditionalWorkflow(Workflow):
            def create_workflow(self):
                self.workflow = StateGraph(self.state_schema)

                def check_condition(state):
                    return state

                def path_a(state):
                    return {"output": "Path A"}

                def path_b(state):
                    return {"output": "Path B"}

                self.workflow.add_node("check", check_condition)
                self.workflow.add_node("path_a", path_a)
                self.workflow.add_node("path_b", path_b)

                self.workflow.set_entry_point("check")
                self.workflow.add_conditional_edges(
                    "check",
                    lambda state: (
                        "path_a" if state["input"].startswith("a:") else "path_b"
                    ),
                    {"path_a": "path_a", "path_b": "path_b"},
                )

                self.graph = self.workflow.compile(checkpointer=MemorySaver())

        return ConditionalWorkflow(name="conditional", debug_mode=True)

    def test_basic_execution(self, basic_workflow):
        """Test basic workflow execution"""
        result = basic_workflow.run({"input": "test"})
        assert "input" in result
        assert result["input"] == "test"
        assert len(basic_workflow.state_history) == 2  # initial state + final state

    def test_conditional_execution(self, conditional_workflow):
        """Test conditional workflow execution"""
        # Test path A
        result = conditional_workflow.run({"input": "a: test"})
        assert result["output"] == "Path A"

        # Test path B
        result = conditional_workflow.run({"input": "b: test"})
        assert result["output"] == "Path B"

    def test_state_update_within_node_execution(self):
        """Test that state updates within a node are applied and recorded correctly."""

        class StateUpdateWorkflow(Workflow):
            def create_workflow(self):
                self.workflow = StateGraph(self.state_schema)

                def track_states(state):
                    state = state.copy()
                    state["output"] = f"Processed {state['input']}"
                    return state

                self.workflow.add_node("track", track_states)
                self.workflow.set_entry_point("track")
                self.graph = self.workflow.compile(checkpointer=MemorySaver())

        # Initialize the workflow
        workflow = StateUpdateWorkflow(
            name="state_update_workflow",
            llm_name="mock",
            debug_mode=True,
        )

        # Run the workflow
        result = workflow.run({"input": "test"})

        # Verify the output
        assert result["output"] == "Processed test"
        assert (
            len(workflow.state_history) == 2
        )  # initial + final state after track node

    def test_human_in_the_loop_workflow(self):
        """Test workflow execution with human-in-the-loop interactions"""

        # Define a human-in-the-loop workflow
        class HumanInLoopWorkflow(Workflow):
            def create_workflow(self):
                self.workflow = StateGraph(self.state_schema)

                # Define nodes
                def recieve_input(state):
                    pass

                def process_input(state):
                    state["output"] = f"Processed input: {state['human_input']}"
                    return state

                # Add nodes to the workflow
                self.workflow.add_node("recieve_input", recieve_input)
                self.workflow.add_node("process_input", process_input)

                # Define the interrupt point before 'process' node
                interrupt_before_list = ["recieve_input"]

                # Add edges
                self.workflow.add_edge("recieve_input", "process_input")
                self.workflow.add_edge("process_input", END)

                # Set entry point
                self.workflow.set_entry_point("recieve_input")

                # Compile the workflow with interrupt
                self.graph = self.workflow.compile(
                    checkpointer=MemorySaver(), interrupt_before=interrupt_before_list
                )

        # Initialize the workflow
        workflow = HumanInLoopWorkflow(
            name="human_in_loop_workflow",
            llm_name="mock",
            debug_mode=True,
        )

        # Mock the input function to simulate human input
        with patch("builtins.input", return_value="Test input"):
            result = workflow.run({"human_input": "initial input"})

        # Verify the output
        assert result["output"] == "Processed input: Test input"
        assert result["human_input"] == "Test input"
        assert (
            len(workflow.state_history) == 4
        )  # initial + receive_input + process_input + final

    def test_multiple_human_inputs_workflow(self):
        """Test workflow with multiple human-in-the-loop interactions"""

        # Define a workflow with multiple interrupt points
        class MultiHumanInputWorkflow(Workflow):
            def create_workflow(self):
                self.workflow = StateGraph(self.state_schema)

                # Define nodes
                def input_1(state):
                    pass

                def node_a(state):
                    state["output"] = f"Node A seen: {state['human_input']}"
                    return state

                def node_b(state):
                    state["output"] = f"Node B seen: {state['human_input']}"
                    return state

                def input_2(state):
                    pass

                def node_c(state):
                    state["output"] = f"Node C seen: {state['human_input']}"
                    return state

                # Add nodes to the workflow
                self.workflow.add_node("input_1", input_1)
                self.workflow.add_node("node_a", node_a)
                self.workflow.add_node("node_b", node_b)
                self.workflow.add_node("input_2", input_2)
                self.workflow.add_node("node_c", node_c)

                # Define interrupt points
                interrupt_before_list = ["input_1", "input_2"]

                # Add edges
                self.workflow.add_edge("input_1", "node_a")
                self.workflow.add_edge("node_a", "node_b")
                self.workflow.add_edge("node_b", "input_2")
                self.workflow.add_edge("input_2", "node_c")
                self.workflow.add_edge("node_c", END)

                # Set entry point
                self.workflow.set_entry_point("input_1")

                # Compile the workflow with interrupts
                self.graph = self.workflow.compile(
                    checkpointer=MemorySaver(), interrupt_before=interrupt_before_list
                )

        # Initialize the workflow
        workflow = MultiHumanInputWorkflow(
            name="multi_human_input_workflow",
            llm_name="mock",
            debug_mode=True,
        )

        # Mock the input function to simulate human inputs
        inputs = iter(["First input", "Second input"])
        with patch("builtins.input", lambda _: next(inputs)):
            result = workflow.run({"human_input": "start input"})

        # Verify the output
        assert result["output"] == "Node C seen: Second input"
        assert (
            len(workflow.state_history) == 8
        )  # initial + input1 + nodeA + nodeB + input2 + nodeC + final

    def test_workflow_exit(self):
        """Test workflow handles both custom exit condition and built-in exit_commands correctly"""

        class ExitWorkflow(Workflow):
            def create_workflow(self):
                self.workflow = StateGraph(self.state_schema)

                def recieve_input(state):
                    pass

                def continue_or_end(state):
                    state["output"] = "Continue or end"
                    return state

                def process_input(state):
                    state["output"] = f"Processed input: {state['human_input']}"
                    return state

                # Add nodes
                self.workflow.add_node("recieve_input", recieve_input)
                self.workflow.add_node("process_input", process_input)
                self.workflow.add_node("continue_or_end", continue_or_end)

                # Add conditional edge that either continues or ends
                self.workflow.add_edge("recieve_input", "continue_or_end")
                self.workflow.add_conditional_edges(
                    "continue_or_end",
                    lambda state: (
                        "end"
                        if state["human_input"].lower() == "magic_word"
                        else "continue"
                    ),
                    {"continue": "process_input", "end": END},
                )
                self.workflow.add_edge("process_input", END)

                # Set entry point
                self.workflow.set_entry_point("recieve_input")

                # Compile with interrupt
                self.graph = self.workflow.compile(
                    checkpointer=MemorySaver(), interrupt_before=["recieve_input"]
                )

        # Test Case 1: Exit via magic_word
        workflow1 = ExitWorkflow(
            name="exit_1",
            exit_commands=["stop workflow", "quit now"],  # Custom exit commands
            debug_mode=True,
        )

        # Add more inputs to prevent StopIteration
        with patch("builtins.input", lambda _: "magic_word"):
            result1 = workflow1.run({"human_input": "initial"})

        # Verify magic_word exit
        assert result1["output"] == "Continue or end"
        assert (
            len(workflow1.state_history) == 4
        )  # initial + receive_input + continue_or_end + final

        # Test Case 2: Exit via built-in exit command
        workflow2 = ExitWorkflow(
            name="exit_2",
            exit_commands=["stop workflow", "quit now"],
            debug_mode=True,
        )

        with patch("builtins.input", lambda _: "quit now"):
            result2 = workflow2.run({"human_input": "initial", "output": "initial"})

        # Verify exit_command exit (should exit before processing)
        assert result2["output"] == "initial"
        assert len(workflow2.state_history) == 2  # initial + exit

        # Test Case 3: Exit until end
        workflow3 = ExitWorkflow(
            name="exit_3",
            debug_mode=True,
        )

        with patch("builtins.input", lambda _: "continue"):
            result3 = workflow3.run({"human_input": "initial", "output": "initial"})

        assert result3["output"] == "Processed input: continue"
        assert (
            len(workflow3.state_history) == 5
        )  # initial + receive_input + continue_or_end + process_input + final


class TestWorkflowErrorHandling(TestWorkflowBase):
    """Tests for error handling and recovery"""

    @pytest.fixture
    def error_workflow(self):
        """Workflow that raises errors"""

        class ErrorWorkflow(Workflow):
            def create_workflow(self):
                self.workflow = StateGraph(self.state_schema)

                def error_node(state):
                    raise ValueError("Test error")

                self.workflow.add_node("error", error_node)
                self.workflow.set_entry_point("error")
                self.graph = self.workflow.compile(checkpointer=MemorySaver())

        return ErrorWorkflow(name="error_test", debug_mode=True)

    def test_error_handling_debug_mode(self, error_workflow):
        """Test error handling in debug mode"""
        with pytest.raises(ValueError, match="Test error"):
            error_workflow.run({"input": "test"})

    def test_error_handling_production_mode(self, error_workflow):
        """Test error handling in production mode"""
        error_workflow.debug_mode = False
        result = error_workflow.run({"input": "test"})
        assert "error" in result
        assert "Test error" in result["error"]

    def test_state_recovery_after_error(self, basic_workflow):
        """Test state recovery after errors"""
        basic_workflow.initialize({"input": "initial"})
        basic_workflow.save_state()

        # Simulate error during state update
        with pytest.raises(ValueError):
            basic_workflow.update_state({"invalid_field": "value"})

        # State should be restored
        assert basic_workflow.state_history[-1].values["input"] == "initial"

    def test_checkpoint_recovery(self, basic_workflow, temp_dir):
        """Test recovery from checkpoint after error"""
        basic_workflow.log_path = temp_dir
        basic_workflow.save_artifacts = True

        # Create initial state and checkpoint
        basic_workflow.initialize({"input": "checkpoint_test"})
        basic_workflow._create_checkpoint()

        # Simulate error and verify recovery
        try:
            basic_workflow.update_state({"invalid": "state"})
        except:
            basic_workflow._restore_last_valid_state()

        assert basic_workflow.state_history[-1].values["input"] == "checkpoint_test"

    def test_error_logging(self, error_workflow, temp_dir):
        """Test error logging functionality"""
        # Set log path and debug mode before any logging setup
        error_workflow.log_path = temp_dir
        error_workflow.debug_mode = False
        error_workflow.save_artifacts = True

        # Re-initialize logging with new settings
        error_workflow._setup_logging(base_dir=temp_dir)

        result = error_workflow.run({"input": "test"})

        # Check log file exists and contains error
        log_files = list(Path(temp_dir).glob("**/*.log"))
        assert len(log_files) > 0
        with open(log_files[0]) as f:
            log_content = f.read()
            assert "Test error" in log_content


class TestTemplateValidation(TestWorkflowBase):
    """Tests for template validation functionality"""

    @pytest.fixture
    def basic_template(self):
        """Basic valid template fixture"""
        return {
            "name": "test_workflow",
            "state_defs": [("input", "str"), ("output", "str")],
            "nodes": {
                "start": {
                    "type": "prompt",
                    "template": "Process: {input}",
                    "next": "end",
                },
                "end": {"type": "prompt", "template": "Final: {output}", "next": "END"},
            },
            "entry_point": "start",
        }

    def test_template_structure_validation(self, basic_template):
        """Test basic template structure validation"""
        # Test valid template
        _validate_template_structure(basic_template)

        # Test missing required fields
        for field in ["name", "state_defs", "nodes", "entry_point"]:
            invalid = basic_template.copy()
            del invalid[field]
            with pytest.raises(ValueError, match=f".*{field}.*"):
                _validate_template_structure(invalid)

    def test_node_validation(self, basic_template):
        """Test node configuration validation"""
        nodes = basic_template["nodes"]
        node_registry = {"prompt": lambda x: x}

        # Test valid nodes
        _validate_nodes(nodes, node_registry)

        # Test missing type field
        invalid_nodes = {"bad_node": {"template": "test", "next": "end"}}
        with pytest.raises(ValueError, match="missing 'type' field"):
            _validate_nodes(invalid_nodes, node_registry)

        # Test unknown node type
        invalid_nodes = {"bad_node": {"type": "unknown_type", "next": "end"}}
        with pytest.raises(ValueError, match="Unknown node type"):
            _validate_nodes(invalid_nodes, node_registry)

    def test_condition_expression_validation(self):
        """Test condition expression validation"""
        # Test valid expressions
        valid_expressions = [
            "status == 'active'",
            "count > 0",
            "len(messages) > 0",
            "all(x > 0 for x in numbers)",
            "'key' in data",
            "status in ['success', 'pending']",
        ]

        for expr in valid_expressions:
            assert _validate_condition_expr(expr) is True

        # Test invalid expressions
        invalid_expressions = [
            "import os",
            "os.system('cmd')",
            "__import__('os')",
            "globals()",
            "eval('1+1')",
            "lambda x: x",
            "def func(): pass",
        ]

        for expr in invalid_expressions:
            with pytest.raises(ValueError):
                _validate_condition_expr(expr)

    def test_prompt_node_validation(self):
        """Test prompt node configuration validation"""
        # Test valid configuration
        valid_config = {
            "type": "prompt",
            "template": "Test prompt",
            "next": "end",
            "image_keys": ["img1", "img2"],
        }
        _validate_prompt_node("test_node", valid_config)

        # Test missing template
        invalid_config = {"type": "prompt", "next": "end"}
        with pytest.raises(ValueError, match="missing 'template' field"):
            _validate_prompt_node("test_node", invalid_config)

        # Test invalid image_keys
        invalid_config = {
            "type": "prompt",
            "template": "test",
            "next": "end",
            "image_keys": "not_a_list",
        }
        with pytest.raises(ValueError, match="image_keys must be a list"):
            _validate_prompt_node("test_node", invalid_config)

    def test_validate_state_definitions(self):
        """Test state definition validation"""

        class DemoState(State):
            demo_field: str

        # Valid state definitions
        valid_states = [
            ["input", "str"],
            ["count", "int"],
            "DemoState",  # Referencing existing state
        ]
        _validate_state_definitions(valid_states, {"DemoState": DemoState})

        # Invalid state definitions
        with pytest.raises(ValueError, match="Invalid state definition format"):
            _validate_state_definitions([123], {})  # Invalid format

        with pytest.raises(ValueError, match="Unknown state type"):
            _validate_state_definitions(["UnknownState"], {})  # Unknown state

        with pytest.raises(ValueError, match="Cannot resolve state type"):
            _validate_state_definitions([["field", "invalid_type"]], {})

    def test_validate_node_transitions(self):
        """Test node transition validation"""
        # Test simple transition
        _validate_node_transitions("test_node", "next_node")

        # Test valid conditional transition
        valid_condition = {"condition": "count > 0", "then": "path_a", "else": "path_b"}
        _validate_node_transitions("test_node", valid_condition)

        # Test missing condition
        with pytest.raises(ValueError, match="missing 'condition'"):
            _validate_node_transitions("test_node", {"then": "a", "else": "b"})

        # Test missing paths
        with pytest.raises(ValueError, match="missing then/else paths"):
            _validate_node_transitions("test_node", {"condition": "x > 0"})

        # Test invalid condition
        with pytest.raises(ValueError, match="Invalid condition"):
            _validate_node_transitions(
                "test_node",
                {
                    "condition": "import os",  # Unsafe expression
                    "then": "a",
                    "else": "b",
                },
            )

    def test_eval_condition(self):
        """Test condition evaluation"""
        test_state = {
            "count": 5,
            "status": "active",
            "items": ["a", "b", "c"],
            "data": {"key": "value"},
        }

        # Test basic comparisons
        assert _eval_condition("count > 3", test_state) == "true"
        assert _eval_condition("count < 3", test_state) == "false"
        assert _eval_condition("status == 'active'", test_state) == "true"

        # Test list operations
        assert _eval_condition("len(items) == 3", test_state) == "true"
        assert _eval_condition("'a' in items", test_state) == "true"

        # Test dict operations
        assert _eval_condition("'key' in data", test_state) == "true"
        assert _eval_condition("data['key'] == 'value'", test_state) == "true"

        # Test complex conditions
        assert _eval_condition("count > 0 and status == 'active'", test_state) == "true"
        assert _eval_condition("len(items) > 5 or count < 10", test_state) == "true"

        # Test invalid expressions
        with pytest.raises(ValueError):
            _eval_condition("os.system('ls')", test_state)  # Unsafe operation

        with pytest.raises(ValueError):
            _eval_condition("lambda x: x", test_state)  # Lambda not allowed

    def test_interpolate_variables(self):
        """Test variable interpolation in templates"""
        template = {
            "name": "${workflow_name}",
            "config": {"model": "${model_name}", "temperature": 0.7},
            "nodes": {
                "start": {
                    "template": "Using ${model_name} with ${temperature}",
                    "next": "END",
                }
            },
            "list_values": ["${value1}", "${value2}"],
        }

        variables = {
            "workflow_name": "test_workflow",
            "model_name": "mock",
            "temperature": 0.5,
            "value1": "a",
            "value2": "b",
        }

        result = _interpolate_variables(template, variables)

        assert result["name"] == "test_workflow"
        assert result["config"]["model"] == "mock"
        assert result["config"]["temperature"] == 0.7  # Unchanged
        assert "mock" in result["nodes"]["start"]["template"]
        assert result["list_values"] == ["a", "b"]

        # Test missing variable
        with pytest.raises(ValueError, match="Required variable .* not found"):
            _interpolate_variables({"name": "${missing_var}"}, {})

        # Test nested interpolation
        nested = {"outer": {"inner": "${var}", "list": ["${var}"]}}
        result = _interpolate_variables(nested, {"var": "value"})
        assert result["outer"]["inner"] == "value"
        assert result["outer"]["list"][0] == "value"


class TestTemplateLoading(TestWorkflowBase):
    """Tests for template loading functionality"""

    @pytest.fixture
    def template_files(self, temp_dir):
        """Create test template files in temporary directory"""
        templates = {}

        # Basic template
        basic = {
            "name": "basic_workflow",
            "state_defs": [["input", "str"], ["output", "str"]],
            "nodes": {
                "start": {
                    "type": "prompt",
                    "template": "Input: {input}",
                    "next": "end",
                },
                "end": {
                    "type": "prompt",
                    "template": "Output: {output}",
                    "next": "END",
                },
            },
            "entry_point": "start",
        }
        basic_path = os.path.join(temp_dir, "basic.yaml")
        with open(basic_path, "w") as f:
            yaml.dump(basic, f)
        templates["basic"] = basic_path

        # Template with variables
        variable = {
            "name": "${workflow_name}",
            "llm": "${model_name}",
            "state_defs": [["input", "str"], ["output", "str"]],
            "nodes": {
                "process": {
                    "type": "prompt",
                    "template": "Using ${model_name}",
                    "next": "END",
                }
            },
            "entry_point": "process",
        }
        variable_path = os.path.join(temp_dir, "variable.yaml")
        with open(variable_path, "w") as f:
            yaml.dump(variable, f)
        templates["variable"] = variable_path

        # Complex template with intervention points and conditional branching
        complex_template = {
            "name": "${workflow_name}",
            "state_defs": ["HilpState"],
            "nodes": {
                "start": {
                    "type": "prompt",
                    "template": "Initial prompt: {human_input}",
                    "sink": "output",
                    "next": {
                        "condition": "'path_a' in output",
                        "then": "path_a",
                        "else": "path_b",
                    },
                },
                "path_a": {
                    "type": "prompt",
                    "template": "Processing path A: {human_input}",
                    "sink": "output",
                    "next": "final",
                },
                "path_b": {
                    "type": "prompt",
                    "template": "Processing path B: {human_input}",
                    "sink": "output",
                    "next": "final",
                },
                "final": {
                    "type": "prompt",
                    "template": "Final output: {human_input}",
                    "sink": "output",
                    "next": "END",
                },
            },
            "entry_point": "start",
            "intervene_before": ["start", "final"],
        }
        complex_path = os.path.join(temp_dir, "complex.yaml")
        with open(complex_path, "w") as f:
            yaml.dump(complex_template, f)
        templates["complex"] = complex_path

        # Custom components template
        custom_template = {
            "name": "custom_workflow",
            "state_defs": [["value", "int"], ["status", "str"], ["output", "str"]],
            "nodes": {
                "start": {
                    "type": "prompt",
                    "template": "Test prompt",
                    "sink": "output",
                    "next": "process",
                },
                "process": {"type": "custom_node", "next": "validate"},
                "validate": {
                    "type": "custom_validator",
                    "next": {
                        "condition": "value > 10",
                        "then": "success",
                        "else": "start",
                    },
                },
                "success": {
                    "type": "prompt",
                    "template": "Success! Final value: {value}",
                    "sink": "output",
                    "next": "END",
                },
            },
            "entry_point": "start",
        }
        custom_path = os.path.join(temp_dir, "custom.yaml")
        with open(custom_path, "w") as f:
            yaml.dump(custom_template, f)
        templates["custom"] = custom_path

        # Return both templates dict and temp_dir for cleanup
        yield templates

        # Cleanup is handled by temp_dir fixture

    def test_basic_template_loading(self, template_files, temp_dir):
        """Test loading basic template"""
        workflow = load_workflow_from_template(
            template_files["basic"],
            debug_mode=True,  # Add debug_mode to prevent log file creation
        )

        # Test basic workflow properties
        assert workflow.name == "basic_workflow"
        assert isinstance(workflow.graph, CompiledStateGraph)
        assert hasattr(workflow, "template")

        # Test template structure
        assert "nodes" in workflow.template
        assert "state_defs" in workflow.template
        assert "entry_point" in workflow.template

        # Test nodes configuration
        assert len(workflow.template["nodes"]) == 2
        assert "start" in workflow.template["nodes"]
        assert "end" in workflow.template["nodes"]

        # Test node structure
        start_node = workflow.template["nodes"]["start"]
        assert start_node["type"] == "prompt"
        assert start_node["template"] == "Input: {input}"
        assert start_node["next"] == "end"

        end_node = workflow.template["nodes"]["end"]
        assert end_node["type"] == "prompt"
        assert end_node["template"] == "Output: {output}"
        assert end_node["next"] == END

        # Test state definitions
        assert len(workflow.template["state_defs"]) == 2
        assert ["input", "str"] in workflow.template["state_defs"]
        assert ["output", "str"] in workflow.template["state_defs"]

        # Test entry point
        assert workflow.template["entry_point"] == "start"

        # Test workflow execution
        result = workflow.run({"input": "test input", "output": ""})
        assert "input" in result
        assert "output" in result
        assert result["input"] == "test input"

        # Test state schema
        annotations = get_type_hints(workflow.state_schema)
        assert "input" in annotations
        assert "output" in annotations
        assert annotations["input"] == str
        assert annotations["output"] == str

    def test_interpolate_variables(self, temp_dir):
        """Test variable interpolation when loading workflow templates"""
        # Create template with variables
        template = {
            "name": "${workflow_name}",
            "llm": "${llm}",
            "vlm": "${vlm}",
            "state_defs": [["input", "str"], ["output", "str"]],
            "nodes": {
                "start": {
                    "type": "prompt",
                    "template": "Using ${llm} and ${vlm} to process ${data_type} data",
                    "next": "END",
                }
            },
            "entry_point": "start",
            "exit_commands": ["${exit_cmd1}", "${exit_cmd2}"],
        }

        # Write template to YAML file
        template_path = os.path.join(temp_dir, "variable_template.yaml")
        with open(template_path, "w") as f:
            yaml.dump(template, f)

        # Test successful interpolation
        workflow = load_workflow_from_template(
            template_path,
            workflow_name="test_workflow",
            llm="mock",
            vlm="mock_vlm",
            data_type="video",
            exit_cmd1="stop",
            exit_cmd2="quit",
            intervene_before1="node_a",
            intervene_before2="node_b",
            debug_mode=True,  # Add debug_mode
        )

        # Verify interpolated values in loaded workflow
        assert workflow.name == "test_workflow"
        assert workflow.llm_client.model_name == "mock"
        assert workflow.vlm_client.model_name == "mock_vlm"
        assert (
            workflow.template["nodes"]["start"]["template"]
            == "Using mock and mock_vlm to process video data"
        )
        assert workflow.template["exit_commands"] == ["stop", "quit"]

    def test_complex_template_loading(self, template_files, temp_dir):
        """Test loading complex template with intervention points and conditional branching"""
        workflow = load_workflow_from_template(
            template_files["complex"],
            state_registry={"HilpState": HilpState},
            workflow_name="complex_workflow",
            llm_name="mock",
            debug_mode=True,  # Add debug_mode
        )

        # Test path A execution
        inputs = iter(["path_a:test", "final input"])
        with patch("builtins.input", lambda _: next(inputs)):
            result_a = workflow.run({"human_input": "", "output": ""})

        # Verify path A execution
        assert result_a["current_node_type"] == "final"
        assert result_a["previous_node_type"] == "path_a"
        assert (
            len(result_a["messages"]) == 5
        )  # Initial input + 2 assistant msgs + final input + final assistant msg
        assert result_a["messages"][0]["role"] == "user"
        assert result_a["messages"][0]["content"] == "path_a:test"
        assert result_a["messages"][1]["content"] == "start started."
        assert result_a["messages"][2]["content"] == "path_a started."
        assert result_a["messages"][3]["content"] == "final input"
        assert result_a["messages"][4]["content"] == "final started."

        # Update workflow for path B
        workflow.update_state({"human_input": "", "output": ""})

        # Test path B execution
        inputs = iter(["path_b:test", "final input"])
        with patch("builtins.input", lambda _: next(inputs)):
            result_b = workflow.run({"human_input": "", "output": ""})

        # Verify path B execution
        assert result_b["current_node_type"] == "final"
        assert result_b["previous_node_type"] == "path_b"
        assert (
            len(result_b["messages"]) == 10
        )  # Previous 5 msgs + new input + 2 assistant msgs + final input + final assistant msg
        assert result_b["messages"][5]["role"] == "user"
        assert result_b["messages"][5]["content"] == "path_b:test"
        assert result_b["messages"][6]["content"] == "start started."
        assert result_b["messages"][7]["content"] == "path_b started."
        assert result_b["messages"][8]["content"] == "final input"

        # Verify state history length for both paths
        assert (
            len(workflow.state_history) >= 6
        )  # Should have multiple state transitions for both paths

    def test_custom_components_loading(self, template_files, temp_dir):
        """Test loading template with custom nodes and states"""

        def custom_node(state, client, **kwargs):
            """Custom node that doubles the value"""
            state = state.copy()
            state["value"] = state["value"] * 2
            state["status"] = "processed"
            return state

        def custom_validator(state, client, **kwargs):
            """Custom validator that checks if value > 10"""
            state = state.copy()
            if state["value"] > 10:
                state["status"] = "validated"
            else:
                state["status"] = "failed_validation"
            return state

        # Create registry with custom nodes
        custom_nodes = {
            "custom_node": custom_node,
            "custom_validator": custom_validator,
        }

        # Load workflow with custom nodes
        workflow = load_workflow_from_template(
            template_files["custom"],
            node_registry={**custom_nodes},
            llm_name="mock",
            debug_mode=True,  # Add debug_mode
            save_artifacts=False,  # Add save_artifacts
            log_path=temp_dir,  # Add log_path to use temp directory
        )

        # Test case 1: Value that needs to loop (5 * 2 * 2 = 20 > 10)
        result1 = workflow.run({"value": 5, "status": "initial", "output": ""})

        # Verify final state for test case 1
        assert result1["value"] == 20
        assert result1["status"] == "validated"
        assert result1["output"] == "user: Success! Final value: 20"

        # Verify state history length for the loop case
        # Initial -> start -> process -> validate -> start -> process -> validate -> success
        assert len(workflow.state_history) >= 8

        # Test case 2: Value that passes validation directly (6 * 2 = 12 > 10)
        result2 = workflow.run({"value": 6, "status": "initial", "output": ""})

        # Verify final state for test case 2
        assert result2["value"] == 12
        assert result2["status"] == "validated"
        assert result2["output"] == "user: Success! Final value: 12"

        # Verify state history for direct pass case
        # Initial -> start -> process -> validate -> success
        assert len(workflow.state_history) >= 5
