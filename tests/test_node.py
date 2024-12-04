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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import pytest
from typing import Any, Dict, List
import numpy as np
from nodeology.client import VLM_Client
from nodeology.node import (
    Node,
    as_node,
    record_messages,
    remove_markdown_blocks_formatting,
)
from nodeology.state import State
from nodeology.log import add_logging_level

add_logging_level("PRINTLOG", logging.INFO + 5)
add_logging_level("LOGONLY", logging.INFO + 1)


# Basic Node Tests
class TestBasicNodeFunctionality:
    def test_node_creation(self):
        """Test basic node creation and configuration"""
        node = Node(
            node_type="test_node",
            prompt_template="Test prompt with {key1} and {key2}",
            sink=["output"],
        )

        assert node.node_type == "test_node"
        assert "key1" in node.required_keys
        assert "key2" in node.required_keys
        assert node.sink == ["output"]

    def test_node_error_handling(self):
        """Test node error handling for missing required keys"""
        node = Node(
            node_type="test_node", prompt_template="Test {required_key}", sink="output"
        )

        state = State()
        state["messages"] = []

        with pytest.raises(ValueError, match="Required key 'required_key' not found"):
            node(state, None)

    def test_empty_sink_list(self):
        """Test node behavior with empty sink list"""
        node = Node(node_type="test_node", prompt_template="Test", sink=[])

        state = State()
        state["messages"] = []

        result_state = node(state, lambda **k: "response")
        assert result_state == state  # State should remain unchanged

    def test_state_type_tracking_chain(self):
        """Test node type tracking through multiple nodes"""
        node1 = Node(node_type="node1", prompt_template="Test", sink="output1")
        node2 = Node(node_type="node2", prompt_template="Test", sink="output2")

        state = State()
        state["messages"] = []

        state = node1(state, lambda **k: "response1")
        assert state["current_node_type"] == "node1"
        assert state["previous_node_type"] == ""

        state = node2(state, lambda **k: "response2")
        assert state["current_node_type"] == "node2"
        assert state["previous_node_type"] == "node1"

    def test_state_preservation(self):
        """Test that unrelated state data is preserved"""
        node = Node(node_type="test_node", prompt_template="Test", sink="output")

        state = State()
        state["messages"] = []
        state["preserved_key"] = "preserved_value"
        state["input"] = "test_input"

        result_state = node(state, lambda **k: "response")
        assert result_state["preserved_key"] == "preserved_value"
        assert result_state["input"] == "test_input"

    def test_state_immutability(self):
        """Test that original state is not modified"""
        node = Node(node_type="test_node", prompt_template="Test", sink="output")

        original_state = State()
        original_state["messages"] = []
        original_state["value"] = "original"

        state_copy = original_state.copy()
        result_state = node(state_copy, lambda **k: "response")

        assert original_state["value"] == "original"
        assert original_state != result_state


# Execution Tests
class TestNodeExecution:
    class MockClient:
        def __call__(self, messages, **kwargs):
            return "Test response"

    @pytest.fixture
    def basic_state(self):
        state = State()
        state["input"] = "value"
        state["messages"] = []
        return state

    def test_basic_execution(self, basic_state):
        """Test basic node execution"""
        node = Node(
            node_type="test_node", prompt_template="Test {input}", sink="output"
        )

        result_state = node(basic_state, self.MockClient())
        assert result_state["output"] == "Test response"

    def test_multiple_sinks(self, basic_state):
        """Test node with multiple output sinks"""

        class MultiResponseMock:
            def __call__(self, messages, **kwargs):
                return ["Response 1", "Response 2"]

        node = Node(
            node_type="test_node",
            prompt_template="Test {input}",
            sink=["output1", "output2"],
        )

        result_state = node(basic_state, MultiResponseMock())
        assert result_state["output1"] == "Response 1"
        assert result_state["output2"] == "Response 2"

    def test_sink_format(self, basic_state):
        """Test node with specific sink format"""

        class FormatMock:
            def __call__(self, messages, **kwargs):
                assert kwargs.get("format") == "json"
                return '{"key": "value"}'

        node = Node(
            node_type="test_node",
            prompt_template="Test {input}",
            sink="output",
            sink_format="json",
        )

        result_state = node(basic_state, FormatMock())
        assert result_state["output"] == '{"key": "value"}'

    def test_invalid_sink_format(self):
        """Test handling of invalid sink format specification"""
        node = Node(
            node_type="test_node",
            prompt_template="Test",
            sink="output",
            sink_format="invalid_format",
        )

        state = State()
        state["messages"] = []

        # The client should handle invalid format
        class FormatTestClient:
            def __call__(self, messages, **kwargs):
                assert kwargs.get("format") == "invalid_format"
                return "response"

        result_state = node(state, FormatTestClient())
        assert result_state["output"] == "response"

    def test_mismatched_sink_response(self):
        """Test handling of mismatched sink and response count"""
        node = Node(
            node_type="test_node", prompt_template="Test", sink=["output1", "output2"]
        )

        state = State()
        state["messages"] = []

        with pytest.raises(
            ValueError, match="Number of responses .* doesn't match number of sink"
        ):
            node(state, lambda **k: ["single_response"])


# Expression Evaluation Tests
class TestExpressionEvaluation:
    @pytest.fixture
    def sample_state(self):
        class SampleState(State):
            items: List[str]
            numbers: np.ndarray
            text: str
            data: Dict[str, Any]
            key: str

        state = SampleState()
        state["messages"] = []
        state["items"] = ["a", "b", "c", "d"]
        state["numbers"] = np.array([1, 2, 3, 4, 5])
        state["text"] = "Hello World"
        state["data"] = {"name": "John", "age": 30}
        state["key"] = "name"
        return state

    def test_basic_indexing(self, sample_state):
        """Test basic list indexing"""
        node = Node(
            node_type="test_node",
            prompt_template="First: {items[0]}, Last: {items[-1]}",
            sink="output",
        )

        result_state = node(sample_state, lambda **k: "response")
        assert "First: a, Last: d" in result_state["messages"][-1]["content"]

    def test_list_slicing(self, sample_state):
        """Test list slicing operations"""
        node = Node(
            node_type="test_node",
            prompt_template="Slice: {items[1:3]}, Reverse: {items[::-1]}",
            sink="output",
        )

        result_state = node(sample_state, lambda **k: "response")
        assert (
            "Slice: ['b', 'c'], Reverse: ['d', 'c', 'b', 'a']"
            in result_state["messages"][-1]["content"]
        )

    def test_string_methods(self, sample_state):
        """Test string method calls"""
        node = Node(
            node_type="test_node",
            prompt_template="""
            Upper: {text.upper()}
            Lower: {text.lower()}
            Title: {text.title()}
            Strip: {text.strip()}
            """,
            sink="output",
        )

        result_state = node(sample_state, lambda **k: "response")
        message = result_state["messages"][-1]["content"]
        assert "Upper: HELLO WORLD" in message
        assert "Lower: hello world" in message
        assert "Title: Hello World" in message

    def test_built_in_functions(self, sample_state):
        """Test allowed built-in function calls"""
        node = Node(
            node_type="test_node",
            prompt_template="""
            Length: {len(items)}
            Maximum: {max(numbers)}
            Minimum: {min(numbers)}
            Sum: {sum(numbers)}
            Absolute: {abs(-42)}
            """,
            sink="output",
        )

        result_state = node(sample_state, lambda **k: "response")
        message = result_state["messages"][-1]["content"]
        assert "Length: 4" in message
        assert "Maximum: 5" in message
        assert "Minimum: 1" in message
        assert "Sum: 15" in message
        assert "Absolute: 42" in message

    def test_dict_access(self, sample_state):
        """Test dictionary access methods"""
        node = Node(
            node_type="test_node",
            prompt_template="""
            Direct key: {data['name']}
            Variable key: {data[key]}
            """,
            sink="output",
        )

        result_state = node(sample_state, lambda **k: "response")
        message = result_state["messages"][-1]["content"]
        assert "Direct key: John" in message
        assert "Variable key: John" in message

    def test_type_conversions(self, sample_state):
        """Test type conversion functions"""
        sample_state["value"] = "42"
        node = Node(
            node_type="test_node",
            prompt_template="""
            Integer: {int(value)}
            Float: {float(value)}
            String: {str(numbers[0])}
            """,
            sink="output",
        )

        result_state = node(sample_state, lambda **k: "response")
        message = result_state["messages"][-1]["content"]
        assert "Integer: 42" in message
        assert "Float: 42.0" in message
        assert "String: 1" in message

    def test_invalid_expressions(self, sample_state):
        """Test error handling for invalid expressions"""
        # Test invalid function
        with pytest.raises(ValueError, match="Function not allowed: print"):
            node = Node(
                node_type="test_node", prompt_template="{print(text)}", sink="output"
            )
            node(sample_state, lambda **k: "response")

        # Test invalid method
        with pytest.raises(ValueError, match="String method not allowed: split"):
            node = Node(
                node_type="test_node", prompt_template="{text.split()}", sink="output"
            )
            node(sample_state, lambda **k: "response")

        # Test invalid index
        with pytest.raises(ValueError):
            node = Node(
                node_type="test_node", prompt_template="{items[10]}", sink="output"
            )
            node(sample_state, lambda **k: "response")

        # Test invalid key
        with pytest.raises(ValueError):
            node = Node(
                node_type="test_node",
                prompt_template="{data['invalid_key']}",
                sink="output",
            )
            node(sample_state, lambda **k: "response")


# Pre/Post Processing Tests
class TestPrePostProcessing:
    @pytest.fixture
    def processed_list(self):
        return []

    def test_pre_post_processing(self, processed_list):
        """Test node with pre and post processing functions"""

        def pre_process(state, client, **kwargs):
            processed_list.append("pre")
            return state

        def post_process(state, client, **kwargs):
            processed_list.append("post")
            return state

        node = Node(
            node_type="test_node",
            prompt_template="Test {input}",
            sink="output",
            pre_process=pre_process,
            post_process=post_process,
        )

        state = State()
        state["input"] = "value"
        state["messages"] = []

        node(state, lambda **k: "response")
        assert processed_list == ["pre", "post"]

    def test_none_pre_post_process(self):
        """Test node behavior when pre/post process returns None"""

        def pre_process(state, client, **kwargs):
            return None

        def post_process(state, client, **kwargs):
            return None

        node = Node(
            node_type="test_node",
            prompt_template="Test {input}",
            sink="output",
            pre_process=pre_process,
            post_process=post_process,
        )

        state = State()
        state["input"] = "value"
        state["messages"] = []

        result_state = node(state, lambda **k: "response")
        assert result_state == state


# Source Mapping Tests
class TestSourceMapping:
    @pytest.fixture
    def state_with_mapping(self):
        state = State()
        state["different_key"] = "mapped value"
        state["input_key"] = "value"
        state["messages"] = []
        return state

    def test_dict_source_mapping(self, state_with_mapping):
        """Test node with source key mapping"""
        node = Node(
            node_type="test_node", prompt_template="Test {value}", sink="output"
        )

        result_state = node(
            state_with_mapping,
            lambda **k: "Test response",
            source={"value": "different_key"},
        )
        assert result_state["output"] == "Test response"

    def test_string_source_mapping(self, state_with_mapping):
        """Test node with string source mapping"""
        node = Node(
            node_type="test_node", prompt_template="Test {source}", sink="output"
        )

        result_state = node(
            state_with_mapping, lambda **k: "response", source="input_key"
        )
        assert result_state["output"] == "response"

    def test_invalid_source_mapping(self):
        """Test handling of invalid source mapping"""
        node = Node(
            node_type="test_node", prompt_template="Test {value}", sink="output"
        )

        state = State()
        state["messages"] = []

        with pytest.raises(ValueError):
            node(state, None, source={"value": "nonexistent_key"})


# VLM Integration Tests
class TestVLMIntegration:
    class MockVLMClient(VLM_Client):
        def __init__(self):
            super().__init__()

        def process_images(self, messages, images, **kwargs):
            # Verify images are valid paths (for testing)
            assert all(isinstance(img, str) for img in images)
            return messages

        def __call__(self, messages, images=None, **kwargs):
            if images is not None:
                messages = self.process_images(messages, images)
            return "Image description response"

    def test_vlm_execution(self):
        """Test node execution with VLM client"""
        node = Node(
            node_type="test_vlm_node",
            prompt_template="Describe this image",
            sink="output",
            image_keys=["image_path"],
        )

        state = State()
        state["messages"] = []

        # Create a temporary test image file
        test_image_path = "test_image.jpg"
        with open(test_image_path, "w") as f:
            f.write("dummy image content")

        try:
            result_state = node(state, self.MockVLMClient(), image_path=test_image_path)
            assert result_state["output"] == "Image description response"
        finally:
            # Clean up the test image
            if os.path.exists(test_image_path):
                os.remove(test_image_path)

    def test_vlm_multiple_images(self):
        """Test VLM node with multiple image inputs"""
        node = Node(
            node_type="test_vlm_node",
            prompt_template="Describe these images",
            sink="output",
            image_keys=["image1", "image2"],
        )

        state = State()
        state["messages"] = []

        # Create temporary test image files
        test_images = ["test1.jpg", "test2.jpg"]
        for img_path in test_images:
            with open(img_path, "w") as f:
                f.write(f"dummy image content for {img_path}")

        try:
            result_state = node(
                state,
                self.MockVLMClient(),
                image1=test_images[0],
                image2=test_images[1],
            )
            assert result_state["output"] == "Image description response"
        finally:
            # Clean up test images
            for img_path in test_images:
                if os.path.exists(img_path):
                    os.remove(img_path)

    def test_vlm_missing_image(self):
        """Test VLM node execution without required image"""
        node = Node(
            node_type="test_vlm_node",
            prompt_template="Describe this image",
            sink="output",
            image_keys=["image_path"],
        )

        state = State()
        state["messages"] = []

        with pytest.raises(ValueError, match="At least one image key must be provided"):
            node(state, self.MockVLMClient())

    def test_vlm_invalid_image_path(self):
        """Test VLM node with invalid image path"""
        node = Node(
            node_type="test_vlm_node",
            prompt_template="Test",
            sink="output",
            image_keys=["image"],
        )

        state = State()
        state["messages"] = []

        with pytest.raises(TypeError, match="should be string"):
            node(state, self.MockVLMClient(), image=None)


# Decorator Tests
class TestDecorators:
    def test_as_node_decorator(self):
        """Test @as_node decorator functionality"""

        @as_node(["output"])
        def test_function(input_value):
            return f"Processed {input_value}"

        state = State()
        state["input_value"] = "test"
        state["messages"] = []

        result_state = test_function(state, None)
        assert result_state["output"] == "Processed test"

    def test_custom_function_defaults(self):
        """Test node with custom function having default parameters"""

        def custom_func(required_param, optional_param="default"):
            return f"{required_param}-{optional_param}"

        node = Node(
            node_type="test_node",
            prompt_template="",
            sink="output",
            custom_function=custom_func,
        )

        state = State()
        state["required_param"] = "value"
        state["messages"] = []

        result_state = node(state, None)
        assert result_state["output"] == "value-default"

    def test_invalid_custom_function_args(self):
        """Test handling of custom function with missing required arguments"""

        def custom_func(required_arg):
            return required_arg

        node = Node(
            node_type="test_node",
            prompt_template="",
            sink="output",
            custom_function=custom_func,
        )

        state = State()
        state["messages"] = []

        with pytest.raises(ValueError, match="Required key 'required_arg' not found"):
            node(state, None)

    def test_as_node_with_multiple_sinks(self):
        """Test @as_node decorator with multiple output sinks"""

        @as_node(["output1", "output2"])
        def multi_output_function(value):
            return [f"First {value}", f"Second {value}"]

        state = State()
        state["value"] = "test"
        state["messages"] = []

        result_state = multi_output_function(state, None)
        assert result_state["output1"] == "First test"
        assert result_state["output2"] == "Second test"

    def test_as_node_with_pre_post_processing(self):
        """Test @as_node decorator with pre and post processing"""
        processed = []

        def pre_process(state, client, **kwargs):
            processed.append("pre")
            return state

        def post_process(state, client, **kwargs):
            processed.append("post")
            return state

        @as_node(["output"], pre_process=pre_process, post_process=post_process)
        def test_function(value):
            processed.append("main")
            return f"Result: {value}"

        state = State()
        state["value"] = "test"
        state["messages"] = []

        result_state = test_function(state, None)
        assert result_state["output"] == "Result: test"
        assert processed == ["pre", "main", "post"]

    def test_as_node_with_multiple_parameters(self):
        """Test @as_node decorator with function having multiple parameters"""

        @as_node(["output"])
        def multi_param_function(param1, param2, param3="default"):
            return f"{param1}-{param2}-{param3}"

        state = State()
        state["param1"] = "value1"
        state["param2"] = "value2"
        state["messages"] = []

        result_state = multi_param_function(state, None)
        assert result_state["output"] == "value1-value2-default"

    def test_as_node_as_function_flag(self):
        """Test @as_node decorator with as_function flag"""

        @as_node(["output"], as_function=True)
        def test_function(value):
            return f"Processed {value}"

        assert callable(test_function)
        assert hasattr(test_function, "node_type")
        assert hasattr(test_function, "sink")
        assert test_function.node_type == "test_function"
        assert test_function.sink == ["output"]

    def test_as_node_error_handling(self):
        """Test @as_node decorator error handling for missing parameters"""

        @as_node(["output"])
        def error_function(required_param):
            return f"Value: {required_param}"

        state = State()
        state["messages"] = []

        with pytest.raises(ValueError, match="Required key 'required_param' not found"):
            error_function(state, None)


# Utility Function Tests
class TestUtilityFunctions:
    def test_record_messages(self):
        """Test message recording functionality"""
        state = State()
        state["messages"] = []

        messages = [
            ("assistant", "Test message", "green"),
            ("user", "Test response", "blue"),
        ]

        record_messages(state, messages)
        assert len(state["messages"]) == 2
        assert state["messages"][0]["role"] == "assistant"
        assert state["messages"][1]["role"] == "user"

    def test_remove_markdown_blocks(self):
        """Test markdown block removal"""
        text = "Normal text\n```python\ndef test():\n    pass\n```\nMore text"
        result = remove_markdown_blocks_formatting(text)
        assert result == "Normal text\nMore text"
