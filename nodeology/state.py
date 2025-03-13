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

import json
import logging
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger(__name__)
from typing import TypedDict, List, Dict, Union, Any, get_origin, get_args
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import msgpack

StateBaseT = Union[str, int, float, bool, np.ndarray]

"""
State management module for nodeology.
Handles type definitions, state processing, and state registry management.
"""


class State(TypedDict):
    """
    Base state class representing the core state structure.
    Contains node information, input/output data, and message history.
    """

    current_node_type: str
    previous_node_type: str
    human_input: str
    input: str
    output: str
    messages: List[dict]


def _split_by_top_level_comma(s: str) -> List[str]:
    """Helper function to split by comma while respecting brackets"""
    parts = []
    current = []
    bracket_count = 0

    for char in s:
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
        elif char == "," and bracket_count == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)

    if current:
        parts.append("".join(current).strip())
    return parts


def _resolve_state_type(type_str: str):
    """
    Resolve string representations of types to actual Python types.
    """
    if not hasattr(_resolve_state_type, "_cache"):
        _resolve_state_type._cache = {}

    if type_str in _resolve_state_type._cache:
        return _resolve_state_type._cache[type_str]

    try:
        # Handle basic types
        if type_str in (
            "str",
            "int",
            "float",
            "bool",
            "dict",
            "list",
            "bytes",
            "tuple",
            "ndarray",
        ):
            if type_str == "ndarray":
                return np.ndarray
            return eval(type_str)

        if type_str.startswith("List[") and type_str.endswith("]"):
            inner_type = type_str[5:-1]
            return List[_resolve_state_type(inner_type)]

        elif type_str.startswith("Dict[") and type_str.endswith("]"):
            inner_str = type_str[5:-1]
            parts = _split_by_top_level_comma(inner_str)
            if len(parts) != 2:
                raise ValueError(f"Invalid Dict type format: {type_str}")

            key_type = _resolve_state_type(parts[0])
            value_type = _resolve_state_type(parts[1])
            return Dict[key_type, value_type]

        elif type_str.startswith("Union[") and type_str.endswith("]"):
            inner_str = type_str[6:-1]
            types = [
                _resolve_state_type(t) for t in _split_by_top_level_comma(inner_str)
            ]
            return Union[tuple(types)]

        else:
            raise ValueError(f"Unknown state type: {type_str}")

    except Exception as e:
        raise ValueError(f"Failed to resolve type '{type_str}': {str(e)}")


def _process_dict_state_def(state_def: Dict) -> tuple:
    """
    Process a dictionary-format state definition.

    Supports both formats:
    - {'name': 'type'} format
    - {'name': str, 'type': str} format

    Args:
        state_def (Dict): Dictionary containing state definition

    Returns:
        tuple: (name, resolved_type)

    Raises:
        ValueError: If state definition is missing required fields
    """
    if len(state_def) == 1:
        # Handle {'name': 'type'} format
        name, type_str = next(iter(state_def.items()))
    else:
        # Handle {'name': str, 'type': str} format
        name = state_def.get("name")
        type_str = state_def.get("type")
        if not name or not type_str:
            raise ValueError(f"Invalid state definition: {state_def}")

    state_type = _resolve_state_type(type_str)
    return (name, state_type)


def _process_list_state_def(state_def: List) -> List:
    """
    Process a list-format state definition.

    Supports two formats:
    1. Single definition: [name, type_str]
    2. Multiple definitions: [[name1, type_str1], [name2, type_str2], ...]

    Args:
        state_def (List): List containing state definitions

    Returns:
        List[tuple]: List of (name, resolved_type) tuples

    Raises:
        ValueError: If state definition format is invalid
    """
    if len(state_def) == 2 and isinstance(state_def[0], str):
        # Single list format [name, type_str]
        name, type_str = state_def
        state_type = _resolve_state_type(type_str)
        return [(name, state_type)]
    else:
        processed_lists = []
        for item in state_def:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"Invalid state definition item: {item}")
            name, type_str = item
            state_type = _resolve_state_type(type_str)
            processed_lists.append((name, state_type))
        return processed_lists


def process_state_definitions(state_defs: List, state_registry: dict):
    """
    Process state definitions from template format to internal format.

    Supports multiple input formats:
    - Dictionary format: {'name': 'type'} or {'name': str, 'type': str}
    - List format: [name, type_str] or [[name1, type_str1], ...]
    - String format: References to pre-defined states in state_registry

    Args:
        state_defs (List): List of state definitions in various formats
        state_registry (dict): Registry of pre-defined states

    Returns:
        List[tuple]: List of processed (name, type) pairs

    Raises:
        ValueError: If state definition format is invalid or state type is unknown
    """
    processed_state_defs = []

    for state_def in state_defs:
        if isinstance(state_def, dict):
            processed_state_defs.append(_process_dict_state_def(state_def))
        elif isinstance(state_def, list):
            processed_state_defs.extend(_process_list_state_def(state_def))
        elif isinstance(state_def, str):
            if state_def in state_registry:
                processed_state_defs.append(state_registry[state_def])
            else:
                raise ValueError(f"Unknown state type: {state_def}")
        else:
            raise ValueError(
                f"Invalid state definition format: {state_def}. Must be a string, "
                "[name, type] list, or {'name': 'type'} dictionary"
            )

    return processed_state_defs


def _type_from_str(type_obj: type) -> str:
    """
    Convert a Python type object to a string representation that _resolve_state_type can parse.
    """
    # Add handling for numpy arrays
    if type_obj is np.ndarray:
        return "ndarray"

    # Handle basic types
    if type_obj in (str, int, float, bool, dict, list, bytes, tuple):
        return type_obj.__name__

    # Get the origin type
    origin = get_origin(type_obj)
    if origin is None:
        # More explicit handling of unknown types
        logger.warning(f"Unknown type {type_obj}, defaulting to None")
        return None

    # Handle List types
    if origin is list or origin is List:
        args = get_args(type_obj)
        if not args:
            return "list"  # Default to list if no type args
        inner_type = _type_from_str(args[0])
        if inner_type is None:
            return "list"
        return f"List[{inner_type}]"

    # Handle Dict types
    if origin is dict or origin is Dict:
        args = get_args(type_obj)
        if not args or len(args) != 2:
            return "dict"  # Default if no/invalid type args
        key_type = _type_from_str(args[0])  # Recursive call for key type
        value_type = _type_from_str(args[1])  # Recursive call for value type
        if key_type is None or value_type is None:
            return "dict"
        return f"Dict[{key_type}, {value_type}]"

    # Handle Union types
    if origin is Union:
        args = get_args(type_obj)
        if not args:
            return "tuple"
        types = [_type_from_str(arg) for arg in args]
        if any(t is None for t in types):
            return "tuple"
        return f"Union[{','.join(types)}]"

    # Default case
    return "str"


class StateEncoder(json.JSONEncoder):
    """Custom JSON encoder for serializing workflow states."""

    def default(self, obj):
        try:
            if isinstance(obj, np.ndarray):
                return {
                    "__type__": "ndarray",
                    "data": obj.tolist(),
                    "dtype": str(obj.dtype),
                }
            if isinstance(obj, go.Figure):
                return {
                    "__type__": "plotly_figure",
                    "data": pio.to_json(obj),
                }
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if isinstance(obj, bytes):
                return obj.decode("utf-8")
            if isinstance(obj, set):
                return list(obj)
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return super().default(obj)
        except TypeError as e:
            logger.warning(f"Could not serialize object of type {type(obj)}: {str(e)}")
            return str(obj)


class CustomSerializer(JsonPlusSerializer):
    NDARRAY_EXT_TYPE = 42  # Ensure this code doesn't conflict with other ExtTypes
    PLOTLY_FIGURE_EXT_TYPE = 43  # New extension type for Plotly figures

    def _default(self, obj: Any) -> Union[str, Dict[str, Any]]:
        if isinstance(obj, np.ndarray):
            return {
                "lc": 2,
                "type": "ndarray",
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, go.Figure):
            return {
                "lc": 2,
                "type": "plotly_figure",
                "data": pio.to_json(obj),
            }
        return super()._default(obj)

    def _reviver(self, value: Dict[str, Any]) -> Any:
        if value.get("lc", None) == 2:
            if value.get("type", None) == "ndarray":
                return np.array(value["data"], dtype=value["dtype"])
            elif value.get("type", None) == "plotly_figure":
                return pio.from_json(value["data"])
        return super()._reviver(value)

    # Override dumps_typed to use instance method _msgpack_enc
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        if isinstance(obj, bytes):
            return "bytes", obj
        elif isinstance(obj, bytearray):
            return "bytearray", obj
        else:
            try:
                return "msgpack", self._msgpack_enc(obj)
            except UnicodeEncodeError:
                return "json", self.dumps(obj)

    # Provide instance-level _msgpack_enc
    def _msgpack_enc(self, data: Any) -> bytes:
        enc = msgpack.Packer(default=self._msgpack_default)
        return enc.pack(data)

    # Provide instance-level _msgpack_default
    def _msgpack_default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            # Prepare metadata for ndarray
            metadata = {
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
            metadata_packed = msgpack.packb(metadata, use_bin_type=True)
            data_packed = obj.tobytes()
            combined = metadata_packed + data_packed
            return msgpack.ExtType(self.NDARRAY_EXT_TYPE, combined)
        elif isinstance(obj, np.number):
            # Handle NumPy scalar types
            return obj.item()
        elif isinstance(obj, go.Figure):
            figure_json = pio.to_json(obj)
            figure_packed = msgpack.packb(figure_json, use_bin_type=True)
            return msgpack.ExtType(self.PLOTLY_FIGURE_EXT_TYPE, figure_packed)

        return super()._msgpack_default(obj)

    # Provide instance-level loads_typed
    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, data_ = data
        if type_ == "bytes":
            return data_
        elif type_ == "bytearray":
            return bytearray(data_)
        elif type_ == "json":
            return self.loads(data_)
        elif type_ == "msgpack":
            return msgpack.unpackb(
                data_, ext_hook=self._msgpack_ext_hook, strict_map_key=False
            )
        else:
            raise NotImplementedError(f"Unknown serialization type: {type_}")

    # Provide instance-level _msgpack_ext_hook
    def _msgpack_ext_hook(self, code: int, data: bytes) -> Any:
        if code == self.NDARRAY_EXT_TYPE:
            # Unpack metadata
            unpacker = msgpack.Unpacker(use_list=False, raw=False)
            unpacker.feed(data)
            metadata = unpacker.unpack()
            buffer_offset = unpacker.tell()
            array_data = data[buffer_offset:]
            array = np.frombuffer(array_data, dtype=metadata["dtype"])
            array = array.reshape(metadata["shape"])
            return array
        elif code == self.PLOTLY_FIGURE_EXT_TYPE:
            figure_json = msgpack.unpackb(
                data, strict_map_key=False, ext_hook=self._msgpack_ext_hook
            )
            return pio.from_json(figure_json)
        else:
            return super()._msgpack_ext_hook(code, data)


def convert_serialized_objects(obj):
    """
    Convert serialized objects back to their original form.
    Currently handles:
    - NumPy arrays (serialized as {"__type__": "ndarray", "data": [...], "dtype": "..."})
    - Plotly figures (serialized as {"__type__": "plotly_figure", "data": "..."})

    Args:
        obj: The object to convert, which may contain serialized objects

    Returns:
        The object with any serialized objects converted back to their original form
    """
    if isinstance(obj, dict):
        if "__type__" in obj:
            if obj["__type__"] == "ndarray":
                return np.array(obj["data"], dtype=obj["dtype"])
            elif obj["__type__"] == "plotly_figure":
                return pio.from_json(obj["data"])
        return {k: convert_serialized_objects(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_serialized_objects(item) for item in obj]
    return obj


if __name__ == "__main__":
    serializer = CustomSerializer()
    original_data = {
        "array": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
        "scalar": np.float32(7.5),
        "message": "Test serialization",
        "nested": {
            "array": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
            "scalar": np.float32(7.5),
            "list_of_arrays": [
                np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
                np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float64),
            ],
        },
    }

    # Create a more complex figure with multiple traces and customization
    fig = go.Figure()

    # Add a scatter plot with markers
    fig.add_trace(
        go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[4, 5.2, 6, 3.2, 8],
            mode="markers+lines",
            name="Series A",
            marker=dict(size=10, color="blue", symbol="circle"),
        )
    )

    # Add a bar chart
    fig.add_trace(
        go.Bar(
            x=[1, 2, 3, 4, 5], y=[2, 3, 1, 5, 3], name="Series B", marker_color="green"
        )
    )

    # Add a line plot with different style
    fig.add_trace(
        go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[7, 6, 9, 8, 7],
            mode="lines",
            name="Series C",
            line=dict(width=3, dash="dash", color="red"),
        )
    )

    # Update layout with title and axis labels
    fig.update_layout(
        title="Complex Test Figure",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        legend_title="Legend",
        template="plotly_white",
    )
    original_data["figure"] = fig

    # Serialize the data
    _, serialized = serializer.dumps_typed(original_data)
    # Deserialize the data
    deserialized_data = serializer.loads_typed(("msgpack", serialized))

    # Assertions
    assert isinstance(deserialized_data["array"], np.ndarray)
    assert np.array_equal(deserialized_data["array"], original_data["array"])
    assert isinstance(deserialized_data["scalar"], float)
    assert deserialized_data["scalar"] == float(original_data["scalar"])
    assert deserialized_data["message"] == original_data["message"]
    assert isinstance(deserialized_data["nested"]["array"], np.ndarray)
    assert np.array_equal(
        deserialized_data["nested"]["array"], original_data["nested"]["array"]
    )
    assert isinstance(deserialized_data["nested"]["scalar"], float)
    assert deserialized_data["nested"]["scalar"] == float(
        original_data["nested"]["scalar"]
    )
    assert isinstance(deserialized_data["nested"]["list_of_arrays"], list)
    assert all(
        isinstance(arr, np.ndarray)
        for arr in deserialized_data["nested"]["list_of_arrays"]
    )
    assert all(
        np.array_equal(arr, original_arr)
        for arr, original_arr in zip(
            deserialized_data["nested"]["list_of_arrays"],
            original_data["nested"]["list_of_arrays"],
        )
    )

    assert isinstance(deserialized_data["figure"], go.Figure)
    assert len(deserialized_data["figure"].data) == len(fig.data)
    for i, trace in enumerate(fig.data):
        assert deserialized_data["figure"].data[i].type == trace.type
        # Compare x and y data if they exist
        if hasattr(trace, "x") and trace.x is not None:
            assert np.array_equal(deserialized_data["figure"].data[i].x, trace.x)
        if hasattr(trace, "y") and trace.y is not None:
            assert np.array_equal(deserialized_data["figure"].data[i].y, trace.y)

    # Compare layout properties
    assert deserialized_data["figure"].layout.title.text == fig.layout.title.text

    print("Serialization and deserialization test passed.")
