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

from typing import TypedDict, List, Dict, Union

StateBaseT = Union[str, int, float, bool]

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


def _resolve_state_type(type_str: str):
    """
    Resolve string representations of types to actual Python types.
    
    Supports basic types (str, int, float, bool), List, Dict, and Union types.
    Includes caching for performance optimization.
    
    Args:
        type_str (str): String representation of the type (e.g., "str", "List[int]")
        
    Returns:
        type: Resolved Python type
        
    Raises:
        ValueError: If type string is malformed or unsupported
    """
    # Add caching to avoid repeated resolution
    if not hasattr(_resolve_state_type, "_cache"):
        _resolve_state_type._cache = {}

    if type_str in _resolve_state_type._cache:
        return _resolve_state_type._cache[type_str]

    # Add error handling for malformed type strings
    try:
        # Handle basic types
        if type_str in ("str", "int", "float", "bool"):
            return eval(type_str)
        # Handle List types
        elif type_str.startswith("List[") and type_str.endswith("]"):
            inner_type = type_str[5:-1]
            return List[_resolve_state_type(inner_type)]
        # Handle Dict types
        elif type_str.startswith("Dict[") and type_str.endswith("]"):
            starting_index = 5  # Position after 'Dict['
            inner_str = type_str[starting_index:-1]  # Extract inner content
            bracket_count = 0
            split_pos = -1
            for i, char in enumerate(inner_str):
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                elif char == "," and bracket_count == 0:
                    split_pos = i
                    break

            if split_pos == -1:
                raise ValueError(f"Invalid Dict type format: {type_str}")

            key_type_str = inner_str[:split_pos].strip()
            value_type_str = inner_str[split_pos + 1 :].strip()

            key_type = _resolve_state_type(key_type_str)
            value_type = _resolve_state_type(value_type_str)

            return Dict[key_type, value_type]
        # Handle Union types
        elif type_str.startswith("Union[") and type_str.endswith("]"):
            types = [_resolve_state_type(t.strip()) for t in type_str[6:-1].split(",")]
            return Union[tuple(types)]
        else:
            raise ValueError(f"Unknown state type: {type_str}")
    except Exception as e:
        raise ValueError(f"Failed to resolve type '{type_str}': {str(e)}")


def _process_dict_state_def(state_def: Dict) -> tuple:
    """
    Process a dictionary-format state definition.
    
    Args:
        state_def (Dict): Dictionary containing 'name' and 'type' keys
        
    Returns:
        tuple: (name, resolved_type)
        
    Raises:
        ValueError: If state definition is missing required fields
    """
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
    - Dictionary format: {'name': str, 'type': str}
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
            raise ValueError(f"Invalid state definition format: {state_def}")

    return processed_state_defs
