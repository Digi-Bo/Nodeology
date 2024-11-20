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
from typing import List, Dict, Union
import pytest
from nodeology.state import process_state_definitions, _resolve_state_type


class TestTypeResolution:
    """Tests for basic type resolution functionality"""

    def test_primitive_types(self):
        """Test resolution of primitive types"""
        assert _resolve_state_type("str") == str
        assert _resolve_state_type("int") == int
        assert _resolve_state_type("float") == float
        assert _resolve_state_type("bool") == bool

    def test_list_types(self):
        """Test resolution of List types"""
        assert _resolve_state_type("List[str]") == List[str]
        assert _resolve_state_type("List[int]") == List[int]
        assert _resolve_state_type("List[bool]") == List[bool]

    def test_dict_types(self):
        """Test resolution of Dict types"""
        assert _resolve_state_type("Dict[str, int]") == Dict[str, int]
        assert _resolve_state_type("Dict[str, List[int]]") == Dict[str, List[int]]
        assert (
            _resolve_state_type("Dict[str, Dict[str, bool]]")
            == Dict[str, Dict[str, bool]]
        )

    def test_nested_types(self):
        """Test resolution of deeply nested types"""
        complex_type = "Dict[str, List[Dict[str, Union[int, str]]]]"
        expected = Dict[str, List[Dict[str, Union[int, str]]]]
        assert _resolve_state_type(complex_type) == expected


class TestErrorHandling:
    """Tests for error handling in type resolution"""

    def test_invalid_type_names(self):
        """Test handling of invalid type names"""
        with pytest.raises(ValueError, match="Unknown state type"):
            _resolve_state_type("InvalidType")

        with pytest.raises(ValueError, match="Unknown state type"):
            _resolve_state_type("List[InvalidType]")

    def test_malformed_type_strings(self):
        """Test handling of malformed type strings"""
        invalid_types = [
            "List[str",  # Missing closing bracket
            "Dict[str]",  # Missing value type
            "Dict[str,]",  # Empty value type
            "Union[]",  # Empty union
            "List[]",  # Empty list type
            "[str]",  # Invalid format
        ]

        for invalid_type in invalid_types:
            with pytest.raises(ValueError):
                _resolve_state_type(invalid_type)

    def test_invalid_dict_formats(self):
        """Test handling of invalid dictionary formats"""
        with pytest.raises(ValueError):
            _resolve_state_type("Dict")

        with pytest.raises(ValueError):
            _resolve_state_type("Dict[str, int, bool]")


class TestStateDefinitionProcessing:
    """Tests for state definition processing"""

    def test_dict_state_definition(self):
        """Test processing of dictionary state definitions"""
        state_def = {"name": "test_field", "type": "str"}
        result = process_state_definitions([state_def], {})
        assert result == [("test_field", str)]

    def test_custom_type_processing(self):
        """Test processing with custom types in registry"""

        class CustomType:
            pass

        registry = {"CustomType": CustomType}

        # Test direct custom type reference
        assert process_state_definitions(["CustomType"], registry) == [CustomType]

        # Test custom type in dictionary definition
        state_def = {
            "name": "custom_field",
            "type": "str",
        }  # Can't use CustomType directly in type string
        result = process_state_definitions([state_def], registry)
        assert result == [("custom_field", str)]

    def test_list_state_definition(self):
        """Test processing of list format state definitions"""
        # Single list definition
        state_def = ["test_field", "List[int]"]
        result = process_state_definitions([state_def], {})
        assert result == [("test_field", List[int])]

        # Multiple list definitions
        state_defs = [["field1", "str"], ["field2", "List[int]"]]
        result = process_state_definitions(state_defs, {})
        assert result == [("field1", str), ("field2", List[int])]

    def test_process_mixed_definitions(self):
        """Test processing mixed format state definitions"""
        state_defs = [
            {"name": "field1", "type": "str"},
            ["field2", "List[int]"],
            {"name": "field3", "type": "Dict[str, bool]"},
        ]
        result = process_state_definitions(state_defs, {})
        assert result == [
            ("field1", str),
            ("field2", List[int]),
            ("field3", Dict[str, bool]),
        ]
