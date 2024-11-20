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
from typing import Dict, List
from nodeology.state import State
from nodeology.node import Node, record_messages


class DiagnosisState(State):
    image_path: str
    diagnosis: str
    diagnosis_history: List[str]
    diagnosis_questions: str
    diagnoser_knowledge: str
    example_diagnosis: Dict[str, str]


commentator = Node(
    name="commentator",
    prompt_template="""# EXAMPLE DIAGNOSIS:
{example_diagnosis_string}

# USER IMAGE:
{image_path}

# Instructions:
Generate a diagnosis for the USER IMAGE following the EXAMPLE DIAGNOSIS.
Description should be concise and to the point.""",
    sink=["diagnosis"],
    image_keys=["image_path"],
)


def commentator_pre_process(state, client, **kwargs):
    record_messages(
        state, [("assistant", "I will generate a diagnosis for the image.", "green")]
    )

    # Handle source path
    image_path = state["result_path"]
    if "result2image_convert_func" in kwargs:
        image_path = kwargs["result2image_convert_func"](image_path)
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])

    state["image_path"] = image_path
    record_messages(state, [("assistant", f"Analyzing {image_path}...", "green")])

    # Prepare example diagnosis string
    state["example_diagnosis_string"] = "\n".join(
        [f"{path}: {desc}" for path, desc in state.get("example_diagnoses", {}).items()]
    )

    return state


def commentator_post_process(state, client, **kwargs):
    record_messages(state, [("assistant", state["diagnosis"], "blue")])
    return state


commentator.pre_process = commentator_pre_process
commentator.post_process = commentator_post_process
