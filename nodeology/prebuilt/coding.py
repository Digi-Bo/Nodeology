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

import subprocess
from typing import Optional, Annotated, List
from nodeology.state import State
from nodeology.node import (
    Node,
    as_node,
    record_messages,
)


class CodingState(State):
    code_example: str
    code: str
    code_explanation: str
    code_id: int
    code_path: str
    driver_path: str
    execution_success: bool
    execution_error: str
    retry_count: int
    result_path: str
    code_history: List[str]
    code_explanation_history: List[str]
    execution_history: List[str]


def _capture_result_path(terminal_output):
    return terminal_output.strip().split("\n")[-1]


def execute_command(command, use_shell=True, capture_output_func=_capture_result_path):
    process = subprocess.run(command, shell=use_shell, capture_output=True)
    terminal_output = process.stdout.decode("utf8")
    if process.returncode:
        raise Exception(process.stderr.decode("utf8"))
    else:
        return capture_output_func(terminal_output)


@as_node(
    name="code_executor",
    sink=["result_path", "execution_success", "retry_count", "execution_error"],
)
def execute_code(
    executable_path: Annotated[str, "Executable path"],
    code_path: Annotated[str, "Path to code file"],
    use_shell: Annotated[Optional[bool], "Use shell"] = True,
    capture_output_func: Annotated[
        callable, "Output capture function"
    ] = _capture_result_path,
    flags_n_options: Annotated[Optional[str], "Additional flags"] = None,
):
    command = f"{executable_path}"
    if flags_n_options:
        command += f" {flags_n_options}"
    command += f" {code_path}"

    try:
        current_result_path = execute_command(command, use_shell, capture_output_func)
        return current_result_path, True, 0, ""
    except Exception as e:
        return "", False, 1, str(e)


code_rewriter = Node(
    name="code_rewriter",
    prompt_template="""# EXAMPLE CODE:
{code_example}

# CONTEXT:
{context}

# PARAMETERS:
{params}

# Instructions:
Generate code following the EXAMPLE CODE, only replace and fill in values from PARAMETERS, keep other code and values. Do not make up values. Do not add new lines. Do not add new code.
Output must be code only. Do not add explanation before or after the code.""",
    sink=["code"],
)


def code_rewriter_pre_process(state, client, **kwargs):
    state["code_id"] = state.get("code_id", 0) + 1
    record_messages(
        state,
        [
            (
                "assistant",
                "I will generate new code based on the provided template and values.",
                "green",
            )
        ],
    )
    return state


def code_rewriter_post_process(state, client, **kwargs):
    code = state["code"]
    filepath = f"{kwargs['code_path']}/{kwargs['code_base_name']}_rewritten_{state['code_id']}{kwargs['code_ext']}"

    with open(filepath, "w") as f:
        f.write(code)

    record_messages(
        state,
        [
            ("assistant", "Here is the new code:", "green"),
            ("assistant", code, "blue"),
            ("assistant", f"The new code has been generated at {filepath}", "green"),
        ],
    )
    state["code_path"] = filepath
    return state


code_rewriter.pre_process = code_rewriter_pre_process
code_rewriter.post_process = code_rewriter_post_process


error_corrector = Node(
    name="error_corrector",
    prompt_template="""# CODE:
{code}

# ERROR MESSAGE:
{execution_error}

# Instructions:
Correct the CODE according to the ERROR MESSAGE.
Output the corrected CODE only. Do not add any explanation before or after the CODE.""",
    sink=["code"],
)


def error_corrector_pre_process(state, client, **kwargs):
    state["retry_count"] = state.get("retry_count", 0) + 1
    record_messages(state, [("assistant", f"I will correct the code.", "green")])
    return state


def error_corrector_post_process(state, client, **kwargs):
    record_messages(
        state,
        [
            ("assistant", "Here is the corrected code:", "green"),
            ("assistant", state["code"], "blue"),
        ],
    )
    state["execution_error"] = ""
    return state


error_corrector.pre_process = error_corrector_pre_process
error_corrector.post_process = error_corrector_post_process


code_tweaker = Node(
    name="code_tweaker",
    prompt_template="""# CODE:
{code}

# Instructions:
Tweak the CODE to improve the performance.
{tweak_instructions}
Output the tweaked CODE only. Do not add any explanation before or after the CODE.""",
    sink=["code"],
)


def code_tweaker_pre_process(state, client, **kwargs):
    record_messages(state, [("assistant", f"I will tweak the code.", "green")])
    return state


def code_tweaker_post_process(state, client, **kwargs):
    record_messages(
        state,
        [
            ("assistant", "Here is the tweaked code:", "green"),
            ("assistant", state["code"], "blue"),
        ],
    )
    return state


code_tweaker.pre_process = code_tweaker_pre_process
code_tweaker.post_process = code_tweaker_post_process


code_explainer = Node(
    name="code_explainer",
    prompt_template="""# CODE:
{code}

# Instructions:
Explain the CODE.
{explanation_instructions}
Do not include any code or code blocks. Do not repeat the code. Output MUST be text only and in a single paragraph.""",
    sink=["code_explanation"],
)


def code_explainer_pre_process(state, client, **kwargs):
    record_messages(state, [("assistant", f"I will explain the code.", "green")])
    return state


def code_explainer_post_process(state, client, **kwargs):
    record_messages(state, [("assistant", state["code_explanation"], "blue")])

    if "code_explanation_history" in state:
        state["code_explanation_history"] = state["code_explanation_history"][
            -state.get("history_length", 10) :
        ] + [state["code_explanation"]]
    return state


code_explainer.pre_process = code_explainer_pre_process
code_explainer.post_process = code_explainer_post_process


code_designer = Node(
    name="code_designer",
    prompt_template="""# EXAMPLE CODE:
{code_example}

# PLAN:
{plan}

# Instructions:
You are a developer with expertise in {expertise}. 
{design_objective}
Read the EXAMPLE CODE and PLAN to suggest a new CODE.
{functional_requirements}
{output_instructions}
Output code in correct syntax ONLY, do not add any other explanation before or after the code.""",
    sink=["code"],
)


def code_designer_pre_process(state, client, **kwargs):
    state["code_id"] = state.get("code_id", 0) + 1
    record_messages(
        state,
        [
            (
                "assistant",
                f"I will generate CODE based on my expertise in {kwargs['expertise']}.",
                "green",
            )
        ],
    )
    return state


def code_designer_post_process(state, client, **kwargs):
    record_messages(
        state,
        [
            ("assistant", "Here is the new code:", "green"),
            ("assistant", state["code"], "blue"),
        ],
    )

    history_key = "code_history"
    if history_key in state:
        state[history_key] = state[history_key][-state["history_length"] :] + [
            state["code"]
        ]

    return code_explainer(state, client, explanation_instructions="")


code_designer.pre_process = code_designer_pre_process
code_designer.post_process = code_designer_post_process
