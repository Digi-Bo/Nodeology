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

from typing import List
from nodeology.state import State
from nodeology.node import Node, record_messages


class HilpState(State):
    begin_conversation: bool
    end_conversation: bool
    conversation: List[dict]
    conversation_summary: str


conversation_summarizer = Node(
    node_type="summarizer",
    prompt_template="""# Instructions:
Summarize the previous conversation and output a summary of key points in bullet points.
Each bullet point should be a complete sentence and contain only one key point.
Do not add new information. Do not make up information. Do not change the order of information.
For numbers, use the exact values from the conversation. Do not make up numbers.
Output MUST be bullet points ONLY, do not add explanation before or after.""",
)


def conversation_summarizer_pre_process(state, client, **kwargs):
    record_messages(
        state, [("assistant", "I will summarize the previous conversation.", "green")]
    )
    return state


def conversation_summarizer_post_process(state, client, **kwargs):
    record_messages(state, [("assistant", state["conversation_summary"], "blue")])
    state["conversation"] = []
    return state


conversation_summarizer.pre_process = conversation_summarizer_pre_process
conversation_summarizer.post_process = conversation_summarizer_post_process


survey = Node(
    node_type="survey",
    prompt_template="""# QUESTIONS:
{questions}

# Instructions:
Ask ALL questions from pre-defined QUESTIONS one by one.
Ask ONLY ONE question at a time following the pre-defined order.
YOU NEED TO ASK ALL QUESTIONS! DO NOT SKIP QUESTIONS! DO NOT CHANGE ORDER OF QUESTIONS! DO NOT REWRITE QUESTIONS!
If all questions have been asked, output exactly "COLLECT_COMPLETE".""",
    sink=None,
    use_conversation=True,
)


def survey_pre_process(state, client, **kwargs):
    if "source_questions" not in kwargs:
        raise ValueError("source_questions parameter is required")

    source_questions = kwargs["source_questions"]
    if source_questions not in state:
        raise ValueError(f"Question source '{source_questions}' not found in state")

    if len(state.get("conversation", [])) == 0:
        state["conversation"] = []
        record_messages(
            state,
            [
                ("assistant", "I'd like to ask some questions", "green"),
                (
                    "assistant",
                    'You can say "terminate pear" to terminate the workflow at any time.',
                    "yellow",
                ),
            ],
        )
        state["begin_conversation"] = True
        state["end_conversation"] = False
        return state
    return None


def survey_post_process(state, client, **kwargs):
    collector_output = state["collector_output"]

    if "COLLECT_COMPLETE" in collector_output:
        record_messages(state, [("assistant", "Thank you for your answers!", "green")])
        state["conversation"].append(
            {"role": "assistant", "content": "Thank you for your answers!"}
        )
        state["begin_conversation"] = False
        state["end_conversation"] = True
        return conversation_summarizer(state, client, **kwargs)

    record_messages(state, [("assistant", collector_output, "green")])
    state["conversation"].append({"role": "assistant", "content": collector_output})
    return state


survey.pre_process = survey_pre_process
survey.post_process = survey_post_process
