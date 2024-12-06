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
from typing import Dict, List
from nodeology.state import State, StateBaseT
from nodeology.node import Node, record_messages


class ParamsOptState(State):
    data_path: str
    params_desc: str
    params: Dict[str, StateBaseT]
    quality: str
    quality_history: List[str]
    params_questions: str


class RecommendationState(State):
    recommendation: str
    recommender_knowledge: str
    example_recommendations: Dict[str, str]


# Formatter Template
formatter = Node(
    node_type="formatter",
    prompt_template="""# PARAMS DESCRIPTION: 
{params_desc}

# SOURCE:
{source}

# Instructions:
Extract parameters from SOURCE into JSON. Include ALL parameters described in PARAMS DESCRIPTION in the JSON. 
If a parameter is not mentioned in SOURCE, use the default values in the PARAMS DESCRIPTION. Do not make up values. 
Output MUST be JSON ONLY, do not add explanation before or after the JSON.""",
    sink=["params"],
    sink_format="json",
)


def formatter_pre_process(state, client, **kwargs):
    record_messages(
        state, [("assistant", f"I will extract parameters from the source.", "green")]
    )
    return state


def formatter_post_process(state, client, **kwargs):
    record_messages(
        state,
        [
            ("assistant", "Here are the current parameters:", "green"),
            ("assistant", state["params"], "blue"),
        ],
    )
    return state


formatter.pre_process = formatter_pre_process
formatter.post_process = formatter_post_process

# Recommender Template
recommender = Node(
    node_type="recommender",
    prompt_template="""# RECOMMENDER KNOWLEDGE:
{recommender_knowledge}

# SOURCE:
{source}

# Instructions:
Check SOURCE, recommend parameters according to RECOMMENDER KNOWLEDGE.""",
    sink=["recommendation"],
)


def recommender_pre_process(state, client, **kwargs):
    record_messages(
        state,
        [("assistant", f"I will recommend some parameters based on SOURCE.", "green")],
    )
    return state


def recommender_post_process(state, client, **kwargs):
    record_messages(state, [("assistant", state["recommendation"], "blue")])
    return state


recommender.pre_process = recommender_pre_process
recommender.post_process = recommender_post_process

# Updater Template
updater = Node(
    node_type="updater",
    prompt_template="""# PARAMS DESCRIPTION:
{params_desc}

# CURRENT PARAMETERS:
{params}

# SOURCE:
{source}

# Instructions:
Modify CURRENT PARAMETERS according to SOURCE.
Output an updated JSON following PARAMS DESCRIPTION. 
Output MUST be JSON ONLY, do not add explanation before or after the JSON.""",
    sink=["params"],
    sink_format="json",
)


def updater_pre_process(state, client, **kwargs):
    source_type = kwargs.get("source", "recommendation")

    if source_type == "human_input":
        if not state["begin_conversation"]:
            if state["previous_node_type"] != "updater":
                if state["previous_node_type"] != "formatter":
                    record_messages(
                        state,
                        [
                            ("assistant", "Here are the current parameters:", "green"),
                            (
                                "assistant",
                                json.dumps(state["params"], indent=2),
                                "blue",
                            ),
                        ],
                    )
                record_messages(
                    state,
                    [
                        (
                            "assistant",
                            'Let me know if you want to make any more changes. If all looks good, please say "LGTM".',
                            "green",
                        )
                    ],
                )
            record_messages(
                state,
                [
                    (
                        "assistant",
                        'You can say "terminate pear" to terminate the workflow at any time.',
                        "yellow",
                    )
                ],
            )
            state["begin_conversation"] = True
            state["end_conversation"] = False
            state["conversation"] = []
            return None  # Signal to skip LLM call

        if any(
            keyword in state["human_input"].lower()
            for keyword in ["confirm", "quit", "exit", "bye", "lgtm", "terminate"]
        ):
            record_messages(
                state, [("assistant", "Thank you for your feedback!", "green")]
            )
            state["begin_conversation"] = False
            state["end_conversation"] = True
            state["conversation"] = []
            return None  # Signal to skip LLM call
    else:
        record_messages(
            state,
            [
                (
                    "assistant",
                    f"I will update the parameters according to {source_type}.",
                    "green",
                )
            ],
        )
    return state


def updater_post_process(state, client, **kwargs):
    record_messages(
        state,
        [
            ("assistant", "Here are the updated parameters:", "green"),
            ("assistant", json.dumps(state["params"], indent=2), "blue"),
            (
                "assistant",
                'Let me know if you want to make any more changes. If all looks good, please say "LGTM".',
                "green",
            ),
        ],
    )
    return state


updater.pre_process = updater_pre_process
updater.post_process = updater_post_process
