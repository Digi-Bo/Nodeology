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

### Initial Author <2025>: Xiangyu Yin

import json
from nodeology.state import State
from nodeology.node import Node, as_node
from nodeology.workflow import Workflow

import chainlit as cl
from chainlit import Message, AskActionMessage, run_sync
from langgraph.graph import END


# 1. Define your state
class TextAnalysisState(State):
    analysis: dict  # Analysis results
    text: str  # Enhanced text
    continue_improving: bool  # Whether to continue improving


# 2. Create nodes
@as_node(sink="text")
def parse_human_input(human_input: str):
    return human_input


analyze_text = Node(
    prompt_template="""Text to analyze: {text}

Analyze the above text for:
- Clarity (1-10)
- Grammar (1-10)
- Style (1-10)
- Suggestions for improvement

Output as JSON:
{{
    "clarity_score": int,
    "grammar_score": int,
    "style_score": int,
    "suggestions": str
}}
""",
    sink="analysis",
    sink_format="json",
)


def report_analysis(state, client, **kwargs):
    analysis = json.loads(state["analysis"])
    run_sync(
        Message(
            content="Below is the analysis of the text:",
            elements=[cl.CustomElement(name="DataDisplay", props={"data": analysis})],
        ).send()
    )
    return state


analyze_text.post_process = report_analysis

improve_text = Node(
    prompt_template="""Text to improve: {text}

Analysis: {analysis}

Rewrite the text incorporating the suggestions while maintaining the original meaning.
Focus on clarity, grammar, and style improvements. Return the improved text only.""",
    sink="text",
)


def report_improvement(state, client, **kwargs):
    text_md = f"{state['text']}"
    run_sync(
        Message(
            content="Below is the improved text:", elements=[cl.Text(content=text_md)]
        ).send()
    )
    return state


improve_text.post_process = report_improvement


@as_node(sink="continue_improving")
def ask_continue_improve():
    res = run_sync(
        AskActionMessage(
            content="Would you like to further improve the text?",
            timeout=300,
            actions=[
                cl.Action(
                    name="continue",
                    payload={"value": "continue"},
                    label="Continue Improving",
                ),
                cl.Action(
                    name="finish",
                    payload={"value": "finish"},
                    label="Finish",
                ),
            ],
        ).send()
    )

    # Return the user's choice
    if res and res.get("payload").get("value") == "continue":
        return True
    else:
        return False


# 3. Create workflow
class TextEnhancementWorkflow(Workflow):
    state_schema = TextAnalysisState

    def create_workflow(self):
        # Add nodes
        self.add_node("parse_human_input", parse_human_input)
        self.add_node("analyze", analyze_text)
        self.add_node("improve", improve_text)
        self.add_node("ask_continue", ask_continue_improve)

        # Connect nodes
        self.add_flow("parse_human_input", "analyze")
        self.add_flow("analyze", "improve")
        self.add_flow("improve", "ask_continue")

        # Add conditional flow based on user's choice
        self.add_conditional_flow(
            "ask_continue",
            "continue_improving",
            "analyze",
            END,
        )

        # Set entry point
        self.set_entry("parse_human_input")

        # Compile workflow
        self.compile(
            interrupt_before=["parse_human_input"],
            interrupt_before_phrases={
                "parse_human_input": "Please enter the text to analyze."
            },
        )


# 4. Run workflow
workflow = TextEnhancementWorkflow(
    llm_name="gemini/gemini-2.0-flash", save_artifacts=True
)

if __name__ == "__main__":
    result = workflow.run(ui=True)
