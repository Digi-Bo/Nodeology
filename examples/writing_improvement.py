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
