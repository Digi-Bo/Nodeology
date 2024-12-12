from nodeology.workflow import Workflow, Node
from nodeology.state import State


# 1. Define your state
class TextAnalysisState(State):
    text: str  # Input text
    analysis: dict  # Analysis results
    improved_text: str  # Enhanced text


# 2. Create nodes
analyze_text = Node(
    prompt_template="""Analyze the following text for:
- Clarity (1-10)
- Grammar (1-10)
- Style (1-10)
- Suggestions for improvement

Output as JSON:
{{
    "clarity_score": score,
    "grammar_score": score,
    "style_score": score,
    "suggestions": ["suggestion1", "suggestion2"]
}}

Text to analyze: {text}""",
    sink="analysis",
    sink_format="json",
)

improve_text = Node(
    prompt_template="""Original text: {text}

Analysis results: {analysis}

Rewrite the text incorporating the suggestions while maintaining the original meaning.
Focus on clarity, grammar, and style improvements.""",
    sink="improved_text",
)


# 3. Create workflow
class TextEnhancementWorkflow(Workflow):
    state_schema = TextAnalysisState

    def create_workflow(self):
        # Add nodes
        self.add_node("analyze", analyze_text)
        self.add_node("improve", improve_text)

        # Connect nodes
        self.add_flow("analyze", "improve")

        # Set entry point
        self.set_entry("analyze")

        # Compile workflow
        self.compile()


# 4. Run workflow
workflow = TextEnhancementWorkflow(
    llm_name="gpt-4o", save_artifacts=True  # Or your preferred model
)

initial_state = {
    "text": "AI technology have huge impact on science research but we must use it carefully and effective."
}

result = workflow.run(initial_state)

# Access results
print("Analysis:", result["analysis"])
print("\nImproved text:", result["improved_text"])
