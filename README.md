> [!IMPORTANT]
> This package is actively in development, and breaking changes may occur.

<div align="center">
  <img src="assets/logo.png" alt="Nodeology Logo" width="300"/>
  <h3></h3>
</div>

## 🤖 Foundation AI-Enhanced Scientific Workflow

Foundation AI has immense potential to supercharge scientific research in processing unstructured information (papers, images), enabling fuzzy logic-based automation, and providing intuitive human-computer interfaces, with some key challenges:

- Scientific research encompasses modalities beyond natural language or images, such as spectroscopic data and diffraction patterns
- Model hallucination remains a significant concern that can compromise scientific accuracy
- Models currently lack the sophistication needed for highly specialized scientific problems requiring domain-specific methods
- Traditional methods remain more appropriate for well-understood problems with established programming solutions
- Full automation is not feasible for high-stakes scientific problems requiring expert judgment

The solution lies in leveraging foundation AI's strengths while integrating them with existing scientific ecosystems and expert guidance to create enhanced and automated scientific workflows. While traditional DAG-based automation workflows have their place, state machines prove more effective for complex scientific workflows that require:

- Iterative optimization loops
- Result-based conditional branching
- Human input interruptions
- Error recovery and retry mechanisms

`nodeology` empowers researchers to rapidly develop, test, adapt, and execute foundation AI-integrated scientific workflows by leveraging `langgraph`'s state machine framework. The framework offers great potential for complex research tasks through streamlined, AI-enhanced processes, including:

- Scientific literature analysis and parameter extraction
- Hypothesis generation and programmatic validation
- Human-in-the-loop experiment planning and monitoring
- Knowledge-based parameter recommendation and result diagnosis

Through its composable node architecture, graph-based workflow orchestration, and template system, `nodeology` can open new research avenues in:

- Novel human-AI copilot interfaces for complex scientific tasks beyond chat interfaces
- Graph-based state variable feedback propagation for prompt optimization
- Foundation model performance and reliability assessment within scientific and engineering context
- AI-driven workflow planning via workflow yaml template generation

## 🧭 Design Philosophy

- **Selective AI Integration**: Seamlessly integrate with existing scientific tools, reserving AI for tasks where it excels. Not every problem requires an AI solution - traditional methods often prove more effective.

- **Simplicity at Core**: Complex workflows shouldn't demand complex code. Create nodes directly from prompts or existing functions, construct workflows via graphs or YAML templates. Avoid unnecessary abstraction and expose low-level prompts for precise control.

- **Human-Centric Design**: Scientific workflows often require expert oversight at critical decision points. Built-in support of human inputs for expert guidance, ground truth validation, error detection etc.

- **Modular Node Architecture**: Build complex processes from fundamental, reusable elements. Purpose-built components for research tasks with broad applicability, forming self-contained units that combine into sophisticated workflows.

- **Workflow Portability**: Enable workflow sharing through templates while safeguarding sensitive implementations. Choose the appropriate level of detail to share - from high-level workflow to specific prompts.

## 🚀 Getting Started

### Installation

To use the latest development version:

```bash
pip install git+https://github.com/xyin-anl/Nodeology.git
```

To use the latest release version:

```bash
pip install nodeology
```

### Environment Setup

Nodeology supports various cloud-based/local foundation models via [LiteLLM](https://docs.litellm.ai/docs/), see [provider list](https://docs.litellm.ai/docs/providers). Most of cloud-based models usage requires setting up API key. For example:

```bash
# For OpenAI models
export OPENAI_API_KEY='your-api-key'

# For Anthropic models
export ANTHROPIC_API_KEY='your-api-key'

# For Gemini models
export GEMINI_API_KEY='your-api-key'

# For Together AI hosted open weight models
export TOGETHER_API_KEY='your-api-key'
```

The field of foundation models is evolving rapidly with new and improved models emerging frequently. As of **February 2025**, we recommend the following models based on their strengths:

- **gpt-4o**: Excellent for broad general knowledge, writing tasks, and conversational interactions
- **o3-mini**: Good balance of math, coding, and reasoning capabilities at a lower price point
- **anthropic/claude-3.7**: Strong performance in general knowledge, math, science, and coding with well-constrained outputs
- **gemini/gemini-2.0-flash**: Effective for general knowledge tasks with a large context window for processing substantial information
- **together_ai/deepseek-ai/DeepSeek-R1**: Exceptional reasoning, math, science, and coding capabilities with transparent thinking processes

### Basic Usage

Here's a bare minimum example that demonstrates the core concepts of nodeology - creating a workflow that analyzes and improves text:

```python
from nodeology.workflow import Workflow, Node
from nodeology.state import State

# 1. Define your state
class TextAnalysisState(State):
    text: str                # Input text
    analysis: dict           # Analysis results
    improved_text: str       # Enhanced text

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
    sink_format="json"
)

improve_text = Node(
    prompt_template="""Original text: {text}

Analysis results: {analysis}

Rewrite the text incorporating the suggestions while maintaining the original meaning.
Focus on clarity, grammar, and style improvements.""",
    sink="improved_text"
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
    llm_name="gpt-4o",  # Or your preferred model
    save_artifacts=True,
    tracing=True  # Enable Langfuse tracing (default)
)

initial_state = {
    "text": "AI technology have huge impact on science research but we must use it carefully and effective."
}

result = workflow.run(initial_state)

# Access results
print("Analysis:", result["analysis"])
print("\nImproved text:", result["improved_text"])
```

This example demonstrates:

- Creating a custom state with type hints
- Building nodes with prompt templates
- Defining a simple workflow with two connected nodes
- Running the workflow and accessing results

#### Langfuse Tracing (Optional)

Nodeology supports [Langfuse](https://langfuse.com/) for observability and tracing of LLM/VLM calls. To use Langfuse:

1. Set up a Langfuse account and get your API keys
2. Configure Langfuse with your keys:

```bash
# Set environment variables
export LANGFUSE_PUBLIC_KEY='your-public-key'
export LANGFUSE_SECRET_KEY='your-secret-key'
export LANGFUSE_HOST='https://cloud.langfuse.com'  # Or your self-hosted URL
```

Or configure programmatically:

```python
from nodeology.client import configure_langfuse

configure_langfuse(
    public_key='your-public-key',
    secret_key='your-secret-key',
    host='https://cloud.langfuse.com'  # Optional
)
```

You can enable or disable tracing when initializing a workflow:

```python
# Create workflow with tracing disabled (default)
workflow = MyWorkflow(
    llm_name="gpt-4o",
    tracing=False  # This is the default
)

# Create workflow with tracing enabled
workflow_w_tracing = MyWorkflow(
    llm_name="gpt-4o",
    tracing=True
)
```

### Next Steps

After getting familiar with the basics:

1. Continue reading this readme to build a particle trajectory analysis workflow and learn about nodeology concepts along the way
2. Check out the [nodeology examples repository](https://github.com/xyin-anl/nodeology_examples) for nodes and workflow examples

## 💡 Concepts

To illustrate nodeology's concepts, we'll build a complex workflow for analyzing charged particle trajectories in electromagnetic fields, as illustrated in the flowchart below. This example demonstrates key features of nodelogy such as **conditional branching**, **numerical calculation integration**, **human in the loop**, **closed-loop iterations**, **vision models usage**, **structured generation** etc. The complete script is available at `workflows/particle_trajectory_analysis.py` in the [nodeology examples repository](https://github.com/xyin-anl/nodeology_examples).

<div align="center">
  <img src="https://raw.githubusercontent.com/xyin-anl/nodeology_examples/main/workflows/particle_trajectory_analysis.png" alt="Example Workflow" width="240"/>
  <h3></h3>
</div>

### 🔣 State

At its core, Nodeology uses a state machine to manage workflow execution. State specifies a list of values to be stored and updated along the workflow progression. This approach offers several advantages:

- **Consistent Data Access**: All nodes in the workflow can access the same state variables
- **History Tracking**: State changes are recorded, enabling data collection, debugging and optimization
- **Conditional Logic**: Workflow paths can branch based on state values
- **Recovery Support**: System can restore previous states if errors occur

In `nodeology`, the base `State` is a type-hinted dictionary that defines the minimum list of required values that common workflow needs:

```python
class State(TypedDict):
    """
    Base state class representing the core state structure.
    Contains node information, input/output data, and message history.
    """
    current_node_type: str     # Tracks currently executing node
    previous_node_type: str    # Records previously executed node
    human_input: str           # Stores user input when needed
    input: str                 # General input data field
    output: str                # General output data field
    messages: List[dict]       # Tracks conversation history
```

To extend the base `State` for specific custom applications, inherit from it and add problem-specific fields. Here we define ParticleState for our particle trajectory analysis example:

```python
from typing import List
import numpy as np
from nodeology.state import State

class ParticleState(State):
    """State for particle trajectory analysis workflow"""
    # Physics parameters
    initial_position: np.ndarray    # Initial position vector [x,y,z]
    initial_velocity: np.ndarray    # Initial velocity vector [vx,vy,vz]
    mass: float                     # Particle mass in kg
    charge: float                   # Particle charge in Coulombs
    E_field: np.ndarray             # Electric field vector [Ex,Ey,Ez] in N/C
    B_field: np.ndarray             # Magnetic field vector [Bx,By,Bz] in Tesla

    # Analysis results
    validation_response: str        # Parameter validation results
    positions: List[np.ndarray]     # List of position vectors over time
    velocities: List[np.ndarray]    # List of velocity vectors over time
    accelerations: List[np.ndarray] # List of acceleration vectors
    energies: List[float]           # Total energy at each timestep
    calculation_warnings: List[str] # Numerical warnings/issues
    trajectory_plot: str            # Path to trajectory plot image
    analysis_result: dict           # Trajectory analysis results
    updated_parameters: dict        # Updated parameter values
```

In the [nodeology examples repository](https://github.com/xyin-anl/nodeology_examples), some example states are available at `states.py` as starting points for users to explore and start building their own states for special scientific problems

**Notes on state usage**:

- States should contain all data needed by workflow nodes
- Update state values through dictionary operations
- Use type validtion to catch errors early
- Consider adding history tracking for important values
- Include metadata needed for workflow decisions

### 🧩 Nodes

Nodes are individual functionality units that access the state values, carry out processing and update those values. Think of nodes as the building blocks of your workflow - each one accesses certain state values and performs a specific task. To define a node, the three main pieces of information include:

1. What state variables it should access (inputs)
2. The processing logic to execute
3. What state variables it should update with results (outputs)

In `nodeology` we hope to combine foundational AI logics (usually defined by a prompt) with traditional logics (usually defined by a function). And we provide shortcut methods for both cases for you to easily define a `Node` based on your own processing logic.

#### Prompt-based `Node` for using foundation AI

In the prompt that specifies the processing logic, we can use the `{var}` syntax to specify what state variables we need to access (i.e., source state variables). During workflow execution, every time those nodes are invoked those `{var}` placeholders will be replaced by the latest state variable values. Essentially you define a prompt template with this syntax. Along with a prompt template, you also need to specify where the `Node` should put the foundation model's response using the `sink` argument.

In our particle trajectory workflow, we use prompt-based nodes for parameter validation, trajectory analysis, and parameter updates. Here's the validation node:

```python
from nodeology.node import Node

validate_params = Node(
    prompt_template="""# Current Parameters:
Mass (mass): {mass} kg
Charge (charge): {charge} C
Initial Position (initial_position): {initial_position} m
Initial Velocity (initial_velocity): {initial_velocity} m/s
E-field (E_field): {E_field} N/C
B-field (B_field): {B_field} T

# Physics Constraints:
1. Mass must be positive and typically between 1e-31 kg (electron) and 1e-27 kg (proton)
2. Charge typically between -1.6e-19 C (electron) and 1.6e-19 C (proton)
3. Velocity must be non-zero and typically < 1e8 m/s
4. Field must be physically reasonable:
    - E-field typically < 1e8 N/C for each direction
    - B-field typically < 100 T for each direction

# Instructions:
Validate current parameters against the above physical constraints carefully.
Output a JSON object with the following structure:
{{
    "validation_passed": true/false,
    "adjustments_needed": [
        {
            "parameter": "parameter_name",
            "reason": "reason for adjustment"
        },
    ]
}}""",
    sink="validation_response",
    sink_format="json",
    sink_transform=lambda x: json.loads(x),
)
```

Note that the `Node` accepts additional arguments for more complicated prompt-based node building. The `sink_format` specifies how the output should be structured as a JSON object. The `image_keys` is used when working with vision language models (VLMs) to specify the state keys to images. The `pre_process` and `post_process` allow you to specify a function that runs before/after the main LLM/VLM logic executes. Those functions can be used to validate inputs, transform data, update the state, and handle logging operations etc. The `sink_transform` is a shortcut argument for transforming model output so that you do not need to define full postprocessing functions. Last but not least, `use_conversatoin` determines the visibility of the mode, so that you can choose if a node sees previous conversation or just the current prompt. You can see some of those additonal arguments in action in the trajectory analysis node and parameter update node in the same example.

#### Function-based `Node` for calling custom functions

To transform custom Python functions into nodes that are compatible with `langgraph` and `nodeology`, we use the `@as_node` decorator. You can specify the name, sink and optionally pre_process and post_process as before. The function's signature is used to define source state variables. For example, if the function recieves `var1` and `var2` arguments, the decorator will try to fetch state variables with name `var1` and `var2` and execute the function, then put the funtion return values into sink state variables specified by the user. Note that for functions that return multiple values, we can specify multiple sinks to store each output in the appropriate state field.

The trajectory calculation node is implemented as a function since it involves numerical computation, this represents common scenarios in scientific workflows where you want to call existing simulators/models/algorithms:

```python
from nodeology.node import as_node
from scipy.integrate import solve_ivp

@as_node(sink=["positions", "velocities", "accelerations", "energies", "calculation_warnings"])
def calculate_trajectory(
    mass: float,
    charge: float,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    E_field: np.ndarray,
    B_field: np.ndarray
) -> tuple:
    """Calculate particle trajectory using equations of motion"""
    warnings = []

    def force(t, state):
        # Extract position and velocity
        pos = state[:3]
        vel = state[3:]

        # Lorentz force calculation
        F = charge * (E_field + np.cross(vel, B_field))

        # Return derivatives [velocity, acceleration]
        return np.concatenate([vel, F/mass])

    # Initial state vector
    y0 = np.concatenate([initial_position, initial_velocity])

    # Solve equations of motion
    t_span = [0, 1.0]  # 1 second trajectory
    t_eval = np.linspace(*t_span, 1000)

    try:
        sol = solve_ivp(force, t_span, y0, t_eval=t_eval, method='RK45')

        # Extract results
        positions = [sol.y[:3,i] for i in range(sol.y.shape[1])]
        velocities = [sol.y[3:,i] for i in range(sol.y.shape[1])]
        accelerations = [force(t, sol.y[:,i])[3:] for i,t in enumerate(sol.t)]

        # Calculate energy at each point
        energies = [0.5 * mass * np.dot(v,v) + charge * np.dot(E_field, p)
                   for p,v in zip(positions, velocities)]

        if not sol.success:
            warnings.append(f"Integration warning: {sol.message}")

    except Exception as e:
        warnings.append(f"Calculation error: {str(e)}")
        return [], [], [], [], [str(e)]

    return positions, velocities, accelerations, energies, warnings
```

In the example, we also used a function-based node for plotting and designed a mini terminal program to carry out pre-defined user interactions. Again, we aim to use foundation AI only when it is necessary.

#### Example research `Node`s

To help users get started and design their own nodes, we have some example research `Node`s at `nodes` directory in the [nodeology examples repository](https://github.com/xyin-anl/nodeology_examples), below is a list of them:

```python
planner,                    # Experiment planning and optimization
execute_code,               # Safe code execution with error handling
code_rewriter,              # Code adaptation and optimization
error_corrector,            # Automated error fixing
code_tweaker,               # Performance optimization
code_explainer,             # Code documentation generation
pdf2md,                     # PDF to markdown conversion
content_summarizer,         # Document summarization
attributes_extractor,       # Extract structured information
effect_analyzer,            # Analyze cause-effect relationships
questions_generator,        # Generate research questions
insights_extractor,         # Extract key insights
context_retriever,          # Knowledge base search
context_augmented_generator,# Context-aware text generation
formatter,                  # Parameter formatting and validation
recommender,                # Parameter recommendations
updater,                    # Parameter updates with validation
conversation_summarizer,    # Dialog management
survey,                     # Systematic data collection
commentator                 # Result analysis and feedback
```

**Notes on node usage**: User defined nodes are to be combined and composed into a workflow. But those nodes by themselves are also mini programs that can run on their own. To run a single node, you just need to provide:

1. `state`: `State` object containing required state variables
2. `client`: LLM or VLM client for AI operations
3. `sink`: Optional for overriding where to store results
4. `source`: Optional for overriding prompt state names
5. `user_conversation`: model visibility
6. optional `kwargs` needed for complex processing logic

### ⛓️ Workflows

With our nodes defined, let's create the complete particle trajectory workflow. `nodeology` is based on `langgraph`'s workflow framework, which is inspired by Google's large scale graph processing system `pregel`. For more information on the design and concepts of `langgraph`, please check their [documentation](https://langchain-ai.github.io/langgraph/concepts/). Every node you defined using `nodeology` is readily compatible with `langgraph`. So you can orchestrate a bunch of `nodeology` nodes from ground up using `langgraph`'s syntax following their tutorial and examples.

#### Define a custom workflow from base `Workflow` class

To streamline the process of building workflow, `nodeology` pre-defines a base `Workflow` class. After inheriting from the base `workflow` class, user just need to define a single member method `create_workflow` and the base class will handle all the other functionalities including:

- State management and validation
- Error handling and recovery
- Workflow execution logic
- Human-in-the-loop support
- Checkpointing and logging
- Template creation and loading

Here's how we define our workflow for particle trajectory analysis:

```python
from nodeology import Workflow

class ParticleTrajectoryWorkflow(Workflow):
    """Workflow for analyzing charged particle trajectories"""
    def create_workflow(self):
        # Add nodes
        self.add_node("validate_parameters", validate_params)
        self.add_node("calculate_trajectory", calculate_trajectory)
        self.add_node("plot_trajectory", plot_trajectory)
        self.add_node("analyze_trajectory", analyze_trajectory)
        self.add_node("update_parameters", update_params)

        # Add edges with conditional logic
        self.add_conditional_flow(
            "validate_parameters",
            'validation_response["validation_passed"]',
            "calculate_trajectory",
            "adjust_parameters"
        )
        self.add_flow("calculate_trajectory", "plot_trajectory")
        self.add_flow("plot_trajectory", "analyze_trajectory")
        self.add_flow("analyze_trajectory", "update_parameters")
        self.add_conditional_flow(
            "update_parameters",
            'not updated_parameters["stop_workflow"]',
            "validate_parameters",
            END
        )

        # Set entry point
        self.set_entry("validate_parameters")

        # Compile with intervention points
        self.compile(interrupt_before=["update_parameters"])
```

After defining the workflow, you can instantiate with custom configurations

```python
    workflow = TrajectoryWorkflow(
        state_defs=TrajectoryState,
        llm_name="gpt-4o",
        vlm_name="gpt-4o",
        exit_commands='stop analysis',
        save_artifacts=True,
        max_history=500
    )
```

You can first try to visualize the workflow to see if there are any structrual problems (need `pygraphviz` installed)

```python
    workflow.graph.get_graph().draw_mermaid_png(output_file_path="particle_trajectory_analysis.png")
```

Then you can run the workflow with initial solutions

```python
initial_state = {
    "mass": 9.1093837015e-31,  # electron mass
    "charge": -1.60217663e-19,  # electron charge
    "initial_position": np.array([0., 0., 0.]),
    "initial_velocity": np.array([1e6, 0., 0.]),
    "E_field": np.array([0., 1e4, 0.]),
    "B_field": np.array([0., 0., 0.1])
}
result = workflow.run(initial_state)
```

#### Export and load workflow yaml templates (experimental feature)

One of the design philosophies is that workflows should be sharable and reusable. To work towards that goal, `nodeology` provides a yaml template system that lets you:

1. Generate yaml templates from existing workflows
2. Declare workflows using yaml templates
3. Share workflow designs in a form of yaml file
4. Load workflows from yaml templates

To generate a yaml workflow template

```python
workflow.to_yaml("particle_trajectory_analysis.yaml")
```

Then you will get something like:

```yaml
name: TrajectoryWorkflow_12_04_2024_00_45_04
state_defs:
  - current_node_type: str
  - previous_node_type: str
  - human_input: str
  - input: str
  - output: str
  - messages: List[dict]
  - initial_position: ndarray
  - initial_velocity: ndarray
  - mass: float
  - charge: float
  - E_field: ndarray
  - B_field: ndarray
  - validation_response: str
  - positions: List[ndarray]
  - velocities: List[ndarray]
  - accelerations: List[ndarray]
  - energies: List[float]
  - calculation_warnings: List[str]
  - trajectory_plot: str
  - analysis_result: dict
  - updated_parameters: dict
nodes:
  validate_parameters:
    type: prompt
    template:
      '# Current Parameters: Mass (mass): {mass} kg Charge (charge): {charge}
      C Initial Position (initial_position): {initial_position} m Initial Velocity
      (initial_velocity): {initial_velocity} m/s E-field (E_field): {E_field} N/C
      B-field (B_field): {B_field} T # Physics Constraints: 1. Mass must be positive
      and typically between 1e-31 kg (electron) and 1e-27 kg (proton) 2. Charge typically
      between -1.6e-19 C (electron) and 1.6e-19 C (proton) 3. Velocity must be non-zero
      and typically < 1e8 m/s 4. Field must be physically reasonable: - E-field typically
      < 1e8 N/C for each direction - B-field typically < 100 T for each direction
      # Instructions: Validate current parameters against the above physical constraints
      carefully to see if there are any violations. Pay special attention to the field
      magnitudes. Output a JSON object with the following structure: {{ "validation_passed":
      true/false, "adjustments_needed": [ { "parameter": "parameter_name", "reason":
      "reason for adjustment" }, ] }} If all parameters are valid, set "validation_passed"
      to true and "adjustments_needed" to an empty list. Otherwise, set "validation_passed"
      to false and list each parameter that needs adjustment with the reason why.'
    sink: validation_response
    next:
      condition: validation_response["validation_passed"]
      then: calculate_trajectory
      otherwise: adjust_parameters
  adjust_parameters:
    type: adjust_parameters
    sink:
      - mass
      - charge
      - initial_position
      - initial_velocity
      - E_field
      - B_field
    next: validate_parameters
  calculate_trajectory:
    type: calculate_trajectory
    sink:
      - positions
      - velocities
      - accelerations
      - energies
      - calculation_warnings
    next: plot_trajectory
  plot_trajectory:
    type: plot_trajectory
    sink: trajectory_plot
    next: analyze_trajectory
  analyze_trajectory:
    type: prompt
    template:
      'Analyze this particle trajectory plot. Please determine: 1. The type
      of motion (linear, circular, helical, or chaotic) 2. Key physical features (radius,
      period, pitch angle if applicable) 3. Explanation of the motion 4. Anomalies
      in the motion Output in JSON format: {{ "trajectory_type": "type_name", "key_features":
      { "feature1": value, "feature2": value }, "explanation": "detailed explanation",
      "anomalies": "anomaly description" }}'
    sink: analysis_result
    image_keys: trajectory_plot
    next: update_parameters
  update_parameters:
    type: prompt
    template:
      'Current parameters: Mass: {mass} kg Charge: {charge} C Initial Position:
      {initial_position} m Initial Velocity: {initial_velocity} m/s E-field: {E_field}
      N/C B-field: {B_field} T Question: Do you want to update the parameters and
      try again? If so, let me know what you''d like to change. Answer: {human_input}
      Based on the answer, decide whether to try again or stop the workflow. Output
      a JSON with parameters and a boolean to stop the workflow or not, IMPORTANT:
      Keep existing values if not mentioned in the answer, do not make up new values:
      {{ "mass": float_value, "charge": float_value, "initial_position": [x, y, z],
      "initial_velocity": [vx, vy, vz], "E_field": [Ex, Ey, Ez], "B_field": [Bx, By,
      Bz], "stop_workflow": false/true }}'
    sink: updated_parameters
    next:
      condition: not updated_parameters["stop_workflow"]
      then: validate_parameters
      otherwise: END
entry_point: validate_parameters
llm: gpt-4o
vlm: gpt-4o
exit_commands: [stop workflow, quit workflow, terminate workflow]
intervene_before: [update_parameters]
```

To reload and run the template:

```python
from nodeology import load_workflow_from_template

workflow = load_workflow_from_template(
    "particle_trajectory.yaml",
    custom_nodes={
        "calculate_trajectory": calculate_trajectory,
        "adjust_parameters": adjust_parameters,
        "plot_trajectory": plot_trajectory}
)
result = workflow.run(initial_state)
```

**Notes on workflow usage**

- Functions need runtime implementation for security consideration, this includes function-based nodes, pre and post processes
- `Nodeology` serializes `np.ndarray` specially, so if your node works with numpy array, it may not work with pure `langgraph` directly
- Currently supports basic workflows, `langgraph`'s advanced patterns like subgraphs not yet supported
- It is advised to review the template carefully before loading for security reasons

## 🔬 Featured Applications

- [PEAR: Ptychography automation framework](https://github.com/xyin-anl/nodeology_examples/blob/main/workflows/ptycho_params_opt.py)
- [AutoScriptCopilot: TEM experiment control](https://github.com/xyin-anl/AutoScriptCopilot)

## 👥 Contributing & Collaboration

We welcome comments, feedbacks, bugs report, code contributions and research collaborations. Please refer to CONTRIBUTING.md

If you find `nodeology` useful and may inspire your research, please use the **Cite this repository** function
