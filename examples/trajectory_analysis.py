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
import tempfile
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Dict
from langgraph.graph import END
import chainlit as cl
from chainlit import Message, AskUserMessage, AskActionMessage, run_sync
from nodeology.state import State
from nodeology.node import Node, as_node
from nodeology.workflow import Workflow
import plotly.graph_objects as go


class TrajectoryState(State):
    """State for particle trajectory analysis workflow"""

    # Parameters
    mass: float  # Particle mass (kg)
    charge: float  # Particle charge (C)
    initial_velocity: np.ndarray  # Initial velocity vector [vx, vy, vz]
    E_field: np.ndarray  # Electric field vector [Ex, Ey, Ez]
    B_field: np.ndarray  # Magnetic field vector [Bx, By, Bz]

    # Confirm parameters
    confirm_parameters: bool

    # Parameters updater
    parameters_updater_output: str

    # Calculation results
    positions: List[np.ndarray]  # Position vectors at each time point

    # Image
    trajectory_plot: str
    trajectory_plot_path: str

    # Analysis results
    analysis_result: Dict

    # Continue simulation
    continue_simulation: bool


@as_node(sink=[])
def display_parameters(
    mass: float,
    charge: float,
    initial_velocity: np.ndarray,
    E_field: np.ndarray,
    B_field: np.ndarray,
):
    # Create a dictionary of parameters for the custom element
    parameters = {
        "Mass (kg)": mass,
        "Charge (C)": charge,
        "Initial Velocity (m/s)": initial_velocity.tolist(),
        "Electric Field (N/C)": E_field.tolist(),
        "Magnetic Field (T)": B_field.tolist(),
    }

    # Use the custom element to display parameters
    run_sync(
        Message(
            content="Below are the current simulation parameters:",
            elements=[
                cl.CustomElement(
                    name="DataDisplay",
                    props={
                        "data": parameters,
                        "title": "Particle Parameters",
                        "badge": "Configured",
                        "showScrollArea": False,
                    },
                )
            ],
        ).send()
    )
    return


@as_node(sink="confirm_parameters")
def ask_confirm_parameters():
    res = run_sync(
        AskActionMessage(
            content="Are you happy with the parameters?",
            timeout=300,
            actions=[
                cl.Action(
                    name="yes",
                    payload={"value": "yes"},
                    label="Yes",
                ),
                cl.Action(
                    name="no",
                    payload={"value": "no"},
                    label="No",
                ),
            ],
        ).send()
    )
    if res and res.get("payload").get("value") == "yes":
        return True
    else:
        return False


@as_node(sink=["human_input"])
def ask_parameters_input():
    human_input = run_sync(
        AskUserMessage(
            content="Please let me know how you want to change any of the parameters :)",
            timeout=300,
        ).send()
    )["output"]
    return human_input


parameters_updater = Node(
    node_type="parameters_updater",
    prompt_template="""Update the parameters based on the user's input.

Current parameters:
mass: {mass}
charge: {charge}
initial_velocity: {initial_velocity}
E_field: {E_field}
B_field: {B_field}

User input:
{human_input}

Please return the updated parameters in JSON format.
{{
    "mass": float,
    "charge": float,
    "initial_velocity": list[float],
    "E_field": list[float],
    "B_field": list[float]
}}
    """,
    sink="parameters_updater_output",
    sink_format="json",
)


def parameters_updater_transform(state, client, **kwargs):
    params_dict = json.loads(state["parameters_updater_output"])
    state["mass"] = params_dict["mass"]
    state["charge"] = params_dict["charge"]
    state["initial_velocity"] = np.array(params_dict["initial_velocity"])
    state["E_field"] = np.array(params_dict["E_field"])
    state["B_field"] = np.array(params_dict["B_field"])
    return state


parameters_updater.post_process = parameters_updater_transform


@as_node(sink=["positions"])
def calculate_trajectory(
    mass: float,
    charge: float,
    initial_velocity: np.ndarray,
    E_field: np.ndarray,
    B_field: np.ndarray,
) -> List[np.ndarray]:
    """Calculate particle trajectory under Lorentz force with automatic time steps"""
    B_magnitude = np.linalg.norm(B_field)
    if B_magnitude == 0 or charge == 0:
        # Handle the case where B=0 or charge=0 (no magnetic force)
        cyclotron_period = 1e-6  # Arbitrary time scale
    else:
        cyclotron_frequency = np.abs(charge) * B_magnitude / mass
        cyclotron_period = 2 * np.pi / cyclotron_frequency

    # Determine total simulation time and time steps
    num_periods = 5  # Simulate over 5 cyclotron periods
    num_points_per_period = 100  # At least 100 points per period
    total_time = num_periods * cyclotron_period
    total_points = int(num_periods * num_points_per_period)
    time_points = np.linspace(0, total_time, total_points)

    def lorentz_force(t, state):
        """Compute acceleration from Lorentz force"""
        vel = state[3:]
        force = charge * (E_field + np.cross(vel, B_field))
        acc = force / mass
        return np.concatenate([vel, acc])

    # Initial state vector [x, y, z, vx, vy, vz]
    initial_position = np.array([0.0, 0.0, 0.0])
    initial_state = np.concatenate([initial_position, initial_velocity])

    # Solve equations of motion
    solution = solve_ivp(
        lorentz_force,
        (time_points[0], time_points[-1]),
        initial_state,
        t_eval=time_points,
        method="RK45",
        rtol=1e-8,
    )

    if not solution.success:
        return [np.zeros(3) for _ in range(len(time_points))]

    return [solution.y[:3, i] for i in range(len(time_points))]


@as_node(sink=["trajectory_plot", "trajectory_plot_path"])
def plot_trajectory(positions: List[np.ndarray]) -> str:
    """Plot 3D particle trajectory and save to temp file

    Returns:
        tuple: (Plotly figure object, path to saved plot image)
    """
    positions = np.array(positions)

    # Create a Plotly 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="lines",
                line=dict(width=4, color="green"),
            )
        ]
    )

    # Update layout
    fig.update_layout(
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
    )

    # Save to temp file
    temp_path = tempfile.mktemp(suffix=".png")
    fig.write_image(temp_path)

    run_sync(
        Message(
            content="Below is the trajectory plot of the particle:",
            elements=[cl.Plotly(figure=fig)],
        ).send()
    )

    return fig, temp_path


trajectory_analyzer = Node(
    node_type="trajectory_analyzer",
    prompt_template="""Analyze this particle trajectory plot.

Please determine:
1. The type of motion (linear, circular, helical, or chaotic)
2. Key physical features (radius, period, pitch angle if applicable)
3. Explanation of the motion
4. Anomalies in the motion
Output in JSON format:
{{
    "trajectory_type": "type_name",
    "key_features": {
        "feature1": value,
        "feature2": value
    },
    "explanation": "detailed explanation",
    "anomalies": "anomaly description"
}}""",
    sink="analysis_result",
    sink_format="json",
    image_keys=["trajectory_plot_path"],
)


def display_trajectory_analyzer_result(state, client, **kwargs):
    state["analysis_result"] = json.loads(state["analysis_result"])

    # Use the custom element to display analysis results
    run_sync(
        Message(
            content="Below is the analysis of the particle trajectory:",
            elements=[
                cl.CustomElement(
                    name="DataDisplay",
                    props={
                        "data": state["analysis_result"],
                        "title": "Trajectory Analysis",
                        "badge": state["analysis_result"].get(
                            "trajectory_type", "Unknown"
                        ),
                        "maxHeight": "400px",
                    },
                )
            ],
        ).send()
    )
    return state


trajectory_analyzer.post_process = display_trajectory_analyzer_result


@as_node(sink="continue_simulation")
def ask_continue_simulation():
    res = run_sync(
        AskActionMessage(
            content="Would you like to continue the simulation?",
            timeout=300,
            actions=[
                cl.Action(
                    name="continue",
                    payload={"value": "continue"},
                    label="Continue Simulation",
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


class TrajectoryWorkflow(Workflow):
    """Workflow for particle trajectory analysis"""

    def create_workflow(self):
        """Define the workflow graph structure"""
        # Add nodes
        self.add_node("display_parameters", display_parameters)
        self.add_node("ask_confirm_parameters", ask_confirm_parameters)
        self.add_node("ask_parameters_input", ask_parameters_input)
        self.add_node("update_parameters", parameters_updater)
        self.add_node("calculate_trajectory", calculate_trajectory)
        self.add_node("plot_trajectory", plot_trajectory)
        self.add_node("analyze_trajectory", trajectory_analyzer)
        self.add_node("ask_continue_simulation", ask_continue_simulation)

        self.add_flow("display_parameters", "ask_confirm_parameters")
        self.add_conditional_flow(
            "ask_confirm_parameters",
            "confirm_parameters",
            then="calculate_trajectory",
            otherwise="ask_parameters_input",
        )
        self.add_flow("ask_parameters_input", "update_parameters")
        self.add_flow("update_parameters", "display_parameters")
        self.add_flow("calculate_trajectory", "plot_trajectory")
        self.add_flow("plot_trajectory", "analyze_trajectory")
        self.add_flow("analyze_trajectory", "ask_continue_simulation")
        self.add_conditional_flow(
            "ask_continue_simulation",
            "continue_simulation",
            then="display_parameters",
            otherwise=END,
        )

        # Set entry point
        self.set_entry("display_parameters")

        # Compile workflow
        self.compile()


if __name__ == "__main__":
    workflow = TrajectoryWorkflow(
        state_defs=TrajectoryState,
        llm_name="gemini/gemini-2.0-flash",
        vlm_name="gemini/gemini-2.0-flash",
        debug_mode=False,
    )

    # # Export workflow to YAML file
    # workflow.to_yaml("particle_trajectory_analysis.yaml")

    # # Print workflow graph
    # workflow.graph.get_graph().draw_mermaid_png(
    #     output_file_path="particle_trajectory_analysis.png"
    # )

    initial_state = {
        "mass": 9.1093837015e-31,  # electron mass in kg
        "charge": -1.602176634e-19,  # electron charge in C
        "initial_velocity": np.array([1e6, 1e6, 1e6]),  # 1e6 m/s in each direction
        "E_field": np.array([5e6, 1e6, 5e6]),  # 1e6 N/C in y-direction
        "B_field": np.array(
            [0.0, 0.0, 50000.0]
        ),  # deliberately typo to be caught by validation
    }

    result = workflow.run(init_values=initial_state, ui=True)
