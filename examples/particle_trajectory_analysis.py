import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from typing import List, Dict, Tuple
from nodeology.state import State
from nodeology.node import Node, as_node
from nodeology.workflow import Workflow
from langgraph.graph import END
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Define the state class
class TrajectoryState(State):
    """State for particle trajectory analysis workflow"""

    # Input parameters
    initial_position: np.ndarray  # Initial position vector [x, y, z]
    initial_velocity: np.ndarray  # Initial velocity vector [vx, vy, vz]
    mass: float  # Particle mass (kg)
    charge: float  # Particle charge (C)
    E_field: np.ndarray  # Electric field vector [Ex, Ey, Ez]
    B_field: np.ndarray  # Magnetic field vector [Bx, By, Bz]

    # Validation feedback
    validation_response: str

    # Calculation results
    positions: List[np.ndarray]  # Position vectors at each time point
    velocities: List[np.ndarray]  # Velocity vectors at each time point
    accelerations: List[np.ndarray]  # Acceleration vectors at each time point
    energies: List[float]  # Total energy at each time point
    calculation_warnings: List[str]  # Numerical calculation warnings

    # Image path
    trajectory_plot: str

    # Analysis results
    analysis_result: Dict  # Analysis result

    # Updated parameters
    updated_parameters: Dict


# Define the nodes
parameter_validator = Node(
    node_type="parameter_validator",
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
Validate current parameters against the above physical constraints carefully to see if there are any violations.
Pay special attention to the field magnitudes.
Output a JSON object with the following structure:
{{
    "validation_passed": true/false,
    "adjustments_needed": [
        {
            "parameter": "parameter_name",
            "reason": "reason for adjustment"
        },
    ]
}}
If all parameters are valid, set "validation_passed" to true and "adjustments_needed" to an empty list.
Otherwise, set "validation_passed" to false and list each parameter that needs adjustment with the reason why.
""",
    sink="validation_response",
    sink_format="json",
    sink_transform=lambda x: json.loads(x),
)


@as_node(
    sink=[
        "mass",
        "charge",
        "initial_position",
        "initial_velocity",
        "E_field",
        "B_field",
    ]
)
def parameter_adjuster(
    validation_response: str,
    mass: float,
    charge: float,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    E_field: np.ndarray,
    B_field: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adjust parameters based on validation feedback and user input"""
    adjustments = validation_response["adjustments_needed"]

    # If no adjustments needed (shouldn't happen due to conditional flow)
    if not adjustments:
        raise ValueError("Parameter adjuster called with no adjustments needed")

    # Dictionary to store updated parameters
    updated_params = {
        "mass": mass,
        "charge": charge,
        "initial_position": initial_position,
        "initial_velocity": initial_velocity,
        "E_field": E_field,
        "B_field": B_field,
    }

    for adj in adjustments:
        param_name = adj["parameter"]
        print(f"{param_name} needs adjustment: {adj['reason']}")
        if param_name == "mass":
            print(f"Current value: {mass} kg")
        elif param_name == "charge":
            print(f"Current value: {charge} C")
        elif param_name == "initial_position":
            print(f"Current value: {initial_position} m")
        elif param_name == "initial_velocity":
            print(f"Current value: {initial_velocity} m/s")
        elif param_name == "E_field":
            print(f"Current value: {E_field} N/C")
        elif param_name == "B_field":
            print(f"Current value: {B_field} T")

        if param_name in ["mass", "charge"]:
            # Handle scalar values
            while True:
                try:
                    value = float(input(f"Enter new value for {param_name}: "))
                    updated_params[param_name] = value
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")

        elif param_name in [
            "initial_position",
            "initial_velocity",
            "E_field",
            "B_field",
        ]:
            # Handle vector inputs
            while True:
                try:
                    print(f"Enter space-separated values for {param_name} (x y z):")
                    values = list(map(float, input().split()))
                    if len(values) != 3:
                        raise ValueError("Must provide exactly 3 values")
                    updated_params[param_name] = np.array(values)
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}. Please try again.")

    return (
        updated_params["mass"],
        updated_params["charge"],
        updated_params["initial_position"],
        updated_params["initial_velocity"],
        updated_params["E_field"],
        updated_params["B_field"],
    )


@as_node(
    sink=[
        "positions",
        "velocities",
        "accelerations",
        "energies",
        "calculation_warnings",
    ],
)
def calculate_trajectory(
    mass: float,
    charge: float,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    E_field: np.ndarray,
    B_field: np.ndarray,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[float],
    List[str],
]:
    """Calculate particle trajectory under Lorentz force with automatic time steps"""
    warnings = []

    # Calculate cyclotron frequency and period
    B_magnitude = np.linalg.norm(B_field)
    if B_magnitude == 0 or charge == 0:
        # Handle the case where B=0 or charge=0 (no magnetic force)
        cyclotron_period = 1e-6  # Arbitrary time scale
        warnings.append(
            "Cyclotron period is set arbitrarily due to zero B-field or charge."
        )
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
        pos = state[:3]
        vel = state[3:]
        force = charge * (E_field + np.cross(vel, B_field))
        acc = force / mass
        return np.concatenate([vel, acc])

    # Initial state vector [x, y, z, vx, vy, vz]
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
        warnings.append(f"Integration warning: {solution.message}")

    # Extract results
    positions = [solution.y[:3, i] for i in range(len(time_points))]
    velocities = [solution.y[3:, i] for i in range(len(time_points))]

    # Calculate accelerations
    accelerations = []
    for i in range(len(time_points)):
        vel = velocities[i]
        acc = charge * (E_field + np.cross(vel, B_field)) / mass
        accelerations.append(acc)

    # Calculate total energy at each point
    energies = [0.5 * mass * np.dot(vel, vel) for vel in velocities]

    return positions, velocities, accelerations, energies, warnings


@as_node(sink=["trajectory_plot"])
def plot_trajectory(positions: List[np.ndarray]) -> str:
    """Plot 3D particle trajectory and save to temp file

    Returns:
        str: Path to saved plot image
    """
    positions = np.array(positions)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Particle Trajectory")

    # Save to temp file
    import tempfile

    temp_path = tempfile.mktemp(suffix=".png")
    plt.savefig(temp_path)
    plt.close()
    return temp_path


def display_analysis_result(state, client, **kwargs):
    # Display the trajectory plot
    plt.figure(figsize=(10, 10))
    img = plt.imread(state["trajectory_plot"])
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    # Optional: Display energy conservation
    plt.figure(figsize=(10, 5))
    plt.plot(state["energies"])
    plt.title("Total Energy vs Time Step")
    plt.xlabel("Time Step")
    plt.ylabel("Energy (J)")
    plt.grid(True)
    plt.show()

    # Display results
    print("\n=== Trajectory Analysis ===")
    print(json.dumps(state["analysis_result"], indent=2))
    print(
        "Do you want to update the parameters and try again? If so, let me know what you'd like to change."
    )


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
    image_keys=["trajectory_plot"],
    sink_transform=lambda x: json.loads(x),
    post_process=display_analysis_result,
)


def update_parameter_values(state, client, **kwargs):
    state["mass"] = state["updated_parameters"]["mass"]
    state["charge"] = state["updated_parameters"]["charge"]
    state["initial_position"] = np.array(
        state["updated_parameters"]["initial_position"]
    )
    state["initial_velocity"] = np.array(
        state["updated_parameters"]["initial_velocity"]
    )
    state["E_field"] = np.array(state["updated_parameters"]["E_field"])
    state["B_field"] = np.array(state["updated_parameters"]["B_field"])
    if state["updated_parameters"]["stop_workflow"]:
        print("Workflow terminated.")
    else:
        print("Parameters updated and workflow will continue.")
        print("Updated parameters:")
        print("Mass: ", state["mass"])
        print("Charge: ", state["charge"])
        print("Initial position: ", state["initial_position"])
        print("Initial velocity: ", state["initial_velocity"])
        print("E-field: ", state["E_field"])
        print("B-field: ", state["B_field"])
    return state


update_parameters = Node(
    node_type="parameter_updater",
    prompt_template="""Current parameters:
Mass: {mass} kg
Charge: {charge} C
Initial Position: {initial_position} m
Initial Velocity: {initial_velocity} m/s
E-field: {E_field} N/C
B-field: {B_field} T

Question: Do you want to update the parameters and try again? If so, let me know what you'd like to change.
Answer: {human_input}

Based on the answer, decide whether to try again or stop the workflow. 
Output a JSON with parameters and a boolean to stop the workflow or not, 
IMPORTANT: Keep existing values if not mentioned in the answer, do not make up new values:
{{
    "mass": float_value,
    "charge": float_value,
    "initial_position": [x, y, z],
    "initial_velocity": [vx, vy, vz],
    "E_field": [Ex, Ey, Ez],
    "B_field": [Bx, By, Bz],
    "stop_workflow": false/true
}}""",
    sink="updated_parameters",
    sink_format="json",
    sink_transform=lambda x: json.loads(x),
    post_process=update_parameter_values,
)


class TrajectoryWorkflow(Workflow):
    """Workflow for particle trajectory analysis"""

    def create_workflow(self):
        """Define the workflow graph structure"""
        # Add nodes
        self.add_node("validate_parameters", parameter_validator)
        self.add_node("adjust_parameters", parameter_adjuster)
        self.add_node("calculate_trajectory", calculate_trajectory)
        self.add_node("plot_trajectory", plot_trajectory)
        self.add_node("analyze_trajectory", trajectory_analyzer)
        self.add_node("update_parameters", update_parameters)

        # Add conditional flow for validation
        self.add_conditional_flow(
            "validate_parameters",
            lambda state: state["validation_response"]["validation_passed"],
            then="calculate_trajectory",
            otherwise="adjust_parameters",
        )

        # Add remaining edges
        self.add_flow("adjust_parameters", "validate_parameters")
        self.add_flow("calculate_trajectory", "plot_trajectory")
        self.add_flow("plot_trajectory", "analyze_trajectory")
        self.add_flow("analyze_trajectory", "update_parameters")

        # Modify this conditional flow to handle missing keys safely
        self.add_conditional_flow(
            "update_parameters",
            lambda state: not state["updated_parameters"]["stop_workflow"],
            then="validate_parameters",
            otherwise=END,
        )

        # Set entry point
        self.set_entry("validate_parameters")

        # Compile workflow with interrupt for validation confirmation
        self.compile(checkpointer="memory", interrupt_before=["update_parameters"])


if __name__ == "__main__":
    initial_state = {
        "mass": 9.1093837015e-31,  # electron mass in kg
        "charge": -1.602176634e-19,  # electron charge in C
        "initial_position": np.array([0.0, 0.0, 0.0]),
        "initial_velocity": np.array([1e6, 1e6, 1e6]),  # 1e6 m/s in each direction
        "E_field": np.array([5e6, 1e6, 5e6]),  # 1e6 N/C in y-direction
        "B_field": np.array(
            [0.0, 0.0, 50000.0]
        ),  # deliberately typo to be caught by validation
    }

    workflow = TrajectoryWorkflow(
        state_defs=TrajectoryState,
        llm_name="gpt-4o",
        vlm_name="gpt-4o",
        debug_mode=False,
    )

    # result = workflow.run(initial_state)

    workflow.to_yaml("particle_trajectory_analysis.yaml")

    print(workflow.graph.get_graph().draw_ascii())
    workflow.graph.get_graph().draw_mermaid_png(
        output_file_path="particle_trajectory_analysis.png"
    )
