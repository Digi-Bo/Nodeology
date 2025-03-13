# Nodeology Examples

This directory contains example applications built with Nodeology, demonstrating features of the framework.

## Prerequisites

Before running the examples, ensure you have `nodeology` installed

```bash
pip install nodeology
```

## Directory Structure

- `writing_improvement.py` - Text analysis and improvement workflow
- `trajectory_analysis.py` - Particle trajectory simulation and visualization
- `public/` - Static assets for the examples (needed for `nodeology` UI elements)
- `.chainlit/` - Chainlit configuration files (needed for `nodeology` UI settings)

## Available Examples

### 1. Writing Improvement (`writing_improvement.py`)

An interactive application that helps users improve their writing through analysis and suggestions. This example demonstrates:

- State management with Nodeology
- Interactive user input handling
- Text analysis workflow
- Chainlit UI integration

To run this example:

```bash
cd examples
python writing_improvement.py
```

### 2. Particle Trajectory Analysis (`trajectory_analysis.py`)

A scientific application that simulates and visualizes particle trajectories under electromagnetic fields. This example showcases:

- Complex scientific calculations
- Interactive parameter input
- Data visualization
- State management for scientific workflows
- Advanced Chainlit UI features

To run this example:

```bash
cd examples
python trajectory_analysis.py
```

## Usage Tips

1. Each example will open in your default web browser when launched
2. Follow the interactive prompts in the Chainlit UI
3. You can modify parameters and experiment with different inputs
4. Use the chat interface to interact with the applications

## License

These examples are provided under the same license as the main Nodeology project. See the license headers in individual files for details.
