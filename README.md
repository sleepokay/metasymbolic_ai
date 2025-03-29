# Metasymbolic AI Field Simulation

This project implements a low-dimensional shader simulation for the Metasymbolic AI framework. It demonstrates how local field component interactions can give rise to emergent attractor dynamics, providing visual evidence of how symbol-like behavior can emerge naturally from neural field interactions.

## Features

- GPU-accelerated simulation using WebGL shaders
- Multiple rule sets (Gray-Scott, Kuramoto, Turing patterns)
- Interactive parameter controls
- Various visualization modes
- Foundation for a marker system to identify emergent attractors

## Setup and Running

### Prerequisites

- Node.js and npm installed

### Installation

1. Clone the repository
```
git clone https://github.com/yourusername/metasymbolic-simulation.git
cd metasymbolic-simulation
```

2. Install dependencies
```
npm install
```

### Running the Simulation

Start the development server:
```
npm run dev
```

This will open the simulation in your default browser. If it doesn't open automatically, navigate to `http://localhost:8080`.

## Controls

- **Rule Set**: Select the simulation rule set (Gray-Scott, Kuramoto, Turing)
- **Visualization**: Choose different ways to visualize the field
- **Parameters**: Adjust parameters specific to each rule set
- **Play/Pause**: Toggle simulation running state
- **Reset**: Reset the simulation to initial conditions
- **Randomize**: Randomize parameter values

## Rule Sets

### Gray-Scott Reaction-Diffusion

This rule set simulates a two-component reaction-diffusion system that creates patterns like spots, stripes, and waves.

Parameters:
- **Feed Rate**: Rate at which substance A is fed into the system
- **Kill Rate**: Rate at which substance B is removed
- **Diffusion Rate A/B**: How quickly substances A and B diffuse
- **Timestep**: Controls simulation speed

### Kuramoto Oscillators

This rule set simulates coupled oscillators that can synchronize their phases, demonstrating emergent coherence.

Parameters:
- **Coupling**: Strength of coupling between oscillators
- **Noise Strength**: Random perturbations added to phases
- **Natural Frequency**: Base oscillation frequency
- **Frequency Variation**: Variation in natural frequencies
- **Diffusion**: Spatial coupling strength

### Turing Patterns

This rule set implements an activator-inhibitor system that produces Turing-like patterns through diffusion-driven instability.

Parameters:
- **Activator Rate**: Production rate of the activator
- **Inhibitor Rate**: Production rate of the inhibitor
- **Activator Diffusion**: Diffusion rate of the activator
- **Inhibitor Diffusion**: Diffusion rate of the inhibitor
- **Scale Factor**: Controls pattern scale

## Project Structure

- `index.html`: Main entry point
- `js/`: JavaScript source files
  - `main.js`: Application initialization
  - `renderer.js`: Three.js setup and rendering
  - `field.js`: Field component data structures
  - `simulation.js`: Simulation logic and rule sets
  - `ui.js`: User interface controls
  - `utils/`: Utility functions
- `shaders/`: GLSL shader files
  - `field/`: Update shaders for different rule sets
  - `visualization/`: Visualization shaders
  - `common/`: Common shader utilities

## Extending

To add a new rule set:

1. Create update and visualization shaders in the `shaders/` directory
2. Add the rule set definition to `simulation.js`
3. Update the UI to include the new rule set

## License

MIT