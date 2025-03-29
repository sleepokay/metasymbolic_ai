/**
 * Handles simulation logic and rule sets for the field
 */
class Simulation {
    /**
     * Create a new simulation
     * @param {Renderer} renderer - The renderer instance
     * @param {Field} field - The field to simulate
     */
    constructor(renderer, field) {
        this.renderer = renderer;
        this.field = field;
        this.running = false;
        this.time = 0;
        
        // Register available rule sets
        this.ruleSets = {
            'gray-scott': {
                name: 'Gray-Scott Reaction-Diffusion',
                shaderPath: 'shaders/field/gray-scott.frag',
                visualizationPath: 'shaders/visualization/gray-scott.frag',
                parameters: {
                    feedRate: { value: 0.055, min: 0.01, max: 0.1, step: 0.001 },
                    killRate: { value: 0.062, min: 0.01, max: 0.1, step: 0.001 },
                    diffusionRateA: { value: 1.0, min: 0.1, max: 2.0, step: 0.01 },
                    diffusionRateB: { value: 0.5, min: 0.1, max: 2.0, step: 0.01 },
                    timestep: { value: 1.0, min: 0.1, max: 2.0, step: 0.01 }
                },
                init: (field) => {
                    // Initialize field with a pattern suitable for Gray-Scott
                    field.reset((data, resolution) => {
                        // Fill with values suitable for Gray-Scott
                        for (let y = 0; y < resolution; y++) {
                            for (let x = 0; x < resolution; x++) {
                                const i = (y * resolution + x) * 4;
                                
                                // Initialize with uniform state (A=1, B=0)
                                data[i] = 1.0;     // A component
                                data[i + 1] = 0.0; // B component
                                data[i + 2] = 0.0; // Unused
                                data[i + 3] = 1.0; // Unused
                            }
                        }
                        
                        // Add some random perturbations in the center
                        const centerSize = Math.floor(resolution * 0.2);
                        const centerStart = Math.floor((resolution - centerSize) / 2);
                        const centerEnd = centerStart + centerSize;
                        
                        for (let y = centerStart; y < centerEnd; y++) {
                            for (let x = centerStart; x < centerEnd; x++) {
                                const i = (y * resolution + x) * 4;
                                
                                // Add B component in the center
                                if (Math.random() < 0.5) {
                                    data[i] = 0.5;
                                    data[i + 1] = 0.5;
                                }
                            }
                        }
                    });
                }
            },
            'kuramoto': {
                name: 'Kuramoto Oscillators',
                shaderPath: 'shaders/field/kuramoto.frag',
                visualizationPath: 'shaders/visualization/kuramoto.frag',
                parameters: {
                    coupling: { value: 0.5, min: 0.0, max: 2.0, step: 0.01 },
                    noiseStrength: { value: 0.01, min: 0.0, max: 0.1, step: 0.001 },
                    naturalFrequency: { value: 0.2, min: 0.01, max: 1.0, step: 0.01 },
                    frequencyVariation: { value: 0.1, min: 0.0, max: 0.5, step: 0.01 },
                    diffusion: { value: 0.1, min: 0.0, max: 0.5, step: 0.01 }
                },
                init: (field) => {
                    // Initialize with random phases
                    field.reset((data, resolution) => {
                        for (let y = 0; y < resolution; y++) {
                            for (let x = 0; x < resolution; x++) {
                                const i = (y * resolution + x) * 4;
                                
                                // Phase (0 to 2π)
                                data[i] = Math.random() * Math.PI * 2;
                                
                                // Natural frequency
                                data[i + 1] = 0.15 + Math.random() * 0.1;
                                
                                // Synchronization indicator (will be computed by shader)
                                data[i + 2] = 0.0;
                                
                                // Unused
                                data[i + 3] = 1.0;
                            }
                        }
                    });
                }
            },
            'turing': {
                name: 'Turing Patterns',
                shaderPath: 'shaders/field/turing.frag',
                visualizationPath: 'shaders/visualization/turing.frag',
                parameters: {
                    activatorRate: { value: 0.1, min: 0.01, max: 0.2, step: 0.001 },
                    inhibitorRate: { value: 0.05, min: 0.01, max: 0.2, step: 0.001 },
                    activatorDiffusion: { value: 0.1, min: 0.01, max: 0.5, step: 0.01 },
                    inhibitorDiffusion: { value: 0.3, min: 0.01, max: 1.0, step: 0.01 },
                    scaleFactor: { value: 2.0, min: 1.0, max: 10.0, step: 0.1 }
                },
                init: (field) => {
                    // Initialize with small random perturbations
                    field.reset((data, resolution) => {
                        for (let y = 0; y < resolution; y++) {
                            for (let x = 0; x < resolution; x++) {
                                const i = (y * resolution + x) * 4;
                                
                                // Activator (with small random variations)
                                data[i] = 0.5 + (Math.random() - 0.5) * 0.01;
                                
                                // Inhibitor (with small random variations)
                                data[i + 1] = 0.5 + (Math.random() - 0.5) * 0.01;
                                
                                // Unused
                                data[i + 2] = 0.0;
                                data[i + 3] = 1.0;
                            }
                        }
                    });
                }
            }
        };
        
        // Default rule set
        this.currentRuleSet = 'gray-scott';
    }
    
    /**
     * Initialize the simulation
     */
    async init() {
        // Load the shader for the current rule set
        await this.loadRuleSet(this.currentRuleSet);
        
        // Add field visualization to the scene
        this.renderer.scene.add(this.field.getVisualizationMesh());
    }
    
    /**
     * Load a rule set by name
     * @param {string} name - The name of the rule set to load
     */
    async loadRuleSet(name) {
        if (!this.ruleSets[name]) {
            console.error(`Rule set ${name} not found`);
            return;
        }
        
        this.currentRuleSet = name;
        const ruleSet = this.ruleSets[name];
        
        try {
            // Load the update shader
            await shaderLoader.load(name + '-update', ruleSet.shaderPath);
            const updateShader = shaderLoader.get(name + '-update');
            this.field.setUpdateShader(updateShader);
            
            // Load the visualization shader
            await shaderLoader.load(name + '-viz', ruleSet.visualizationPath);
            const vizShader = shaderLoader.get(name + '-viz');
            this.field.setVisualizationShader(vizShader);
            
            // Set up parameters
            this.setupParameters(ruleSet.parameters);
            
            // Initialize the field for this rule set
            if (typeof ruleSet.init === 'function') {
                ruleSet.init(this.field);
            }
            
            console.log(`Loaded rule set: ${ruleSet.name}`);
        } catch (error) {
            console.error(`Error loading rule set ${name}:`, error);
        }
    }
    
    /**
     * Set up parameters for the current rule set
     * @param {Object} parameters - The parameters configuration
     */
    setupParameters(parameters) {
        // Set uniform values in the shader
        for (const [name, config] of Object.entries(parameters)) {
            this.field.setUpdateUniform(name, config.value);
        }
        
        // Trigger UI update (if the UI module is available)
        if (typeof ui !== 'undefined') {
            ui.updateParameters(parameters);
        }
    }
    
    /**
     * Update parameter value
     * @param {string} name - The parameter name
     * @param {number} value - The new value
     */
    updateParameter(name, value) {
        const ruleSet = this.ruleSets[this.currentRuleSet];
        if (ruleSet.parameters[name]) {
            ruleSet.parameters[name].value = value;
            this.field.setUpdateUniform(name, value);
        }
    }
    
    /**
     * Start the simulation
     */
    start() {
        this.running = true;
    }
    
    /**
     * Pause the simulation
     */
    pause() {
        this.running = false;
    }
    
    /**
     * Toggle the simulation state (running/paused)
     */
    toggle() {
        this.running = !this.running;
        return this.running;
    }
    
    /**
     * Reset the simulation
     */
    reset() {
        const ruleSet = this.ruleSets[this.currentRuleSet];
        if (typeof ruleSet.init === 'function') {
            ruleSet.init(this.field);
        }
        this.time = 0;
    }
    
    /**
     * Randomize the simulation parameters
     */
    randomize() {
        const ruleSet = this.ruleSets[this.currentRuleSet];
        
        // Randomize each parameter within its range
        for (const [name, config] of Object.entries(ruleSet.parameters)) {
            const randomValue = config.min + Math.random() * (config.max - config.min);
            ruleSet.parameters[name].value = randomValue;
            this.field.setUpdateUniform(name, randomValue);
        }
        
        // Trigger UI update
        if (typeof ui !== 'undefined') {
            ui.updateParameters(ruleSet.parameters);
        }
    }
    
    /**
     * Get the current visualization mesh
     * @returns {THREE.Mesh} The visualization mesh
     */
    getVisualizationMesh() {
        return this.field.getVisualizationMesh();
    }
    
    /**
     * Update the simulation
     * @param {number} deltaTime - Time since last update
     */
    update(deltaTime) {
        if (!this.running) return;
        
        // Update the simulation time
        this.time += deltaTime;
        
        // Update the field
        this.field.update(this.time, deltaTime);
    }
}