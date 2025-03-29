/**
 * Handles user interface controls and interactions
 */
class UI {
    /**
     * Create a new UI manager
     * @param {Simulation} simulation - The simulation instance
     * @param {Renderer} renderer - The renderer instance
     */
    constructor(simulation, renderer) {
        this.simulation = simulation;
        this.renderer = renderer;
        
        // Set up DOM element references
        this.setupDOMReferences();
        
        // Set up event listeners
        this.setupEventListeners();
    }
    
    /**
     * Set up references to DOM elements
     */
    setupDOMReferences() {
        // Control elements
        this.ruleSelect = document.getElementById('rule-select');
        this.visualizationMode = document.getElementById('visualization-mode');
        this.dynamicControls = document.getElementById('dynamic-controls');
        this.playPauseButton = document.getElementById('play-pause');
        this.resetButton = document.getElementById('reset');
        this.randomizeButton = document.getElementById('randomize');
        
        // Info elements
        this.infoDisplay = document.getElementById('info-display');
    }
    
    /**
     * Set up event listeners for UI controls
     */
    setupEventListeners() {
        // Rule set selection
        this.ruleSelect.addEventListener('change', () => {
            const ruleSet = this.ruleSelect.value;
            this.simulation.loadRuleSet(ruleSet);
        });
        
        // Visualization mode selection
        this.visualizationMode.addEventListener('change', () => {
            const mode = parseInt(this.visualizationMode.value, 10);
            this.simulation.field.setVisualizationUniform('visualizationMode', mode);
        });
        
        // Play/pause button
        this.playPauseButton.addEventListener('click', () => {
            const isRunning = this.simulation.toggle();
            this.playPauseButton.textContent = isRunning ? 'Pause' : 'Play';
        });
        
        // Reset button
        this.resetButton.addEventListener('click', () => {
            this.simulation.reset();
        });
        
        // Randomize button
        this.randomizeButton.addEventListener('click', () => {
            this.simulation.randomize();
        });
        
        // Listen for visualization mesh interactions (for marker placement)
        this.setupVisualizationInteractions();
    }
    
    /**
     * Set up interactions with the visualization mesh
     */
    setupVisualizationInteractions() {
        // Raycaster for detecting clicks on the visualization mesh
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // Mouse event listeners
        this.renderer.renderer.domElement.addEventListener('click', (event) => {
            this.handleMeshClick(event);
        });
        
        this.renderer.renderer.domElement.addEventListener('mousemove', (event) => {
            this.handleMouseMove(event);
        });
    }
    
    /**
     * Handle clicks on the visualization mesh
     * @param {MouseEvent} event - The mouse click event
     */
    handleMeshClick(event) {
        // Get normalized device coordinates
        this.updateMousePosition(event);
        
        // Cast ray from camera
        this.raycaster.setFromCamera(this.mouse, this.renderer.activeCamera);
        
        // Check if ray intersects the visualization mesh
        const intersects = this.raycaster.intersectObject(this.simulation.getVisualizationMesh());
        
        if (intersects.length > 0) {
            const intersect = intersects[0];
            const uv = intersect.uv;
            
            // Here we would use the UV coordinates to place a marker or get field info
            // For now, just display the position info
            this.displayPositionInfo(uv);
        }
    }
    
    /**
     * Handle mouse movement over the visualization mesh
     * @param {MouseEvent} event - The mouse move event
     */
    handleMouseMove(event) {
        // Get normalized device coordinates
        this.updateMousePosition(event);
        
        // Additional hover effects could be implemented here
    }
    
    /**
     * Update mouse position in normalized device coordinates
     * @param {MouseEvent} event - The mouse event
     */
    updateMousePosition(event) {
        const rect = this.renderer.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    }
    
    /**
     * Display information about a position in the field
     * @param {THREE.Vector2} uv - The UV coordinates of the position
     */
    displayPositionInfo(uv) {
        const x = Math.floor(uv.x * this.simulation.field.resolution);
        const y = Math.floor(uv.y * this.simulation.field.resolution);
        
        // Info display (could be expanded with actual field values in the future)
        this.infoDisplay.innerHTML = `
            <div>Position: (${x}, ${y})</div>
            <div>UV: (${uv.x.toFixed(3)}, ${uv.y.toFixed(3)})</div>
        `;
        
        // This would be a good place to add marker placement in the future
        console.log(`Position clicked: (${x}, ${y}), UV: (${uv.x.toFixed(3)}, ${uv.y.toFixed(3)})`);
    }
    
    /**
     * Update the parameters UI based on the current rule set
     * @param {Object} parameters - The parameters configuration
     */
    updateParameters(parameters) {
        // Clear existing controls
        this.dynamicControls.innerHTML = '';
        
        // Create controls for each parameter
        for (const [name, config] of Object.entries(parameters)) {
            const controlDiv = document.createElement('div');
            controlDiv.className = 'parameter-control';
            
            // Create label
            const label = document.createElement('label');
            label.htmlFor = `param-${name}`;
            label.textContent = this.formatParameterName(name);
            controlDiv.appendChild(label);
            
            // Create slider
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.id = `param-${name}`;
            slider.min = config.min;
            slider.max = config.max;
            slider.step = config.step;
            slider.value = config.value;
            
            // Create value display
            const valueDisplay = document.createElement('div');
            valueDisplay.className = 'value-display';
            
            const minSpan = document.createElement('span');
            minSpan.textContent = config.min;
            
            const valueSpan = document.createElement('span');
            valueSpan.textContent = config.value.toFixed(3);
            valueSpan.id = `param-${name}-value`;
            
            const maxSpan = document.createElement('span');
            maxSpan.textContent = config.max;
            
            valueDisplay.appendChild(minSpan);
            valueDisplay.appendChild(valueSpan);
            valueDisplay.appendChild(maxSpan);
            
            // Update parameter on slider change
            slider.addEventListener('input', () => {
                const value = parseFloat(slider.value);
                this.simulation.updateParameter(name, value);
                valueSpan.textContent = value.toFixed(3);
            });
            
            controlDiv.appendChild(slider);
            controlDiv.appendChild(valueDisplay);
            this.dynamicControls.appendChild(controlDiv);
        }
    }
    
    /**
     * Format a parameter name for display
     * @param {string} name - The parameter name
     * @returns {string} The formatted name
     */
    formatParameterName(name) {
        // Convert camelCase to Title Case with spaces
        return name
            .replace(/([A-Z])/g, ' $1')
            .replace(/^./, (str) => str.toUpperCase());
    }
}