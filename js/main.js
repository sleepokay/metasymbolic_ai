/**
 * Main application initialization and loop
 */
class Application {
    constructor() {
        this.init();
    }
    
    /**
     * Initialize the application
     */
    async init() {
        // Create the renderer
        const container = document.getElementById('simulation-container');
        this.renderer = new Renderer(container);
        
        // Create the field
        this.field = new Field(this.renderer, {
            resolution: 512,
            gridSize: 2.0
        });
        
        // Create the simulation
        this.simulation = new Simulation(this.renderer, this.field);
        
        // Create the UI
        this.ui = new UI(this.simulation, this.renderer);
        
        // Make UI available globally for other modules
        window.ui = this.ui;
        
        // Load initial shaders and setup
        await this.loadShaders();
        
        // Initialize the simulation
        await this.simulation.init();
        
        // Start the animation loop
        this.simulation.start();
        this.animate();
        
        console.log('Application initialized');
    }
    
    /**
     * Load shader files
     */
    async loadShaders() {
        try {
            // Common shaders
            await shaderLoader.load('common', 'shaders/common/common.glsl');
            
            // Rule set shaders - will be loaded by the simulation when a rule set is selected
            
            console.log('Shaders loaded successfully');
        } catch (error) {
            console.error('Error loading shaders:', error);
        }
    }
    
    /**
     * Animation loop
     */
    animate() {
        requestAnimationFrame(this.animate.bind(this));
        
        // Update the simulation
        this.simulation.update(0.016); // Approximately 60 FPS
        
        // Render the scene
        this.renderer.render();
    }
}

// Wait for DOM to be loaded before initializing
document.addEventListener('DOMContentLoaded', () => {
    // Check for WebGL support
    if (!window.WebGLRenderingContext) {
        alert('Your browser does not support WebGL. Please use a WebGL-compatible browser.');
        return;
    }
    
    // Create and initialize the application
    window.app = new Application();
});