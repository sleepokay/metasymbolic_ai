/**
 * Manages field component data, textures, and ping-pong buffers
 */
class Field {
    /**
     * Create a new field
     * @param {Renderer} renderer - The renderer instance
     * @param {Object} options - Field configuration options
     */
    constructor(renderer, options = {}) {
        this.renderer = renderer;
        
        // Default options
        const defaultOptions = {
            resolution: 512,      // Resolution of the field (width/height in pixels)
            gridSize: 1.0,        // Size of the field in world units
            numComponents: 4,     // Number of components per field element (RGBA)
            initialState: null    // Initial state generator function
        };
        
        this.options = { ...defaultOptions, ...options };
        
        this.resolution = this.options.resolution;
        this.gridSize = this.options.gridSize;
        this.numComponents = this.options.numComponents;
        
        this.init();
    }
    
    /**
     * Initialize the field data structures and buffers
     */
    init() {
        // Create textures for the field state
        this.createFieldTextures();
        
        // Create ping-pong render targets
        this.createRenderTargets();
        
        // Create scene and camera for field updates
        this.createUpdateScene();
        
        // Create visualization scene
        this.createVisualizationScene();
        
        // Current read/write targets for ping-pong rendering
        this.read = 0;
        this.write = 1;
    }
    
    /**
     * Create the textures for the field state, orientation, and stability
     */
    createFieldTextures() {
        // Create state texture
        const stateData = this.createInitialStateData();
        this.stateTexture = new THREE.DataTexture(
            stateData,
            this.resolution,
            this.resolution,
            THREE.RGBAFormat,
            THREE.FloatType
        );
        this.stateTexture.needsUpdate = true;
        
        // Create orientation texture (for relational signatures)
        const orientationData = new Float32Array(this.resolution * this.resolution * 4);
        // Initialize with random unit vectors
        for (let i = 0; i < orientationData.length; i += 4) {
            const angle = Math.random() * Math.PI * 2;
            orientationData[i] = Math.cos(angle);
            orientationData[i + 1] = Math.sin(angle);
            orientationData[i + 2] = 0; // z-component (unused in 2D)
            orientationData[i + 3] = 1; // w-component (unused)
        }
        this.orientationTexture = new THREE.DataTexture(
            orientationData,
            this.resolution,
            this.resolution,
            THREE.RGBAFormat,
            THREE.FloatType
        );
        this.orientationTexture.needsUpdate = true;
        
        // Create stability texture
        const stabilityData = new Float32Array(this.resolution * this.resolution * 4);
        // Initialize with uniform stability
        for (let i = 0; i < stabilityData.length; i += 4) {
            stabilityData[i] = 0.5;     // Stability value
            stabilityData[i + 1] = 0.1;  // Additional parameter 1
            stabilityData[i + 2] = 0.1;  // Additional parameter 2
            stabilityData[i + 3] = 1.0;  // Additional parameter 3
        }
        this.stabilityTexture = new THREE.DataTexture(
            stabilityData,
            this.resolution,
            this.resolution,
            THREE.RGBAFormat,
            THREE.FloatType
        );
        this.stabilityTexture.needsUpdate = true;
    }
    
    /**
     * Create the initial state data for the field
     * @returns {Float32Array} The initial state data
     */
    createInitialStateData() {
        const data = new Float32Array(this.resolution * this.resolution * 4);
        
        // If a custom initializer is provided, use it
        if (typeof this.options.initialState === 'function') {
            this.options.initialState(data, this.resolution);
            return data;
        }
        
        // Default initialization (small random values)
        for (let i = 0; i < data.length; i += 4) {
            data[i] = Math.random() * 0.1;
            data[i + 1] = Math.random() * 0.1;
            data[i + 2] = Math.random() * 0.1;
            data[i + 3] = 1.0;
        }
        
        return data;
    }
    
    /**
     * Create the render targets for ping-pong rendering
     */
    createRenderTargets() {
        // Create two render targets for ping-pong rendering
        this.renderTargets = [
            this.renderer.createRenderTarget(this.resolution, this.resolution),
            this.renderer.createRenderTarget(this.resolution, this.resolution)
        ];
        
        // Additional render targets for orientation and stability updates
        this.orientationTarget = this.renderer.createRenderTarget(this.resolution, this.resolution);
        this.stabilityTarget = this.renderer.createRenderTarget(this.resolution, this.resolution);
    }
    
    /**
     * Create the scene and materials for field updates
     */
    createUpdateScene() {
        // Create a scene for updating the field
        this.updateScene = new THREE.Scene();
        
        // Create a camera for rendering to the field textures
        this.updateCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
        this.updateCamera.position.z = 1;
        
        // Create a plane that covers the entire view
        const geometry = new THREE.PlaneBufferGeometry(2, 2);
        
        // Create update shaders (will be set by the simulation)
        this.updateMaterial = new THREE.ShaderMaterial({
            uniforms: {
                stateTexture: { value: this.stateTexture },
                orientationTexture: { value: this.orientationTexture },
                stabilityTexture: { value: this.stabilityTexture },
                resolution: { value: this.resolution },
                time: { value: 0.0 },
                deltaTime: { value: 0.0 }
            },
            vertexShader: `
                varying vec2 vUv;
                
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                // This will be replaced with actual update logic
                varying vec2 vUv;
                uniform sampler2D stateTexture;
                
                void main() {
                    vec4 state = texture2D(stateTexture, vUv);
                    gl_FragColor = state;
                }
            `
        });
        
        // Create mesh for field updates
        this.updateMesh = new THREE.Mesh(geometry, this.updateMaterial);
        this.updateScene.add(this.updateMesh);
    }
    
    /**
     * Create the scene and materials for field visualization
     */
    createVisualizationScene() {
        // Create geometry for visualization
        // Using a plane for 2D visualization and potentially a more complex geometry for 3D
        const planeGeometry = new THREE.PlaneBufferGeometry(this.gridSize, this.gridSize);
        
        // Default visualization material (just shows state values as colors)
        this.visualizationMaterial = new THREE.ShaderMaterial({
            uniforms: {
                stateTexture: { value: null },  // Will be set during rendering
                orientationTexture: { value: this.orientationTexture },
                stabilityTexture: { value: this.stabilityTexture },
                visualizationMode: { value: 0 }, // 0: state, 1: orientation, 2: stability, 3: flow
                colorMap: { value: null }        // Optional color map texture
            },
            vertexShader: `
                varying vec2 vUv;
                
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                varying vec2 vUv;
                uniform sampler2D stateTexture;
                uniform int visualizationMode;
                
                void main() {
                    vec4 state = texture2D(stateTexture, vUv);
                    
                    // Basic visualization - using RGB components directly
                    if (visualizationMode == 0) {
                        gl_FragColor = vec4(state.rgb, 1.0);
                    } 
                    // Other visualization modes will be implemented later
                    else {
                        gl_FragColor = state;
                    }
                }
            `
        });
        
        // Create the visualization mesh
        this.visualizationMesh = new THREE.Mesh(planeGeometry, this.visualizationMaterial);
        
        // We don't add to scene here - this will be done by the simulation or main class
    }
    
    /**
     * Set the update shader for the field
     * @param {string} fragmentShader - The fragment shader source for field updates
     */
    setUpdateShader(fragmentShader) {
        // Update the material with the new shader
        this.updateMaterial.fragmentShader = fragmentShader;
        this.updateMaterial.needsUpdate = true;
    }
    
    /**
     * Set the visualization shader for the field
     * @param {string} fragmentShader - The fragment shader source for visualization
     */
    setVisualizationShader(fragmentShader) {
        // Update the material with the new shader
        this.visualizationMaterial.fragmentShader = fragmentShader;
        this.visualizationMaterial.needsUpdate = true;
    }
    
    /**
     * Set a uniform value in the update shader
     * @param {string} name - The name of the uniform
     * @param {any} value - The value to set
     */
    setUpdateUniform(name, value) {
        if (this.updateMaterial.uniforms[name] !== undefined) {
            this.updateMaterial.uniforms[name].value = value;
        } else {
            this.updateMaterial.uniforms[name] = { value };
        }
    }
    
    /**
     * Set a uniform value in the visualization shader
     * @param {string} name - The name of the uniform
     * @param {any} value - The value to set
     */
    setVisualizationUniform(name, value) {
        if (this.visualizationMaterial.uniforms[name] !== undefined) {
            this.visualizationMaterial.uniforms[name].value = value;
        } else {
            this.visualizationMaterial.uniforms[name] = { value };
        }
    }
    
    /**
     * Reset the field to its initial state
     * @param {Function} initialStateFunc - Optional function to generate new initial state
     */
    reset(initialStateFunc) {
        // If a new initializer is provided, use it
        if (typeof initialStateFunc === 'function') {
            const data = new Float32Array(this.resolution * this.resolution * 4);
            initialStateFunc(data, this.resolution);
            
            this.stateTexture.image.data = data;
            this.stateTexture.needsUpdate = true;
        } else {
            // Otherwise use the default initializer
            this.stateTexture.image.data = this.createInitialStateData();
            this.stateTexture.needsUpdate = true;
        }
        
        // Reset orientation and stability if needed
        // For now, we'll keep these as they are
    }
    
    /**
     * Update the field state using the current update shader
     * @param {number} time - Current simulation time
     * @param {number} deltaTime - Time since last update
     */
    update(time, deltaTime) {
        // Update time uniforms
        this.setUpdateUniform('time', time);
        this.setUpdateUniform('deltaTime', deltaTime);
        
        // Set the current state texture as input
        this.setUpdateUniform('stateTexture', this.renderTargets[this.read].texture);
        
        // Render to the write target
        this.renderer.renderToTarget(this.updateScene, this.updateCamera, this.renderTargets[this.write]);
        
        // Swap read and write targets
        this.swapBuffers();
    }
    
    /**
     * Swap the read and write buffers
     */
    swapBuffers() {
        this.read = 1 - this.read;
        this.write = 1 - this.write;
    }
    
    /**
     * Get the current state texture
     * @returns {THREE.Texture} The current state texture
     */
    getStateTexture() {
        return this.renderTargets[this.read].texture;
    }
    
    /**
     * Get the visualization mesh for rendering
     * @returns {THREE.Mesh} The visualization mesh
     */
    getVisualizationMesh() {
        // Update the visualization material with the current state texture
        this.setVisualizationUniform('stateTexture', this.getStateTexture());
        return this.visualizationMesh;
    }
}