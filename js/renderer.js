/**
 * Handles Three.js setup, rendering, and camera controls
 */
class Renderer {
    constructor(container) {
        this.container = container;
        this.width = container.clientWidth;
        this.height = container.clientHeight;
        this.clock = new THREE.Clock();
        
        this.init();
        this.setupStats();
        this.setupEventListeners();
    }

    /**
     * Initialize the Three.js scene, camera, and renderer
     */
    init() {
        // Create scene
        this.scene = new THREE.Scene();
        
        // Create camera (perspective for 3D view)
        this.camera = new THREE.PerspectiveCamera(
            75, // Field of view
            this.width / this.height, // Aspect ratio
            0.1, // Near clipping plane
            1000 // Far clipping plane
        );
        this.camera.position.z = 5;
        
        // Also create an orthographic camera for 2D view
        this.orthoCamera = new THREE.OrthographicCamera(
            -1, 1, 1, -1, 0.1, 10
        );
        this.orthoCamera.position.z = 5;
        
        // Current active camera (default to perspective)
        this.activeCamera = this.camera;
        
        // Create WebGL renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setClearColor(0x000000);
        
        // Check if we have floating point texture support
        const gl = this.renderer.getContext();
        this.hasFloatTextures = !!gl.getExtension('OES_texture_float');
        if (!this.hasFloatTextures) {
            console.warn('OES_texture_float not supported - simulation may not work correctly');
        }
        
        // Add renderer to container
        this.container.appendChild(this.renderer.domElement);
        
        // Create orbit controls for camera
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.25;
        
        // Create orthographic controls
        this.orthoControls = new THREE.OrbitControls(this.orthoCamera, this.renderer.domElement);
        this.orthoControls.enableRotate = false; // Disable rotation for 2D view
        this.orthoControls.enableDamping = true;
        this.orthoControls.dampingFactor = 0.25;
        
        // Active controls (default to 3D)
        this.activeControls = this.controls;
    }

    /**
     * Set up stats.js for performance monitoring
     */
    setupStats() {
        this.stats = new Stats();
        this.stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
        const statsContainer = document.getElementById('stats-container');
        if (statsContainer) {
            statsContainer.appendChild(this.stats.dom);
        }
    }

    /**
     * Set up event listeners for window resize
     */
    setupEventListeners() {
        window.addEventListener('resize', this.onResize.bind(this));
    }

    /**
     * Handle window resize
     */
    onResize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        
        // Update perspective camera
        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();
        
        // Update orthographic camera
        const aspect = this.width / this.height;
        this.orthoCamera.left = -aspect;
        this.orthoCamera.right = aspect;
        this.orthoCamera.top = 1;
        this.orthoCamera.bottom = -1;
        this.orthoCamera.updateProjectionMatrix();
        
        // Update renderer
        this.renderer.setSize(this.width, this.height);
    }

    /**
     * Create a render target for offscreen rendering
     * @param {number} width - Width of the render target
     * @param {number} height - Height of the render target
     * @param {Object} options - Options for the render target
     * @returns {THREE.WebGLRenderTarget} The created render target
     */
    createRenderTarget(width, height, options = {}) {
        const defaultOptions = {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            format: THREE.RGBAFormat,
            type: this.hasFloatTextures ? THREE.FloatType : THREE.HalfFloatType,
            stencilBuffer: false,
            depthBuffer: false
        };
        
        const mergedOptions = { ...defaultOptions, ...options };
        return new THREE.WebGLRenderTarget(width, height, mergedOptions);
    }

    /**
     * Toggle between 2D and 3D view
     * @param {boolean} use2D - Whether to use 2D view
     */
    toggle2D(use2D) {
        if (use2D) {
            this.activeCamera = this.orthoCamera;
            this.activeControls = this.orthoControls;
        } else {
            this.activeCamera = this.camera;
            this.activeControls = this.controls;
        }
    }

    /**
     * Render the scene
     */
    render() {
        this.stats.begin();
        
        const deltaTime = this.clock.getDelta();
        
        // Update active controls
        this.activeControls.update();
        
        // Render scene with active camera
        this.renderer.render(this.scene, this.activeCamera);
        
        this.stats.end();
    }

    /**
     * Render to a specific render target
     * @param {THREE.Scene} scene - The scene to render
     * @param {THREE.Camera} camera - The camera to use
     * @param {THREE.WebGLRenderTarget} target - The render target
     */
    renderToTarget(scene, camera, target) {
        this.renderer.setRenderTarget(target);
        this.renderer.render(scene, camera);
        this.renderer.setRenderTarget(null);
    }
}