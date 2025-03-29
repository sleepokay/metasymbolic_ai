// Gray-Scott Visualization Shader
//
// This shader visualizes the state of the Gray-Scott reaction-diffusion system
// with several visualization modes.

varying vec2 vUv;
uniform sampler2D stateTexture;
uniform sampler2D orientationTexture;
uniform sampler2D stabilityTexture;
uniform int visualizationMode;

// Color mapping functions
vec3 viridis(float t) {
    // Viridis color map approximation
    const vec3 c0 = vec3(0.2777273272, 0.0054929071, 0.3292213618);
    const vec3 c1 = vec3(0.1050930431, 1.4040130041, 1.3838529381);
    const vec3 c2 = vec3(-0.3308618287, 0.2144421625, 0.0942338638);
    const vec3 c3 = vec3(-4.6340841254, -5.1798880482, -5.7177108140);
    const vec3 c4 = vec3(6.2834246873, 5.5386535571, 10.7554246980);
    const vec3 c5 = vec3(4.2399093050, 2.9317169692, 4.2482959839);
    const vec3 c6 = vec3(4.7741854511, 4.2754230750, 3.6572624669);
    
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // Get state values
    vec4 state = texture2D(stateTexture, vUv);
    float a = state.r; // Chemical A concentration
    float b = state.g; // Chemical B concentration
    
    // Get orientation and stability
    vec4 orientation = texture2D(orientationTexture, vUv);
    vec4 stability = texture2D(stabilityTexture, vUv);
    
    // Output color based on visualization mode
    vec3 color;
    
    // Mode 0: Basic chemical visualization (U as blue, V as red)
    if (visualizationMode == 0) {
        // Highlight B chemical with vibrant colors
        // We use a blue-red color scheme
        float intensity = smoothstep(0.0, 0.8, b);
        color = mix(
            vec3(0.0, 0.0, 0.2), // Dark blue for low concentration
            vec3(1.0, 0.0, 0.0), // Bright red for high concentration
            intensity
        );
        
        // Add some white highlights for very high concentrations
        if (b > 0.7) {
            color = mix(color, vec3(1.0), (b - 0.7) / 0.3);
        }
    }
    // Mode 1: Chemical gradient visualization
    else if (visualizationMode == 1) {
        // Use viridis colormap to visualize chemical B
        color = viridis(b);
    }
    // Mode 2: Stability visualization
    else if (visualizationMode == 2) {
        // Visualize stability as a heatmap
        float s = stability.r;
        color = hsv2rgb(vec3(0.6 - s * 0.6, 0.8, s * 0.8 + 0.2));
    }
    // Mode 3: Orientation visualization
    else if (visualizationMode == 3) {
        // Visualize orientation as hue, with saturation and value based on magnitude
        float angle = atan(orientation.y, orientation.x) / (2.0 * 3.14159) + 0.5;
        float magnitude = length(orientation.xy);
        color = hsv2rgb(vec3(angle, magnitude, 0.8));
    }
    // Default: Grayscale visualization of B chemical
    else {
        color = vec3(b);
    }
    
    gl_FragColor = vec4(color, 1.0);
}