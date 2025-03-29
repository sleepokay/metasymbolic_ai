// Turing Pattern Visualization Shader
//
// This shader visualizes the state of the Turing pattern reaction-diffusion system
// with several visualization modes.

varying vec2 vUv;
uniform sampler2D stateTexture;
uniform sampler2D orientationTexture;
uniform sampler2D stabilityTexture;
uniform int visualizationMode;

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Inferno colormap approximation
vec3 inferno(float t) {
    const vec3 c0 = vec3(0.0002189403, 0.0002050442, 0.0136440758);
    const vec3 c1 = vec3(0.1057264752, 0.0568058041, 1.0000971986);
    const vec3 c2 = vec3(2.5880021273, 1.4861939986, -0.7017955726);
    const vec3 c3 = vec3(-2.7702232991, -2.9009746027, -2.4440126480);
    const vec3 c4 = vec3(1.3144556824, 1.8495966508, 0.0476831077);
    const vec3 c5 = vec3(-0.1986710240, -0.4127034217, 2.0086226704);
    
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
}

void main() {
    // Get state values
    vec4 state = texture2D(stateTexture, vUv);
    float activator = state.r;
    float inhibitor = state.g;
    
    // Get orientation and stability
    vec4 orientation = texture2D(orientationTexture, vUv);
    vec4 stability = texture2D(stabilityTexture, vUv);
    
    // Output color based on visualization mode
    vec3 color;
    
    // Mode 0: Activator visualization with custom colormap
    if (visualizationMode == 0) {
        // Use inferno colormap for activator
        color = inferno(activator);
    }
    // Mode 1: Inhibitor visualization
    else if (visualizationMode == 1) {
        // Visualize inhibitor with a cool blue palette
        color = vec3(0.0, 0.0, 0.2) + inhibitor * vec3(0.0, 0.5, 0.8);
    }
    // Mode 2: Activator-inhibitor ratio
    else if (visualizationMode == 2) {
        // Calculate ratio (constrained to prevent division by zero)
        float ratio = activator / max(inhibitor, 0.01);
        ratio = clamp(ratio, 0.0, 2.0) / 2.0; // Normalize to 0-1
        
        // Use a diverging colormap (blue to white to red)
        if (ratio < 0.5) {
            // Blue to white (0.0 - 0.5 maps to ratio 0.0 - 1.0)
            float t = ratio * 2.0;
            color = mix(vec3(0.0, 0.0, 0.8), vec3(1.0), t);
        } else {
            // White to red (0.5 - 1.0 maps to ratio 0.0 - 1.0)
            float t = (ratio - 0.5) * 2.0;
            color = mix(vec3(1.0), vec3(0.8, 0.0, 0.0), t);
        }
    }
    // Mode 3: Pattern highlight (edges and transitions)
    else if (visualizationMode == 3) {
        // Calculate gradient magnitude using simple central differences
        float texelSize = 1.0 / 512.0; // Assuming 512x512 resolution
        
        float centerA = activator;
        float rightA = texture2D(stateTexture, vUv + vec2(texelSize, 0.0)).r;
        float leftA = texture2D(stateTexture, vUv + vec2(-texelSize, 0.0)).r;
        float upA = texture2D(stateTexture, vUv + vec2(0.0, texelSize)).r;
        float downA = texture2D(stateTexture, vUv + vec2(0.0, -texelSize)).r;
        
        float dx = (rightA - leftA) / (2.0 * texelSize);
        float dy = (upA - downA) / (2.0 * texelSize);
        
        float gradientMagnitude = sqrt(dx * dx + dy * dy);
        
        // Highlight edges with a colorful scheme
        float hue = activator * 0.8; // Base hue on activator value
        float saturation = 0.6 + gradientMagnitude * 2.0; // Higher saturation at edges
        float value = 0.5 + 0.5 * gradientMagnitude; // Brighter at edges
        
        // Clamp values
        saturation = clamp(saturation, 0.0, 1.0);
        value = clamp(value, 0.0, 1.0);
        
        color = hsv2rgb(vec3(hue, saturation, value));
    }
    // Default: Show activator in grayscale
    else {
        color = vec3(activator);
    }
    
    gl_FragColor = vec4(color, 1.0);
}