// Kuramoto Oscillator Visualization Shader
//
// This shader visualizes the state of the Kuramoto oscillator system
// with several visualization modes.

varying vec2 vUv;
uniform sampler2D stateTexture;
uniform sampler2D orientationTexture;
uniform sampler2D stabilityTexture;
uniform int visualizationMode;

// Constants
#define PI 3.1415926535897932384626433832795
#define TWO_PI 6.2831853071795864769252867665590

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Viridis color map
vec3 viridis(float t) {
    const vec3 c0 = vec3(0.2777273272, 0.0054929071, 0.3292213618);
    const vec3 c1 = vec3(0.1050930431, 1.4040130041, 1.3838529381);
    const vec3 c2 = vec3(-0.3308618287, 0.2144421625, 0.0942338638);
    const vec3 c3 = vec3(-4.6340841254, -5.1798880482, -5.7177108140);
    const vec3 c4 = vec3(6.2834246873, 5.5386535571, 10.7554246980);
    const vec3 c5 = vec3(4.2399093050, 2.9317169692, 4.2482959839);
    const vec3 c6 = vec3(4.7741854511, 4.2754230750, 3.6572624669);
    
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

void main() {
    // Get state values
    vec4 state = texture2D(stateTexture, vUv);
    float phase = state.r;            // Current phase (0 to 2π)
    float frequency = state.g;        // Natural frequency
    float synchronization = state.b;  // Local synchronization
    
    // Get orientation and stability
    vec4 orientation = texture2D(orientationTexture, vUv);
    vec4 stability = texture2D(stabilityTexture, vUv);
    
    // Output color based on visualization mode
    vec3 color;
    
    // Mode 0: Phase as hue
    if (visualizationMode == 0) {
        // Map phase (0 to 2π) to hue (0 to 1)
        float hue = phase / TWO_PI;
        
        // Map synchronization to brightness
        float value = 0.5 + synchronization * 0.5;
        
        // Set saturation to full
        float saturation = 0.8;
        
        // Convert HSV to RGB
        color = hsv2rgb(vec3(hue, saturation, value));
    }
    // Mode 1: Synchronization visualization
    else if (visualizationMode == 1) {
        // Visualize local synchronization using viridis colormap
        color = viridis(synchronization);
    }
    // Mode 2: Frequency visualization
    else if (visualizationMode == 2) {
        // Normalize frequency to 0-1 range assuming frequencies are around naturalFrequency
        float normalizedFreq = (frequency - 0.1) / 0.2; // Rough normalization
        
        // Visualize as a cool-to-warm color gradient
        color = mix(
            vec3(0.0, 0.0, 0.8), // Cool blue for low frequencies
            vec3(0.8, 0.0, 0.0), // Warm red for high frequencies
            clamp(normalizedFreq, 0.0, 1.0)
        );
    }
    // Mode 3: Phase gradient visualization
    else if (visualizationMode == 3) {
        float texelSize = 1.0 / 512.0; // Assuming 512x512 resolution
        
        // Sample neighboring phases
        float phaseRight = texture2D(stateTexture, vUv + vec2(texelSize, 0.0)).r;
        float phaseLeft = texture2D(stateTexture, vUv + vec2(-texelSize, 0.0)).r;
        float phaseUp = texture2D(stateTexture, vUv + vec2(0.0, texelSize)).r;
        float phaseDown = texture2D(stateTexture, vUv + vec2(0.0, -texelSize)).r;
        
        // Calculate phase gradient (handling phase wrapping)
        float dx = mod(phaseRight - phaseLeft + 3.0 * PI, TWO_PI) - PI;
        float dy = mod(phaseUp - phaseDown + 3.0 * PI, TWO_PI) - PI;
        
        // Normalize gradient
        float gradientMagnitude = sqrt(dx * dx + dy * dy) / (2.0 * PI);
        
        // Calculate gradient direction
        float angle = atan(dy, dx) / TWO_PI + 0.5;
        
        // Visualize gradient as direction (hue) and magnitude (brightness)
        color = hsv2rgb(vec3(angle, 0.8, gradientMagnitude * 2.0));
    }
    // Default: Simple grayscale phase
    else {
        color = vec3(phase / TWO_PI);
    }
    
    gl_FragColor = vec4(color, 1.0);
}