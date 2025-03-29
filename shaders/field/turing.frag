//
// Turing Pattern Reaction-Diffusion System
//
// This shader implements a classic activator-inhibitor system
// that produces Turing patterns.
//

varying vec2 vUv;
uniform sampler2D stateTexture;
uniform sampler2D orientationTexture;
uniform sampler2D stabilityTexture;
uniform float time;
uniform float deltaTime;
uniform float resolution;

// Parameters
uniform float activatorRate;       // Rate of activator production
uniform float inhibitorRate;       // Rate of inhibitor production
uniform float activatorDiffusion;  // Diffusion rate of activator
uniform float inhibitorDiffusion;  // Diffusion rate of inhibitor
uniform float scaleFactor;         // Scale factor for patterns

// Include common utilities
// #include <common>

// Laplacian operator for diffusion
vec2 laplacian(vec2 uv) {
    float texelSize = 1.0 / resolution;
    
    // Get values at the center and neighboring points
    vec2 center = texture2D(stateTexture, uv).rg;
    vec2 right = texture2D(stateTexture, uv + vec2(texelSize, 0.0)).rg;
    vec2 left = texture2D(stateTexture, uv + vec2(-texelSize, 0.0)).rg;
    vec2 up = texture2D(stateTexture, uv + vec2(0.0, texelSize)).rg;
    vec2 down = texture2D(stateTexture, uv + vec2(0.0, -texelSize)).rg;
    
    // 5-point stencil Laplacian
    return right + left + up + down - 4.0 * center;
}

void main() {
    // Scale UV coordinates for pattern size control
    vec2 scaledUV = vUv * scaleFactor;
    
    // Add some positional variation
    float variation = 0.05 * sin(scaledUV.x * 10.0) * sin(scaledUV.y * 10.0);
    
    // Get current state
    vec4 state = texture2D(stateTexture, vUv);
    float a = state.r; // Activator
    float i = state.g; // Inhibitor
    
    // Compute Laplacian for diffusion
    vec2 lap = laplacian(vUv);
    
    // Reaction terms (based on activator-inhibitor dynamics)
    float activatorReaction = activatorRate * (a * a / i - a);
    float inhibitorReaction = inhibitorRate * (a * a - i);
    
    // Update activator (a) and inhibitor (i) using reaction-diffusion equations
    float da = activatorDiffusion * lap.x + activatorReaction;
    float di = inhibitorDiffusion * lap.y + inhibitorReaction;
    
    // Apply updates with time step
    float newA = a + da * deltaTime;
    float newI = i + di * deltaTime;
    
    // Clamp values to prevent instability
    newA = clamp(newA, 0.01, 1.0);
    newI = clamp(newI, 0.01, 1.0);
    
    // Output updated state
    gl_FragColor = vec4(newA, newI, 0.0, 1.0);
}