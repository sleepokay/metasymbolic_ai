//
// Kuramoto Oscillator Model
//
// This shader implements a spatially-coupled 2D lattice of Kuramoto oscillators.
// Each oscillator has a phase and a natural frequency, and is coupled to its neighbors.
//

varying vec2 vUv;
uniform sampler2D stateTexture;
uniform sampler2D orientationTexture;
uniform sampler2D stabilityTexture;
uniform float time;
uniform float deltaTime;
uniform float resolution;

// Parameters
uniform float coupling;          // Strength of coupling between oscillators
uniform float noiseStrength;     // Strength of random perturbations
uniform float naturalFrequency;  // Base frequency of oscillators
uniform float frequencyVariation; // Variation in natural frequencies
uniform float diffusion;         // Spatial diffusion rate

// Include common utilities
// #include <common>

// Compute phase difference, accounting for wrap-around
float phaseDifference(float phi1, float phi2) {
    float diff = phi1 - phi2;
    
    // Wrap to [-PI, PI]
    if (diff > PI) diff -= TWO_PI;
    if (diff < -PI) diff += TWO_PI;
    
    return diff;
}

// Random function for adding noise
float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // Texel size for neighbor sampling
    float texelSize = 1.0 / resolution;
    
    // Get current state
    vec4 state = texture2D(stateTexture, vUv);
    float phase = state.r;            // Current phase
    float frequency = state.g;        // Natural frequency
    float synchronization = state.b;  // Local synchronization indicator
    
    // Sample neighbor phases
    float phaseSum = 0.0;
    float neighborCount = 0.0;
    
    // Sample in a 3x3 neighborhood
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            // Skip center point (self)
            if (i == 0 && j == 0) continue;
            
            vec2 offset = vec2(float(i), float(j)) * texelSize;
            vec2 neighborUV = vUv + offset;
            
            // Skip if outside texture bounds
            if (neighborUV.x < 0.0 || neighborUV.x > 1.0 || 
                neighborUV.y < 0.0 || neighborUV.y > 1.0) continue;
            
            float neighborPhase = texture2D(stateTexture, neighborUV).r;
            
            // Compute phase difference
            float diff = phaseDifference(neighborPhase, phase);
            
            // Sum up the sine of phase differences (Kuramoto model)
            phaseSum += sin(diff);
            neighborCount += 1.0;
        }
    }
    
    // If we have neighbors, compute coupling effect
    float couplingEffect = 0.0;
    if (neighborCount > 0.0) {
        couplingEffect = coupling * phaseSum / neighborCount;
    }
    
    // Add some noise
    float noise = (rand(vUv + vec2(time * 0.01, time * 0.007)) * 2.0 - 1.0) * noiseStrength;
    
    // Update phase using Kuramoto model
    float phaseVelocity = frequency + couplingEffect + noise;
    float newPhase = phase + phaseVelocity * deltaTime;
    
    // Wrap phase to [0, 2*PI]
    newPhase = mod(newPhase, TWO_PI);
    
    // Calculate local synchronization based on neighbor phase coherence
    float sinSum = 0.0;
    float cosSum = 0.0;
    
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            vec2 offset = vec2(float(i), float(j)) * texelSize;
            vec2 neighborUV = vUv + offset;
            
            // Skip if outside texture bounds
            if (neighborUV.x < 0.0 || neighborUV.x > 1.0 || 
                neighborUV.y < 0.0 || neighborUV.y > 1.0) continue;
            
            float neighborPhase = texture2D(stateTexture, neighborUV).r;
            
            // Sum sin and cos components for order parameter calculation
            sinSum += sin(neighborPhase);
            cosSum += cos(neighborPhase);
        }
    }
    
    // Calculate order parameter (normalized magnitude of phase vectors)
    float r = sqrt(sinSum * sinSum + cosSum * cosSum) / 25.0; // 5x5 grid = 25 oscillators
    
    // Smooth synchronization value for visualization stability
    float newSynchronization = mix(synchronization, r, 0.1);
    
    // Output updated state
    gl_FragColor = vec4(newPhase, frequency, newSynchronization, 1.0);
}