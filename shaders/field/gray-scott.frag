// Gray-Scott Reaction-Diffusion System
//
// This shader implements the Gray-Scott reaction-diffusion model:
// dA/dt = Da * Laplacian(A) - A*B^2 + f * (1 - A)
// dB/dt = Db * Laplacian(B) + A*B^2 - (f + k) * B
//
// Where:
// - A and B are chemical concentrations
// - Da and Db are diffusion rates
// - f is the feed rate
// - k is the kill rate

varying vec2 vUv;
uniform sampler2D stateTexture;
uniform sampler2D orientationTexture;
uniform sampler2D stabilityTexture;
uniform float time;
uniform float deltaTime;
uniform float resolution;

// Parameters
uniform float feedRate;
uniform float killRate;
uniform float diffusionRateA;
uniform float diffusionRateB;
uniform float timestep;

// Compute Laplacian using a 3x3 filter kernel
vec2 laplacian(sampler2D tex, vec2 uv) {
    float texelSize = 1.0 / resolution;
    
    // 3x3 convolution kernel approximating the Laplacian
    vec2 sum = vec2(0.0);
    
    // Center point (weighted by -4 in Laplacian kernel)
    vec2 center = texture2D(tex, uv).rg;
    sum += center * -4.0;
    
    // Adjacent points (weighted by 1 in Laplacian kernel)
    sum += texture2D(tex, uv + vec2(texelSize, 0.0)).rg;  // Right
    sum += texture2D(tex, uv + vec2(-texelSize, 0.0)).rg; // Left
    sum += texture2D(tex, uv + vec2(0.0, texelSize)).rg;  // Up
    sum += texture2D(tex, uv + vec2(0.0, -texelSize)).rg; // Down
    
    return sum;
}

void main() {
    // Get current state values
    vec4 state = texture2D(stateTexture, vUv);
    float a = state.r; // Chemical A concentration
    float b = state.g; // Chemical B concentration
    
    // Compute Laplacian for diffusion terms
    vec2 lap = laplacian(stateTexture, vUv);
    
    // Reaction-diffusion equations
    float reaction = a * b * b;
    
    // dA/dt = Da * Laplacian(A) - A*B^2 + f * (1 - A)
    float da = diffusionRateA * lap.r - reaction + feedRate * (1.0 - a);
    
    // dB/dt = Db * Laplacian(B) + A*B^2 - (f + k) * B
    float db = diffusionRateB * lap.g + reaction - (feedRate + killRate) * b;
    
    // Update concentrations using forward Euler method
    float newA = a + da * timestep * deltaTime;
    float newB = b + db * timestep * deltaTime;
    
    // Clamp values to valid range [0, 1]
    newA = clamp(newA, 0.0, 1.0);
    newB = clamp(newB, 0.0, 1.0);
    
    // Output updated state
    gl_FragColor = vec4(newA, newB, 0.0, 1.0);
}