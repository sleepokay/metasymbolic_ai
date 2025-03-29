// Common GLSL functions and utilities for field simulations

// Constants
#define PI 3.1415926535897932384626433832795
#define TWO_PI 6.2831853071795864769252867665590

// Random function based on the state
float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

// 2D Perlin noise (simplified)
float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    
    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));
    
    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    // Mix 4 corners
    return mix(
        mix(a, b, u.x),
        mix(c, d, u.x),
        u.y
    );
}

// Smooth step function with smoother transition
float smootherstep(float edge0, float edge1, float x) {
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

// Map a value from one range to another
float map(float value, float inMin, float inMax, float outMin, float outMax) {
    return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

// Convert HSV to RGB
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Convert RGB to HSV
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// Compute Laplacian using a 3x3 filter kernel
vec4 laplacian(sampler2D tex, vec2 uv, float texelSize) {
    // 3x3 convolution kernel approximating the Laplacian
    vec4 sum = vec4(0.0);
    
    // Center point (weighted by -4 in Laplacian kernel)
    vec4 center = texture2D(tex, uv);
    sum += center * -4.0;
    
    // Adjacent points (weighted by 1 in Laplacian kernel)
    sum += texture2D(tex, uv + vec2(texelSize, 0.0));  // Right
    sum += texture2D(tex, uv + vec2(-texelSize, 0.0)); // Left
    sum += texture2D(tex, uv + vec2(0.0, texelSize));  // Up
    sum += texture2D(tex, uv + vec2(0.0, -texelSize)); // Down
    
    return sum;
}

// Compute gradient using central differences
vec2 gradient(sampler2D tex, vec2 uv, float texelSize, int component) {
    vec2 grad;
    
    // Sample neighboring pixels
    float right = texture2D(tex, uv + vec2(texelSize, 0.0))[component];
    float left = texture2D(tex, uv + vec2(-texelSize, 0.0))[component];
    float up = texture2D(tex, uv + vec2(0.0, texelSize))[component];
    float down = texture2D(tex, uv + vec2(0.0, -texelSize))[component];
    
    // Compute central differences
    grad.x = (right - left) / (2.0 * texelSize);
    grad.y = (up - down) / (2.0 * texelSize);
    
    return grad;
}

// Viridis color map approximation
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

// Plasma color map approximation
vec3 plasma(float t) {
    const vec3 c0 = vec3(0.0506352, 0.0302344, 0.5576087);
    const vec3 c1 = vec3(2.1720491, 1.3118511, 5.4372489);
    const vec3 c2 = vec3(0.5677561, 1.5128502, 0.0673455);
    const vec3 c3 = vec3(-2.7518816, -0.4015211, 4.6661530);
    const vec3 c4 = vec3(-0.1984204, -1.0652691, -3.8271928);
    const vec3 c5 = vec3(2.6297774, 0.4282363, -0.1164618);
    
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
}

// Inferno color map approximation
vec3 inferno(float t) {
    const vec3 c0 = vec3(0.0002189403, 0.0002050442, 0.0136440758);
    const vec3 c1 = vec3(0.1057264752, 0.0568058041, 1.0000971986);
    const vec3 c2 = vec3(2.5880021273, 1.4861939986, -0.7017955726);
    const vec3 c3 = vec3(-2.7702232991, -2.9009746027, -2.4440126480);
    const vec3 c4 = vec3(1.3144556824, 1.8495966508, 0.0476831077);
    const vec3 c5 = vec3(-0.1986710240, -0.4127034217, 2.0086226704);
    
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
}