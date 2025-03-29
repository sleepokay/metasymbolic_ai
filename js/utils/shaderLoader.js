/**
 * Utility class for loading and managing GLSL shaders
 */
class ShaderLoader {
    constructor() {
        this.shaders = {};
        this.loadPromises = {};
    }

    /**
     * Load a shader file from the specified URL
     * @param {string} name - The name to reference the shader by
     * @param {string} url - The URL of the shader file
     * @returns {Promise} A promise that resolves when the shader is loaded
     */
    load(name, url) {
        // If we've already started loading this shader, return the existing promise
        if (this.loadPromises[name]) {
            return this.loadPromises[name];
        }

        // Start a new load request
        const promise = fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to load shader ${name} from ${url}: ${response.statusText}`);
                }
                return response.text();
            })
            .then(text => {
                this.shaders[name] = text;
                return text;
            })
            .catch(error => {
                console.error(`Error loading shader ${name}:`, error);
                throw error;
            });

        this.loadPromises[name] = promise;
        return promise;
    }

    /**
     * Get a loaded shader by name
     * @param {string} name - The name of the shader to retrieve
     * @returns {string} The shader source code
     */
    get(name) {
        if (!this.shaders[name]) {
            throw new Error(`Shader ${name} has not been loaded`);
        }
        return this.shaders[name];
    }

    /**
     * Check if a shader has been loaded
     * @param {string} name - The name of the shader to check
     * @returns {boolean} True if the shader has been loaded
     */
    isLoaded(name) {
        return !!this.shaders[name];
    }

    /**
     * Load multiple shaders and wait for all to complete
     * @param {Object} shaderMap - Map of shader names to URLs
     * @returns {Promise} A promise that resolves when all shaders are loaded
     */
    loadAll(shaderMap) {
        const promises = Object.entries(shaderMap).map(
            ([name, url]) => this.load(name, url)
        );
        return Promise.all(promises);
    }

    /**
     * Process shader source to include other shader chunks
     * @param {string} source - The shader source code
     * @returns {string} The processed shader with includes resolved
     */
    processIncludes(source) {
        const includeRegex = /#include\s+<(.+)>/g;
        let match;
        let result = source;

        while ((match = includeRegex.exec(source)) !== null) {
            const includeName = match[1];
            if (!this.isLoaded(includeName)) {
                console.warn(`Shader include ${includeName} has not been loaded`);
                continue;
            }

            const includeSource = this.get(includeName);
            result = result.replace(match[0], includeSource);
        }

        return result;
    }
}

// Create and export a singleton instance
const shaderLoader = new ShaderLoader();