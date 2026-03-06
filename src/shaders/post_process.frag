#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 3) uniform sampler2D sceneTexture;

void main() {
    vec2 texelSize = 1.0 / textureSize(sceneTexture, 0);

    // 5x5 Gaussian blur kernel
    vec3 result = vec3(0.0);
    float weights[5] = float[](0.06136, 0.24477, 0.38774, 0.24477, 0.06136);

    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            float w = weights[x + 2] * weights[y + 2];
            result += texture(sceneTexture, uv + offset).rgb * w;
        }
    }

    outColor = vec4(result, 1.0);
}
