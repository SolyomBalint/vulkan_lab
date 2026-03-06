#version 450

#include "forward_common.glsl"

// vertex inputs
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;

// uniform data
layout(set = 2, binding = 0) uniform UniformBufferObject {
    PerObjectUniformBufferObject perObject;
};

// vertex shader outputs == fragment shader inputs
layout(location = 0) out vec3 viewSpacePosition;
layout(location = 1) out vec3 viewSpaceNormal;
layout(location = 2) out vec2 texCoord;
layout(location = 3) out mat3 TBN;

void main() {
    viewSpacePosition = (perObject.mv * vec4(inPosition, 1)).xyz;
    mat3 normalMx = mat3(perObject.mvInvT);
    vec3 N = normalize(normalMx * inNormal);
    vec3 T = normalize(normalMx * inTangent.xyz);
    T = normalize(T - dot(T, N) * N);
    vec3 B = normalize(cross(N, T) * inTangent.w);

    viewSpaceNormal = N;
    texCoord = inTexCoord;
    TBN = mat3(T, B, N);
    gl_Position = perObject.mvp * vec4(inPosition, 1.0);
}
