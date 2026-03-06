#version 450

#include "forward_common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;

layout(set = 2, binding = 0) uniform UniformBufferObject {
    PerObjectUniformBufferObject perObject;
};

layout(location = 0) out vec3 viewSpacePosition;
layout(location = 1) out vec3 viewSpaceNormal;
layout(location = 2) out vec2 texCoord;
layout(location = 3) out mat3 TBN;

void main() {
    viewSpacePosition = (perObject.mv * vec4(inPosition, 1)).xyz;
    viewSpaceNormal = (perObject.mvInvT * vec4(inNormal, 0)).xyz;
    texCoord = inTexCoord;

    vec3 N = normalize((perObject.mvInvT * vec4(inNormal, 0)).xyz);
    vec3 T = normalize((perObject.mvInvT * vec4(inTangent.xyz, 0)).xyz);
    vec3 B = cross(N, T) * inTangent.w;
    TBN = mat3(T, B, N);

    gl_Position = perObject.mvp * vec4(inPosition, 1.0);
}
