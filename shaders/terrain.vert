#version 450

layout(push_constant) uniform Push {
    mat4 mvp;
    mat4 model;
    vec4 color;
} push;

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inNormal;

layout(location = 0) out vec3 fragNormal;

void main() {
    gl_Position = push.mvp * vec4(inPosition.xyz, 1.0);
    fragNormal = mat3(push.model) * inNormal.xyz;
}