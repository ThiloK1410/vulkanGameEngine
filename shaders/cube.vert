#version 450

layout(push_constant) uniform Push {
    mat4 mvp;
} push;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = push.mvp * vec4(inPosition, 1.0);
}
