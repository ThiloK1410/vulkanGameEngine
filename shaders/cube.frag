#version 450

layout(push_constant) uniform Push {
    mat4 mvp;   // occupied by vertex shader, but must be declared for correct offset
    vec4 color;
} push;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = push.color;
}
