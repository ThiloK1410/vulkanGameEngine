#version 450

layout(push_constant) uniform Push {
    mat4 mvp;
    mat4 model;
    vec4 color;
} push;

layout(location = 0) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5));
    vec3 n = normalize(fragNormal);
    float diffuse = max(dot(n, lightDir), 0.0);
    float ambient = 0.15;
    outColor = vec4(push.color.rgb * (ambient + diffuse * 0.85), 1.0);
}