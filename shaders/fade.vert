#version 450

void main() {
    // generate a fullscreen triangle from vertex index alone (no vertex buffer needed)
    // indices 0,1,2 produce vertices at (-1,-1), (3,-1), (-1,3) covering the whole screen
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}