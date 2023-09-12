#pragma once
#include <glad/gl.h>
inline constexpr GLchar copy_vert_src[] = R"(
#version 400 compatibility
out vec2 texcoord;
void main() {
   if(gl_VertexID == 0){ texcoord = vec2(0.0, 2.0); }
   if(gl_VertexID == 1){ texcoord = vec2(0.0, 0.0); }
   if(gl_VertexID == 2){ texcoord = vec2(2.0, 0.0); }
   if(gl_VertexID == 0){ gl_Position = vec4(-1.0,  3.0, 0.0, 1.0); }
   if(gl_VertexID == 1){ gl_Position = vec4(-1.0, -1.0, 0.0, 1.0); }
   if(gl_VertexID == 2){ gl_Position = vec4( 3.0, -1.0, 0.0, 1.0); }
}
)";

inline constexpr GLchar copy_frag_src[] = R"(
#version 400 compatibility
in vec2 texcoord;
uniform sampler2D renderbuffer;
void main() {
    gl_FragColor.xyz = texture(renderbuffer, texcoord).xyz;
}
)";