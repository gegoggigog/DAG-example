#pragma once
#include <string>
#include <glad/gl.h>

using gl_string = std::basic_string<GLchar>;
gl_string get_file_contents(const std::string& filename);
GLuint compile_shader(GLenum shader_type, const GLchar* source);
GLuint link_shaders_vs_fs(GLuint vs, GLuint fs);
