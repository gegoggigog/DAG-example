#pragma once
#include <string>
#include <glad/gl.h>

using gl_string = std::basic_string<GLchar>;
inline std::string getShaderLog(GLuint shader_id)
{
    GLint bufflen{ 0 };
    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &bufflen);
    if (bufflen > 1) {
        std::vector<GLchar> log_string(static_cast<std::size_t>(bufflen + 1));
        glGetShaderInfoLog(shader_id, bufflen, nullptr, log_string.data());
        return std::string{ log_string.begin(), log_string.end() };
    }
    return "";
}

inline std::string getProgramLog(GLuint program_id)
{
    GLint bufflen{ 0 };
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &bufflen);
    if (bufflen > 1) {
        std::vector<GLchar> log_string(static_cast<std::size_t>(bufflen + 1));
        glGetProgramInfoLog(program_id, bufflen, nullptr, log_string.data());
        return std::string{ log_string.begin(), log_string.end() };
    }
    return "";
}

inline gl_string get_file_contents(const std::string& filename)
{
    if (auto ifs = std::ifstream{ filename }) {
        using it = std::istreambuf_iterator<gl_string::value_type>;
        return gl_string{ it{ ifs }, it{} };
    }
    throw std::runtime_error{ std::string{ "Failed to read file: " } + filename };
}

inline GLuint compile_shader(GLenum shader_type, const GLchar* source)
{
    GLuint shader_id = glCreateShader(shader_type);
    glShaderSource(shader_id, 1, &source, nullptr);
    glCompileShader(shader_id);

    GLint compile_status = GL_FALSE;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compile_status);
    const auto shader_log = getShaderLog(shader_id);
    if (compile_status != GL_TRUE)
    {
        std::cerr << "Shader failed to compile:\n" << source << '\n' << shader_log;
        glDeleteShader(shader_id);
        throw std::runtime_error("Failed to compile shaders");
    }
    std::cout << shader_log;
    return shader_id;
}

inline GLuint link_shaders_vs_fs(GLuint vs, GLuint fs) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);

    glLinkProgram(program);
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (linkStatus != GL_TRUE) {
        std::cout << "Shader failed to link: " << getProgramLog(program);
        glDeleteProgram(program);
        throw std::runtime_error("Failed to link shaders");
    }

    return program;
}
