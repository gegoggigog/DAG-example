#include <iostream>
#include <glad/gl.h>
//#include "../CudaHelpers.h" //FIXME: Proper search paths
#include <GLFW/glfw3.h>
#include "utils/DAG.h"
//#include "gl_debug_callback.hpp"
#include <voxelize_and_merge.h>

static void error_callback(int error, const char* description)
{
	std::cerr << "Error<" << error << ">: " << description << '\n';
}
int main(int argc, char* argv[]) {
	glfwSetErrorCallback(error_callback);
	if (!glfwInit()) exit(EXIT_FAILURE);


	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	GLFWwindow* window = glfwCreateWindow(1, 1, "...", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	gladLoadGL(glfwGetProcAddress);

	{
		const GLubyte* vendor = glGetString(GL_VENDOR);
		const GLubyte* renderer = glGetString(GL_RENDERER);
		std::cout << "vendor: " << vendor << '\n';
		std::cout << "renderer: " << renderer << '\n';
	}
	//glEnable(GL_DEBUG_OUTPUT);
	//glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	//glDebugMessageCallback(glDebugOutput, nullptr);
	//glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
	constexpr int dag_resolution{ 1 << 12 };
	std::cout << "Resolution: " << dag_resolution << std::endl;
	std::optional<dag::DAG> dag = DAG_from_scene(dag_resolution, R"(C:\Users\dan\garbage_collector\DAG_Compression\assets\Sponza\glTF\)", "Sponza.gltf");
	

glfwDestroyWindow(window);
glfwTerminate();
std::cout << "Program completed successfully\n";
return EXIT_SUCCESS;
}
