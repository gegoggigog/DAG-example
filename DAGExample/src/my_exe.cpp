#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <stack>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <glm/gtc/type_ptr.hpp>

#include "utils/DAG.h"
#include "DAGLoader/DAGLoader.h"
#include "DAGTracer/DAGTracer.h"

#include "utils/glTFLoader/glTFLoader.h"
#include "voxelize_and_merge.h"
#include "ColorCompression/ours_varbit.h"
#include "tracy/Tracy.hpp"

#include "utils/shader_helpers.hpp"
#include "utils/gl_debug_callback.hpp"

#include "shader_sources.hpp"
#include "app.hpp"

using glm::ivec2;
using glm::vec2;
using glm::vec3;
using glm::vec4;

ivec2 screen_dim{1024, 1024};
GLuint copy_shader{0};
GLint renderbuffer_uniform{-1};

constexpr bool load_cached{ false };
constexpr bool load_compressed{ false };

const char* dag_file              = R"(cache/dag16k.bin)";
//const char* raw_color_file        = R"(cache/raw16k.bin)";
const char* raw_color_file        = R"(raw16k.bin)";
const char* compressed_color_file = R"(cache/compressed16k.bin)";

template<typename T>
static void write_vector_to_disc(const std::string file, const std::vector<T>& vec)
{
	if (auto ofs = std::ofstream{ file, std::ofstream::binary | std::ofstream::out }; ofs) {
		ofs.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
	}
	else throw "Failed to open file";
}

// NOTE: "T* &d_vec" - We do **not** want a copy of the pointer.
template<typename T>
void upload_vector(const std::vector<T>& h_vec, T*& d_vec) {
	const size_t count = h_vec.size() * sizeof(T);
	cudaMallocManaged(&d_vec, count);

	printf("Allocating %zu things (%fMiB)\n", h_vec.size(), h_vec.size() * sizeof(T) / double(1 << 20));
	cudaMemcpy(d_vec, h_vec.data(), count, cudaMemcpyHostToDevice);
}

// NOTE: "T* &d_vec" - We do **not** want a copy of the pointer.
template<typename T>
void free_and_clear_cudaptr(T*& ptr) {
	if (ptr) {
		cudaFree(ptr);
		ptr = nullptr;
	}
}

void free_gpu(ColorData* dst) {
	free_and_clear_cudaptr(dst->d_block_headers);
	free_and_clear_cudaptr(dst->d_block_colors);
	free_and_clear_cudaptr(dst->d_weights);
	free_and_clear_cudaptr(dst->d_macro_w_offset);
}

void upload_to_gpu(const ours_varbit::OursData& src, ColorData *dst) {
	free_gpu(dst);
	upload_vector(src.h_block_headers,  dst->d_block_headers);
	upload_vector(src.h_block_colors,   dst->d_block_colors);
	upload_vector(src.h_weights,        dst->d_weights);
	upload_vector(src.h_macro_w_offset, dst->d_macro_w_offset);
	dst->nof_blocks = src.nof_blocks;
	dst->nof_colors = src.nof_colors;
}


static void error_callback(int error, const char* description)
{
	std::cerr << "Error<" << error << ">: " << description << '\n';
}
int main(int argc, char* argv[]) {
	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwSetErrorCallback(error_callback);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	GLFWwindow* window = glfwCreateWindow(screen_dim.x, screen_dim.y, "DAG Example", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	gladLoadGL(glfwGetProcAddress);

	// No v-sync.
	glfwSwapInterval(0);
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

	AppState app;
	app.camera.lookAt(vec3{ 0.0f, 1.0f, 0.0f }, vec3{ 0.0f, 0.0f, 0.0f });
	app.window = window;

	{
		GLuint vs = compile_shader(GL_VERTEX_SHADER, copy_vert_src);
		GLuint fs = compile_shader(GL_FRAGMENT_SHADER, copy_frag_src);
		copy_shader = link_shaders_vs_fs(vs, fs);
		glDeleteShader(vs);
		glDeleteShader(fs);
	}

	renderbuffer_uniform = glGetUniformLocation(copy_shader, "renderbuffer");

	ZoneScoped;
	
	auto dag = [] {
		constexpr int dag_resolution{ 1 << 8 };
		std::cout << "Resolution: " << dag_resolution << std::endl;
		auto optional_dag = DAG_from_scene(dag_resolution, R"(C:\Users\dan\garbage_collector\DAG_Compression\assets\Sponza\glTF\)", "Sponza.gltf");
		if (!optional_dag) {
			std::cerr << "Could not construct dag, assert file path.";
			exit(-1);
		}
		return *optional_dag;
	}();

	write_vector_to_disc(raw_color_file, dag.m_base_colors);
	ours_varbit::OursData compressed_color = ours_varbit::compressColors(disc_vector<uint32_t>{ raw_color_file, macro_block_size }, 0.05f, ColorLayout::RGB_5_6_5);

	DAGTracer dag_tracer;
	dag_tracer.resize(screen_dim.x, screen_dim.y);

	upload_to_gpu(compressed_color, &dag_tracer.m_compressed_colors);
	upload_to_gpu(dag);

	while (!glfwWindowShouldClose(window))
	{
		app.frame_timer.start();
		app.handle_events();


		const int color_lookup_lvl = dag.nofGeometryLevels();
		dag_tracer.resolve_paths(dag, app.camera, color_lookup_lvl);
		dag_tracer.resolve_colors(dag, color_lookup_lvl);

		glViewport(0, 0, screen_dim.x, screen_dim.y);
		glUseProgram(copy_shader);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, dag_tracer.m_color_buffer.m_gl_idx);
		glUniform1i(renderbuffer_uniform, 0);
		glActiveTexture(GL_TEXTURE0);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		glfwSwapBuffers(window);
		app.frame_timer.end();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	return EXIT_SUCCESS;
}

//// Load from file
//#include "DAGLoader/DAGLoader.h"
//#include "DAGConstructor/DAGConstructor.h"
//#include "morton.h"
//...
//constexpr int GRID_RESOLUTION = 512;
//constexpr float GRID_RESOLUTION_FLOAT = static_cast<float>(GRID_RESOLUTION);
//...
//bool load_entire_dag{false};
//dag::DAG dag;
//if (load_entire_dag) 
//{
//	dag          = dag::cerealization::bin::load("../../cache/dag.bin");
//	dag.m_colors = dag::cerealization::bin::load_vec<uint32_t>("../../cache/colors.raw.bin");
//}
//else
//{
//	auto points = dag::cerealization::bin::load_vec<glm::vec3>("../../cache/positions");
//	auto colors = dag::cerealization::bin::load_vec<float>("../../cache/colors");
//
//	auto make_square_aabb = [](chag::Aabb aabb) {
//		const glm::vec3 hsz    = aabb.getHalfSize();
//		const glm::vec3 centre = aabb.getCentre();
//		const glm::vec3 c{glm::max(hsz.x, glm::max(hsz.y, hsz.z))};
//		aabb.min = centre - c;
//		aabb.max = centre + c;
//		return aabb;
//	};
//
//	chag::Aabb aabb = std::accumulate(
//			begin(points), end(points),
//			chag::make_aabb(vec3{std::numeric_limits<float>::max()}, vec3{std::numeric_limits<float>::lowest()}),
//			[](const chag::Aabb &lhs, const vec3 &rhs) {
//				chag::Aabb result;
//				result.min.x = std::min(lhs.min.x, rhs.x);
//				result.min.y = std::min(lhs.min.y, rhs.y);
//				result.min.z = std::min(lhs.min.z, rhs.z);
//
//				result.max.x = std::max(lhs.max.x, rhs.x);
//				result.max.y = std::max(lhs.max.y, rhs.y);
//				result.max.z = std::max(lhs.max.z, rhs.z);
//				return result;
//			});
//
//	chag::Aabb square_aabb = make_square_aabb(aabb);
//
//	std::vector<uint32_t> morton(points.size());
//	std::transform(begin(points), end(points), begin(morton), [square_aabb](const vec3 &pos) {
//					// First make sure the positions are in the range [0, GRID_RESOLUTION-1].
//		const vec3 corrected_pos = clamp(
//			GRID_RESOLUTION_FLOAT * ((pos - square_aabb.min) / (square_aabb.max - square_aabb.min)), 
//			vec3(0.0f), 
//			vec3(GRID_RESOLUTION_FLOAT - 1.0f)
//		);
//		return morton_encode_32(
//			static_cast<uint32_t>(corrected_pos.x),
//			static_cast<uint32_t>(corrected_pos.y),
//			static_cast<uint32_t>(corrected_pos.z)
//		);
//	});
//
//	// Need to make sure colors and morton key and colors are sorted according to morton.
//	{
//		struct sort_elem 
//		{
//			uint32_t morton;
//			vec4 color;
//		};
//		std::vector<sort_elem> sortme(morton.size());
//		for(size_t i{0}; i<sortme.size(); ++i)
//		{
//			sortme[i].morton = morton[i];
//			sortme[i].color  = vec4{ colors[4 * i + 0], colors[4 * i + 1], colors[4 * i + 2], colors[4 * i + 3] };
//		}
//		std::sort(begin(sortme), end(sortme), [](const sort_elem &lhs, const sort_elem &rhs){ return lhs.morton < rhs.morton; });
//		for (size_t i{0}; i < sortme.size(); ++i) 
//		{
//			morton[i] = sortme[i].morton;
//			colors[4 * i + 0] = sortme[i].color.x;
//			colors[4 * i + 1] = sortme[i].color.y;
//			colors[4 * i + 2] = sortme[i].color.z;
//			colors[4 * i + 3] = sortme[i].color.w;
//		}
//	}
//	DAGConstructor tmp;
//	// The log2(GRID_RESOLUTION / 4) + 0 is because we use 4x4x4 leafs instead of 2x2x2. 
//	// The final + 0 is a placeholder for when we need to merge dags.
//	dag = tmp.build_dag(morton, colors, static_cast<int>(morton.size()), static_cast<int>(log2(GRID_RESOLUTION / 4) + 0), square_aabb);
//}
