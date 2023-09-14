#include "Voxelizer.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <string>

#include "utils/shader_helpers.hpp"

#include "voxelize.vert"
#include "voxelize.geom"
#include "voxelize.frag"

namespace voxelizer {
GLuint link_shaders(GLuint vs, GLuint fs, GLuint gs)
{
	GLuint program_id = glCreateProgram();
	glAttachShader(program_id, vs);
	glAttachShader(program_id, fs);
	glAttachShader(program_id, gs);

	glProgramParameteri(program_id, GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
	glProgramParameteri(program_id, GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
	glProgramParameteri(program_id, GL_GEOMETRY_VERTICES_OUT_EXT, 3);
	
	glLinkProgram(program_id);

	GLint linkStatus = GL_FALSE;
	glGetProgramiv(program_id, GL_LINK_STATUS, &linkStatus);

	if (linkStatus != GL_TRUE) { throw std::runtime_error("Failed to link shaders"); }

	return program_id;
}

Context create_context(unsigned grid_size)
{
	assert(grid_size > 0 /* && ctx.m_grid_size < 1025*/);
	Context ctx;
	ctx.m_grid_size = grid_size;

	// Check for extension GL_NV_conservative_raster
	{
		GLint no_of_extensions = 0;
		glGetIntegerv(GL_NUM_EXTENSIONS, &no_of_extensions);
		std::set<std::string> ogl_extensions;
		for (int i = 0; i < no_of_extensions; ++i)
			ogl_extensions.insert((const char*)glGetStringi(GL_EXTENSIONS, i));
		ctx.SUPPORTED_GL_NV_conservative_raster = ogl_extensions.find("GL_NV_conservative_raster") != ogl_extensions.end();
		std::cout << "GL_NV_conservative_raster: " << (ctx.SUPPORTED_GL_NV_conservative_raster ? "SUPPORTED!\n" : "NOT SUPPORTED!\n");
	}

	// Create shader
	{
		GLuint vs_shader = compile_shader(GL_VERTEX_SHADER, voxelize_vert_src);
		GLuint fs_shader = compile_shader(GL_FRAGMENT_SHADER, voxelize_frag_src);
		GLuint gs_shader = compile_shader(GL_GEOMETRY_SHADER, voxelize_geom_src);
		ctx.voxelize_shader.program = link_shaders(vs_shader, fs_shader, gs_shader);
		ctx.voxelize_shader.uniform_aabb_size    = glGetUniformLocation(ctx.voxelize_shader.program, "aabb_size");
		ctx.voxelize_shader.uniform_grid_dim     = glGetUniformLocation(ctx.voxelize_shader.program, "grid_dim");
		ctx.voxelize_shader.uniform_proj_x       = glGetUniformLocation(ctx.voxelize_shader.program, "proj_x");
		ctx.voxelize_shader.uniform_proj_y       = glGetUniformLocation(ctx.voxelize_shader.program, "proj_y");
		ctx.voxelize_shader.uniform_proj_z       = glGetUniformLocation(ctx.voxelize_shader.program, "proj_z");
		ctx.voxelize_shader.uniform_model_matrix = glGetUniformLocation(ctx.voxelize_shader.program, "model_matrix");
		glDeleteShader(vs_shader);
		glDeleteShader(fs_shader);
		glDeleteShader(gs_shader);
	}

	{
		// Atomic counter.
		glGenBuffers(1, &ctx.m_frag_ctr_buffer);
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, ctx.m_frag_ctr_buffer);
		glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_READ);
		GLuint just_zero(0);
		glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &just_zero);
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

		// Data buffer (pos).
		glGenBuffers(1, &ctx.m_position_ssbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx.m_position_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, ctx.m_tex_dim * sizeof(uint64_t), NULL, GL_STATIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

#if 0
		// Data buffer (base color)
		glGenBuffers(1, &m_base_color_ssbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_base_color_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, m_tex_dim * sizeof(uint32_t), NULL, GL_STATIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
#endif

		// Dummy framebuffer.
		glGenFramebuffers(1, &ctx.m_dummy_fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, ctx.m_dummy_fbo);
		glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, ctx.m_grid_size);
		glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, ctx.m_grid_size);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	return ctx;
}

uint32_t generate_voxels(Context& ctx, const chag::Aabb aabb, int grid_resolution, Drawfunc draw_func, void* draw_func_user_data)
{
	// Get ortho camera for given axis.
	enum class Axis { X, Y, Z };
	auto get_camera = [&aabb](Axis axis) {
		// clang-format off
		const glm::vec3 pos = axis == Axis::X ?    aabb.getCentre() - aabb.getHalfSize().x * glm::vec3{ 1.f, 0.f, 0.f } :
			                  axis == Axis::Y ?    aabb.getCentre() - aabb.getHalfSize().y * glm::vec3{ 0.f, 1.f, 0.f } :
			               /* axis == Axis::Z ? */ aabb.getCentre() - aabb.getHalfSize().z * glm::vec3{ 0.f, 0.f, 1.f };
		const glm::vec3 up = axis == Axis::X ?    glm::vec3{ 0.f, 1.f, 0.f } :
			                 axis == Axis::Y ?    glm::vec3{ 0.f, 0.f, 1.f } :
			              /* axis == Axis::Z ? */ glm::vec3{ 1.f, 0.f, 0.f };
		// clang-format on

		// Figure out clipping planes.
		const std::array<const glm::vec3, 8> points{
			glm::vec3{ aabb.min.x, aabb.max.y, aabb.min.z }, glm::vec3{ aabb.min.x, aabb.max.y, aabb.max.z },
			glm::vec3{ aabb.min.x, aabb.min.y, aabb.min.z }, glm::vec3{ aabb.min.x, aabb.min.y, aabb.max.z },
			glm::vec3{ aabb.max.x, aabb.max.y, aabb.min.z }, glm::vec3{ aabb.max.x, aabb.max.y, aabb.max.z },
			glm::vec3{ aabb.max.x, aabb.min.y, aabb.min.z }, glm::vec3{ aabb.max.x, aabb.min.y, aabb.max.z } };

		float min_x = std::numeric_limits<float>::max();
		float min_y = std::numeric_limits<float>::max();
		float min_z = std::numeric_limits<float>::max();
		float max_x = std::numeric_limits<float>::lowest();
		float max_y = std::numeric_limits<float>::lowest();
		float max_z = std::numeric_limits<float>::lowest();

		chag::orthoview result;
		result.lookAt(pos, aabb.getCentre(), up);
		{
			const glm::mat4 MV = result.get_MV();
			for (const auto& v : points) {
				const glm::vec4 vec = MV * glm::vec4{ v, 1.0f };

				min_x = std::min(min_x, vec.x);
				min_y = std::min(min_y, vec.y);
				min_z = std::min(min_z, vec.z);
				max_x = std::max(max_x, vec.x);
				max_y = std::max(max_y, vec.y);
				max_z = std::max(max_z, vec.z);
			}
		}

		result.m_right = max_x;
		result.m_left = min_x;
		result.m_top = max_y;
		result.m_bottom = min_y;
		// TODO: Remember my reason for this..
		result.m_far = min_z;
		result.m_near = max_z;

		return result;
		};
	const chag::orthoview o_x = get_camera(Axis::X);
	const chag::orthoview o_y = get_camera(Axis::Y);
	const chag::orthoview o_z = get_camera(Axis::Z);

	glUseProgram(ctx.voxelize_shader.program);
	glUniformMatrix4fv(ctx.voxelize_shader.uniform_proj_x, 1, false, glm::value_ptr(o_x.get_MVP()));
	glUniformMatrix4fv(ctx.voxelize_shader.uniform_proj_y, 1, false, glm::value_ptr(o_y.get_MVP()));
	glUniformMatrix4fv(ctx.voxelize_shader.uniform_proj_z, 1, false, glm::value_ptr(o_z.get_MVP()));
	glUniform1i(ctx.voxelize_shader.uniform_grid_dim, grid_resolution);
	glUniform3fv(ctx.voxelize_shader.uniform_aabb_size, 1, glm::value_ptr(aabb.getHalfSize()));

	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, ctx.m_frag_ctr_buffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ctx.m_position_ssbo);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_base_color_ssbo);

	glBindFramebuffer(GL_FRAMEBUFFER, ctx.m_dummy_fbo);
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glClearColor(1, 0, 0, 1);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	{
		glViewport(0, 0, grid_resolution, grid_resolution);
		glDisable(GL_CULL_FACE);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_MULTISAMPLE);

		if (ctx.SUPPORTED_GL_NV_conservative_raster) glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			draw_func(draw_func_user_data, ctx.voxelize_shader.uniform_model_matrix);
		}
		if (ctx.SUPPORTED_GL_NV_conservative_raster) glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Get number of written fragments.
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, ctx.m_frag_ctr_buffer);
	GLuint frag_count = 0;
	glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &frag_count);
	uint32_t zero_counter(0);
	glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &zero_counter);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
	glUseProgram(0);

	//assert(frag_count < ctx.m_tex_dim);
	return frag_count;
}

void destroy_context(Context& ctx)
{
	glDeleteBuffers(1, &ctx.m_frag_ctr_buffer);
	glDeleteBuffers(1, &ctx.m_position_ssbo);
	//glDeleteBuffers(1, &ctx.m_base_color_ssbo);
	glDeleteBuffers(1, &ctx.m_dummy_fbo);
}
}  // namespace voxelizer
