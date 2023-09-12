#pragma once
#include <array>
#include <functional>
#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>

#include "utils/Aabb.h"
#include "utils/view.h"

#ifndef FRAMEBUFFER_PROGRAMMABLE_SAMPLE_LOCATIONS_NV
#define FRAMEBUFFER_PROGRAMMABLE_SAMPLE_LOCATIONS_NV 0x9342
#endif

namespace voxelizer
{
	// This rasterizer method call an arbitrary draw function.. something something we need to pass modelmatrix.
	using Drawfunc = void(void* data, GLuint uniform_location_modelMatrix);
	struct VoxelizeShader {
		GLuint program              = 0xFFFFFFFF;
		GLuint uniform_proj_x       = 0xFFFFFFFF;
		GLuint uniform_proj_y       = 0xFFFFFFFF;
		GLuint uniform_proj_z       = 0xFFFFFFFF;
		GLuint uniform_grid_dim     = 0xFFFFFFFF;
		GLuint uniform_aabb_size    = 0xFFFFFFFF;
		GLuint uniform_model_matrix = 0xFFFFFFFF;
	};
	struct Context
	{
		GLuint m_frag_ctr_buffer = 0xFFFFFFFF;
		GLuint m_dummy_fbo = 0xFFFFFFFF;
		GLuint tex = 0xFFFFFFFF;
		VoxelizeShader voxelize_shader;

		int m_tex_dim = 128 * 1024 * 1024;
		GLuint m_grid_size = 0xFFFFFFFF;

		GLuint m_position_ssbo = 0xFFFFFFFF;
		//GLuint m_base_color_ssbo = 0xFFFFFFFF;
		GLuint m_mask_ssbo = 0xFFFFFFFF;
		GLuint m_frag_count = 0xFFFFFFFF;
		GLuint m_num_colors = 0xFFFFFFFF;

		bool SUPPORTED_GL_NV_conservative_raster = false;
	};
	Context create_context(unsigned grid_size);
	void destroy_context(Context& ctx);
	uint32_t generate_voxels(Context& ctx, const chag::Aabb aabb, int grid_resolution, Drawfunc func, void* data);
}  // namespace voxelizer
