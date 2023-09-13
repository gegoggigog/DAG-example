#include "voxelize_and_merge.h"
#include <stack>

#include <glad/gl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "DAG/DAG.h"
#include "DAGConstructor/DAGConstructor.h"
#include "Voxelizer/Voxelizer.h"
#include "glTFLoader/glTFLoader.h"
#ifndef NO_TRACE
#include "tracy/Tracy.hpp"
#else
#define ZoneScoped {}
#define ZoneScopedN(x) {}
#define ZoneText(x, y) {}
#define ZoneName(x,y) {}
#endif // !NO_TRACE

using glm::ivec2;
using glm::vec2;
using glm::vec3;
using glm::vec4;

// In case of morton encoding into 32 bits, this must be at most 1024 (2^(floor(32/3)) == 1024)
// We could go higher here, but this will probably generate more voxels than our output buffer 
// m_tex_dim == 128*1024*1024. 
constexpr int max_subdag_resolution{2048};

chag::Aabb make_square_aabb(chag::Aabb aabb) {
	const glm::vec3 centre        = aabb.getCentre();
	const glm::vec3 half_sz       = aabb.getHalfSize();
	const glm::vec3 max_half_size{ glm::max(half_sz.x, glm::max(half_sz.y, half_sz.z)) };
	aabb.min = centre - max_half_size;
	aabb.max = centre + max_half_size;
	return aabb;
}

struct UserData{
	const glTFLoader::Scene *scene;
	std::vector<std::size_t> renderable_nodes;
};

UserData get_draw_data(glTFLoader::Scene& scene)
{
	UserData result;
	result.scene = &scene;
	// Fetch the nodes we want to render.
	struct StackElement {
		glm::mat4 model_matrix{ 1.0f };
		std::size_t node_id;
	};
	std::stack<StackElement> stack;

	for (const auto& node : scene.scene_nodes_roots) { stack.push({ glm::mat4{1.0f}, node }); }
	while (!stack.empty()) {
		using namespace glTFLoader;
		using F = SceneNode::Flag;
		auto& nodes = scene.scene_nodes;
		StackElement node = stack.top();
		stack.pop();
		SceneNode& scene_node = nodes[node.node_id];

		// If we use the nodes transform, we need to recalculate the AABB.
		// ... So let's not.
		node.model_matrix =
			glm::mat4{ 1.0f };
		//node.model_matrix *
		//glm::translate(glm::mat4{1.0f}, scene_node.translation) *
		//glm::mat4_cast(scene_node.rotation) *
		//glm::scale(glm::mat4{1.0f}, scene_node.scale);

		if (SceneNode::hasProperties(scene_node, F::CHILDREN)) {
			for (std::size_t child_index : scene_node.children) { stack.push({ node.model_matrix, child_index }); }
		}

		if (SceneNode::hasProperties(scene_node, F::MESH)) {
			nodes[node.node_id].cached_transform = node.model_matrix;
			result.renderable_nodes.push_back(node.node_id);
		}
	}
	return result;
}

void draw_gltf_scene(void *draw_data, GLuint unform_location_modelMatrix) {
	const UserData* ud = reinterpret_cast<UserData*>(draw_data);
	const auto &renderable_nodes = ud->renderable_nodes;
	const auto &scene = *ud->scene;
	for (const auto& rel : renderable_nodes) {
		const auto& node = scene.scene_nodes[rel];
		const auto& mesh = scene.meshes[node.mesh_id];
		const auto& prims = mesh.primitives;
		using namespace glTFLoader;
		glBindVertexArray(MASTER_VAO);
		int has_TEXCOORD1 = 0;
		for (const auto& prim : prims) {
			for (const auto& attribute : prim.attributes) {
				if (attribute.index == Attribute::ERROR_TYPE) continue;
				const auto& ac = scene.accsessors[attribute.accsessor];
				const auto& bv = scene.bufferViews[ac.bufferView];
				const auto& bo = scene.buffer_objects[bv.buffer];
				glEnableVertexAttribArray(attribute.index);
				glBindBuffer(GL_ARRAY_BUFFER, bo);
				void* offset = (void*)(ac.byteOffset + bv.byteOffset);
				glVertexAttribPointer(attribute.index, ac.typeSize, ac.componentType, GL_FALSE, 0, offset);
			}

			auto bindTexture = [&](GLuint texture_index, GLint unit)
				{
					if (texture_index != NO_ID_SENTINEL)
					{
						auto texture = scene.textures[texture_index];
						glActiveTexture(GL_TEXTURE0 + unit);
						glBindTexture(GL_TEXTURE_2D, texture);
					}
				};

			glUniformMatrix4fv(unform_location_modelMatrix, 1, false, glm::value_ptr(node.cached_transform));

			const auto& material = scene.materials[prim.material];
			if (material.type == Material::Type::METALLIC_ROUGHNESS) {
				const auto& pbr = material.metallic_roughness;
				bindTexture(pbr.baseColorTexture, 0);
			}
			else if (material.type == Material::Type::SPECULAR_GLOSSINESS) {
				const auto& pbr = material.specular_glossiness;
				bindTexture(pbr.diffuseTexture, 0);
			}

			const auto& ind = scene.accsessors[prim.indices];
			const auto& bv = scene.bufferViews[ind.bufferView];
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene.buffer_objects[bv.buffer]);
			void* offset = (void*)(ind.byteOffset + bv.byteOffset);
			glDrawElements(GL_TRIANGLES, ind.count, ind.componentType, offset);
		}
	}
}

std::optional<dag::DAG> DAG_from_scene(const int dag_resolution, const std::string scene_folder, const std::string scene_file)
{
	ZoneScoped;
	
	std::cout << "Loading scene... " << scene_folder << scene_file << std::endl;
	glTFLoader::Scene scene = glTFLoader::load(scene_folder, scene_file);
	std::cout << "done.\n";
	voxelizer::Context voxelizer_context = voxelizer::create_context(std::min(dag_resolution, max_subdag_resolution));

	// Map the resources on the GPU
	cudaGraphicsResource *cuda_position_resource;
	//cudaGraphicsResource *cuda_base_color_resource;
	//uint32_t *d_base_color{nullptr};
	uint64_t *d_positions{nullptr};

	auto map_and_register = [](GLuint ssbo, cudaGraphicsResource *&resource, auto *&dev_ptr){
		std::size_t num_bytes{0};
		cudaGraphicsGLRegisterBuffer(&resource, ssbo, cudaGraphicsMapFlagsNone);
		cudaGraphicsMapResources(1, &resource);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&dev_ptr), &num_bytes, resource);
	};

	map_and_register(voxelizer_context.m_position_ssbo,   cuda_position_resource,   d_positions );
	//map_and_register( voxel_generator.m_base_color_ssbo, cuda_base_color_resource, d_base_color);

	// The DAGConstructor will call a function, here `get_voxels`, which will return a struct with the raw data
	// used for construction. The raw data in this case, we get from calling the `generate_voxels` function
	// of the voxelizer, which will write to the previously mapped resources.
	UserData draw_data = get_draw_data(scene);
	auto get_voxels = [&](const chag::Aabb &aabb, int resolution) {
		ZoneScopedN("get_voxels");
		auto count = voxelizer::generate_voxels(voxelizer_context, aabb, resolution, draw_gltf_scene, reinterpret_cast<void*>(&draw_data));
		return DAGConstructor::RawData{
			d_positions,
			//d_base_color,
			count,
			DAGConstructor::RawData::ON_GPU
		};
	};

	// Parameters for construction
	DAGConstructor dag_constructor;
	const int subdag_levels_excluding_64bit_leaf = static_cast<int>(std::log2(std::min(dag_resolution, max_subdag_resolution) / 4));

	// Calculate aabb which overlaps every node
	chag::Aabb combined_aabb = chag::make_inverse_extreme_aabb();
	for(const auto &node_idx : scene.scene_nodes_roots )
	{
		combined_aabb = chag::combine(combined_aabb, scene.scene_nodes[node_idx].aabb);
	}

	chag::Aabb square_dag_aabb = make_square_aabb(combined_aabb);



	// Try to actually construct the DAG
	std::cout << "Start constructing DAG...\n";
	auto maybe_dag = dag_constructor.generate_DAG(
			get_voxels,
			dag_resolution,
			max_subdag_resolution,
			subdag_levels_excluding_64bit_leaf,
			square_dag_aabb
		);


	//cudaGraphicsUnmapResources(1, &cuda_base_color_resource);
	cudaGraphicsUnmapResources(1, &cuda_position_resource);

	voxelizer::destroy_context(voxelizer_context);
	glTFLoader::free_scene(scene);

	return maybe_dag;
}
