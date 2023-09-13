#pragma once
#include <cinttypes>
#include <thrust/device_vector.h>
#include "DAG/DAG.h"
#include "utils/Aabb.h"

struct DAGConstructor_impl {
	std::size_t m_num_colors;
	int m_cached_num_colors;
	int m_cached_frag_count;

	thrust::device_vector<uint64_t> compact_masks;
	thrust::device_vector<uint64_t> child_sort_key;
	thrust::device_vector<uint64_t> parent_sort_key;
	thrust::device_vector<uint32_t> unique_pos;
	thrust::device_vector<uint32_t> first_child_pos;
	thrust::device_vector<uint32_t> child_dag_idx;
	thrust::device_vector<uint32_t> compact_dag;
	thrust::device_vector<uint64_t> parent_paths;
	thrust::device_vector<uint32_t> parent_node_size;
	thrust::device_vector<uint32_t> parent_svo_idx;
	thrust::device_vector<uint32_t> sorted_orig_pos;
	thrust::device_vector<uint32_t> sorted_parent_node_size;
	thrust::device_vector<uint32_t> sorted_parent_svo_idx;
	thrust::device_vector<uint32_t> unique_size;
	thrust::device_vector<uint32_t> parent_dag_idx;
	thrust::device_vector<uint32_t> parent_svo_nodes;

	thrust::device_vector<uint64_t> path;
	//thrust::device_vector<uint32_t> base_color;

	int m_parent_svo_size;
	std::size_t m_child_svo_size;

	void map_resources(size_t child_svo_size);
	thrust::device_vector<uint32_t> initDag(int* child_level_start_offset, int* parent_level_start_offset);

	using DAGHashResult = std::pair<
		std::vector<thrust::device_vector<uint32_t>>,
		std::vector<thrust::device_vector<uint64_t>>
	>;
	DAGHashResult buildDAG(
		int bottomLevel,
		int* parent_level_start_offset,
		int* child_level_start_offset);

	void build_parent_level(int lvl, int bottomLevel);
	thrust::device_vector<uint32_t> create_dag_nodes(int size);

	std::size_t sort_and_merge_fragments(std::size_t count);
	dag::DAG build_dag(int count, int depth, const chag::Aabb& aabb);

	void upload_path(uint64_t* pos, int count);
	void upload_colors(uint32_t* col, int count);
};
