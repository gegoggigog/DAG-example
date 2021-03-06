#include "DAGConstructor.h"
// C-STD
#include <stdint.h>
//#include <intrin.h>
#include <algorithm>
// CUDA
#include <cuda_runtime.h>
#include <math.h>
// THRUST
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/adjacent_difference.h>
// PROJECT
#include "../CudaHelpers.h" //FIXME: Proper search paths
#include "../hash.h"        //FIXME: Proper search paths

uint32_t count_child_nodes(int lvl, int bottomlevel, uint32_t node_idx, std::vector<std::vector<uint32_t>> *dag) {
	if (lvl == bottomlevel) {
		auto a = __popcnt((*dag)[lvl][node_idx]);
		auto b = __popcnt((*dag)[lvl][node_idx + 1]);
		return a + b;
	}
	uint32_t n_children = 0;

	uint32_t set_bit            = 1;
	uint32_t idx                = 0;
	uint32_t *current_node_mask = &(*dag)[lvl][node_idx];
	do {
		if (set_bit & *current_node_mask) {
			const auto &child_idx = (*dag)[lvl][++idx + node_idx];
			n_children += count_child_nodes(lvl + 1, bottomlevel, child_idx, dag);
		}
		set_bit = set_bit << 1;
	} while (set_bit != (1 << 8));

	*current_node_mask |= n_children << 8;

	return n_children;
}

struct DAGConstructor::impl {
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
	thrust::device_vector<uint32_t> parent_paths;
	thrust::device_vector<uint32_t> parent_node_size;
	thrust::device_vector<uint32_t> parent_svo_idx;
	thrust::device_vector<uint32_t> sorted_orig_pos;
	thrust::device_vector<uint32_t> sorted_parent_node_size;
	thrust::device_vector<uint32_t> sorted_parent_svo_idx;
	thrust::device_vector<uint32_t> unique_size;
	thrust::device_vector<uint32_t> parent_dag_idx;
	thrust::device_vector<uint32_t> parent_svo_nodes;

	thrust::device_vector<uint32_t> path;
	thrust::device_vector<uint32_t> base_color;

	int m_parent_svo_size;
	std::size_t m_child_svo_size;

	void map_resources(size_t child_svo_size);
	thrust::device_vector<uint32_t> initDag(int *child_level_start_offset, int *parent_level_start_offset);

	std::pair<
		std::vector<thrust::device_vector<uint32_t>>,
		std::vector<thrust::device_vector<uint64_t>>
	> buildDAG(
		int bottomLevel,
		int *parent_level_start_offset,
		int *child_level_start_offset);

	void build_parent_level(int lvl, int bottomLevel);
	thrust::device_vector<uint32_t> create_dag_nodes(int size);

	std::size_t sort_and_merge_fragments(std::size_t count);
	dag::DAG build_dag(int count, int depth, const chag::Aabb &aabb);
};

__device__ uint32_t childBits(const uint32_t &val) { return uint32_t(val & 0x7); }

__global__ void build_parent_level_kernel(
	uint32_t *d_parent_svo_nodes,
	uint32_t *d_parent_svo_idx,
	uint32_t *d_child_paths, 
	uint32_t *d_first_child_pos,
	uint32_t *d_child_dag_idx,
	uint64_t *d_child_sort_key,
	uint64_t *d_parent_sort_key, 
	int parent_svo_size,
	int child_svo_size,
	int lvl,
	int bottomLevel
) 
{
	int thread = getGlobalIdx_1D_1D();
	if (thread < parent_svo_size) {
		uint32_t nof_children;
		if (thread == parent_svo_size - 1)
			nof_children = child_svo_size - d_first_child_pos[thread];
		else
			nof_children = d_first_child_pos[thread + 1] - d_first_child_pos[thread];

		uint64_t hash_key[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
		int ctr              = 0;

		d_parent_svo_nodes[d_parent_svo_idx[thread]] = 0;
		for (uint32_t i = 0; i < nof_children; i++) {
			uint32_t childpath = d_child_paths[d_first_child_pos[thread] + i];
			uint32_t child_idx = childBits(childpath);
			d_parent_svo_nodes[d_parent_svo_idx[thread]] |= (1 << child_idx);
			if (lvl != bottomLevel) {
				d_parent_svo_nodes[d_parent_svo_idx[thread] + 1 + i] = d_child_dag_idx[d_first_child_pos[thread] + i];
				hash_key[ctr++]                                      = d_child_sort_key[d_first_child_pos[thread] + i];
			}
		}
		if (lvl == bottomLevel) {
			d_parent_sort_key[thread] = d_parent_svo_nodes[d_parent_svo_idx[thread]];
		} else {
			hash_key[ctr++] = d_parent_svo_nodes[d_parent_svo_idx[thread]];
			uint64_t hash_value[2];
			MurmurHash3_x64_128(&hash_key[0], 9 * sizeof(uint64_t), 0x0, hash_value);
			d_parent_sort_key[thread] = hash_value[0];
		}
	}
}

__global__ void create_dag_nodes_kernel(
	uint64_t *d_parent_sort_key, 
	uint32_t *d_parent_node_size,
	uint32_t *d_unique_pos, 
	uint32_t *d_parent_svo_idx,
	uint32_t *d_parent_svo_nodes,
	uint32_t *d_parent_dag_idx,
	uint32_t *d_compact_dag, 
	uint32_t *d_sorted_orig_pos,
	int parent_svo_size)
{
	int thread = getGlobalIdx_1D_1D();
	if (thread < parent_svo_size) {
		bool first_unique = true;
		if (thread != 0) 
		{
			first_unique =
				d_parent_sort_key[thread] != d_parent_sort_key[thread - 1]; 
		}

		uint32_t dag_index = d_parent_dag_idx[d_sorted_orig_pos[thread]] = d_unique_pos[thread];
		if (first_unique) {
			// Write childmask
			d_compact_dag[dag_index] = d_parent_svo_nodes[d_parent_svo_idx[thread]];
			// Write indices
			for (uint32_t i = 1; i < d_parent_node_size[thread]; i++) {
				d_compact_dag[dag_index + i] = d_parent_svo_nodes[d_parent_svo_idx[thread] + i];
			}
		}
	}
}

void DAGConstructor::impl::build_parent_level(int lvl, int bottomLevel) {
	dim3 blockDim = dim3(256);
	dim3 gridDim = dim3((m_parent_svo_size +  blockDim.x - 1)/ blockDim.x);
	build_parent_level_kernel<<<gridDim, blockDim>>>(
		parent_svo_nodes.data().get(),
		parent_svo_idx.data().get(),
		path.data().get(),
		first_child_pos.data().get(),
		child_dag_idx.data().get(),
		child_sort_key.data().get(),
		parent_sort_key.data().get(),
		m_parent_svo_size,
		m_child_svo_size,
		lvl,
		bottomLevel
		);
}


thrust::device_vector<uint32_t> DAGConstructor::impl::create_dag_nodes(int size) {
	thrust::device_vector<uint32_t> result(size);
	dim3 blockDim = dim3(256);
	dim3 gridDim = dim3((m_parent_svo_size +  blockDim.x - 1)/ blockDim.x);

	create_dag_nodes_kernel<<<gridDim, blockDim>>>(
		parent_sort_key.data().get(),
		parent_node_size.data().get(),
		unique_pos.data().get(),
		parent_svo_idx.data().get(),
		parent_svo_nodes.data().get(),
		parent_dag_idx.data().get(),
		result.data().get(),
		sorted_orig_pos.data().get(),
		m_parent_svo_size
		);
	return result;
}

void DAGConstructor::impl::map_resources(size_t child_svo_size) {
	m_child_svo_size = child_svo_size;
	size_t free, total;
	cudaMemGetInfo(&free, &total);

	compact_dag.resize(10*1024*1024);
	parent_svo_nodes.resize(3*m_child_svo_size);
	sorted_parent_node_size.resize(m_child_svo_size);
	sorted_parent_svo_idx.resize(m_child_svo_size);
	parent_node_size.resize(m_child_svo_size);
	first_child_pos.resize(m_child_svo_size);
	parent_sort_key.resize(m_child_svo_size);
	sorted_orig_pos.resize(m_child_svo_size);
	parent_svo_idx.resize(m_child_svo_size);
	parent_dag_idx.resize(m_child_svo_size);
	child_sort_key.resize(m_child_svo_size);
	child_dag_idx.resize(m_child_svo_size);
	parent_paths.resize(m_child_svo_size);
	unique_size.resize(m_child_svo_size);
	unique_pos.resize(m_child_svo_size);

	cudaMemGetInfo(&free, &total);
}

thrust::device_vector<uint32_t> DAGConstructor::impl::initDag(int *child_level_start_offset, int *parent_level_start_offset) {
	thrust::device_vector<uint32_t> result;
	constexpr std::size_t child_start_offset = 9;
	// Used for later
	thrust::copy_n(
		compact_masks.cbegin(),
		m_child_svo_size,
		child_sort_key.begin()
	);

	// Index to masks
	thrust::sequence(
		unique_pos.begin(),
		unique_pos.begin() + m_child_svo_size
	);

	// Sort masks AND index
	thrust::sort_by_key(
		compact_masks.begin(),
		compact_masks.begin() + m_child_svo_size,
		unique_pos.begin()
	);

	// Mark unique masks as 2, 
	// since the 64-bit mask will be represented as 2x32-bit words in the DAG.
	thrust::adjacent_difference(
		compact_masks.cbegin(),
		compact_masks.cbegin() + m_child_svo_size,
		first_child_pos.begin(),
		2 * (thrust::placeholders::_1 != thrust::placeholders::_2)
	);

	// Set first to zero for inclusive scan (prefix sum)
	first_child_pos[0] = 0;

	// child_dag_idx[unique_pos[i]] = sum_at[i]
	thrust::inclusive_scan(
		first_child_pos.cbegin(),
		first_child_pos.cbegin() + m_child_svo_size,
		thrust::make_permutation_iterator(
			child_dag_idx.begin(), unique_pos.cbegin()
		)
	);

	// Reduce to unique masks
	const std::size_t num_unique = thrust::distance(
		compact_masks.begin(),
		thrust::unique(
			compact_masks.begin(),
			compact_masks.begin() + m_child_svo_size
		)
	);

	// Quick and dirty memcpy as I do not know the thrust way of doing this..
	result.resize(2 * num_unique);
	std::size_t count = result.size() * sizeof(uint32_t);
	 cudaMemcpy(
		result.data().get(),
		compact_masks.data().get(),
		count,
		cudaMemcpyDeviceToDevice
	);

	// Not used in first pass
	*child_level_start_offset  = child_start_offset;
	// Make room for a copied root node
	*parent_level_start_offset = child_start_offset + 2 * num_unique;
	return result;
}

namespace IdentifyParents{
using InParam  = thrust::tuple<uint32_t, uint32_t>;
struct equal_to {
	__host__ __device__ bool
		operator()(const InParam &lhs, const InParam &rhs) const
	{
		return lhs.get<1>() == rhs.get<1>();
	}
};

using Iter = thrust::device_vector<uint32_t>::iterator;
auto InBegin(Iter path){
	return thrust::make_zip_iterator(
		thrust::make_tuple(
			thrust::make_counting_iterator(0ull),
			thrust::make_transform_iterator(path, thrust::placeholders::_1 >> 3)
		)
	);
}
auto InEnd(Iter path, std::size_t count) {
	return thrust::make_zip_iterator(
		thrust::make_tuple(
			thrust::make_counting_iterator(count),
			thrust::make_transform_iterator(path + count, thrust::placeholders::_1 >> 3)
		)
	);
}
auto OutBegin(Iter first_child_pos_, Iter parent_paths_){
	return thrust::make_zip_iterator(
		thrust::make_tuple(first_child_pos_, parent_paths_)
	);
}
}  // namespace IdentifyParents

namespace FirstUnique {
using InParam1 = uint64_t;
using InParam2 = thrust::tuple<uint64_t, uint32_t>;
using OutParam = uint32_t;
struct functor : public thrust::binary_function<InParam1, InParam2, OutParam> {
	__host__ __device__ OutParam
		operator()(const InParam1 &lhs, const InParam2 &rhs) const
	{
		return lhs != rhs.get<0>() ? rhs.get<1>() : 0;
	}
};
}  // namespace FirstUnique

std::pair<
	std::vector<thrust::device_vector<uint32_t>>,
	std::vector<thrust::device_vector<uint64_t>>
>
DAGConstructor::impl::buildDAG(
	int bottomLevel,
	int *parent_level_start_offset,
	int *child_level_start_offset
)
{
	std::pair<
		std::vector<thrust::device_vector<uint32_t>>,
		std::vector<thrust::device_vector<uint64_t>>
	> result;

	auto &dag_lvls = result.first;
	auto &hash_lvls = result.second;

	dag_lvls.emplace_back(initDag(child_level_start_offset, parent_level_start_offset));
	hash_lvls.emplace_back(dag_lvls[0].size()/2);
	cudaMemcpy(
		hash_lvls.back().data().get(),
		dag_lvls[0].data().get(),
		dag_lvls[0].size()*sizeof(uint32_t),
		cudaMemcpyDeviceToDevice
	);

	for (int lvl = bottomLevel - 1; lvl >= 0; lvl--) {
		int final_level_size;
		{
			// Copy unique parent paths and their children indexes
			{
				auto out_iterator = IdentifyParents::OutBegin(first_child_pos.begin(), parent_paths.begin());
				m_parent_svo_size = thrust::distance(
					out_iterator, 
					thrust::unique_copy(
						IdentifyParents::InBegin(path.begin()), 
						IdentifyParents::InEnd(path.begin(), m_child_svo_size),
						out_iterator,
						IdentifyParents::equal_to()
					)
				);
			}

			// (1)
			// Figure out how large each node is (mask + children, i.e. 1+N where N is the number of children)
			{
				using namespace thrust::placeholders;
				auto first_child_pos_back   = first_child_pos.begin()  + m_parent_svo_size-1;
				auto first_child_pos_end    = first_child_pos.begin()  + m_parent_svo_size;
				auto first_child_pos_second = first_child_pos.begin()  + 1;
				auto parent_node_size_back  = parent_node_size.begin() + m_parent_svo_size-1;
				thrust::transform(first_child_pos_second, first_child_pos_end, first_child_pos.begin(), parent_node_size.begin(), _1 - _2 + 1);
				*parent_node_size_back = m_child_svo_size - (*first_child_pos_back) + 1;
			}
			// (2)
			// Figure out index offset of these nodes
			thrust::exclusive_scan(parent_node_size.begin(), parent_node_size.begin() + m_parent_svo_size, parent_svo_idx.begin());
			// FIXME: (1) & (2) could be replaced by
			//        parent_node_size[i] = first_child_pos[i] + i

			build_parent_level(lvl, bottomLevel);

			// Need to copy m_parent_sort_key to m_child_sort_key now, as m_parent_sort_key will become sorted.
			// We don't want in m_child_sort_key to be sorted..
			thrust::copy_n(parent_sort_key.begin(), m_parent_svo_size, child_sort_key.begin());

			// Create mapping according to parent sort key
			thrust::sequence(sorted_orig_pos.begin(), sorted_orig_pos.begin() + m_parent_svo_size);
			thrust::sort_by_key(
				parent_sort_key.begin(),
				parent_sort_key.begin() + m_parent_svo_size,
				thrust::make_zip_iterator(
					thrust::make_tuple(
						sorted_orig_pos.begin(),
						parent_node_size.begin(),
						parent_svo_idx.begin()
					)
				)
			);

			// FIXME: Could this entire copy & resize be optimized?
			hash_lvls.emplace_back(m_parent_svo_size);
			thrust::copy_n(
				parent_sort_key.begin(),
				m_parent_svo_size,
				hash_lvls.back().begin()
			);
			hash_lvls.back().resize(
				thrust::distance(hash_lvls.back().begin(),
					thrust::unique(hash_lvls.back().begin(),
						hash_lvls.back().end()
					)
				)
			);

			// Compares parent_sort_key[i+1] to parent_sort_key[i]
			// If equal, write 0 to unique_size, else
			// write sorted_parent_node_size[i].
			// TODO: I could rewrite this function to be more clear.
			//       i.e., the zip iterator etc.
			thrust::transform(
				parent_sort_key.begin() + 1,
				parent_sort_key.begin() + m_parent_svo_size,
				thrust::make_zip_iterator(
					thrust::make_tuple(
						parent_sort_key.begin(),
						parent_node_size.begin()
					)
				),
				unique_size.begin() + 1,
				FirstUnique::functor()
			);

			unique_size[0] = 0;

			// Sum up sizes to figure out offsets
			thrust::inclusive_scan(unique_size.begin(), unique_size.begin() + m_parent_svo_size, unique_pos.begin());

			final_level_size = unique_pos[m_parent_svo_size-1] + parent_node_size[m_parent_svo_size-1];

			dag_lvls.emplace_back(create_dag_nodes(final_level_size));

			thrust::swap(parent_dag_idx, child_dag_idx);
			thrust::swap(parent_paths, path);
		}

		///////////////////////////////////////////////////////////////////
		// New level
		///////////////////////////////////////////////////////////////////
		m_child_svo_size          = m_parent_svo_size;
		*child_level_start_offset = *parent_level_start_offset;
		*parent_level_start_offset += final_level_size;
	}

	// DAG is built bottom up. Reverse to have coarser levels "on top"
	std::reverse(begin(dag_lvls), end(dag_lvls));
	std::reverse(begin(hash_lvls), end(hash_lvls));
	return result;
}

__device__ inline float4 &operator+=(float4 &a, const float4 &b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

__device__ inline float4 operator+(const float4 &a, const float4 &b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

//FIXME: thrust unary deprecated use std..
struct AverageData : public thrust::unary_function<float4, float4> {
	__device__ float4 operator()(const float4 &a) { return make_float4(a.x / a.w, a.y / a.w, a.z / a.w, 1.0f); }
};

//FIXME: thrust unary deprecated use std..
template<bool ignore_black>
struct ToFloat4 : public thrust::unary_function<uint32_t, float4>
{
	__device__
	float4 operator()(uint32_t x) const
	{
		return make_float4(
			float((x >> 24) & 0xFF)/255.0f,
			float((x >> 16) & 0xFF)/255.0f,
			float((x >> 8 ) & 0xFF)/255.0f,
			!ignore_black || x != 0 ? 1.0f : 0.0f
		);
	}
};

__device__ uint32_t to_uint(const float x) { 
	return thrust::min(
		(uint32_t)round(thrust::max(x*255.0f, 0.0f)),
		255u
	);
}
//FIXME: thrust unary deprecated use std..
struct ToUint32 : public thrust::unary_function<float4, uint32_t>
{
	__device__
	uint32_t operator()(const float4 &v) const
	{
		return to_uint(v.z / v.w) << 16 | to_uint(v.y / v.w) << 8 | to_uint(v.x / v.w);
	}
};


std::size_t DAGConstructor::impl::sort_and_merge_fragments(std::size_t count) {
	// These could be cached as well, to reduce reallocations.
	thrust::device_vector<uint32_t> th_sorted_index(count, 0);
	thrust::device_vector<uint32_t> th_unique(count,0);
	thrust::device_vector<uint32_t> th_pre_inc_sum(count,0);
	thrust::device_vector<uint64_t> th_sorted_masks(count,0);

	// IOTA.
	thrust::sequence(
		th_sorted_index.begin(),
		th_sorted_index.end()
	);

	// Sorts value AND key.
	thrust::sort_by_key(
		path.begin(),
		path.begin() + count,
		th_sorted_index.begin()
	);

	// NOTE: This and color sorting should be possible to 
	//       do under different streams. However, this is
	//       not really a bottleneck, so meh.
	// Figure out set bit in 64-bit leaf
	thrust::transform(
		path.begin(),
		path.begin() + count,
		th_sorted_masks.begin(),
		1ull << (thrust::placeholders::_1 & 0x3F)
	);

	// Mark unique paths with 1
	thrust::adjacent_difference(
		path.begin(),
		path.begin() + count,
		th_unique.begin(),
		thrust::not_equal_to<uint32_t>()
	);

	// First element need to be zero for inclusive scan (prefix sum)
	th_unique[0] = 0;

	// Creates a mapping to compact elements
	std::size_t unique_frag =
		1 + *(thrust::inclusive_scan(th_unique.cbegin(), th_unique.cend(), th_pre_inc_sum.begin()) - 1);

	m_num_colors = unique_frag;

	const auto compaction = [&](thrust::device_vector<uint32_t> &vec_in, const bool ignore_black){
		// This could be cached as well, to reduce reallocations.
		thrust::device_vector<float4> th_sorted_data(count);
		thrust::device_vector<float4> th_compact_data(m_num_colors);

		// Sort data from uint32_t to float4.
		// If we want to ignore completely black values in the averaging,
		// ToFloat4 will set the w component to 0, instead of 1.
		if(ignore_black)
		{
			thrust::gather(
				th_sorted_index.cbegin(),
				th_sorted_index.cend(),
				thrust::make_transform_iterator(vec_in.begin(), ToFloat4<true>()),
				th_sorted_data.begin()
			);
		} else
		{
			thrust::gather(
				th_sorted_index.cbegin(),
				th_sorted_index.cend(),
				thrust::make_transform_iterator(vec_in.begin(), ToFloat4<false>()),
				th_sorted_data.begin()
			);
		}

		// TODO: These can be done in paralell by later transforms..
		// Reduce identical colors by sum.
		auto compact_out_end = 
			thrust::reduce_by_key(
				th_pre_inc_sum.cbegin(),
				th_pre_inc_sum.cend(),
				th_sorted_data.begin(),
				thrust::make_discard_iterator(),
				th_compact_data.begin()
			).second;

		// And average the sum (N is in color.w)
		thrust::transform(
			th_compact_data.begin(),
			compact_out_end,
			th_compact_data.begin(),
			AverageData()
		);

		// Pack back to vector
		thrust::transform(
			th_compact_data.begin(),
			compact_out_end,
			vec_in.begin(),
			ToUint32()
		);
	};

	compaction(base_color, false);

	// Truncate positions and collect unique
	// Calculate parent paths
	thrust::transform(
		path.begin(),
		path.begin() + count,
		path.begin(),
		thrust::placeholders::_1 >> 6
	);

	// Mark unique parents with 1
	thrust::adjacent_difference(
		path.begin(),
		path.begin() + count,
		th_unique.begin(),
		thrust::not_equal_to<uint32_t>()
	);

	// First element need to be zero for inclusive scan (prefix sum)
	th_unique[0] = 0;

	{
		// scopedStream s1, s2;
		// TODO: Read up on streams
		// Creates a mapping to compact elements
		thrust::inclusive_scan(
			th_unique.cbegin(),
			th_unique.cend(),
			th_pre_inc_sum.begin()
		);

		// Reduce identical parents
		m_child_svo_size = 
			thrust::distance(
				path.begin(),
				thrust::unique(
					path.begin(),
					path.begin() + count)
			);
	}

	// FIXME: Guess we need to move these allocations elsewhere..
	if (compact_masks.size() < m_child_svo_size) {
		compact_masks.resize(m_child_svo_size);
	}

	// Reduce them according to parents by bitwise or
	thrust::reduce_by_key(
		th_pre_inc_sum.cbegin(),
		th_pre_inc_sum.cend(),
		th_sorted_masks.cbegin(),
		thrust::make_discard_iterator(),
		compact_masks.begin(),
		thrust::equal_to<uint32_t>(),
		thrust::bit_or<uint64_t>()
	);
	return m_child_svo_size;
}

dag::DAG DAGConstructor::impl::build_dag(int count, int depth, const chag::Aabb &aabb) {
	dag::DAG result;
	result.m_levels = depth;
	result.m_aabb   = aabb;
	// TODO: Return n_paths **and** n_colors to avoid relying on internal state.
	size_t n_paths  = sort_and_merge_fragments(count);

	///////////////////////////////////////////////////////////////////////////
	// Setup
	///////////////////////////////////////////////////////////////////////////
	map_resources(n_paths);
	int child_level_start_offset;
	int parent_level_start_offset;
	auto out = buildDAG(result.m_levels, &parent_level_start_offset,&child_level_start_offset);
	auto &the_dag = out.first;
	auto &the_hash = out.second;
	{
		// Copy DAG.
		result.m_data.reserve(the_dag.size());
		for (int i{0}; i<the_dag.size(); ++i) {
			const std::size_t nelem = the_dag[i].size();
			const std::size_t count = nelem * sizeof(uint32_t);
			result.m_data.emplace_back(std::vector<uint32_t>(nelem));
			cudaMemcpy(
				result.m_data[i].data(),
				the_dag[i].data().get(),
				count,
				cudaMemcpyDeviceToHost
			);
		}
		// Copy hash
		result.m_hashes.reserve(the_hash.size());
		for (int i{0}; i < the_hash.size(); ++i) {
			const std::size_t nelem = the_hash[i].size();
			const std::size_t count = nelem * sizeof(uint64_t);
			result.m_hashes.emplace_back(std::vector<uint64_t>(nelem));
			cudaMemcpy(
				result.m_hashes[i].data(),
				the_hash[i].data().get(),
				count,
				cudaMemcpyDeviceToHost
			);
		}

		// Copy data.
		auto copy_to_host = [&](std::vector<uint32_t> &host_vec, const thrust::device_vector<uint32_t> &th_vec) {
			host_vec.resize(m_num_colors);
			cudaMemcpy(
				host_vec.data(),
				th_vec.data().get(),
				host_vec.size() * sizeof(uint32_t),
				cudaMemcpyDeviceToHost
			);
		};
		copy_to_host(result.m_base_colors, base_color);


		count_child_nodes(0, result.m_levels, 0, &result.m_data);
		result.m_data.shrink_to_fit();
		result.m_base_colors.shrink_to_fit();
	}
	return result;
}


template<typename Dev_t, typename Host_t>
void copy_to_device(thrust::device_vector<Dev_t> &dev, const std::vector<Host_t> &host, const std::size_t size){
	dev.resize(size);
	cudaMemcpy(
		dev.data().get(),
		host.data(),
		size * sizeof(Dev_t),
		cudaMemcpyHostToDevice
	);
}

template<typename Dev_t>
void copy_to_device(thrust::device_vector<Dev_t> &dev, Dev_t *dev_ptr, const std::size_t size){
	dev.resize(size);
	thrust::copy_n(
		thrust::device_ptr<Dev_t>{dev_ptr},
		size,
		dev.begin()
	);
}

// Base wrapper.
dag::DAG DAGConstructor::build_dag(
	const std::vector<uint32_t> &morton_paths,
	const std::vector<uint32_t> &base_color,
	int count,
	int depth,
	const chag::Aabb &aabb
)
{
	copy_to_device(p_impl_->path,       morton_paths, count);
	copy_to_device(p_impl_->base_color, base_color,   count);
	
	return p_impl_->build_dag(count, depth, aabb);
}

dag::DAG DAGConstructor::build_dag(
	uint32_t *d_pos,
	uint32_t *d_base_color,
	int count,
	int depth,
	const chag::Aabb &aabb
)
{
	copy_to_device(p_impl_->path,       d_pos,        count);
	copy_to_device(p_impl_->base_color, d_base_color, count);

	return p_impl_->build_dag(count, depth, aabb);
}
DAGConstructor::DAGConstructor() : p_impl_{std::make_unique<impl>()} {}
DAGConstructor::~DAGConstructor() {}
