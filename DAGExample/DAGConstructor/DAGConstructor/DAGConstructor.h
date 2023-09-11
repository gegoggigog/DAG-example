#pragma once
#include <functional>
#include <memory>
#include <optional>
#include <DAG/DAG.h>

struct DAGConstructor_impl;
class DAGConstructor {
	std::unique_ptr<DAGConstructor_impl> cuda_builder;

 public:
	DAGConstructor();
	~DAGConstructor();
	DAGConstructor(DAGConstructor &&)      = delete;
	DAGConstructor(const DAGConstructor &) = delete;
	DAGConstructor &operator=(DAGConstructor &&) = delete;
	DAGConstructor &operator=(const DAGConstructor &) = delete;

	struct RawData {
		uint64_t* positions;
		//uint32_t *base_color;
		uint32_t count;
		enum {
			ON_GPU,
			ON_CPU
		} residency;
	};

	using GetVoxelFunction = std::function<RawData(const chag::Aabb& aabb, int resolution)>;

	std::optional<dag::DAG> generate_DAG(
		GetVoxelFunction get_voxels,
		int geometry_resolution,
		int max_subdag_resolution,
		int LevelsExcluding64BitLeafs,
		chag::Aabb aabb_in
	);
};
