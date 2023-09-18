#include "DAGConstructor.h"
#include "DAGConstructor_impl.h"
#include <iostream>
#include <tuple>
#include <vector>
#include <chrono>
#include <glm/gtc/type_ptr.hpp>
#include "Merger.h"

#include <tracy/Tracy.hpp>

inline void printSeconds(double input_seconds) {
	size_t minutes = static_cast<std::size_t>(input_seconds / 60);
	size_t seconds = static_cast<std::size_t>(input_seconds) % 60;
	std::cout << minutes << ":" << seconds << " ";
}

void print_subdag_process(const int it, const int mod, const std::size_t total, std::chrono::steady_clock::time_point startTime) {
	if (it % mod == 0)
	{
		std::cout << "Generating sub DAG " << it << " of " << total << ". ";
		const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startTime).count();
		std::cout << " Elapsed: ";
		printSeconds(elapsed);
		std::cout << "Remaining: ";
		printSeconds(elapsed * total / it - elapsed);
		std::cout << '\n';
	}
}

void print_merge_progress(const std::size_t it, const std::size_t dags_left, std::chrono::steady_clock::time_point passStartTime) {
	std::cout << "Sub - pass " << it + 1 << " of " << dags_left / 8;
	const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - passStartTime).count();
	std::cout << " Elapsed: ";
	printSeconds(elapsed);
	std::cout << "Remaining: ";
	printSeconds(elapsed * dags_left / 8 / (it + 1) - elapsed);
	std::cout << '\n';
}

std::optional<dag::DAG> DAGConstructor::generate_DAG(GetVoxelFunction get_voxels, int geometry_resolution, int max_subdag_resolution, int LevelsExcluding64BitLeafs, chag::Aabb aabb_in)
{
	ZoneScoped;
	using DAG_opt = std::optional<dag::DAG>;

	auto aabb_list = std::vector<chag::Aabb>(1, aabb_in);
	// If the geometry resolution is too high, we need to split the geometry into smaller sub volumes
	// and process them independently. They are later merged to the final result.
	int nof_splits = std::max(0, static_cast<int>(log2(geometry_resolution) - log2(max_subdag_resolution)));
	for (unsigned i = 0; i < (unsigned)nof_splits; ++i) { aabb_list = merger::split_aabb(std::move(aabb_list)); }

	// Create sub DAGs from the sub volumes. Note that not all volumes may contain geometry, hence std::optional.
	std::vector<DAG_opt> dags(aabb_list.size());

	const auto startTime = std::chrono::high_resolution_clock::now();

	uint64_t total_count = 0;
	uint64_t total_unique_count = 0;
	for (int i{ 0 }; i < aabb_list.size(); ++i)
	{
		print_subdag_process(i, 100, aabb_list.size(), startTime);
		auto& aabb = aabb_list[i];
		auto voxels = get_voxels(aabb, std::min(max_subdag_resolution, geometry_resolution));
		if (voxels.count > 0) {
			total_count += voxels.count;
			cuda_builder->upload_path(voxels.positions, voxels.count);
#ifdef DAG_COLORS
			cuda_builder->upload_colors(voxels.base_color, voxels.count);
#endif
			dags[i] = cuda_builder->build_dag(voxels.count, LevelsExcluding64BitLeafs, aabb);

			total_unique_count += dags[i] ? (*dags[i]).m_base_colors.size() : 0ull;
		}
	}
	const auto DAGTimeDone = std::chrono::high_resolution_clock::now();
	std::cout << "done.\n";
	std::cout << "Total voxels: " << total_count << "\n";
	std::cout << "Total unique voxels: " << total_unique_count << "\n";


	// The way the sub volumes are split, is in a morton order.
	// 8 consecutive volumes hence compose a larger super volume.
	// We thus create a batch of 8 subvolumes and merge them,
	// and place them in a new array to be recursively processed in
	// the next iteration.
	std::vector<DAG_opt> merged_dags(dags.size() / 8);
	std::size_t dags_left{ dags.size() };
	std::cout << "Start merging DAGs...\n";
	while (dags_left != 1) {
		std::cout << "Passes left: " << static_cast<int>(std::log(dags_left) / std::log(8)) << ".\n";
		const auto passStartTime = std::chrono::high_resolution_clock::now();
		for (std::size_t i = 0; i < dags_left / 8; ++i) {
			print_merge_progress(i, dags_left, passStartTime);

			std::array<DAG_opt, 8> batch;
			for (int j{ 0 }; j < 8; ++j) {
				batch[j] = std::move(dags[8 * i + j]);
			}
			merged_dags[i] = merger::merge(batch);
		}
		dags_left /= 8;
		std::swap(dags, merged_dags);
	}
	const auto MergeTimeDone = std::chrono::high_resolution_clock::now();
	std::cout << "done.\n";
	std::cout << "Time to DAG: "   << std::chrono::duration_cast<std::chrono::duration<double>>(DAGTimeDone - startTime).count()     << " seconds\n";
	std::cout << "Time to Merge: " << std::chrono::duration_cast<std::chrono::duration<double>>(MergeTimeDone - DAGTimeDone).count() << " seconds\n";

	// When all DAGs have been merged, the result resides in the
	// first slot of the array.
	if (dags[0]) {
		std::memcpy(dags[0]->aabb_min.data(), glm::value_ptr(aabb_in.min), 3 * sizeof(float));
		std::memcpy(dags[0]->aabb_max.data(), glm::value_ptr(aabb_in.max), 3 * sizeof(float));
		return std::move(dags[0]);
	}
	return {};
}
DAGConstructor::DAGConstructor() : cuda_builder{std::make_unique<DAGConstructor_impl>()} {}
DAGConstructor::~DAGConstructor() {}
