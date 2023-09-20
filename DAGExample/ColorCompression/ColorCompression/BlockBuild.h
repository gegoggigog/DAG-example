#ifndef BLOCK_BUILD_H
#define BLOCK_BUILD_H
#include <cstdint>
#include <cuda_runtime.h>
#include <glm/vec3.hpp>
#include <vector>
#include "color_layout.hpp"
struct BlockBuild {
	BlockBuild(size_t blockStart, size_t blockLength) :
		blockStart(blockStart),
		blockLength(blockLength),
		dirty(true)
	{};

	BlockBuild(size_t blockStart) :
		BlockBuild(blockStart, 1)
	{};

	size_t blockStart;
	size_t blockLength;
	bool dirty;
};

void uploadColors(const std::vector<float3> &colors);
void scores_gpu(const std::vector<BlockBuild> &blocks,
				std::vector<float> &scores,
				std::vector<uint8_t> &weights,
				std::vector<float3> &colorRanges,
				float error_treshold,
				ColorLayout layout,
				int max_w,
				bool finalEval = false);
#endif  // BLOCK_BUILD_H
