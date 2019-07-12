#pragma once
#include "BlockBuild.h"
#include <cstdint>
#include <utility>  // std::pair, std::make_pair
#include <vector>

namespace ours_varbit {
using ColorLayout = ColorLayout;
struct OursData {
	uint32_t *d_block_headers  = nullptr;
	uint8_t *d_block_colors    = nullptr;
	uint32_t *d_weights        = nullptr;
	uint64_t *d_macro_w_offset = nullptr;
	uint32_t nof_blocks;
	uint32_t nof_colors;
	uint32_t bits_per_weight;
	ColorLayout color_layout;
	bool use_single_color_blocks;
	std::vector<uint32_t> h_block_headers;
	std::vector<uint8_t> h_block_colors;
	std::vector<uint32_t> h_weights;
	std::vector<uint64_t> h_macro_w_offset;
	float compression;
	float error_threshold;
	int bytes_raw;
	int bytes_compressed;
};
OursData compressColors_alternative_par(std::vector<uint32_t> &original_colors, const float error_threshold,
                                        const ColorLayout layout);

struct CacheHeader {
	uint32_t headers_size;
	uint32_t weights_size;
	uint32_t nof_blocks;
	uint32_t nof_colors;
	uint32_t bits_per_weight;
	bool use_single_color_blocks;
};

bool getErrInfo(const std::vector<uint32_t> &colors, const std::string filename, const ColorLayout layout, float *mse,
                float *maxR, float *maxG, float *maxB, float *maxLength);
float getPSNR(float mse);

void upload_to_gpu(OursData &data);
};  // namespace ours_varbit
