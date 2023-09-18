#pragma once
#include <cstdint>
#include <vector>
#include "utils/disc_vector.hpp"
#include "color_layout.hpp"

constexpr uint64_t macro_block_size = 16ull * 1024ull;
namespace ours_varbit {
  struct OursData {
    uint32_t *d_block_headers = nullptr;
    uint8_t *d_block_colors = nullptr;
    uint32_t *d_weights = nullptr;
    uint64_t *d_macro_w_offset = nullptr;
    uint64_t nof_blocks;
    uint64_t nof_colors;
    uint32_t bits_per_weight;
    ColorLayout color_layout;
    bool use_single_color_blocks;
    std::vector<uint32_t> h_block_headers;
    std::vector<uint8_t> h_block_colors;
    std::vector<uint32_t> h_weights;
    std::vector<uint64_t> h_macro_w_offset;
    float compression;
    float error_threshold;
    uint64_t bytes_raw;
    uint64_t bytes_compressed;
  };

  OursData compressColors(
    disc_vector<uint32_t> &&original_colors,
    const float error_threshold,
    const ColorLayout layout
  );

  bool getErrInfo(
    const std::vector<uint32_t> &colors,
    const std::string filename,
    const ColorLayout layout,
    float *mse,
    float *maxR,
    float *maxG,
    float *maxB,
    float *maxLength
  );

  float getPSNR(float mse);

  void upload_to_gpu(OursData &data);
};  // namespace ours_varbit
