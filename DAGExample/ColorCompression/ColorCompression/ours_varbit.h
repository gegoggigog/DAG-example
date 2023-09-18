#pragma once
#include <cstdint>
#include <vector>
#include "utils/disc_vector.hpp"
#include "color_layout.hpp"

constexpr uint64_t macro_block_size = 16ull * 1024ull;
namespace ours_varbit {
  struct OursData {
    uint64_t nof_blocks;
    uint64_t nof_colors;
    ColorLayout color_layout;
    std::vector<uint32_t> h_block_headers;
    std::vector<uint8_t>  h_block_colors;
    std::vector<uint32_t> h_weights;
    std::vector<uint64_t> h_macro_w_offset;
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
};  // namespace ours_varbit
