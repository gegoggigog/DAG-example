#pragma once
#include "BlockBuild.h"
#include <cstdint>
#include <utility>  // std::pair, std::make_pair
#include <vector>

#include <fstream>
#include <array>
template<typename T>
static void write_to_disc(const std::string file, const std::vector<T> &vec)
{
  std::ofstream ofs{ file, std::ofstream::binary | std::ofstream::out };
  ofs.write(
    reinterpret_cast<const char*>(vec.data()),
    vec.size() * sizeof(T)
  );
}

template<typename T, std::size_t _cache_size>
class disc_array {
public:
  disc_array(const disc_array&) = delete;
  disc_array(disc_array&&) = default;
  disc_array& operator=(const disc_array&) = delete;
  disc_array& operator=(disc_array&&) = default;
  virtual ~disc_array() = default;
  disc_array(const std::string file) : ifs{ file, std::ifstream::binary | std::ifstream::in | std::ifstream::ate }
  {
    const_cast<std::size_t&>(_size) = ifs.tellg() / sizeof(T);
    ifs.seekg(std::ifstream::beg);
  };

  const T operator [] (std::size_t i) {
    if (!is_in_cache(i))
    {
      //std::cout << "Not in cache\n";
      read_block(i);
    }
    return _cache[i % _cache.size()];
  }
  const std::size_t size() const { return _size; }
private:

  bool is_in_cache(const std::size_t i) const {
    //std::cout
    //  << "i: " << i << '\n'
    //  << " _cache_size * _block:" << _cache_size * _block << '\n'
    //  << " _cache_size * (_block + 1): " << _cache_size * (_block + 1) << '\n';
    return
      _has_cache &&
      _cache_size * _block <= i &&
      _cache_size * (_block + 1) > i;
  }

  void read_block(const std::size_t i) {
    //std::cout << "Read\n";
    const std::size_t fasdas = ifs.gcount();
    _block = i / _cache_size;
    std::size_t read_start{ _block * _cache_size * sizeof(T) };
    std::size_t max_read_to = _size * sizeof(T);
    std::size_t bytes_to_read = _cache_size * sizeof(T);
    std::size_t read_end = read_start + bytes_to_read;
    if (read_end > max_read_to)
    {
      bytes_to_read = max_read_to - read_start;
    }
    ifs.seekg(read_start, std::ifstream::beg);
    ifs.read(
      reinterpret_cast<char*>(_cache.data()),
      bytes_to_read
    );
    _has_cache = true;
  }

  std::ifstream ifs;
  std::array<T, _cache_size> _cache{ 0 };
  bool _has_cache{ false };
  std::size_t _block{ 0 };
  const std::size_t _size{ 0 };
};

constexpr uint64_t macro_block_size = 16ull * 1024ull;
namespace ours_varbit {
  using ColorLayout = ColorLayout;
  struct OursData {
    uint32_t *d_block_headers = nullptr;
    uint8_t *d_block_colors = nullptr;
    uint32_t *d_weights = nullptr;
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

  OursData compressColors_alternative_par(
    //std::vector<uint32_t> &original_colors,
    disc_array<uint32_t, macro_block_size> &&original_colors,
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
