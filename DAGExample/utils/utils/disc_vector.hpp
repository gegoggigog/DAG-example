#pragma once
#include <fstream>

template<typename T>
class disc_vector {
public:
    disc_vector(const disc_vector&) = delete;
    disc_vector(disc_vector&&)      = default;
    disc_vector& operator=(const disc_vector&) = delete;
    disc_vector& operator=(disc_vector&&)      = default;
    virtual ~disc_vector() = default;
    disc_vector(const std::string file, std::size_t cache_size) : _ifs{ file, std::ifstream::binary | std::ifstream::in | std::ifstream::ate }
    {
        if (!_ifs.good()) throw "Failed to open/create cache file for disc vector";
        _size = _ifs.tellg() / sizeof(T);
        _ifs.seekg(std::ifstream::beg);
        _cache.resize(cache_size);
    };

    const T operator [] (std::size_t i) {
        if (!is_in_cache(i)) {
            read_block(i);
        }
        return _cache[i % _cache.size()];
    }
    std::size_t size() const { return _size; }
private:

    bool is_in_cache(const std::size_t i) const {
        return
            _has_cache &&
            _cache.size() * _block <= i &&
            _cache.size() * (_block + 1) > i;
    }

    void read_block(const std::size_t i) {
        _block = i / _cache.size();
        std::size_t num_to_read       = _cache.size();
        const std::size_t read_start  = _block * _cache.size();
        const std::size_t read_end    = read_start + num_to_read;
        const std::size_t max_read_to = _size;
        if (read_end > max_read_to) {
            num_to_read = max_read_to - read_start;
        }
        _ifs.seekg(read_start * sizeof(T), std::ifstream::beg);
        _ifs.read(reinterpret_cast<char*>(_cache.data()),
                  num_to_read * sizeof(T));
        _has_cache = true;
    }

    std::ifstream _ifs;
    std::vector<T> _cache;
    bool _has_cache{ false };
    std::size_t _block{ 0 };
    std::size_t _size{ 0 };
};
