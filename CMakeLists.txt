cmake_minimum_required(VERSION 3.24)
project(DAG_Compression LANGUAGES CXX)

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation
# https://en.wikipedia.org/wiki/CUDA
# virtual: lowest to run on
# real:    generate for gpu
# 61: 10XX
# 75: 20XX
# 86: 30XX
# 89: 40XX

# CMake >= 3.18: (Manually specify archs)
# ---------------------------------------
# set(CHAG_CUDA_ARCH 50-virtual 86-real)

# CMake >= 3.23: (Build to support all archs (?))
# -----------------------------------------------
# set(CHAG_CUDA_ARCH all)
# set(CHAG_CUDA_ARCH all-major)

# CMake >= 3.24: (Build only for current arch)
# --------------------------------------------
set(CHAG_CUDA_ARCH native)

include(cmake/CPM.cmake)

CPMAddPackage(
  NAME glm
  GIT_REPOSITORY https://github.com/g-truc/glm
  GIT_TAG 0.9.9.8
  DOWNLOAD_ONLY YES
)
CPMAddPackage(
  NAME stb
  GIT_REPOSITORY https://github.com/nothings/stb
  GIT_TAG master
  DOWNLOAD_ONLY YES
)
CPMAddPackage(
  NAME cereal
  VERSION 1.3.2
  GITHUB_REPOSITORY USCiLab/cereal
  OPTIONS "SKIP_PORTABILITY_TEST ON" "JUST_INSTALL_CEREAL ON"
)
CPMAddPackage(
        NAME glfw
        GITHUB_REPOSITORY glfw/glfw
        GIT_TAG 3.3.8
        OPTIONS
          "GLFW_BUILD_TESTS OFF"
          "GLFW_BUILD_EXAMPLES OFF"
          "GLFW_BULID_DOCS OFF"
        #DOWNLOAD_ONLY YES
)
CPMAddPackage(
  NAME cxxopts
  GITHUB_REPOSITORY jarro2783/cxxopts
  VERSION 2.2.0
  OPTIONS
    "CXXOPTS_BUILD_EXAMPLES Off"
    "CXXOPTS_BUILD_TESTS Off"
)
CPMAddPackage("gh:wolfpld/tracy@0.9.1")

# glad
add_library(glad external/glad/src/gl.c)
target_include_directories(glad PUBLIC external/glad/include)

# stb
file(WRITE ${CMAKE_BINARY_DIR}/stb_image.c 
[=[
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
]=])
add_library(stb ${CMAKE_BINARY_DIR}/stb_image.c)
target_include_directories(stb SYSTEM PUBLIC ${stb_SOURCE_DIR})

# glm
add_library(glm INTERFACE)
target_include_directories(glm SYSTEM INTERFACE ${glm_SOURCE_DIR})

# 

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)
#find_package(Thrust REQUIRED CONFIG)
#thrust_create_target(Thrust)

add_subdirectory(DAGExample)

include_directories(DAGExample)
include_directories(DAGExample/src)
include_directories(DAGExample/utils)
include_directories(DAGExample/glTFLoader)
# Executable ---- DAGBuilderCLI
add_executable(DAGBuilderCLI
    DAGExample/DAGConstructor/DAGConstructor/cli_main.cpp
    DAGExample/glTFLoader/glTFLoader/nlohmann/json.hpp
    DAGExample/glTFLoader/glTFLoader/glTFLoader.cpp
    DAGExample/glTFLoader/glTFLoader/glTFLoader.h
    DAGExample/src/voxelize_and_merge.cpp
    DAGExample/src/voxelize_and_merge.h
    DAGExample/bits_in_uint_array.h
    DAGExample/CudaHelpers.h
    DAGExample/hash.h
    DAGExample/morton.h
    DAGExample/shader_sources.hpp
    DAGExample/app.hpp
)
target_link_libraries(DAGBuilderCLI
    DAGBuilder
    cxxopts
    glfw
    OpenGL::GL
    glad
    glm
    stb
    Tracy::TracyClient
    CUDA::cudart
    cereal
    Voxelizer
    DAG
)

set_target_properties(DAGBuilderCLI
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
)