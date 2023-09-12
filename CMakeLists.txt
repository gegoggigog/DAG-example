cmake_minimum_required(VERSION 3.14)
project(DAG_Compression LANGUAGES CUDA CXX)

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
        NAME glfw_download
        GITHUB_REPOSITORY glfw/glfw
        GIT_TAG 3.3.8
        OPTIONS
          "GLFW_BUILD_TESTS OFF"
          "GLFW_BUILD_EXAMPLES OFF"
          "GLFW_BULID_DOCS OFF"
        DOWNLOAD_ONLY YES
)

CPMAddPackage("gh:wolfpld/tracy@0.9.1")

#glfw
add_subdirectory(${glfw_download_SOURCE_DIR})

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
target_include_directories(stb PUBLIC ${stb_SOURCE_DIR})

# glm
add_library(glm INTERFACE)
target_include_directories(glm SYSTEM INTERFACE ${glm_SOURCE_DIR})

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
#set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

# Ignore some warnings that add a lot of spam and aren't that critical:
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-parentheses -Wall -Wextra -Wconversion")

# find_package(glm REQUIRED)
# find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# set(CMAKE_CUDA_HOST_COMPILER gcc-8)
#XXX add -g to debug
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61)

# Add --expt-relaxed-constexpr, allowing us to call constexpr functions in
# CUDA device code (without warnings)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")


include_directories(DAGExample)
include_directories(DAGExample/ColorCompression/)
include_directories(DAGExample/DAG/)
include_directories(DAGExample/DAGConstructor/)
include_directories(DAGExample/DAGLoader/)
include_directories(DAGExample/DAGTracer)
include_directories(DAGExample/glTFLoader/)
include_directories(DAGExample/src)
include_directories(DAGExample/utils/)
include_directories(DAGExample/Voxelizer/)

add_library(Voxelizer 
    DAGExample/Voxelizer/Voxelizer/Voxelizer.cpp
)
target_include_directories(Voxelizer PUBLIC DAGExample/Voxelizer DAGExample)
target_link_libraries(Voxelizer PUBLIC glad glm)
set_target_properties(Voxelizer
        PROPERTIES
                CXX_STANDARD_REQUIRED ON
                CXX_STANDARD 17
)

add_executable(DAG_Compression
        DAGExample/ColorCompression/ColorCompression/BlockBuild.h
        DAGExample/ColorCompression/ColorCompression/ours.cpp
        DAGExample/ColorCompression/ColorCompression/ours.h
        DAGExample/ColorCompression/ColorCompression/ours.cu
        DAGExample/ColorCompression/ColorCompression/ours_varbit.cpp
        DAGExample/ColorCompression/ColorCompression/ours_varbit.h
        DAGExample/ColorCompression/ColorCompression/svd.cpp
        DAGExample/ColorCompression/ColorCompression/svd.h
        DAGExample/DAG/DAG/DAG.cpp
        DAGExample/DAG/DAG/DAG.h
        DAGExample/DAGConstructor/DAGConstructor/DAGConstructor.cu
        DAGExample/DAGConstructor/DAGConstructor/DAGConstructor.cuh
        DAGExample/DAGConstructor/DAGConstructor/DAGConstructor.cpp
        DAGExample/DAGConstructor/DAGConstructor/DAGConstructor.h
        DAGExample/DAGConstructor/DAGConstructor/Merger.cpp
        DAGExample/DAGConstructor/DAGConstructor/Merger.h
        DAGExample/DAGLoader/DAGLoader/DAGLoader.cpp
        DAGExample/DAGLoader/DAGLoader/DAGLoader.h
        DAGExample/DAGTracer/DAGTracer/DAGTracer.cu
        DAGExample/DAGTracer/DAGTracer/DAGTracer.h
        DAGExample/glTFLoader/glTFLoader/nlohmann/json.hpp
        DAGExample/glTFLoader/glTFLoader/glTFLoader.cpp
        DAGExample/glTFLoader/glTFLoader/glTFLoader.h
        DAGExample/src/my_exe.cpp
        DAGExample/src/voxelize_and_merge.cpp
        DAGExample/src/voxelize_and_merge.h
        DAGExample/utils/utils/Aabb.cpp
        DAGExample/utils/utils/Aabb.h
        DAGExample/utils/utils/glm_extensions.cpp
        DAGExample/utils/utils/glm_extensions.h
        DAGExample/utils/utils/orientation.cpp
        DAGExample/utils/utils/orientation.h
        DAGExample/utils/utils/view.cpp
        DAGExample/utils/utils/view.h
        DAGExample/bits_in_uint_array.h
        DAGExample/CudaHelpers.h
        DAGExample/hash.h
        DAGExample/morton.h
        DAGExample/shader_helpers.cpp
        DAGExample/shader_helpers.hpp
        DAGExample/shader_sources.hpp
        DAGExample/app.hpp
        )


target_link_libraries(DAG_Compression
        glfw
        OpenGL::GL
        glad
        glm
        stb
        Tracy::TracyClient
        CUDA::cudart
        CUDA::cuda_driver
        Thrust
        cereal
        Voxelizer
    )

set(CHAG_CUDA_ARCH
        #  61-real # 1080
        # 75-real # 2060
        86-real # 3090
)
set_target_properties(DAG_Compression
        PROPERTIES
                CXX_STANDARD_REQUIRED ON
                CXX_EXTENSIONS ON
                CXX_STANDARD 17
                CUDA_STANDARD 17
                CUDA_EXTENSIONS ON
                CUDA_SEPARABLE_COMPILATION ON
                # Need quotes here if we want pass multiple architectures. e.g., 61-real;75-real;86-real
                # https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
                # Using CMake >= 3.24 we can replace this with all, all-major, or native.
                CUDA_ARCHITECTURES "${CHAG_CUDA_ARCH}"
)