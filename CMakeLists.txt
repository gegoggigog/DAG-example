cmake_minimum_required(VERSION 3.14)
project(DAG_Compression)

#set(CMAKE_EXE_LINKER_FLAGS "/VERBOSE:LIB")

set(CHAG_CUDA_ARCH
        #  61-real # 1080
        # 75-real # 2060
        86-real # 3090
)

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

#glfw
#add_subdirectory(${glfw_download_SOURCE_DIR})

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


find_package(OpenGL REQUIRED)
#find_package(CUDAToolkit REQUIRED)
#find_package(Thrust REQUIRED CONFIG)
#thrust_create_target(Thrust)

# set(CMAKE_CUDA_HOST_COMPILER gcc-8)
#XXX add -g to debug
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61)

# Add --expt-relaxed-constexpr, allowing us to call constexpr functions in
# CUDA device code (without warnings)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_subdirectory(DAGExample)