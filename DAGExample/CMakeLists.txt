cmake_minimum_required(VERSION 3.18)
enable_language(CUDA CXX)
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#code-changes-for-separate-compilation
# https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(ColorCompression)
include_directories(glTFLoader)
include_directories(src)
#include_directories(DAG)
#include_directories(DAGConstructor)
include_directories(DAGLoader)
include_directories(DAGTracer)
#include_directories(utils)

add_library(CommonCompilerFlags INTERFACE)
target_compile_options(CommonCompilerFlags 
    INTERFACE 
        $<$<COMPILE_LANGUAGE:CUDA>:
            -diag-suppress 20012; # Annoing waring from __device__ explicit constructor something...
            #--expt-relaxed-constexpr; # allowing us to call constexpr functions in CUDA device code (without warnings)
        >
)

# Internal library ---- DAG
add_library(DAG STATIC
    DAG/DAG/DAG.cpp
    DAG/DAG/DAG.h
    utils/utils/Aabb.cpp
    utils/utils/Aabb.h
    utils/utils/glm_extensions.cpp
    utils/utils/glm_extensions.h
    utils/utils/orientation.cpp
    utils/utils/orientation.h
    utils/utils/view.cpp
    utils/utils/view.h
)
target_include_directories(DAG PUBLIC DAG utils)
target_link_libraries(DAG PUBLIC glm)
set_target_properties(DAG
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
)
# Internal library ---- Voxelizer
add_library(Voxelizer STATIC
    Voxelizer/Voxelizer/Voxelizer.cpp
    Voxelizer/Voxelizer/Voxelizer.h
)
target_include_directories(Voxelizer PUBLIC Voxelizer)
target_link_libraries(Voxelizer PUBLIC glad glm DAG)
set_target_properties(Voxelizer
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
)
# Internal library ---- DAGBuilder
add_library(DAGBuilder STATIC
    # Impl
    DAGConstructor/DAGConstructor/DAGConstructor.cpp
    DAGConstructor/DAGConstructor/DAGConstructor_impl.cu
    DAGConstructor/DAGConstructor/Merger.cpp
    # Headers
    DAGConstructor/DAGConstructor/DAGConstructor.h
    DAGConstructor/DAGConstructor/DAGConstructor_impl.h
    DAGConstructor/DAGConstructor/Merger.h
)
target_include_directories(DAGBuilder PUBLIC DAGConstructor)
target_link_libraries(DAGBuilder 
    PUBLIC
        OpenGL::GL
        glad
        glm
        DAG
    PRIVATE
        Tracy::TracyClient
        CommonCompilerFlags
)
set_target_properties(DAGBuilder
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES "${CHAG_CUDA_ARCH}"
)

add_executable(DAG_Compression
    ColorCompression/ColorCompression/BlockBuild.h
    ColorCompression/ColorCompression/ours.cpp
    ColorCompression/ColorCompression/ours.h
    ColorCompression/ColorCompression/ours.cu
    ColorCompression/ColorCompression/ours_varbit.cpp
    ColorCompression/ColorCompression/ours_varbit.h
    ColorCompression/ColorCompression/svd.cpp
    ColorCompression/ColorCompression/svd.h
    DAGLoader/DAGLoader/DAGLoader.cpp
    DAGLoader/DAGLoader/DAGLoader.h
    DAGTracer/DAGTracer/DAGTracer.cu
    DAGTracer/DAGTracer/DAGTracer.h
    glTFLoader/glTFLoader/nlohmann/json.hpp
    glTFLoader/glTFLoader/glTFLoader.cpp
    glTFLoader/glTFLoader/glTFLoader.h
    src/my_exe.cpp
    src/voxelize_and_merge.cpp
    src/voxelize_and_merge.h
    bits_in_uint_array.h
    CudaHelpers.h
    hash.h
    morton.h
    shader_sources.hpp
    shader_helpers.cpp
    shader_helpers.hpp
    app.hpp
)


target_link_libraries(DAG_Compression
    PRIVATE
        glfw
        OpenGL::GL
        glad
        glm
        stb
        Tracy::TracyClient
        cereal
        Voxelizer
        DAG
        DAGBuilder
        CommonCompilerFlags
)

set_target_properties(DAG_Compression
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES "${CHAG_CUDA_ARCH}"
)
