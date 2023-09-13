cmake_minimum_required(VERSION 3.14)
project(Implementations LANGUAGES CUDA CXX)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(ColorCompression/)
include_directories(DAG/)
include_directories(DAGConstructor/)
include_directories(DAGLoader/)
include_directories(DAGTracer)
include_directories(glTFLoader/)
include_directories(src)
include_directories(utils/)

# Internal library ---- DAG
add_library(DAG STATIC
        DAG/DAG/DAG.cpp
        DAG/DAG/DAG.h
        utils/utils/Aabb.cpp
        utils/utils/Aabb.h
        shader_helpers.cpp
        shader_helpers.hpp
        utils/utils/glm_extensions.cpp
        utils/utils/glm_extensions.h
        utils/utils/orientation.cpp
        utils/utils/orientation.h
        utils/utils/view.cpp
        utils/utils/view.h
)
target_include_directories(DAG PUBLIC DAGExample/DAG DAGExample)
target_link_libraries(DAG PUBLIC glm glad) #Needed for chag::Aabb
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
target_include_directories(Voxelizer PUBLIC Voxelizer DAGExample)
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
target_include_directories(DAGBuilder PUBLIC DAGExample/DAGBuilder DAGExample)
target_link_libraries(DAGBuilder 
    PUBLIC
        glfw
        OpenGL::GL
        glad
        glm
        stb
        DAG
    PRIVATE
        Tracy::TracyClient
        #CUDA::cudart
        #CUDA::cuda_driver
        #Thrust
)
set_target_properties(DAGBuilder
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
# Executable ---- DAGBuilderCLI
add_executable(DAGBuilderCLI
    DAGConstructor/DAGConstructor/cli_main.cpp
    #ColorCompression/ColorCompression/BlockBuild.h
    #ColorCompression/ColorCompression/ours.cpp
    #ColorCompression/ColorCompression/ours.h
    #ColorCompression/ColorCompression/ours.cu
    #ColorCompression/ColorCompression/ours_varbit.cpp
    #ColorCompression/ColorCompression/ours_varbit.h
    #ColorCompression/ColorCompression/svd.cpp
    #ColorCompression/ColorCompression/svd.h
    DAGLoader/DAGLoader/DAGLoader.cpp
    DAGLoader/DAGLoader/DAGLoader.h
    DAGTracer/DAGTracer/DAGTracer.cu
    DAGTracer/DAGTracer/DAGTracer.h
    glTFLoader/glTFLoader/nlohmann/json.hpp
    glTFLoader/glTFLoader/glTFLoader.cpp
    glTFLoader/glTFLoader/glTFLoader.h
    src/voxelize_and_merge.cpp
    src/voxelize_and_merge.h
    bits_in_uint_array.h
    CudaHelpers.h
    hash.h
    morton.h
    shader_sources.hpp
    app.hpp
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
        #CUDA::cudart
        #CUDA::cuda_driver
        #Thrust
        cereal
        Voxelizer
        DAG
)
#target_compile_definitions(DAGBuilderCLI PUBLIC NO_TRACE)
set_target_properties(DAGBuilderCLI
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
        #CUDA::cudart
        #CUDA::cuda_driver
        #Thrust
        cereal
        Voxelizer
        DAG
        DAGBuilder
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
