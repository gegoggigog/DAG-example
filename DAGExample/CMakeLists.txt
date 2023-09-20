cmake_minimum_required(VERSION 3.18)
enable_language(CUDA CXX)
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#code-changes-for-separate-compilation
# https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

#find_package(OpenMP)

add_library(CommonCompilerFlags INTERFACE)
target_compile_options(CommonCompilerFlags 
    INTERFACE 
        $<$<COMPILE_LANGUAGE:CUDA>:
            -diag-suppress 20012; # Annoing waring from __device__ explicit constructor something...
            #--expt-relaxed-constexpr; # allowing us to call constexpr functions in CUDA device code (without warnings)
        >
)

# Internal library ---- utils
add_library(utils STATIC
    utils/utils/Aabb.cpp
    utils/utils/DAG.cpp
    utils/utils/glm_extensions.cpp  
    utils/utils/orientation.cpp
    utils/utils/view.cpp
    utils/utils/glTFLoader/glTFLoader.cpp
)
target_include_directories(utils PUBLIC utils)
target_link_libraries(utils 
    PUBLIC glm glad OpenGL::GL
    PRIVATE stb
)
set_target_properties(utils
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
)

# Internal library ---- Voxelizer
add_library(Voxelizer STATIC Voxelizer/Voxelizer/Voxelizer.cpp)
target_include_directories(Voxelizer PUBLIC Voxelizer)
target_link_libraries(Voxelizer PUBLIC OpenGL::GL glad glm utils)
set_target_properties(Voxelizer
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
)

# Internal library ---- DAGBuilder
add_library(DAGBuilder STATIC
    DAGConstructor/DAGConstructor/DAGConstructor.cpp
    DAGConstructor/DAGConstructor/DAGConstructor_impl.cu
    DAGConstructor/DAGConstructor/Merger.cpp
)
target_include_directories(DAGBuilder PUBLIC DAGConstructor)
target_link_libraries(DAGBuilder 
    PUBLIC
        glm
        utils
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

# Internal library ---- ColorCompression
add_library(ColorCompression STATIC
    ColorCompression/ColorCompression/ours.cu
    ColorCompression/ColorCompression/ours_varbit.cpp
)
target_include_directories(ColorCompression PUBLIC ColorCompression)
target_link_libraries(ColorCompression 
    PUBLIC
        glm
        utils
    PRIVATE
        Tracy::TracyClient
        CommonCompilerFlags
        #OpenMP::OpenMP_CXX
)
set_target_properties(ColorCompression
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES "${CHAG_CUDA_ARCH}"
)


add_executable(DAG_Compression
    DAGLoader/DAGLoader/DAGLoader.cpp
    DAGTracer/DAGTracer/DAGTracer.cu
    src/my_exe.cpp
    src/voxelize_and_merge.cpp
)

target_include_directories(DAG_Compression 
    PRIVATE 
        src
        DAGLoader
        DAGTracer
)

target_link_libraries(DAG_Compression
    PRIVATE
        Voxelizer
        DAGBuilder
        ColorCompression
        utils
        glfw
        OpenGL::GL
        glad
        glm
        cereal
        CommonCompilerFlags
        Tracy::TracyClient
)

set_target_properties(DAG_Compression
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES "${CHAG_CUDA_ARCH}"
)
