cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(qwen_thor LANGUAGES CXX CUDA)

# ============================================================================
# 编译标准
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# CUDA 架构: SM110a (Blackwell / Jetson AGX Thor)
# ============================================================================
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_110a,code=sm_110a --expt-relaxed-constexpr -O3 --use_fast_math -Xptxas -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# ============================================================================
# 依赖
# ============================================================================
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# 包含目录
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer/include
)

add_definitions(-DFLAT_SM90A_ENABLED)

# ============================================================================
# 核心引擎静态库 (engine)
# ============================================================================
file(GLOB_RECURSE ENGINE_SOURCES
    "src/engine/*.cpp"
    "src/engine/*.cu"
)

add_library(qwen_thor_engine STATIC ${ENGINE_SOURCES})
target_link_libraries(qwen_thor_engine PRIVATE CUDA::cudart CUDA::cublas rt dl)

# ============================================================================
# 主可执行文件: qwen3-27b-thor
# ============================================================================
add_executable(qwen3-27b-thor
    src/main.cpp
    src/tests.cpp
    src/benchmark.cpp
    src/serve/serve.cpp
    src/tui/tui.cpp
)
target_link_libraries(qwen3-27b-thor PRIVATE qwen_thor_engine)
