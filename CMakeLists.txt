cmake_minimum_required(VERSION 3.16)
project(NanoVectorDB LANGUAGES CXX)  
option(NVDB_WITH_FAISS "Build FAISS-based IVF tools" ON)

# set complier arguments
set(CMAKE_CXX_STANDARD 17)    
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 

option(NVDB_ENABLE_SANITIZER "Enable address/ub sanitizer" OFF)

add_library(nvdb
  src/mmap_file.cpp
  src/vector_dataset.cpp
  src/flat_index.cpp
  src/flat_index_omp.cpp
  src/flat_index_async.cpp
  src/flat_index_pool.cpp
  src/simd_dot.cpp   
)



target_include_directories(nvdb PUBLIC include)

if (NVDB_ENABLE_SANITIZER)
  target_compile_options(nvdb PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(nvdb PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
endif()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP_CXX_FOUND=${OpenMP_CXX_FOUND}")
  target_link_libraries(nvdb PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(nvdb PUBLIC NVDB_HAS_OPENMP=1)
else()
  target_compile_definitions(nvdb PUBLIC NVDB_HAS_OPENMP=0)
endif()

add_executable(nvdb_dump apps/nvdb_dump.cpp)
target_link_libraries(nvdb_dump PRIVATE nvdb)

add_executable(nvdb_sanity apps/nvdb_sanity.cpp)
target_link_libraries(nvdb_sanity PRIVATE nvdb)

add_executable(nvdb_search apps/nvdb_search.cpp)
target_link_libraries(nvdb_search PRIVATE nvdb)

add_executable(nvdb_slice tools/nvdb_slice.cpp)
target_link_libraries(nvdb_slice PRIVATE nvdb)

add_executable(nvdb_make_query tools/nvdb_make_query.cpp)
target_link_libraries(nvdb_make_query PRIVATE nvdb)

add_executable(nvdb_bench apps/nvdb_bench.cpp)
target_link_libraries(nvdb_bench PRIVATE nvdb)

add_executable(nvdb_convert_f16 tools/nvdb_convert_f16.cpp)
target_link_libraries(nvdb_convert_f16 PRIVATE nvdb)

add_executable(nvdb_quantize_i8  apps/nvdb_quantize_i8.cpp)
target_link_libraries(nvdb_quantize_i8 PRIVATE nvdb)

add_executable(nvdb_hnsw_eval apps/nvdb_hnsw_eval.cpp)
target_link_libraries(nvdb_hnsw_eval PRIVATE nvdb)
target_include_directories(nvdb_hnsw_eval PRIVATE third_party/hnswlib)

add_executable(nvdb_hnsw_build apps/nvdb_hnsw_build.cpp)
target_link_libraries(nvdb_hnsw_build PRIVATE nvdb)
target_include_directories(nvdb_hnsw_build PRIVATE third_party/hnswlib)

add_executable(nvdb_hnsw_search apps/nvdb_hnsw_search.cpp)
target_link_libraries(nvdb_hnsw_search PRIVATE nvdb)
target_include_directories(nvdb_hnsw_search PRIVATE third_party/hnswlib)


# ---- FAISS (optional; required for IVF tools) ----
set(FAISS_ROOT "$ENV{FAISS_ROOT}")
if(FAISS_ROOT STREQUAL "")
  set(FAISS_ROOT "$ENV{HOME}/.local")
endif()

# Look for libfaiss in lib or lib64
find_library(FAISS_LIB
  NAMES faiss libfaiss.a libfaiss.so
  HINTS "${FAISS_ROOT}/lib" "${FAISS_ROOT}/lib64"
)

# (optional) FAISS headers check (prevents confusing include errors)
find_path(FAISS_INCLUDE_DIR
  NAMES faiss/Index.h
  HINTS "${FAISS_ROOT}/include"
)

if (FAISS_LIB AND FAISS_INCLUDE_DIR)
  message(STATUS "FAISS_ROOT=${FAISS_ROOT}")
  message(STATUS "FAISS_INCLUDE_DIR=${FAISS_INCLUDE_DIR}")
  message(STATUS "FAISS_LIB=${FAISS_LIB}")

  add_executable(nvdb_ivf_build apps/nvdb_ivf_build.cpp)
  target_include_directories(nvdb_ivf_build PRIVATE "${FAISS_INCLUDE_DIR}")
  target_link_libraries(nvdb_ivf_build PRIVATE nvdb ${FAISS_LIB} openblas)

  add_executable(nvdb_ivf_eval apps/nvdb_ivf_eval.cpp)
  target_include_directories(nvdb_ivf_eval PRIVATE "${FAISS_INCLUDE_DIR}")
  target_link_libraries(nvdb_ivf_eval PRIVATE nvdb ${FAISS_LIB} openblas)

  add_executable(nvdb_ivfpq_build apps/nvdb_ivfpq_build.cpp)
  target_include_directories(nvdb_ivfpq_build PRIVATE "${FAISS_INCLUDE_DIR}")
  target_link_libraries(nvdb_ivfpq_build PRIVATE nvdb ${FAISS_LIB} openblas)

else()
  message(WARNING "FAISS not found (FAISS_ROOT=${FAISS_ROOT}). IVF/IVFPQ tools will not be built.")
endif()
if (TARGET nvdb_ivf_eval)
  target_link_libraries(nvdb_ivf_eval PRIVATE nvdb_cuda)
endif()

# ---- Ground truth builder should NOT depend on FAISS ----
add_executable(nvdb_gt_build apps/nvdb_gt_build.cpp)
target_link_libraries(nvdb_gt_build PRIVATE nvdb)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 3080 = sm_86
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(nvdb_cuda_sanity apps/nvdb_cuda_sanity.cu)
target_compile_options(nvdb_cuda_sanity PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)



add_library(nvdb_cuda STATIC src/cuda_refine.cu)
target_include_directories(nvdb_cuda PUBLIC include)
target_compile_options(nvdb_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
# CMake CUDA provides cudart target
target_link_libraries(nvdb_cuda PUBLIC CUDA::cudart)

