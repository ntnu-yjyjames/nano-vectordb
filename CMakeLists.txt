cmake_minimum_required(VERSION 3.16)
project(NanoVectorDB LANGUAGES CXX)  

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
