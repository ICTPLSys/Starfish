cmake_minimum_required(VERSION 3.16)
project(FarLib)

enable_language(CXX ASM)

find_package(Threads REQUIRED)
find_package(hdr_histogram REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenSSL REQUIRED)

# Find PAPI manually since it doesn't provide CMake config files everywhere.
find_path(PAPI_INCLUDE_DIR NAMES papi.h PATHS /usr/local/include /usr/include)
find_library(PAPI_LIBRARY NAMES papi libpapi PATHS /usr/local/lib /usr/lib /usr/lib64)
if(PAPI_INCLUDE_DIR AND PAPI_LIBRARY)
    message(STATUS "Found PAPI include: ${PAPI_INCLUDE_DIR}")
    message(STATUS "Found PAPI library: ${PAPI_LIBRARY}")
    include_directories(${PAPI_INCLUDE_DIR})
    add_compile_definitions(FARLIB_HAS_PAPI=1)
else()
    message(STATUS "PAPI counters disabled (optional)")
endif()

find_library(ISAL_LIBRARY NAMES isal libisal PATHS /usr/local/lib /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu)
find_path(ISAL_INCLUDE_DIR NAMES erasure_code.h PATH_SUFFIXES isa-l)
if(NOT ISAL_LIBRARY OR NOT ISAL_INCLUDE_DIR)
    message(FATAL_ERROR "ISA-L headers and library are required for the EC codec")
endif()
message(STATUS "ISA-L EC codec: ${ISAL_LIBRARY}; headers: ${ISAL_INCLUDE_DIR}")
include_directories(SYSTEM ${ISAL_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
option(FARLIB_BUILD_TESTS "Build FarLib tests" OFF)
option(FARLIB_ENABLE_EVICT_BREAKDOWN "Compile runtime logical-worker eviction timing" OFF)
add_compile_definitions(FARLIB_ENABLE_EVICT_BREAKDOWN=$<BOOL:${FARLIB_ENABLE_EVICT_BREAKDOWN}>)
option(FARLIB_ENABLE_EC_RMW_TIMING
       "Compile sampled EC RMW timing diagnostics" OFF)
add_compile_definitions(FARLIB_ENABLE_EC_RMW_TIMING=$<BOOL:${FARLIB_ENABLE_EC_RMW_TIMING}>)
option(FARLIB_BUILD_WORDCOUNT
       "Build the optional recomputable/recipe Wordcount targets" OFF)
option(FARLIB_BUILD_MG
       "Build the optional MG benchmark target" OFF)
option(FARLIB_BUILD_NQ
       "Build the optional frozen Friendster n-hop graph benchmark" OFF)
option(FARLIB_BUILD_BANDWIDTH_MICROBENCHMARK
       "Build the synthetic 512/4000-byte bandwidth microbenchmark" OFF)
option(FARLIB_BUILD_OBJECT_SIZE
       "Deprecated alias for FARLIB_BUILD_BANDWIDTH_MICROBENCHMARK" OFF)
if(FARLIB_BUILD_OBJECT_SIZE)
    message(DEPRECATION "Use FARLIB_BUILD_BANDWIDTH_MICROBENCHMARK instead")
    set(FARLIB_BUILD_BANDWIDTH_MICROBENCHMARK ON)
endif()
option(FARLIB_BUILD_KVS "Build the aligned 512-byte KV throughput benchmark" OFF)

# This macro changes ConcurrentArrayCache layout. Keep runtime and every
# header consumer on the same numeric value, including standalone diagnostics.
option(FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION
       "Isolate shared cache accounting fields on separate cache lines" ON)
set(FARLIB_CACHE_LAYOUT_DEFINITION
    FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION=$<BOOL:${FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION}>)
add_compile_definitions(${FARLIB_CACHE_LAYOUT_DEFINITION})
message(STATUS "Backup accounting cacheline isolation: ${FARLIB_BACKUP_ACCOUNTING_CACHELINE_ISOLATION}")
if(FARLIB_BUILD_TESTS)
    enable_testing()
endif()

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    /usr/include/
)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

link_directories(/usr/lib/x86_64-linux-gnu/)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_library(libcache STATIC
    ${SRC_DIR}/rdma/client.cpp
    ${SRC_DIR}/rdma/exchange_msg.cpp
    ${SRC_DIR}/cache/cache.cpp
    ${SRC_DIR}/cache/ec_update.cpp
    ${SRC_DIR}/design1/backup_budget.cpp
    ${SRC_DIR}/utils/stats.cpp
    ${SRC_DIR}/utils/control.cpp
    ${SRC_DIR}/utils/uthreads.cpp)
add_library(libasync STATIC ${SRC_DIR}/utils/signal.cpp)
# Also propagate the layout contract to targets outside this directory.
target_compile_definitions(libcache INTERFACE ${FARLIB_CACHE_LAYOUT_DEFINITION})
target_include_directories(libcache PUBLIC ${ISAL_INCLUDE_DIR})
target_link_libraries(libcache PUBLIC ${ISAL_LIBRARY})
set(COMMON_LIBS libcache libasync ibverbs Threads::Threads ${Boost_PROGRAM_OPTIONS_LIBRARY})
if(ISAL_LIBRARY)
    set(COMMON_LIBS ${COMMON_LIBS} ${ISAL_LIBRARY})
endif()

add_executable(server ${SRC_DIR}/rdma/server.cpp ${SRC_DIR}/rdma/exchange_msg.cpp)
target_link_libraries(server ibverbs Threads::Threads)
if(ISAL_LIBRARY)
    target_link_libraries(server ${ISAL_LIBRARY})
endif()

# libfibre, https://git.uwaterloo.ca/mkarsten/libfibre
set(LIBFIBRE_DIR "$ENV{LIBFIBRE_DIR}")
message(STATUS "LibFibre path: " ${LIBFIBRE_DIR})
find_library(LibFibre libfibre.so PATHS ${LIBFIBRE_DIR}/src REQUIRED)
include_directories(SYSTEM ${LIBFIBRE_DIR}/src)
set(COMMON_LIBS ${COMMON_LIBS} ${LibFibre})
if(PAPI_INCLUDE_DIR AND PAPI_LIBRARY)
    set(COMMON_LIBS ${COMMON_LIBS} ${PAPI_LIBRARY})
endif()

if(FARLIB_BUILD_TESTS)
    add_library(cpu_profile_state OBJECT tests/cpu_profile_state.cpp)
    add_subdirectory(tests/design2)
    add_subdirectory(tests/recovery)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/CMakeLists.txt")
        add_subdirectory(tests/integration)
    endif()
endif()

# Applications are shared by the independently built runtimes.
set(FARLIB_APPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../apps"
    CACHE PATH "Shared application source directory")
get_filename_component(FARLIB_APPS_DIR "${FARLIB_APPS_DIR}" ABSOLUTE)
foreach(app IN ITEMS llama bfs)
    if(NOT EXISTS "${FARLIB_APPS_DIR}/${app}/CMakeLists.txt")
        message(FATAL_ERROR "Missing shared application: ${FARLIB_APPS_DIR}/${app}")
    endif()
endforeach()

# Preserve the binary paths used by the existing runners.
set(FARLIB_LLAMA_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/benchmark/llama")
set(FARLIB_BFS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/benchmark/microbenchmarks")
set(FARLIB_NQ_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/benchmark/microbenchmarks")
add_subdirectory("${FARLIB_APPS_DIR}/llama" "${CMAKE_CURRENT_BINARY_DIR}/apps/llama")
add_subdirectory("${FARLIB_APPS_DIR}/bfs" "${CMAKE_CURRENT_BINARY_DIR}/apps/bfs")
if(FARLIB_BUILD_BANDWIDTH_MICROBENCHMARK)
    add_subdirectory("${FARLIB_APPS_DIR}/bandwidth_microbenchmark"
                     "${CMAKE_CURRENT_BINARY_DIR}/apps/bandwidth_microbenchmark")
endif()
if(FARLIB_BUILD_KVS)
    add_subdirectory("${FARLIB_APPS_DIR}/kvs" "${CMAKE_CURRENT_BINARY_DIR}/apps/kvs")
endif()
if(FARLIB_BUILD_NQ)
    if(NOT EXISTS "${FARLIB_APPS_DIR}/nq/CMakeLists.txt")
        message(FATAL_ERROR "Missing optional NQ application sources")
    endif()
    add_subdirectory("${FARLIB_APPS_DIR}/nq"
                     "${CMAKE_CURRENT_BINARY_DIR}/apps/nq")
endif()
if(FARLIB_BUILD_WORDCOUNT)
    set(FARLIB_WORDCOUNT_STARFISH_RUNTIME ON)
    if(NOT EXISTS "${FARLIB_APPS_DIR}/wordcount/CMakeLists.txt")
        message(FATAL_ERROR "Missing optional Wordcount application sources")
    endif()
    set(FARLIB_WORDCOUNT_OUTPUT_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/benchmark/mapreduce")
    add_subdirectory("${FARLIB_APPS_DIR}/wordcount"
                     "${CMAKE_CURRENT_BINARY_DIR}/apps/wordcount")
endif()

if(FARLIB_BUILD_MG)
    if(NOT EXISTS "${FARLIB_APPS_DIR}/mg/CMakeLists.txt")
        message(FATAL_ERROR "Missing optional MG application sources")
    endif()
    set(FARLIB_MG_OUTPUT_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/benchmark/mg")
    add_subdirectory("${FARLIB_APPS_DIR}/mg"
                     "${CMAKE_CURRENT_BINARY_DIR}/apps/mg")
    target_compile_definitions(mg PRIVATE FARLIB_STARFISH_MG=1)
    if(DEFINED MG_OBJECT_ALLOCATION_BYTES)
        if(NOT MG_OBJECT_ALLOCATION_BYTES MATCHES "^[1-9][0-9]*$")
            message(FATAL_ERROR "MG_OBJECT_ALLOCATION_BYTES must be a positive integer")
        endif()
        target_compile_definitions(mg PRIVATE MG_OBJECT_ALLOCATION_BYTES=${MG_OBJECT_ALLOCATION_BYTES})
    endif()
endif()
