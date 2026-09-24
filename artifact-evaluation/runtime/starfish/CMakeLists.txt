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
if(ISAL_LIBRARY)
    message(STATUS "Found ISA-L library: ${ISAL_LIBRARY}")
else()
    message(STATUS "ISA-L not found (optional)")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
option(FARLIB_BUILD_TESTS "Build FarLib tests" OFF)
if(FARLIB_BUILD_TESTS)
    enable_testing()
endif()

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    /usr/include/
)

link_directories(/usr/lib/x86_64-linux-gnu/)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_library(libcache STATIC
    ${SRC_DIR}/rdma/client.cpp
    ${SRC_DIR}/rdma/exchange_msg.cpp
    ${SRC_DIR}/cache/cache.cpp
    ${SRC_DIR}/design1/backup_budget.cpp
    ${SRC_DIR}/utils/stats.cpp
    ${SRC_DIR}/utils/control.cpp
    ${SRC_DIR}/utils/uthreads.cpp)
add_library(libasync STATIC ${SRC_DIR}/utils/signal.cpp)
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
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
        message(FATAL_ERROR "FARLIB_BUILD_TESTS requires ${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
    endif()
    add_subdirectory(test)
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
add_subdirectory("${FARLIB_APPS_DIR}/llama" "${CMAKE_CURRENT_BINARY_DIR}/apps/llama")
add_subdirectory("${FARLIB_APPS_DIR}/bfs" "${CMAKE_CURRENT_BINARY_DIR}/apps/bfs")
