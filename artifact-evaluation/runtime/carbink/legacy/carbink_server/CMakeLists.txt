cmake_minimum_required(VERSION 3.16)

project(carbink_server LANGUAGES CXX)

find_package(Threads REQUIRED)

find_path(CARBINK_ISAL_INCLUDE_DIR
    NAMES erasure_code.h
    PATH_SUFFIXES isa-l
)
find_library(CARBINK_ISAL_LIBRARY NAMES isal)

if(CARBINK_ISAL_INCLUDE_DIR AND
   EXISTS "${CARBINK_ISAL_INCLUDE_DIR}/erasure_code.h" AND
   NOT EXISTS "${CARBINK_ISAL_INCLUDE_DIR}/isa-l/erasure_code.h")
    get_filename_component(CARBINK_ISAL_INCLUDE_DIR
        "${CARBINK_ISAL_INCLUDE_DIR}" DIRECTORY)
endif()

if(NOT CARBINK_ISAL_INCLUDE_DIR OR NOT CARBINK_ISAL_LIBRARY)
    message(FATAL_ERROR
        "ISA-L is required; set CMAKE_INCLUDE_PATH/CMAKE_LIBRARY_PATH or "
        "CARBINK_ISAL_INCLUDE_DIR/CARBINK_ISAL_LIBRARY")
endif()

add_executable(carbink_server
    src/rdma/server.cpp
    src/rdma/exchange_msg.cpp
)

target_compile_features(carbink_server PRIVATE cxx_std_20)

target_include_directories(carbink_server
    BEFORE PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CARBINK_ISAL_INCLUDE_DIR}
)

target_link_libraries(carbink_server
    PRIVATE
        ibverbs
        Threads::Threads
        ${CARBINK_ISAL_LIBRARY}
)
