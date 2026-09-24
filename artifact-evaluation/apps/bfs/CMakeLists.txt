add_executable(gapbs_bfs_chunked gapbs_bfs_chunked.cpp)
target_include_directories(gapbs_bfs_chunked PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(gapbs_bfs_chunked hdr_histogram::hdr_histogram ${COMMON_LIBS})

if(DEFINED FARLIB_BFS_OUTPUT_DIR)
    set_target_properties(gapbs_bfs_chunked
                          PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                     "${FARLIB_BFS_OUTPUT_DIR}")
endif()
