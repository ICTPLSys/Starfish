add_executable(nhop_graph nhop_graph.cpp)
target_include_directories(nhop_graph PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(nhop_graph hdr_histogram::hdr_histogram ${COMMON_LIBS})

if(DEFINED FARLIB_NQ_OUTPUT_DIR)
    set_target_properties(nhop_graph
                          PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                     "${FARLIB_NQ_OUTPUT_DIR}")
endif()
