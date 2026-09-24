add_executable(run_chat_far run_chat_far.cpp)
target_link_libraries(run_chat_far ${COMMON_LIBS} m)

if(DEFINED FARLIB_LLAMA_OUTPUT_DIR)
    set_target_properties(run_chat_far
                          PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                     "${FARLIB_LLAMA_OUTPUT_DIR}")
endif()
