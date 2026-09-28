# Synthetic bandwidth workload; keep the measured algorithm unchanged.
foreach(payload IN ITEMS 4000 512)
    if(payload EQUAL 4000)
        set(target bandwidth_microbenchmark)
        set(legacy_target object_size)
    else()
        set(target bandwidth_microbenchmark_${payload})
        set(legacy_target object_size_${payload})
    endif()
    add_executable(${target} bandwidth_microbenchmark.cpp)
    target_compile_definitions(${target} PRIVATE OBJECT_SIZE_BYTES=${payload})
    target_link_libraries(${target} hdr_histogram ${COMMON_LIBS})
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/benchmark/microbenchmarks")
    # Preserve historical external runners without recompiling the workload.
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "$<TARGET_FILE_NAME:${target}>"
            "$<TARGET_FILE_DIR:${target}>/${legacy_target}"
        BYPRODUCTS "${CMAKE_BINARY_DIR}/benchmark/microbenchmarks/${legacy_target}"
        VERBATIM)
    add_custom_target(${legacy_target} DEPENDS ${target})
endforeach()
