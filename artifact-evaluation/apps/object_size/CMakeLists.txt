# The original object_size workload, unchanged; preserve its runner binary names.
foreach(payload IN ITEMS 4000 512)
    if(payload EQUAL 4000)
        set(target object_size)
    else()
        set(target object_size_${payload})
    endif()
    add_executable(${target} object_size.cpp)
    target_compile_definitions(${target} PRIVATE OBJECT_SIZE_BYTES=${payload})
    target_link_libraries(${target} hdr_histogram ${COMMON_LIBS})
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/benchmark/microbenchmarks")
endforeach()
