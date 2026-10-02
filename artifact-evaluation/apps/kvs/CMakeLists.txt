add_executable(kvs_throughput throughput.cpp)
target_compile_definitions(kvs_throughput PRIVATE FARLIB_KVS_OBJECT_BYTES=512)
# Retain the baseline's locked-GET/deref-stats header without shadowing runtime headers.
target_include_directories(kvs_throughput BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(kvs_throughput hdr_histogram::hdr_histogram ${COMMON_LIBS})
set_target_properties(kvs_throughput PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/benchmark/kvs")

add_executable(test_kvs_logging EXCLUDE_FROM_ALL logging_test.cpp)
target_compile_features(test_kvs_logging PRIVATE cxx_std_20)
target_link_libraries(test_kvs_logging PRIVATE Threads::Threads)
