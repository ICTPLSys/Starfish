foreach(test_name IN ITEMS configuration full_stripe handshake settings request_shards thread_registry write_ring write_fence rollback_layout wait_policy)
    add_executable(test_carbink_${test_name} ${test_name}.cpp)
    target_compile_options(test_carbink_${test_name} PRIVATE -UNDEBUG)
    target_link_libraries(test_carbink_${test_name} PRIVATE Threads::Threads hydra_isal)
    add_test(NAME carbink_${test_name} COMMAND test_carbink_${test_name})
    set_tests_properties(carbink_${test_name} PROPERTIES TIMEOUT 60)
endforeach()
add_test(NAME carbink_settings_override COMMAND test_carbink_settings overrides)
set_tests_properties(carbink_settings_override PROPERTIES
    ENVIRONMENT "FARLIB_COMPACTION_MAX_UPDATES_PER_SEC=50000;FARLIB_COMPACTION_INFLIGHT_UPDATES=64;FARLIB_COMPACTION_DRAIN_TASK_BUDGET=7"
    TIMEOUT 60)

add_executable(test_carbink_entry_guard entry_guard.cpp)
target_compile_options(test_carbink_entry_guard PRIVATE -UNDEBUG)
target_link_libraries(test_carbink_entry_guard PRIVATE ${COMMON_LIBS})
add_test(NAME carbink_entry_guard COMMAND test_carbink_entry_guard)
set_tests_properties(carbink_entry_guard PROPERTIES TIMEOUT 60)

add_executable(test_carbink_stripe_state stripe_state.cpp)
target_compile_options(test_carbink_stripe_state PRIVATE -UNDEBUG)
target_link_libraries(test_carbink_stripe_state PRIVATE ${COMMON_LIBS})
add_test(NAME carbink_stripe_state COMMAND test_carbink_stripe_state)
set_tests_properties(carbink_stripe_state PROPERTIES TIMEOUT 60)

add_executable(test_carbink_server_pending_item server_pending_item.cpp)
target_compile_options(test_carbink_server_pending_item PRIVATE -UNDEBUG)
target_include_directories(test_carbink_server_pending_item BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../legacy/carbink_server/include)
target_link_libraries(test_carbink_server_pending_item PRIVATE Threads::Threads)
add_test(NAME carbink_server_pending_item COMMAND test_carbink_server_pending_item)
set_tests_properties(carbink_server_pending_item PROPERTIES TIMEOUT 60)

# Manual two-physical-host check. Never launch it implicitly with CPU ctest.
add_executable(test_carbink_remote_compact remote_compact.cpp)
target_compile_options(test_carbink_remote_compact PRIVATE -UNDEBUG)
target_link_libraries(test_carbink_remote_compact PRIVATE ${COMMON_LIBS})

# Original-shaped 512-byte workload; run manually with the recorded two-host
# setup. FARLIB_OBJECT_SIZE_VERIFY=1 adds an explicit full-data oracle.
add_executable(carbink_object_size object_size.cpp)
target_compile_options(carbink_object_size PRIVATE -UNDEBUG)
target_link_libraries(carbink_object_size PRIVATE hdr_histogram::hdr_histogram ${COMMON_LIBS})
