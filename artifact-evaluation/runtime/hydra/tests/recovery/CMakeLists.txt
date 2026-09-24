set(RECOVERY_CPU_TESTS
    config_uint8_numeric
    ec_batch_pending_backpressure
    ec_read_context
    ec_read_four_read
    ec_recovery_scratch_growth
    ec_split_foundation
    ec_split_manager
    entry_frequency
    entry_frequency_lifecycle
    entry_frequency_oom
    recomputable_entry
    recompute_recipe)

foreach(test_name IN LISTS RECOVERY_CPU_TESTS)
    add_executable(test_${test_name} ${test_name}.cpp)
    target_compile_options(test_${test_name} PRIVATE -UNDEBUG)
    target_link_libraries(test_${test_name} PRIVATE Threads::Threads hydra_isal)
    add_test(NAME recovery_${test_name} COMMAND test_${test_name})
    set_tests_properties(recovery_${test_name} PROPERTIES TIMEOUT 60)
endforeach()
