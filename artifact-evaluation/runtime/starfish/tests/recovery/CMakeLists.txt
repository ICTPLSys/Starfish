set(RECOVERY_CPU_TESTS
    config_uint8_numeric
    ec_batch_pending_backpressure
    ec_batch_worker_private
    ec_diagnostic_counter
    ec_isal_codec
    ec_update_participant
    ec_slot_reuse
    ec_direct_codec
    ec_direct_write_bank
    ec_direct_group
    ec_direct_size_classes
    ec_size_class_builder
    ec_direct_source_gate
    evict_breakdown
    ec_space_accounting
    ec_space_reporting
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
    target_sources(test_${test_name} PRIVATE $<TARGET_OBJECTS:cpu_profile_state>)
    target_compile_options(test_${test_name} PRIVATE -UNDEBUG)
    target_link_libraries(test_${test_name} PRIVATE Threads::Threads ${ISAL_LIBRARY})
    add_test(NAME recovery_${test_name} COMMAND test_${test_name})
    set_tests_properties(recovery_${test_name} PROPERTIES TIMEOUT 60)
endforeach()

# Count real coefficient expansions in the test, not in the production hot path.
target_link_options(test_ec_isal_codec PRIVATE -Wl,--wrap=ec_init_tables)
set_tests_properties(recovery_evict_breakdown PROPERTIES SKIP_RETURN_CODE 77)

# This test exercises the real allocator helpers and their real runtime deps.
add_executable(test_exclusive_work_alignment exclusive_work_alignment.cpp)
target_compile_options(test_exclusive_work_alignment PRIVATE -UNDEBUG)
target_link_libraries(test_exclusive_work_alignment PRIVATE ${COMMON_LIBS})
add_test(NAME recovery_exclusive_work_alignment COMMAND test_exclusive_work_alignment)
set_tests_properties(recovery_exclusive_work_alignment PROPERTIES TIMEOUT 60)
