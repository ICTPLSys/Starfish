set(RECOVERY_INTEGRATION_TESTS
    test_recomputable_entry_lifecycle
    test_ec_behavior_groups
    test_failed_write_lifecycle)

foreach(test_name IN LISTS RECOVERY_INTEGRATION_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_compile_options(${test_name} PRIVATE -UNDEBUG)
    target_link_libraries(${test_name} PRIVATE Threads::Threads hydra_isal)
    add_test(NAME integration_${test_name} COMMAND ${test_name})
    set_tests_properties(integration_${test_name} PROPERTIES TIMEOUT 90)
endforeach()
