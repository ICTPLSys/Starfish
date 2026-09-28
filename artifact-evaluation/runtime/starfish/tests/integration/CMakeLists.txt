set(RECOVERY_DESIGN2_INTEGRATION_TESTS
    test_behavior_group_runtime
    test_recomputable_design2_entry
    test_ec_behavior_groups
    test_failed_write_design2)

foreach(test_name IN LISTS RECOVERY_DESIGN2_INTEGRATION_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_sources(${test_name} PRIVATE $<TARGET_OBJECTS:cpu_profile_state>)
    target_compile_options(${test_name} PRIVATE -UNDEBUG)
    target_link_libraries(${test_name} PRIVATE Threads::Threads ${ISAL_LIBRARY})
    add_test(NAME integration_${test_name} COMMAND ${test_name})
    set_tests_properties(integration_${test_name} PROPERTIES TIMEOUT 90)
endforeach()

add_executable(test_far_vector_iterator_boundaries
               test_far_vector_iterator_boundaries.cpp)
target_compile_options(test_far_vector_iterator_boundaries PRIVATE -UNDEBUG)
target_link_libraries(test_far_vector_iterator_boundaries PRIVATE ${COMMON_LIBS})
add_test(NAME integration_far_vector_iterator_boundaries
         COMMAND test_far_vector_iterator_boundaries)
set_tests_properties(integration_far_vector_iterator_boundaries PROPERTIES
                     TIMEOUT 30)
