add_executable(mg mg.cpp)

if(DEFINED MG_UTHREAD_COUNT)
    if(NOT MG_UTHREAD_COUNT MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "MG_UTHREAD_COUNT must be a positive integer")
    endif()
    target_compile_definitions(mg PRIVATE MG_UTHREAD_COUNT=${MG_UTHREAD_COUNT})
endif()

add_library(mg_common OBJECT
    common/c_print_results.cpp common/c_randdp.cpp
    common/c_timers.cpp common/wtime.cpp)


target_include_directories(mg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mg_common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)


target_compile_options(mg PRIVATE -Wall -Wextra)
target_compile_options(mg_common PRIVATE -Wall -Wextra)

target_link_libraries(mg PRIVATE ${COMMON_LIBS} $<TARGET_OBJECTS:mg_common>)

add_executable(test_mg_iterator_boundaries EXCLUDE_FROM_ALL tests/iterator_boundaries.cpp)
target_compile_options(test_mg_iterator_boundaries PRIVATE -UNDEBUG)
target_link_libraries(test_mg_iterator_boundaries PRIVATE ${COMMON_LIBS})

if(DEFINED FARLIB_MG_OUTPUT_DIR)
    set_target_properties(mg PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${FARLIB_MG_OUTPUT_DIR}")
endif()
