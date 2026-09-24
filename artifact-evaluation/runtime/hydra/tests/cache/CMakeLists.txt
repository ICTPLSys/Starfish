set(CACHE_UNIT_ENV
    "FARLIB_SIMPLE_LOCAL_RESIDENT=1;FARLIB_SIMPLE_REGION_HEAT=1;FARLIB_SIMPLE_SIX_GROUPS=1;FARLIB_LIST_ONLY_SIX=1;FARLIB_LIST_ONLY_SIX_GROUPS=0;FARLIB_SIMPLE_HOTCOLD=1;FARLIB_SIMPLE_LOCAL_ROUTING=1;FARLIB_SIMPLE_REMOTE_HOTCOLD=1;FARLIB_SIMPLE_HOTCOLD_RELINK=1;FARLIB_SIMPLE_REGION_BUDGET=adaptive;FARLIB_SIMPLE_REGION_BUDGET_FAST=1;FARLIB_SIMPLE_REGION_HEAT_FAST=1;FARLIB_SIMPLE_REGION_HEAT_INTERVAL_MS=100;FARLIB_SIMPLE_REGION_BUDGET_INTERVAL_MS=100;FARLIB_FIXED_SIX_GROUPS=0;FARLIB_FIXED_SIX_POOLS=0")

foreach(check IN ITEMS budget allocator transitions)
    set(target "test_simple_resident_${check}")
    add_executable(${target} "${target}.cpp")
    target_compile_options(${target} PRIVATE -UNDEBUG)
    target_link_libraries(${target} PRIVATE ${COMMON_LIBS})
    add_test(NAME "cache_${check}" COMMAND ${target})
    set_tests_properties("cache_${check}" PROPERTIES
        ENVIRONMENT "${CACHE_UNIT_ENV}" TIMEOUT 30)
endforeach()

# This target requires a separate physical memory server and a site-specific
# config. Never register it as an automatic/same-host CTest.
add_executable(test_simple_resident_integration test_simple_resident_integration.cpp)
target_compile_options(test_simple_resident_integration PRIVATE -O1 -UNDEBUG)
target_link_libraries(test_simple_resident_integration PRIVATE ${COMMON_LIBS})
