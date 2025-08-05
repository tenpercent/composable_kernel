# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_ck_tile_tile_distribution test_tile_distribution.cpp)
    set_property(TARGET test_ck_tile_tile_distribution PROPERTY CXX_STANDARD 20)
else()
    message(DEBUG "Skipping ck_tile tile_distribution tests for current target")
endif()
