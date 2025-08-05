# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_ck_tile_boilerplate ../boilerplate_example.cpp)
    set_property(TARGET test_ck_tile_boilerplate PROPERTY CXX_STANDARD 20)
else()
    message(DEBUG "Skipping ck_tile boilerplate tests for current target")
endif()
