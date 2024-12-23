file(GLOB A_STAR_SOURCES "*.cpp" "*.h")

add_library(a_star SHARED ${A_STAR_SOURCES})

target_link_libraries(a_star PRIVATE pybind11::module)

set_target_properties(a_star PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../modules/)
