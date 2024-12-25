file(GLOB UTILS_SOURCES "*.cpp" "*.h")

add_library(utils SHARED ${UTILS_SOURCES})

target_link_libraries(utils PRIVATE pybind11::module)

set_target_properties(utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../modules/)
