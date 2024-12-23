file(GLOB LOCALIZATION_SOURCES "*.cpp" "*.h" "*.hpp")

add_library(localization SHARED ${LOCALIZATION_SOURCES})

target_link_libraries(localization PRIVATE pybind11::module)

find_package(GSL REQUIRED)

target_link_libraries(localization PRIVATE gsl gslcblas)

set_target_properties(localization PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../modules/)
