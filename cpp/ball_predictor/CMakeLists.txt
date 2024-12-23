file(GLOB BALL_PREDICTOR_SOURCES "*.cpp" "*.h")

add_library(ball_predictor SHARED ${BALL_PREDICTOR_SOURCES})

target_link_libraries(ball_predictor PRIVATE pybind11::module)

set_target_properties(ball_predictor PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../modules/)
