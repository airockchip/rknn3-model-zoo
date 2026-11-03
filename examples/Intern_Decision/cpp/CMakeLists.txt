cmake_minimum_required(VERSION 3.10)
project(rknn_intern_decision_demo)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)
add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${LIBRKNNAPI_INCLUDES} ${LIBTOKENIZER_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/json)
target_link_libraries(${PROJECT_NAME} imageutils fileutils
    ${LIBRKNNAPI} ${LIBTOKENIZER} dl pthread)
install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ../model/customer_service.json ../model/street_scene.json
    ../model/bus.jpg DESTINATION model)
file(GLOB DEPLOY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn"
    "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.weight"
    "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.tokenizer.gguf"
    "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.embed.bin")
install(FILES ${DEPLOY_FILES} DESTINATION model)
