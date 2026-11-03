cmake_minimum_required(VERSION 3.15)
project(rknn_laya_demo)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Laya currently supports Linux AArch64")
endif()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)
set(TOKENIZERS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/tokenizers-cpp)
add_executable(${PROJECT_NAME} main.cpp laya.cpp laya_text.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${LIBRKNNAPI_INCLUDES}
    ${TOKENIZERS_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/json)
target_link_libraries(${PROJECT_NAME} fileutils timeutils ${LIBRKNNAPI}
    ${TOKENIZERS_ROOT}/Linux/aarch64/libtokenizers_c.a dl pthread m)
install(TARGETS ${PROJECT_NAME} DESTINATION .)
# Copy available resources; the demo checks required files when it runs.
file(GLOB MODEL_FILES LIST_DIRECTORIES false
    "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn"
    "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.weight"
    "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.bin"
    "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.json")
install(FILES ${MODEL_FILES} DESTINATION model)
