cmake_minimum_required(VERSION 3.15)
project(rknn_qwen3_forced_aligner_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils utils.out)
add_executable(${PROJECT_NAME} main.cpp audio.cpp text.cpp resource.cpp)
# Allow firmware-matched system runtime selection through LD_LIBRARY_PATH.
target_link_options(${PROJECT_NAME} PRIVATE "-Wl,--enable-new-dtags")
target_include_directories(${PROJECT_NAME} PRIVATE ${LIBRKNNAPI_INCLUDES} ${LIBTOKENIZER_INCLUDES})
target_link_libraries(${PROJECT_NAME} audioutils ${LIBTOKENIZER} ${LIBRKNNAPI} dl pthread)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../data/mel_128_filters.txt DESTINATION .)
file(GLOB MODEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn"
                      "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.weight")
install(FILES ${MODEL_FILES} DESTINATION model)

file(GLOB TEXT_RESOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.embed.bin"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.tokenizer.gguf"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../model/*_config.txt"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../model/*_rules.bin")
install(FILES ${TEXT_RESOURCES} DESTINATION model)
