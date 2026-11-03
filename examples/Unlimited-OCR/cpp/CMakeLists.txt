cmake_minimum_required(VERSION 3.10)

project(rknn_unlimited_ocr_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT DEFINED TARGET_SOC)
    set(TARGET_SOC x86)
endif()

if (TARGET_SOC STREQUAL "x86")
    set(DISABLE_RGA ON)
endif()

if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/vision)
include_directories(${CMAKE_SOURCE_DIR}/llm)

set(vision_file vision/rknn_unlimited_ocr_vision.cc)
set(llm_file llm/rknn_unlimited_ocr_llm.cc)

message(STATUS "LIBRKNNAPI = ${LIBRKNNAPI}")
message(STATUS "LIBRKNNAPI_INCLUDES = ${LIBRKNNAPI_INCLUDES}")
message(STATUS "TARGET_LIB_ARCH = ${TARGET_LIB_ARCH}")

add_executable(${PROJECT_NAME}
    main.cc
    unlimited_ocr.cc
    ${vision_file}
    ${llm_file}
)

target_link_libraries(${PROJECT_NAME}
    timeutils
    imageutils
    fileutils
    ${LIBRKNNAPI}
    ${LIBTOKENIZER}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME} log)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
    ${LIBTOKENIZER_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*/*.rknn")
file(GLOB WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*/Unlimited-OCR-*.weight")
file(GLOB TOKENIZER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.tokenizer.gguf")
file(GLOB EMBED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.embed.bin")
install(FILES ${RKNN_FILES} DESTINATION ./model)
install(FILES ${WEIGHT_FILES} DESTINATION ./model)
install(FILES ${TOKENIZER_FILES} DESTINATION ./model)
install(FILES ${EMBED_FILES} DESTINATION ./model)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../data/test.png" DESTINATION ./)
