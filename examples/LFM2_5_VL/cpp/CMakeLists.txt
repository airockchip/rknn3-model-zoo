cmake_minimum_required(VERSION 3.10)

project(rknn_lfm2_vl_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

include_directories(${CMAKE_SOURCE_DIR}/llm)
include_directories(${CMAKE_SOURCE_DIR}/vision/)

set(llm_file llm/rknn_lfm2_vl_llm.cc)
set(vision_file vision/rknn_lfm2_vl_vision.cc)

message(STATUS "LIBRKNNAPI = ${LIBRKNNAPI}")
message(STATUS "LIBRKNNAPI_INCLUDES = ${LIBRKNNAPI_INCLUDES}")
message(STATUS "TARGET_LIB_ARCH = ${TARGET_LIB_ARCH}")

add_executable(${PROJECT_NAME}
    main.cc
    lfm2_vl.cc
    ${llm_file}
    ${vision_file}
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
    target_link_libraries(${PROJECT_NAME}
    log
)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${PROJECT_NAME}
        pthread
    )
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
    ${LIBTOKENIZER_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
file(GLOB VISION_RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/vision/*.rknn")
file(GLOB VISION_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/vision/*.weight")
file(GLOB LLM_RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*convstream*.rknn")
file(GLOB LLM_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*convstream*.weight")
file(GLOB LLM_TOKENIZER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.tokenizer.gguf")
file(GLOB LLM_EMBED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.embed.bin")
install(FILES ${VISION_RKNN_FILES} DESTINATION ./model/vision)
install(FILES ${VISION_WEIGHT_FILES} DESTINATION ./model/vision)
install(FILES ${LLM_RKNN_FILES} DESTINATION ./model/llm)
install(FILES ${LLM_WEIGHT_FILES} DESTINATION ./model/llm)
install(FILES ${LLM_TOKENIZER_FILES} DESTINATION ./model/llm)
install(FILES ${LLM_EMBED_FILES} DESTINATION ./model/llm)

# explicit runtime paths (relative to this source tree, portable)
set(LFM2_RKNPU_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/rknpu3)
set(LFM2_TOK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/tokenizer)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${LFM2_RKNPU_DIR}/include
  ${LFM2_TOK_DIR}/include
)
target_link_directories(${PROJECT_NAME} PRIVATE
  ${LFM2_RKNPU_DIR}/Linux/aarch64
  ${LFM2_TOK_DIR}/lib/Linux/aarch64
)

# gridfix explicit libs
target_link_libraries(${PROJECT_NAME}
  ${LFM2_RKNPU_DIR}/Linux/aarch64/librknn3_api.so
  ${LFM2_TOK_DIR}/lib/Linux/aarch64/libtokenizer.a
)
