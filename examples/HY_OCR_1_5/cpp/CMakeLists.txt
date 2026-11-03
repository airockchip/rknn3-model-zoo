cmake_minimum_required(VERSION 3.10)

project(rknn_hy_ocr_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

# 构建可执行文件
add_executable(${PROJECT_NAME}
    vision/rknn_hy_ocr_vision.cc
    llm/rknn_hy_ocr_llm.cc
    hy_ocr_vl.cc
    main.cc
)

target_link_libraries(${PROJECT_NAME}
    timeutils
    imageutils
    fileutils
    ${LIBRKNNAPI}
    ${LIBTOKENIZER}
    dl
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vision
    ${CMAKE_CURRENT_SOURCE_DIR}/llm
    ${LIBRKNNAPI_INCLUDES}
    ${LIBTOKENIZER_INCLUDES}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${PROJECT_NAME}
        pthread
)
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../data/demo.jpg DESTINATION ./model)
file(GLOB VISION_RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/vision/*.rknn")
file(GLOB VISION_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/vision/*.weight")
file(GLOB VISION_POS_EMBED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/vision/pos_embed_*.bin")
file(GLOB LLM_RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.rknn")
# Only package the RKNN external weight. The model directory may also contain
# model.language_model.embed_tokens.weight, which is a float32 source tensor
# and must not be deployed as the LLM external weight.
file(GLOB LLM_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*-llm.weight")
file(GLOB LLM_TOKENIZER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.tokenizer.gguf")
file(GLOB LLM_EMBED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.embed.bin")
install(FILES ${VISION_RKNN_FILES} DESTINATION ./model)
install(FILES ${VISION_WEIGHT_FILES} DESTINATION ./model)
install(FILES ${VISION_POS_EMBED_FILES} DESTINATION ./model)
install(FILES ${LLM_RKNN_FILES} DESTINATION ./model)
install(FILES ${LLM_WEIGHT_FILES} DESTINATION ./model)
install(FILES ${LLM_TOKENIZER_FILES} DESTINATION ./model)
install(FILES ${LLM_EMBED_FILES} DESTINATION ./model)
