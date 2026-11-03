cmake_minimum_required(VERSION 3.10)

project(rknn_qwen3_vl_demo)

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

set(llm_file llm/rknn_qwen3_vl_llm.cc)
set(vision_file vision/rknn_qwen3_vl_vision.cc)

message(STATUS "LIBRKNNAPI = ${LIBRKNNAPI}")
message(STATUS "LIBRKNNAPI_INCLUDES = ${LIBRKNNAPI_INCLUDES}")
message(STATUS "TARGET_LIB_ARCH = ${TARGET_LIB_ARCH}")

add_executable(${PROJECT_NAME}
    main.cc
    qwen3_vl.cc
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
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../data/demo.jpg DESTINATION ./model)
file(GLOB LLM_RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../python/llm/*.rknn")
file(GLOB LLM_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../python/llm/*.weight")
file(GLOB LLM_TOKENIZER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../python/llm/*.tokenizer.gguf")
file(GLOB LLM_EMBED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../python/llm/*.embed.bin")
file(GLOB VISION_RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../python/vision/*.rknn")
file(GLOB VISION_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../python/vision/*.weight")
install(FILES ${LLM_RKNN_FILES} DESTINATION ./model)
install(FILES ${LLM_WEIGHT_FILES} DESTINATION ./model)
install(FILES ${LLM_TOKENIZER_FILES} DESTINATION ./model)
install(FILES ${LLM_EMBED_FILES} DESTINATION ./model)
install(FILES ${VISION_RKNN_FILES} DESTINATION ./model)
install(FILES ${VISION_WEIGHT_FILES} DESTINATION ./model)

# 导出产物默认目录：examples/Qwen3_VL_LoRA/model/
# *.lora_weight 不会被 *.weight 匹配到（文件名里是 "_weight" 而非 ".weight"），需要单独收集
file(GLOB MODEL_RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
file(GLOB MODEL_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.weight")
file(GLOB MODEL_LORA_WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.lora_weight")
file(GLOB MODEL_TOKENIZER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.tokenizer.gguf")
file(GLOB MODEL_EMBED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.embed.bin")
# 过滤掉 ONNX external data（language_model.* / onnx__MatMul_*），它们只在导出阶段使用
list(FILTER MODEL_WEIGHT_FILES EXCLUDE REGEX "/language_model\\.|/onnx__")
if (MODEL_RKNN_FILES OR MODEL_WEIGHT_FILES OR MODEL_LORA_WEIGHT_FILES OR MODEL_TOKENIZER_FILES OR MODEL_EMBED_FILES)
    install(FILES ${MODEL_RKNN_FILES} ${MODEL_WEIGHT_FILES} ${MODEL_LORA_WEIGHT_FILES} ${MODEL_TOKENIZER_FILES} ${MODEL_EMBED_FILES} DESTINATION ./model)
endif()
