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
message(STATUS "SPEEDUP_LIB = ${SPEEDUP_LIB}")

add_executable(${PROJECT_NAME}
    main.cc
    qwen3_vl.cc
    ${llm_file}
    ${vision_file}
)

if(ENABLE_SPEEDUP)
    message(STATUS "Found SpeedUP = ${SPEEDUP_LIB}")
    install(PROGRAMS ${SPEEDUP_LIB} DESTINATION lib)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_SPEEDUP)
else()
    message(STATUS "SpeedUP disabled")
    set(SPEEDUP_LIB "")
endif()

target_link_libraries(${PROJECT_NAME}
    timeutils
	imageutils
    fileutils
    ${LIBRKNNAPI}
    ${LIBTOKENIZER}
    ${SPEEDUP_LIB}
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
    ${SPEEDUP_INCLUDES}
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
