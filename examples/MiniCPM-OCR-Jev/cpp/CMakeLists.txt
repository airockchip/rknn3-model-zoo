cmake_minimum_required(VERSION 3.10)

project(rknn_minicpm_ocr_jev_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

file(GLOB llm_file ${CMAKE_CURRENT_SOURCE_DIR}/llm/*.cc)
file(GLOB vision_file ${CMAKE_CURRENT_SOURCE_DIR}/vision/*.cc)

add_executable(${PROJECT_NAME}
    main.cc
    minicpm_ocr_jev.cc
    ${llm_file}
    ${vision_file}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/llm
    ${CMAKE_CURRENT_SOURCE_DIR}/vision
    ${LIBRKNNAPI_INCLUDES}
    ${LIBTOKENIZER_INCLUDES}
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
else ()
    target_link_libraries(${PROJECT_NAME} pthread)
endif ()

install(TARGETS ${PROJECT_NAME} DESTINATION .)

# Sample payload (manifest + images + OCR texts)
install(DIRECTORY ../data/ DESTINATION ./data)

list(FILTER LLM_WEIGHT_FILES EXCLUDE REGEX "embed_tokens")
file(GLOB TOKENIZER_FILES "../model/llm/*.tokenizer.gguf")
file(GLOB EMBED_FILES "../model/llm/*.embed.bin")

install(FILES ${LLM_RKNN_FILES} DESTINATION ./model/llm)
install(FILES ${LLM_WEIGHT_FILES} DESTINATION ./model/llm)
install(FILES ${TOKENIZER_FILES} DESTINATION ./model/llm)
install(FILES ${EMBED_FILES} DESTINATION ./model/llm)

install(DIRECTORY ../model/vision DESTINATION ./model
    FILES_MATCHING PATTERN "*.rknn" PATTERN "*.weight")
