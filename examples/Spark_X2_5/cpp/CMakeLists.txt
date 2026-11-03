cmake_minimum_required(VERSION 3.10)

project(rknn_spark_x2_5_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)

set(DISABLE_RGA ON CACHE BOOL "Disable RGA utils for Spark-X2.5 LLM demo" FORCE)
set(DISABLE_LIBJPEG ON CACHE BOOL "Disable JPEG utils for Spark-X2.5 LLM demo" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

message(STATUS "LIBRKNNAPI = ${LIBRKNNAPI}")
message(STATUS "LIBRKNNAPI_INCLUDES = ${LIBRKNNAPI_INCLUDES}")
message(STATUS "TARGET_LIB_ARCH = ${TARGET_LIB_ARCH}")

if (NOT LIBRKNNAPI_INCLUDES)
    set(LIBRKNNAPI_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/rknpu3/include)
endif()

if (NOT LIBTOKENIZER_INCLUDES)
    set(LIBTOKENIZER_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/tokenizer/include)
endif()

include_directories(${CMAKE_SOURCE_DIR}/llm/)

add_executable(${PROJECT_NAME}
    main.cc
    rknn_spark.cc
    llm/rknn_spark_llm.cc
)

target_link_libraries(${PROJECT_NAME}
    cnpy
    timeutils
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
    ${CMAKE_CURRENT_SOURCE_DIR}/llm
    ${LIBRKNNAPI_INCLUDES}
    ${LIBTOKENIZER_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/json
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.rknn")
file(GLOB WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.weight")
file(GLOB TOKENIZER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.tokenizer.gguf")
file(GLOB EMBED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.embed.bin")
file(GLOB SAFETENSORS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/llm/*.safetensors")
install(FILES ${RKNN_FILES} DESTINATION ./model)
install(FILES ${WEIGHT_FILES} DESTINATION ./model)
install(FILES ${TOKENIZER_FILES} DESTINATION ./model)
install(FILES ${EMBED_FILES} DESTINATION ./model)
install(FILES ${SAFETENSORS_FILES} DESTINATION ./model)
