cmake_minimum_required(VERSION 3.10)

project(rknn_whisper_demo)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(rknpu_whisper_file rknpu3/whisper.cc)

set(RKNN3_MODEL_ZOO_INSTALL_RUNTIME_LIBS OFF CACHE BOOL "Install runtime shared libraries" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(SHARED_SOURCES
    postprocess.cc
    whisper.cc
)


set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/rknpu3/include)
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(RKNN_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/rknpu3/Android/arm64-v8a)
else()
    set(RKNN_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/rknpu3/Linux/aarch64)
endif()


message(STATUS "RKNN_API_PATH: ${RKNN_API_PATH}")
message(STATUS "RKNN_LIB_PATH: ${RKNN_LIB_PATH}")

set(LIBRKNNRT rknn3_api)
set(LIBRKNNRT_INCLUDES ${RKNN_API_PATH})

link_directories(${RKNN_LIB_PATH})

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)



add_executable(rknn_whisper_raw_demo
    main_raw.cc
    process.cc
    ${rknpu_whisper_file}
)

target_link_libraries(rknn_whisper_raw_demo
    fileutils
    audioutils
    timeutils
    ${LIBRKNNRT}
    pocketfft
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(rknn_whisper_raw_demo Threads::Threads)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    # librknn3_api.so(Android)依赖__android_log_print，需要链接liblog
    target_link_libraries(rknn_whisper_raw_demo log)
endif()

target_include_directories(rknn_whisper_raw_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBTIMER_INCLUDES}
)


function(install_real_file_if_exists source_path dest_dir)
    if (EXISTS "${source_path}")
        get_filename_component(source_name "${source_path}" NAME)
        get_filename_component(source_realpath "${source_path}" REALPATH)
        install(FILES "${source_realpath}" DESTINATION "${dest_dir}" RENAME "${source_name}")
    endif()
endfunction()

function(install_model_binary_if_exists file_name)
    set(model_source_path "${CMAKE_CURRENT_SOURCE_DIR}/../model/${file_name}")
    set(export_source_path "${CMAKE_CURRENT_SOURCE_DIR}/../python/whisper-base-model/${file_name}")

    if (EXISTS "${model_source_path}")
        install_real_file_if_exists("${model_source_path}" model)
    elseif (EXISTS "${export_source_path}")
        install_real_file_if_exists("${export_source_path}" model)
    endif()
endfunction()


install(TARGETS rknn_whisper_raw_demo DESTINATION .)


set(MODEL_RUNTIME_FILES
    long_en.wav
    test_en.wav
    test_zh.wav
    vocab_en.txt
    vocab_zh.txt
    mel_80_filters.txt
    mel_128_filters.txt
    prompt_embed.bin
    prompt_embed_en.bin
    prompt_embed_zh.bin
    token_embed.bin
    position_embed.bin
)

foreach(model_runtime_file IN LISTS MODEL_RUNTIME_FILES)
    install_real_file_if_exists("${CMAKE_CURRENT_SOURCE_DIR}/../model/${model_runtime_file}" model)
endforeach()

set(MODEL_BINARY_FILES
    whisper_encoder_base.rknn
    whisper_encoder_base.weight
    whisper_decode0_base.rknn
    whisper_decode0_base.weight
    whisper_decode1_base.rknn
    whisper_decode1_base.weight
)

foreach(model_binary_file IN LISTS MODEL_BINARY_FILES)
    install_model_binary_if_exists("${model_binary_file}")
endforeach()
