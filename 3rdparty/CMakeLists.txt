cmake_minimum_required(VERSION 3.15)

project(rknn3_model_zoo_3rdparty_libs)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set (TARGET_LIB_ARCH ${CMAKE_ANDROID_ARCH_ABI})
else()
    if (TARGET_SOC STREQUAL "x86")
        set (TARGET_LIB_ARCH x86_64)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x86")
        set (TARGET_LIB_ARCH x86_64)
    else()
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set (TARGET_LIB_ARCH aarch64)
        else()
            set (TARGET_LIB_ARCH armhf)
        endif()
        if (CMAKE_C_COMPILER MATCHES "uclibc")
            set (TARGET_LIB_ARCH ${TARGET_LIB_ARCH}_uclibc)
        endif()
    endif()
endif()

# stb image
set(STB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/stb_image)
set(STB_INCLUDES ${STB_PATH} PARENT_SCOPE)

# jpeg turbo
set(JPEG_PATH ${CMAKE_CURRENT_SOURCE_DIR}/jpeg_turbo)
set(LIBJPEG ${JPEG_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libturbojpeg.a PARENT_SCOPE)
set(LIBJPEG_INCLUDES ${JPEG_PATH}/include PARENT_SCOPE)

# miniaudio
set(MINIAUDIO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/miniaudio)
set(MINIAUDIO_INCLUDES ${MINIAUDIO_PATH} PARENT_SCOPE)

# pocketfft
add_subdirectory(pocketfft)

# librknnapi
# for rknpu3
if (TARGET_SOC STREQUAL "rk3588" OR TARGET_SOC STREQUAL "rk3576" OR TARGET_SOC STREQUAL "rk3572" OR TARGET_SOC STREQUAL "x86")
    set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/rknpu3)
    set(LIBRKNNAPI ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknn3_api.so PARENT_SCOPE)
    set(LIBRKNNAPI_INCLUDES ${RKNN_PATH}/include PARENT_SCOPE)

    set(RKNN_INSTALL_LIBS
        ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknn3_api.so)

    if (TARGET_SOC STREQUAL "rk3588" OR TARGET_SOC STREQUAL "rk3576" OR TARGET_SOC STREQUAL "RK3588" OR TARGET_SOC STREQUAL "RK3576" OR TARGET_SOC STREQUAL "x86")
        list(APPEND RKNN_INSTALL_LIBS ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknn3_api_rkcp.so)
    elseif (TARGET_SOC STREQUAL "rk3572" OR TARGET_SOC STREQUAL "RK3572")
        list(APPEND RKNN_INSTALL_LIBS ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknn3_api_native.so)
    endif()
endif()
if (RKNN_INSTALL_LIBS)
    install(PROGRAMS ${RKNN_INSTALL_LIBS} DESTINATION lib)
endif()

# SpeedUP
set(SPEEDUP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/SpeedUP)
set(SPEEDUP_LIB ${SPEEDUP_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libSpeedUP.so PARENT_SCOPE)
set(SPEEDUP_INCLUDES ${SPEEDUP_PATH}/include PARENT_SCOPE)
set(SPEEDUP_AVAILABLE OFF)
if(EXISTS "${SPEEDUP_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libSpeedUP.so"
   AND EXISTS "${SPEEDUP_PATH}/include/speedup.h")
    set(SPEEDUP_AVAILABLE ON)
endif()
if(SPEEDUP_AVAILABLE)
    if(NOT DEFINED ENABLE_SPEEDUP)
        string(TOLOWER "${CMAKE_SYSTEM_NAME}" SPEEDUP_SYSTEM)
        message(STATUS "SpeedUP available but disabled by default; enable with --enable_speedup (for example: ./build-${SPEEDUP_SYSTEM}.sh -t ${TARGET_SOC} -a ${TARGET_LIB_ARCH} -d Qwen3_5_VL --enable_speedup), or -DENABLE_SPEEDUP=ON")
    elseif(NOT ENABLE_SPEEDUP)
        message(STATUS "SpeedUP components available; disabled by request, using normal inference")
    endif()
elseif(NOT ENABLE_SPEEDUP)
    message(WARNING "SpeedUP library or header missing; SpeedUP disabled, using normal inference")
endif()
option(ENABLE_SPEEDUP "Enable SpeedUP" OFF)
if(ENABLE_SPEEDUP AND NOT SPEEDUP_AVAILABLE)
    message(FATAL_ERROR "SpeedUP library or header missing")
endif()

# rga
if (NOT TARGET_SOC STREQUAL "x86" AND NOT TARGET_LIB_ARCH STREQUAL "x86_64")
    set(RGA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librga)
    set(LIBRGA ${RGA_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librga.a PARENT_SCOPE)
    set(LIBRGA_INCLUDES ${RGA_PATH}/include PARENT_SCOPE)
    install(PROGRAMS ${RGA_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librga.so DESTINATION lib)
endif()

# tokenizer
set(TOKENIZER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer)
# Darwin SDK layout uses macOS/arm64 (not CMAKE_SYSTEM_NAME=Darwin)
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(LIBTOKENIZER ${TOKENIZER_PATH}/lib/macOS/arm64/libtokenizer.a PARENT_SCOPE)
else()
    set(LIBTOKENIZER ${TOKENIZER_PATH}/lib/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libtokenizer.a PARENT_SCOPE)
endif()
set(LIBTOKENIZER_INCLUDES ${TOKENIZER_PATH}/include PARENT_SCOPE)

# libsndfile
set(LIBSNDFILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libsndfile)
set(LIBSNDFILE_INCLUDES ${LIBSNDFILE_PATH}/include PARENT_SCOPE)
set(LIBSNDFILE ${LIBSNDFILE_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libsndfile.a PARENT_SCOPE)

# kaldi_native_fbank
set(LIBKALDI_NATIVE_FBANK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/kaldi_native_fbank)
set(LIBKALDI_NATIVE_FBANK_INCLUDES ${LIBKALDI_NATIVE_FBANK_PATH}/include PARENT_SCOPE)
set(LIBKALDI_NATIVE_FBANK ${LIBKALDI_NATIVE_FBANK_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libkaldi-native-fbank-core.a PARENT_SCOPE)

# espeak-ng
set(ESPEAK_NG_PATH ${CMAKE_CURRENT_SOURCE_DIR}/espeak-ng)
set(ESPEAK_NG_INCLUDES ${ESPEAK_NG_PATH}/include PARENT_SCOPE)

# Set espeak-ng library path based on platform
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(ESPEAK_NG_LIB_DIR ${ESPEAK_NG_PATH}/Android/arm64-v8a)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ESPEAK_NG_LIB_DIR ${ESPEAK_NG_PATH}/Linux/aarch64)
endif()

# Define espeak-ng libraries
set(LIBESPEAK_NG ${ESPEAK_NG_LIB_DIR}/libespeak-ng.a PARENT_SCOPE)
set(LIBSPEECH_PLAYER ${ESPEAK_NG_LIB_DIR}/libspeechPlayer.a PARENT_SCOPE)
set(LIBUCD ${ESPEAK_NG_LIB_DIR}/libucd.a PARENT_SCOPE)
