# Copyright (c) 2026 Rockchip Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

project(rknn_paraformer_demo C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT TARGET_SOC STREQUAL "rk3588" AND NOT TARGET_SOC STREQUAL "rk3576")
    message(FATAL_ERROR "Paraformer RK1820 demo requires an RK3588/RK3576 Linux host")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils utils.out)

add_executable(${PROJECT_NAME}
    hotword.cpp
    main.cpp
    paraformer.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../utils
    ${LIBRKNNAPI_INCLUDES}
    ${LIBKALDI_NATIVE_FBANK_INCLUDES}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    audioutils
    timeutils
    ${LIBRKNNAPI}
    ${LIBKALDI_NATIVE_FBANK}
    pthread
    dl
    m
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/paraformer_encoder.rknn
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/paraformer_encoder.weight
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/paraformer_decoder.rknn
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/paraformer_decoder.weight
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/tokens.json
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/asr_example.wav
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/asr_example_hotword.wav
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/hotwords.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/paraformer_decoder_hotword.rknn
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/paraformer_decoder_hotword.weight
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/seaco_decoder.rknn
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/seaco_decoder.weight
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/asf_scorer.rknn
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/asf_scorer.weight
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/bias_encoder.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/seg_dict
    DESTINATION model
    OPTIONAL
)
