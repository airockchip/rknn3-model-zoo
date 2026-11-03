cmake_minimum_required(VERSION 3.10)

project(rknn_wemm_embedding_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

add_executable(${PROJECT_NAME}
    main.cpp
    wemm_embedding.cpp
)

target_link_libraries(${PROJECT_NAME}
    cnpy
    imageutils
    timeutils
    fileutils
    ${LIBRKNNAPI}
    ${LIBTOKENIZER}
    dl
)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME} log)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
    ${LIBTOKENIZER_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/test_image.jpg
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/video_frames.txt
    DESTINATION model
)
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/../model/video_frames
    DESTINATION model
)
