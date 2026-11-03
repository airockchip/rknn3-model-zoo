#!/bin/bash

set -e

echo "$0 $@"
SPEEDUP_CMAKE_ARGS=(-UENABLE_SPEEDUP)
while getopts ":t:a:d:b:m:r:j-:" opt; do
  case $opt in
    -)
      case "$OPTARG" in
        enable_speedup) SPEEDUP_CMAKE_ARGS=(-DENABLE_SPEEDUP=ON) ;;
        disable_speedup) SPEEDUP_CMAKE_ARGS=(-DENABLE_SPEEDUP=OFF) ;;
        *) echo "Invalid option: --$OPTARG"; exit 1 ;;
      esac
      ;;
    t)
      TARGET_SOC=$OPTARG
      ;;
    a)
      TARGET_ARCH=$OPTARG
      ;;
    d)
      BUILD_DEMO_NAME=$OPTARG
      ;;
    b)
      BUILD_TYPE=$OPTARG
      ;;
    m)
      ENABLE_ASAN=ON
      export ENABLE_ASAN=TRUE
      ;;
    r)
      DISABLE_RGA=ON
      ;;
    j)
      DISABLE_LIBJPEG=ON
      ;;
    :)
      echo "Option -$OPTARG requires an argument." 
      exit 1
      ;;
    ?)
      echo "Invalid option: -$OPTARG index:$OPTIND"
      ;;
  esac
done

if [ -z ${TARGET_SOC} ]  || [ -z ${TARGET_ARCH} ] ||  [ -z ${BUILD_DEMO_NAME} ]; then
  echo "$0 -t <target> -a <arch> -d <build_demo_name> [-b <build_type>] [-m] [-r] [-j] [--enable_speedup|--disable_speedup]"
  echo ""
  echo "    -t : target (rk3588/rk3576/rk3572)"
  echo "    -a : arch (arm64-v8a/armeabi-v7a)"
  echo "    -d : demo name"
  echo "    -b : build_type (Debug/Release)"
  echo "    -m : enable address sanitizer, build_type need set to Debug"
  echo "    -r : disable rga, use cpu resize image"
  echo "    -j : disable libjpeg to avoid conflicts between libjpeg and opencv"
  echo "    --enable_speedup : enable SpeedUP (library and header required)"
  echo "    --disable_speedup : build without SpeedUP"
  echo "    Without either flag, SpeedUP is disabled; available files only produce an enablement hint."
  echo "such as: $0  -t rk3588 -a arm64-v8a -d Qwen2_5"
  echo "Note: 'if you want to use opencv to read or save jpg files, use the '-j' option to disable libjpeg"
  echo ""
  exit -1
fi

# Debug / Release / RelWithDebInfo
if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

# Build with Address Sanitizer for memory check, BUILD_TYPE need set to Debug
if [[ -z ${ENABLE_ASAN} ]];then
    ENABLE_ASAN=OFF
fi

if [[ -z ${DISABLE_RGA} ]];then
    DISABLE_RGA=OFF
fi

if [[ -z ${DISABLE_LIBJPEG} ]];then
    DISABLE_LIBJPEG=OFF
fi

for demo_path in `find examples tested_models -name "${BUILD_DEMO_NAME}"`
do
    if [ -d "$demo_path/cpp" ]
    then
        BUILD_DEMO_PATH="$demo_path/cpp"
        break;
    fi
done

if [[ -z "${BUILD_DEMO_PATH}" ]]
then
    echo "Cannot find demo: ${BUILD_DEMO_NAME}, only support:"

    for demo_path in `find examples tested_models -name cpp`
    do
        if [ -d "$demo_path" ]
        then
            dname=`dirname "$demo_path"`
            name=`basename $dname`
            echo "$name"
        fi
    done

    exit
fi

case ${TARGET_SOC} in
    rk3588)
        ;;
    rk3576)
        ;;
    rk3572)
        ;;
    *)
        echo "Invalid target: ${TARGET_SOC}"
        echo "Valid target: rk3588,rk3576,rk3572"
        exit -1
        ;;
esac

TARGET_SDK="rknn_${BUILD_DEMO_NAME}_demo"

TARGET_PLATFORM=${TARGET_SOC}_android
if [[ -n ${TARGET_ARCH} ]];then
TARGET_PLATFORM=${TARGET_PLATFORM}_${TARGET_ARCH}
fi
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
INSTALL_DIR=${ROOT_PWD}/install/${TARGET_PLATFORM}/${TARGET_SDK}
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_SDK}_${TARGET_PLATFORM}_${BUILD_TYPE}

echo "==================================="
echo "BUILD_DEMO_NAME=${BUILD_DEMO_NAME}"
echo "BUILD_DEMO_PATH=${BUILD_DEMO_PATH}"
echo "TARGET_SOC=${TARGET_SOC}"
echo "TARGET_ARCH=${TARGET_ARCH}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "ENABLE_ASAN=${ENABLE_ASAN}"
echo "DISABLE_RGA=${DISABLE_RGA}"
echo "DISABLE_LIBJPEG=${DISABLE_LIBJPEG}"
echo "INSTALL_DIR=${INSTALL_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "ANDROID_NDK_PATH=${ANDROID_NDK_PATH}"
echo "==================================="

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

if [[ -d "${INSTALL_DIR}" ]]; then
  rm -rf ${INSTALL_DIR}
fi

cd ${BUILD_DIR}
cmake ../../${BUILD_DEMO_PATH} \
        -DTARGET_SOC=${TARGET_SOC} \
        -DCMAKE_SYSTEM_NAME=Android \
        -DCMAKE_SYSTEM_VERSION=23 \
        -DCMAKE_ANDROID_ARCH_ABI=${TARGET_ARCH} \
        -DCMAKE_ANDROID_STL_TYPE=c++_static \
        -DCMAKE_ANDROID_NDK=${ANDROID_NDK_PATH} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DENABLE_ASAN=${ENABLE_ASAN} \
        -DDISABLE_RGA=${DISABLE_RGA} \
        -DDISABLE_LIBJPEG=${DISABLE_LIBJPEG} \
        "${SPEEDUP_CMAKE_ARGS[@]}" \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
# make VERBOSE=1
make -j4
make install

# Check if there is a rknn model in the install directory
suffix=".rknn"
shopt -s nullglob
if [ -d "$INSTALL_DIR" ]; then
    files=("$INSTALL_DIR/model/"/*"$suffix")
    shopt -u nullglob

    if [ ${#files[@]} -le 0 ]; then
        echo -e "\e[91mThe RKNN model can not be found in \"$INSTALL_DIR/model\", please check!\e[0m"
    fi
else
    echo -e "\e[91mInstall directory \"$INSTALL_DIR\" does not exist, please check!\e[0m"
fi
