#!/bin/bash

set -e

TARGET_SDK=tokenizer
ENABLE_ASAN=OFF
TARGET_ARCH=
echo "$0 $@"
while getopts "s:a:b:mn:" opt; do
  case $opt in
    n)
      TARGET_SDK=$OPTARG
      ;;
    s)
      TARGET_SYSTEM=$OPTARG
      ;;
    a)
      TARGET_ARCH=$OPTARG
      ;;
    b)
      BUILD_TYPE=$OPTARG
      ;;
    m)
      ENABLE_ASAN=ON
      echo "ENABLE_ASAN"
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

if [ x"${TARGET_SYSTEM}" != x"linux" ]  &&  [ x"${TARGET_SYSTEM}" != x"android" ]  &&  [ x"${TARGET_SYSTEM}" != x"riscv64" ]  &&  [ x"${TARGET_SYSTEM}" != x"cygwin" ]  &&  [ x"${TARGET_SYSTEM}" != x"macos" ]; then
  echo "$0 -s <system> -a <arch> -n <sdk> -b <build_type>"
  echo "Please select config:"
  echo ""
  echo "    -s : system (linux/android/riscv64/cygwin/macos)"
  echo "    -a : arch (linux: aarch64/armhf/x86; android: arm64-v8a/armeabi-v7a; macos: arm64/x86_64)"
  echo "    -n : sdk name(hello)"
  echo "    -b : build_type(Debug/Release/RelWithDebInfo)"
  echo "    -m : enable asan"
  echo ""
  exit -1
fi

# Debug / Release / RelWithDebInfo
if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

if [[ -z ${ENABLE_ASAN} ]];then
    ENABLE_ASAN=OFF
fi

if [[ -z ${TARGET_SDK} ]];then
    TARGET_SDK=llama_tokenizer
fi

if [ "${TARGET_SYSTEM}" == "android" ]; then
    source env_android.sh
fi

if [ "${TARGET_SYSTEM}" == "linux" ]; then
    source env_linux.sh
fi

if [ "${TARGET_SYSTEM}" == "riscv64" ]; then
    source env_riscv64.sh
fi

if [ "${TARGET_SYSTEM}" == "cygwin" ]; then
    source env_cygwin.sh
fi

if [ "${TARGET_SYSTEM}" == "macos" ]; then
    source env_macos.sh
fi

TARGET_PLATFORM=${TARGET_SYSTEM}
if [[ -n ${TARGET_ARCH} ]];then
  TARGET_PLATFORM=${TARGET_PLATFORM}_${TARGET_ARCH}
fi

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
INSTALL_DIR=${ROOT_PWD}/install/${TARGET_SDK}_${TARGET_PLATFORM}
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_SDK}_${TARGET_PLATFORM}_${BUILD_TYPE}

if [ "${TARGET_SYSTEM}" == "linux" ]; then
  if [ "${TARGET_ARCH}" == "aarch64" ]; then
    C_COMPILER=${C_COMPILER_AARCH64}
    CXX_COMPILER=${CXX_COMPILER_AARCH64}
  elif [ "${TARGET_ARCH}" == "armhf" ]; then
    C_COMPILER=${C_COMPILER_ARM32}
    CXX_COMPILER=${CXX_COMPILER_ARM32}
  elif [ "${TARGET_ARCH}" == "x86" ]; then
    C_COMPILER=${C_COMPILER_X86_64}
    CXX_COMPILER=${CXX_COMPILER_X86_64}
  fi
fi

echo "==================================="
echo "TARGET_ARCH=${TARGET_ARCH}"
echo "TARGET_SYSTEM=${TARGET_SYSTEM}"
echo "TARGET_SDK=${TARGET_SDK}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "ENABLE_ASAN=${ENABLE_ASAN}"
echo "C_COMPILER=${C_COMPILER}"
echo "CXX_COMPILER=${CXX_COMPILER}"
echo "INSTALL_DIR=${INSTALL_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "==================================="

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

if [[ -d "${INSTALL_DIR}" ]]; then
  rm -rf ${INSTALL_DIR}
fi

cd ${BUILD_DIR}

if [ "${TARGET_SYSTEM}" == "linux" ]; then
  cmake ../.. \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_C_COMPILER=${C_COMPILER} \
      -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DBUILD_SHARED_LIBS=OFF \
      -DENABLE_ASAN=${ENABLE_ASAN} \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
elif [ "${TARGET_SYSTEM}" == "android" ]; then
  cmake ../.. \
      -DCMAKE_SYSTEM_NAME=Android \
      -DANDROID_PLATFORM=android-26 \
      -DANDROID_ABI=${TARGET_ARCH} \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DENABLE_ASAN=${ENABLE_ASAN} \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
elif [ "${TARGET_SYSTEM}" == "riscv64" ]; then
  cmake ../.. \
      -DCMAKE_SYSTEM_NAME=Generic \
      -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
      -DCMAKE_C_COMPILER=${C_COMPILER} \
      -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DBUILD_SHARED_LIBS=OFF \
      -DENABLE_ASAN=${ENABLE_ASAN} \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
elif [ "${TARGET_SYSTEM}" == "cygwin" ]; then
  cmake ../.. \
      -DCMAKE_SYSTEM_NAME=CYGWIN \
      -DCMAKE_C_COMPILER=${C_COMPILER} \
      -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DBUILD_SHARED_LIBS=OFF \
      -DENABLE_ASAN=${ENABLE_ASAN} \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
elif [ "${TARGET_SYSTEM}" == "x86" ]; then
  cmake ../.. \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_C_COMPILER=${C_COMPILER} \
      -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DBUILD_SHARED_LIBS=OFF \
      -DENABLE_ASAN=${ENABLE_ASAN} \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
elif [ "${TARGET_SYSTEM}" == "macos" ]; then
  MACOS_CMAKE_ARGS=(
      -DCMAKE_SYSTEM_NAME=Darwin
      -DCMAKE_C_COMPILER=${C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CXX_COMPILER}
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
      -DBUILD_SHARED_LIBS=OFF
      -DENABLE_ASAN=${ENABLE_ASAN}
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
  )
  if [[ -n ${TARGET_ARCH} ]]; then
    MACOS_CMAKE_ARGS+=(-DCMAKE_OSX_ARCHITECTURES=${TARGET_ARCH})
  fi
  cmake ../.. "${MACOS_CMAKE_ARGS[@]}"
fi

make -j16
make install
cd -

