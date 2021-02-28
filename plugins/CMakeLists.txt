#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(TensorRT
        LANGUAGES CXX CUDA
        DESCRIPTION "TensorRT is a C++ library that facilitates high performance inference on NVIDIA GPUs and deep learning accelerators."
        HOMEPAGE_URL "https://github.com/NVIDIA/TensorRT")

option(BUILD_PLUGINS "Build TensorRT plugin" ON)


############################################################################################
# Normal Settings

if(NOT DEFINED TRT_OUT_DIR)
    set(TRT_OUT_DIR ${CMAKE_BINARY_DIR})
endif()
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++)
set(CMAKE_SKIP_BUILD_RPATH True)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss")


############################################################################################
# Tensorrt Version and PLATFORM Config

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE TRT_PLATFORM_ID)
message(STATUS "Targeting TRT Platform: ${TRT_PLATFORM_ID}")
find_path(TRT_LIB_DIR libnvinfer_plugin.so
  HINTS ${TENSORRT_PLUGINS} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 lib/${TRT_PLATFORM_ID}-linux-gnu)
execute_process(COMMAND ls ${TRT_LIB_DIR} COMMAND grep "^libnvinfer_plugin.so.[0-9].[0-9].[0-9]$" OUTPUT_VARIABLE TENSORRT_VERSION)
execute_process(COMMAND echo ${TENSORRT_VERSION} COMMAND cut -c 22 COMMAND tr -d '\n' OUTPUT_VARIABLE TRT_MAJOR)
execute_process(COMMAND echo ${TENSORRT_VERSION} COMMAND cut -c 24 COMMAND tr -d '\n' OUTPUT_VARIABLE TRT_MINOR)
execute_process(COMMAND echo ${TENSORRT_VERSION} COMMAND cut -c 26 COMMAND tr -d '\n' OUTPUT_VARIABLE TRT_PATCH)
foreach(TYPE MAJOR MINOR PATCH)
    execute_process(COMMAND sed -i "s/${TYPE} [0-9]/${TYPE} ${TRT_${TYPE}}/g" "${CMAKE_CURRENT_SOURCE_DIR}/include/NvInferVersion.h")
endforeach(TYPE)

set(TRT_VERSION "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}" CACHE STRING "TensorRT project version")
set(TRT_SOVERSION "${TRT_MAJOR}" CACHE STRING "TensorRT library so version")
message(STATUS "Building for TensorRT version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}")


############################################################################################
# Dependencies

find_package(CUDA REQUIRED)
set(CUB_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cub CACHE STRING "directory of CUB installation")
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_ROOT_DIR}/include
)
find_library(CUDNN_LIB cudnn HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
find_library(CUBLAS_LIB cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUBLASLT_LIB cublasLt HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUDART_LIB cudart HINTS 
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)


############################################################################################
# CUDA targets

if (NOT DEFINED GPU_ARCHS)
  list(APPEND GPU_ARCHS 35 53 61 70 75)
  string(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
  if (IS_ARM)
    # Xavier (SM72) only supported for aarch64.
    list(APPEND GPU_ARCHS 72)
  endif()
  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
    # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
    list(APPEND GPU_ARCHS 80)
  else()
    message(WARNING "Detected CUDA version ${CUDA_VERSION} is < 11.0. SM80 not supported.")
  endif()
  message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
else()
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
endif()

# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")


############################################################################################
# Plugin and Parser

if(BUILD_PLUGINS)
    add_subdirectory(plugin)
endif()


############################################################################################
# Installation

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX /usr/local CACHE PATH "TensorRT installation" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)