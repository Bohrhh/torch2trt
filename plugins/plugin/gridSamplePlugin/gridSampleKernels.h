/*
Author: kmlee
Date: 20200828
 */

#ifndef TRT_GRIDSAMPLE_KERNEL_H
#define TRT_GRIDSAMPLE_KERNEL_H
#include "plugin.h"
using namespace nvinfer1;
using namespace nvinfer1::plugin;

enum class GridSamplerInterpolation {Bilinear, Nearest};
enum class GridSamplerPadding {Zeros, Border, Reflection};

cudaError_t grid_sampler_2d_cuda(
    cudaStream_t stream, 
    const void* input, 
    const void* grid, 
    void* output,
    int interpolation_mode, 
    int padding_mode,
    bool align_corners,
    int batch, 
    int C,
    int feat_H,
    int feat_W,
    int grid_H, 
    int grid_W);

cudaError_t grid_sampler_3d_cuda(
    cudaStream_t stream, 
    const void* input, 
    const void* grid, 
    void* output,
    int interpolation_mode, 
    int padding_mode,
    bool align_corners,
    int batch, 
    int C,
    int feat_D,
    int feat_H,
    int feat_W,
    int grid_D,
    int grid_H, 
    int grid_W);

#endif