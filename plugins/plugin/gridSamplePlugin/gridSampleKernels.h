/*
Author: kmlee
Date: 20200828
 */

#ifndef TRT_GRIDSAMPLE_KERNEL_H
#define TRT_GRIDSAMPLE_KERNEL_H

namespace nvinfer1
{
namespace plugin
{

enum class GridSamplerInterpolation {Nearest, Bilinear};
enum class GridSamplerPadding {Zeros, Border, Reflection};

cudaError_t grid_sampler_2d_cuda(
    cudaStream_t stream, 
    const float* input, 
    const float* grid, 
    float* output,
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
    const float* input, 
    const float* grid, 
    float* output,
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

} // namespace plugin
} // namespace nvinfer1

#endif