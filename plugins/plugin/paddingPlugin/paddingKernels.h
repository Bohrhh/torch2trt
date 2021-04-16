/*
Author: kmlee
Date: 20200525
 */

#include "NvInfer.h"
#include "NvInferPlugin.h"

#ifndef TRT_PADDING_KERNEL_H
#define TRT_PADDING_KERNEL_H

namespace nvinfer1
{
namespace plugin
{

enum class PaddingMode {Constant, Replicate, Reflect};

cudaError_t padding_cuda(
    cudaStream_t stream,
    const float* input, 
    float* output,
    const nvinfer1::Dims& dims,
    int* pads,
    PaddingMode mode);

} // namespace plugin
} // namespace nvinfer1

#endif