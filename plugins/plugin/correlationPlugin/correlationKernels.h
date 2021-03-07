/*
Author: kmlee
Date: 20200525
 */

#ifndef TRT_CORRELATION_KERNEL_H
#define TRT_CORRELATION_KERNEL_H

namespace nvinfer1
{
namespace plugin
{

void correlation (
    cudaStream_t stream,
    const float* left, 
    const float* right, 
    float* corr, 
    int32_t batch, 
    int32_t maxdisp,
    int32_t stride,  
    int32_t c, 
    int32_t h, 
    int32_t w,
    bool is_time,
    bool is_mean);

} // namespace plugin
} // namespace nvinfer1

#endif