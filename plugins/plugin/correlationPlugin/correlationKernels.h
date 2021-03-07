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

cudaError_t correlation (
    cudaStream_t stream,
    const float* left, 
    const float* right, 
    float* corr, 
    int batch, 
    int maxdisp,
    int stride,  
    int c, 
    int h, 
    int w,
    bool is_time,
    bool is_mean);

} // namespace plugin
} // namespace nvinfer1

#endif