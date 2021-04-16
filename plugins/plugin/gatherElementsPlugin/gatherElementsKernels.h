/*
Author: kmlee
Date: 20200422
 */

#ifndef TRT_GATHERELEMENTS_KERNEL_H
#define TRT_GATHERELEMENTS_KERNEL_H

namespace nvinfer1
{
namespace plugin
{

cudaError_t gatherElements(
    cudaStream_t stream,    
    const float *input,
    float *output,
    const int *index,
    const nvinfer1::Dims inputSizes,
    const nvinfer1::Dims indexSizes,
    const int dim,
    const int nbDims,
    const int totalElements);

} // namespace plugin
} // namespace nvinfer1

#endif