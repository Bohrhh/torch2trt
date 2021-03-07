/*
Author: kmlee
Date: 20200422
 */

#include <vector>
#include <cassert>
#include <iostream>
#include <NvInferRuntimeCommon.h>
#include "gatherElementsKernels.h"

namespace nvinfer1
{
namespace plugin
{

template <typename IndexType, typename Real>
__global__ void gatherElements_kernel(
    const Real *input,
    Real *output,
    const int *index,
    const nvinfer1::Dims inputSizes,
    const nvinfer1::Dims indexSizes,
    const int dim,
    const int nbDims,
    const IndexType totalElements)
{   
    int indexStrides[8] = {1,1,1,1,1,1,1,1};
    int inputStrides[8] = {1,1,1,1,1,1,1,1};
    for (int i=nbDims-2; i>=0; i--){
        indexStrides[i] = indexStrides[i+1]*indexSizes.d[i+1];
        inputStrides[i] = inputStrides[i+1]*inputSizes.d[i+1];
    }

    for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x; linearId < totalElements; linearId += gridDim.x * blockDim.x) {
        IndexType correntId = linearId;
        IndexType outputOffset = 0;
        IndexType inputOffset = 0;
        IndexType indexOffset = 0;

        for (int d = nbDims - 1; d >= 0; d--) {
            IndexType curDimIndex = correntId % indexSizes.d[d];
            indexOffset += curDimIndex * indexStrides[d];
            outputOffset += curDimIndex * indexStrides[d];
            if (d != dim) {
                inputOffset += curDimIndex * inputStrides[d];
            }
            correntId /= indexSizes.d[d];
        }

        int indexValue = index[indexOffset];
        assert(indexValue < inputSizes.d[dim]);
        inputOffset += indexValue * inputStrides[dim];
        output[outputOffset] = input[inputOffset];
  }
}


cudaError_t gatherElements(
    cudaStream_t stream,    
    const float *input,
    float *output,
    const int *index,
    const nvinfer1::Dims inputSizes,
    const nvinfer1::Dims indexSizes,
    const int dim,
    const int nbDims,
    const int totalElements)
{   
    int block = 32*32;
    int grid = ((int)totalElements + block-1)/block;
    gatherElements_kernel<int, float><<<grid,block,0,stream>>>(    
        input, output, index,
        inputSizes, indexSizes,
        dim, nbDims, totalElements);
    return cudaGetLastError();
}

} // namespace plugin
} // namespace nvinfer1