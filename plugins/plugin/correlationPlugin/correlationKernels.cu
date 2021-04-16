/*
Author: kmlee
Date: 20200525
 */

#include <vector>
#include "correlationKernels.h"

namespace nvinfer1
{
namespace plugin
{

template <typename T>
__global__ void correlation_kernel(
    const T* left, 
    const T* right, 
    T* corr, 
    int batch, 
    int maxdisp, 
    int stride, 
    int c, 
    int h, 
    int w, 
    bool is_time,
    bool is_mean)
{
    const int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int y0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int z0 = blockIdx.z * blockDim.z + threadIdx.z;
    int lStrides[4] = {c*h*w,h*w,w,1};
    int cStrides[4] = {(2*maxdisp/stride+1)*h*w, h*w, w, 1};
    int leftOffset, rightOffset, corrOffset;

    for (int iz=z0; iz<batch; iz+=blockDim.z * gridDim.z){
        for (int iy=y0; iy<h; iy+=blockDim.y * gridDim.y){
            for (int ix=x0; ix<w; ix+=blockDim.x * gridDim.x){
                leftOffset = ix*lStrides[3]+iy*lStrides[2]+iz*lStrides[0];
                corrOffset = ix*cStrides[3]+iy*cStrides[2]+iz*cStrides[0];
                for (int i=-maxdisp; i<maxdisp+1; i+=stride){
                    T s = 0;
                    int corr_C = (i+maxdisp)/stride;
                    if (ix<-i || ix>=w-i){
                        if (is_time)
                            s = 0;
                        else
                            for (int j=0; j<c; ++j)
                                s += abs(left[leftOffset+j*lStrides[1]]);
                    }
                    else{
                        rightOffset = (ix+i)*lStrides[3]+iy*lStrides[2]+iz*lStrides[0];
                        if (is_time)
                            for (int j=0; j<c; ++j)
                                s += left[leftOffset+j*lStrides[1]]*right[rightOffset+j*lStrides[1]];
                        else
                            for (int j=0; j<c; ++j)
                                s += abs(left[leftOffset+j*lStrides[1]]-right[rightOffset+j*lStrides[1]]);
                    }
                    if (is_mean)
                        corr[corrOffset+corr_C*cStrides[1]] = s/c;
                    else
                        corr[corrOffset+corr_C*cStrides[1]] = s;
                }
            }
        }
    }
}


cudaError_t correlation(
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
    bool is_mean)

{
    dim3 block(32, 32, 1);
    dim3 grid((w+block.x-1)/block.x , (h+block.y-1)/block.y, (batch+block.z-1)/block.z);

    correlation_kernel<float><<<grid,block,0,stream>>>(
        left, 
        right, 
        corr, 
        batch, 
        maxdisp, 
        stride,
        c, 
        h, 
        w,
        is_time,
        is_mean);
    return cudaGetLastError();
}

} // namespace plugin
} // namespace nvinfer1