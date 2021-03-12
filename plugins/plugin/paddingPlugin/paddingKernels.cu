/*
Author: kmlee
Date: 20210312

Reference: pytorch
 */

#include <vector>
#include <thrust/pair.h>
#include "paddingKernels.h"

namespace nvinfer1
{
namespace plugin
{

__device__
inline thrust::pair<int64_t, int64_t> get_index_mapping1d(
    int64_t input_w, int64_t output_w,
    int64_t output_x,
    int64_t pad_l) {
  // 3D grid of 1D blocks
  auto input_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * input_w;
  auto output_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * output_w;

  auto i_start_x = ::max(int64_t(0), -pad_l);
  auto o_start_x = ::max(int64_t(0), pad_l);

  int64_t input_x = ::abs(output_x - pad_l)
                    - ::abs(output_x - (input_w + pad_l - 1))
                    - output_x
                    + 2 * pad_l + input_w - 1
                    - o_start_x + i_start_x;

  return thrust::make_pair<int64_t, int64_t>(
    input_offset + input_x, output_offset + output_x);
}

__device__
inline thrust::pair<int32_t, int32_t>  get_index_mapping2d(
    int32_t input_dim_x, int32_t input_dim_y,
    int32_t output_dim_x, int32_t output_dim_y,
    int32_t pad_l, int32_t pad_t,
    int32_t output_xy) {
  // 3D grid of 1D blocks
  auto input_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * input_dim_x * input_dim_y;
  auto output_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * output_dim_x * output_dim_y;

  auto output_x = output_xy % output_dim_x;
  auto output_y = output_xy / output_dim_x;

  auto i_start_x = ::max(int32_t(0), -pad_l);
  auto i_start_y = ::max(int32_t(0), -pad_t);
  auto o_start_x = ::max(int32_t(0), pad_l);
  auto o_start_y = ::max(int32_t(0), pad_t);

  auto input_x = ::abs(output_x - pad_l)
                 - ::abs(output_x - (input_dim_x + pad_l - 1))
                 - output_x
                 + 2 * pad_l + input_dim_x - 1
                 - o_start_x + i_start_x;

  auto input_y = ::abs(output_y - pad_t)
                 - ::abs(output_y - (input_dim_y + pad_t - 1))
                 - output_y
                 + 2 * pad_t + input_dim_y - 1
                 - o_start_y + i_start_y;

  return thrust::make_pair<int32_t, int32_t>(
    input_offset + input_y * input_dim_x + input_x,
    output_offset + output_y * output_dim_x + output_x);
}

template<typename scalar_t>
__global__ void reflection_pad1d_out_kernel(
    scalar_t * input, scalar_t * output,
    int32_t input_w,
    int32_t pad_l, int32_t pad_r) {
  auto output_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_w = input_w + pad_l + pad_r;

  if (output_x < output_w) {
    auto index_pair = get_index_mapping1d(input_w, output_w, output_x, pad_l);
    output[index_pair.second] = input[index_pair.first];
  }
}

template<typename scalar_t>
__global__ void reflection_pad2d_out_kernel(
    scalar_t * input, scalar_t * output,
    int32_t input_dim_x, int32_t input_dim_y,
    int pad_t, int pad_b, int pad_l, int pad_r) {
  auto output_xy = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_dim_x = input_dim_x + pad_l + pad_r;
  auto output_dim_y = input_dim_y + pad_t + pad_b;

  if (output_xy < output_dim_x * output_dim_y) {
    auto index_pair = get_index_mapping2d(
      input_dim_x, input_dim_y,
      output_dim_x, output_dim_y,
      pad_l, pad_t,
      output_xy);

    output[index_pair.second] = input[index_pair.first];
  }
}


cudaError_t padding_cuda(
    cudaStream_t stream,
    const float* input, 
    float* output,
    const nvinfer1::Dims& dims,
    int* pads,
    PaddingMode mode)

{   
    int nbatch = dims.d[0];
    int dim_plane = 1;
    int dim_w = 2;


    ASSERT(dims.nbDims<=2 && "Reflect padding is only implemented for padding the last 2 \
        dimensions of 4D input tensor, or the last dimension of 3D input tensor.");
    ASSERT(mode==PaddingMode::Reflect && "Padding plugin only support reflect padding.");
    
    if (dims.nbDims==3){
        int input_w = dims.d[2];
        int pad_l = pads[0];
        int pad_r = pads[1];
        int  output_w = input_w + pad_l + pad_r;
        dim3 block_size(output_w > 256 ? 256 : output_w);
        dim3 grid_size((int) ::ceil(output_w / 256.0), nplane, nbatch);

        reflection_pad1d_out_kernel<<<grid_size, block_size, 0, stream>>>(
                input, output, input_w, pad_l, pad_r);
    }
    else if (dims.nbDims==4){
        int input_h = dims.d[2];
        int input_w = dims.d[3];
        int pad_l = pads[0];
        int pad_r = pads[1];
        int pad_t = pads[2];
        int pad_b = pads[3];
        int output_h = input_h + pad_t + pad_b;
        int output_w  = input_w + pad_l + pad_r;

        int output_plane_size = output_h * output_w;
        dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);
        dim3 grid_size((int) std::ceil(output_plane_size/256.0), nplane, nbatch);

        reflection_pad2d_out_kernel<<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
            input, output, input_w, input_h, pad_t, pad_b, pad_l, pad_r);
    }

    return cudaGetLastError();
}

} // namespace plugin
} // namespace nvinfer1