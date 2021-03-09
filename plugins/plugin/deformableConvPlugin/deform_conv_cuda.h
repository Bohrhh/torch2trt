#pragma once

#include <cublas_v2.h>


namespace nvinfer1
{
namespace plugin
{

typedef struct _DCN_PARAMS {
  cublasHandle_t cublas_handle;
  int batchSize = 1;
  int inputChannel = 1;
  int inputW = 256;
  int inputH = 256;
  int outputChannel = 1;
  int kernelW = 3;
  int kernelH = 3;
  int kernelChannels = 64;
  int strideW = 1;
  int strideH = 1;
  int padW = 0;
  int padH = 0;
  int dilationW = 1;
  int dilationH = 1;
  int groups = 1;
  int deformable_groups = 1;
  int im2col_step = 64;
} DCN_PARAMS;

void output_add_bias(cudaStream_t stream, 
    float *output, const float *bias, size_t batch,
    size_t channel, size_t height, size_t width);

void deformable_im2col(cudaStream_t stream,
    const float* data_im, const float* data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, float* data_col);

void modulated_deformable_im2col_cuda(cudaStream_t stream,
    const float* data_im, const float* data_offset, const float* data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, float* data_col);


int deform_conv_forward_cuda(cudaStream_t stream, 
                             float *input, float *weight, float *bias,
                             float *offset, float *output, void *workspace,
                             const DCN_PARAMS &dcn_params);

cudaError_t modulated_deform_conv_cuda_forward(cudaStream_t stream,
                                        const float *input, const float *weight,
                                        const float *bias, const float *offset, const float *mask,
                                        float *output, void *workspace,
                                        const DCN_PARAMS &dcn_params);

} // namespace plugin
} // namespace nvinfer1