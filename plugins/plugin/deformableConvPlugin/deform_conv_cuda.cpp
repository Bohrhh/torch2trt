#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "deform_conv_cuda.h"

namespace nvinfer1
{
namespace plugin
{

cudaError_t modulated_deform_conv_cuda_forward(cudaStream_t stream, 
    const float* input, const float* weight, const float* bias,
    const float* offset, const float* mask, float* output, void* workspace,
    const DCN_PARAMS &dcn_params) {

  cublasHandle_t cublas_handle = dcn_params.cublas_handle;
  int channels_kernel = dcn_params.kernelChannels;
  int kernel_w = dcn_params.kernelW;
  int kernel_h = dcn_params.kernelH;
  int channels_out = dcn_params.outputChannel;
  int stride_w = dcn_params.strideW;
  int stride_h = dcn_params.strideH;
  int pad_w = dcn_params.padW;
  int pad_h = dcn_params.padH;
  int dilation_w = dcn_params.dilationW;
  int dilation_h = dcn_params.dilationH;
  int group = dcn_params.groups;
  int deformable_group = dcn_params.deformable_groups;
  int im2col_step = dcn_params.im2col_step;

  int batch = dcn_params.batchSize;
  int channels = dcn_params.inputChannel;
  int height = dcn_params.inputH;
  int width = dcn_params.inputW;
  bool with_bias = (bias != nullptr);

  if (channels != channels_kernel * group){
    printf("Input shape and kernel channels wont match: (%d vs %d).\n", channels, channels_kernel * group);
    exit(0);
  }

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  float *columns = (float *)workspace;

  const size_t input_step = channels * height * width;
  const size_t offset_step =
      deformable_group * kernel_h * kernel_w * 2 * height * width;
  const size_t mask_step =
      deformable_group * kernel_h * kernel_w * height * width;
  const size_t out_step = channels_out * height_out * width_out;
  const size_t out_group_step = out_step / group;
  const size_t col_g_step =
      channels * kernel_w * kernel_h / group * height_out * width_out;
  const size_t weight_g_step =
      channels_out / group * channels / group * kernel_h * kernel_w;

  const int m = channels_out / group;
  const int n = height_out * width_out;
  const int k = channels / group * kernel_h * kernel_w;
  float alpha = 1.;
  float beta = 0.;

  for (int b = 0; b < batch; b++) {

    const float *input_start  = input + b * input_step;
    const float *offset_start = offset + b * offset_step;
    const float *mask_start   = mask + b * mask_step;

    modulated_deformable_im2col_cuda(stream,
        input_start, offset_start, mask_start, 1, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, columns);

    for (int g = 0; g < group; g++) {

      const float *weight_start = weight + g * weight_g_step;
      const float *col_start    = columns + g * col_g_step;
      float *out_buffer_start   = output + b * out_step + g * out_group_step;

      cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                  col_start, n, weight_start, k, &beta, out_buffer_start, n);
    }
  }

  if (with_bias) {
    output_add_bias(stream, output, bias, batch, channels_out, height_out, width_out);
  }

  return cudaGetLastError();
}

} // namespace plugin
} // namespace nvinfer1