/*
Author: kmlee
Mail: likemin@zju.edu.cn
Date: 20200902

Input: 
    feature        -- [batch_size, C,        H,     W    ]
    offset         -- [batch_size, 2*Chan/3, H_off, W_off]
    mask           -- [batch_size, Chan/3,   H_off, W_off]

    Chan  = deformable_group * 3 * kernel_H * kernel_W
    H_off = (H+2*padding-kernel_H+1)/stride
    W_off = (W+2*padding-kernel_W+1)/stride

Args:
    kernel_H         -- int
    kernel_W         -- int
    deformable_group -- int
    dilation         -- int
    groups           -- int
    padding          -- int
    stride           -- int

Output:
    sampled_feature -- shape (batch_size, C_out, H_off, W_off)

Reference: https://github.com/CaoWGG/TensorRT-CenterNet
 */

#ifndef TRT_DCN_V2_PLUGIN_H
#define TRT_DCN_V2_PLUGIN_H

#include <cassert>
#include <cuda_runtime_api.h>
#include <string.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "plugin.h"
#include "serialize.hpp"
#include "dcnV2Kernels.h"


namespace nvinfer1
{
namespace plugin
{
class DCNv2 : public IPluginV2DynamicExt
{
public:
    DCNv2(int in_channel,
          int out_channel,
          int kernel_H,
          int kernel_W,
          int deformable_group,
          int dilation,
          int groups,
          int padding,
          int stride,
          nvinfer1::Weights const& weight,
          nvinfer1::Weights const& bias);
  
    DCNv2(const void* data, size_t length);

    DCNv2() = delete;

    ~DCNv2() override = default;

    int getNbOutputs() const override;

    DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;

    int initialize() override;

    void terminate() override;

    void destroy() override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, 
                const void* const* inputs, void* const* outputs, 
                void* workspace, 
                cudaStream_t stream) override;

    // DynamicExt plugin supportsFormat update.
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    IPluginV2DynamicExt* clone() const override;

    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;


    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
                        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

    void detachFromContext() override;

private:
    int _in_channel;
    int _out_channel;
    int _kernel_H;
    int _kernel_W;
    int _deformable_group;
    int _dilation;
    int _groups; // not use
    int _padding;
    int _stride;
    std::vector<float> _h_weight;
    std::vector<float> _h_bias;
    float* _d_weight;
    float* _d_bias;
    float* _d_ones;
    float *_d_columns;
    int _outC;
    int _outH;
    int _outW;

    bool _initialized=false;
    std::string mNameSpace;

};

class DCNv2PluginCreator : public BaseCreator
{
public:
    DCNv2PluginCreator();

    ~DCNv2PluginCreator(){};

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* data, size_t length) override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_DCN_V2_PLUGIN_H
