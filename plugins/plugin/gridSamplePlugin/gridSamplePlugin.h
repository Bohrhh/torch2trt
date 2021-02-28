/*
Author: kmlee
Date: 20200828

sample from feature according to keypoints

Input: 
    feature        -- [batch_size, C, H, W] or [batch_size, C, D, H, W]
    keypoints      -- [batch_size, H_out, W_out, 2] or [batch_size, D_out, H_out, W_out, 3] range(-1,1)

Args:
    mode           -- 0 ('bilinear'), 1 ('nearest') 
    padding_mode   -- 0 ('zeros'), 1 ('border'), 2 ('reflection')
    align_corner   -- False

Output:
    sampled_feature -- shape (batch_size, C, H_out, W_out) or (batch_size, C, D_out, H_out, W_out)

Reference: pytorch
 */

#ifndef TRT_GRIDSAMPLE_PLUGIN_H
#define TRT_GRIDSAMPLE_PLUGIN_H

#include <cassert>
#include <cuda_runtime_api.h>
#include <string.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "plugin.h"
#include "gridSampleKernels.h"


namespace nvinfer1
{
namespace plugin
{
class GridSample : public IPluginV2DynamicExt
{
public:
    GridSample(int mode, int padding_mode, bool align_corners);

    GridSample(const void* data, size_t length);

    GridSample() = delete;

    ~GridSample() override = default;

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
    int mMode;
    int mPadding_mode;
    bool mAlign_corners;
    std::string mNameSpace;
};

class GridSamplePluginCreator : public BaseCreator
{
public:
    GridSamplePluginCreator();

    ~GridSamplePluginCreator(){};

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* data, size_t length) override;

private:
    static PluginFieldCollection mFC;
    int mMode;
    int mPadding_mode;
    bool mAlign_corners;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_GRIDSAMPLE_PLUGIN_H
