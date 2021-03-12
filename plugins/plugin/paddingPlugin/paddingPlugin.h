#ifndef TRT_PADDING_PLUGIN_H
#define TRT_PADDING_PLUGIN_H

#include <cassert>
#include <cuda_runtime_api.h>
#include <string.h>
#include <string>
#include <vector>

#include "plugin.h"
#include "paddingKernels.h"


namespace nvinfer1
{
namespace plugin
{
class Padding : public IPluginV2DynamicExt
{
public:
    Padding(const int* pads, int mode);

    Padding(const void* data, size_t length);

    Padding() = delete;

    ~Padding() override = default;

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
    int mPads[16];
    int mMode;
    std::string mNameSpace;
};

class PaddingPluginCreator : public BaseCreator
{
public:
    PaddingPluginCreator();

    ~PaddingPluginCreator(){};

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
#endif // TRT_COSTVOLUME_PLUGIN_H
