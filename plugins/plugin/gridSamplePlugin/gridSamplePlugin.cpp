/*
Author: kmlee
Date: 20200828
 */

#include <cuda_runtime_api.h>
#include <iostream>
#include "common.h"
#include "serialize.hpp"
#include "gridSamplePlugin.h"

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::GridSample;
using nvinfer1::plugin::GridSamplePluginCreator;

#define DEBUG
                        
namespace
{
const char* GridSample_PLUGIN_VERSION{"1"};
const char* GridSample_PLUGIN_NAME{"GridSamplePlugin"};
} // namespace

PluginFieldCollection GridSamplePluginCreator::mFC{};
std::vector<PluginField> GridSamplePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GridSamplePluginCreator);

GridSamplePluginCreator::GridSamplePluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("padding_mode", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("align_corners", nullptr, PluginFieldType::kINT32, 0));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GridSamplePluginCreator::getPluginName() const
{
    return GridSample_PLUGIN_NAME;
};

const char* GridSamplePluginCreator::getPluginVersion() const
{
    return GridSample_PLUGIN_VERSION;
};

const PluginFieldCollection* GridSamplePluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2DynamicExt* GridSamplePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "mode"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mMode = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "padding_mode"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mPadding_mode = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "align_corners"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mAlign_corners = *(static_cast<const bool*>(fields[i].data));
        }

    }
    return new GridSample(mMode, mPadding_mode, mAlign_corners);
};

IPluginV2DynamicExt* GridSamplePluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new GridSample(data, length);
};

GridSample::GridSample(int mode, int padding_mode, bool align_corners)
    : mMode(mode), mPadding_mode(padding_mode), mAlign_corners(align_corners)
{   
    ASSERT(mMode <= 1 && mMode >= 0);
    ASSERT(mPadding_mode <= 2 && mPadding_mode >= 0);
};


int GridSample::getNbOutputs() const
{
    return 1;
};

DimsExprs GridSample::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{

    ASSERT(nbInputs == 2);
    ASSERT(outputIndex == 0);
    nvinfer1::DimsExprs output(inputs[0]);
    output.d[2] = inputs[1].d[1];
    output.d[3] = inputs[1].d[2];
    output.d[4] = inputs[1].d[3];

    return output;
};

int GridSample::initialize()
{
    return 0;
};

void GridSample::terminate(){

};

void GridSample::destroy(){

};

size_t GridSample::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const 
{
    return 0;
}

size_t GridSample::getSerializationSize() const
{
    return 2*sizeof(int)+sizeof(bool);
};

void GridSample::serialize(void* buffer) const
{
    serialize_value(&buffer, mMode);
    serialize_value(&buffer, mPadding_mode);
    serialize_value(&buffer, mAlign_corners);
};

GridSample::GridSample(const void* data, size_t length)
{
    deserialize_value(&data, &length, &mMode);
    deserialize_value(&data, &length, &mPadding_mode);
    deserialize_value(&data, &length, &mAlign_corners);
};

const char* GridSample::getPluginType() const
{
    return GridSample_PLUGIN_NAME;
};

const char* GridSample::getPluginVersion() const
{
    return GridSample_PLUGIN_VERSION;
};

IPluginV2DynamicExt* GridSample::clone() const
{
    return new GridSample(*this);
};

void GridSample::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* GridSample::getPluginNamespace() const
{
    return mNameSpace.c_str();
}


bool GridSample::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return (inOut[pos].type == DataType::kFLOAT && inOut[pos].format == PluginFormat::kLINEAR);
}


int GridSample::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    if (inputDesc[0].dims.nbDims==4){
        cudaError_t status = grid_sampler_2d_cuda(stream,
                                    static_cast<const float*>(inputs[0]), 
                                    static_cast<const float*>(inputs[1]), 
                                    static_cast<float*>(outputs[0]),
                                    mMode,
                                    mPadding_mode,
                                    mAlign_corners,
                                    inputDesc[0].dims.d[0], 
                                    inputDesc[0].dims.d[1],
                                    inputDesc[0].dims.d[2],
                                    inputDesc[0].dims.d[3],
                                    inputDesc[1].dims.d[1], 
                                    inputDesc[1].dims.d[2]);  
        CUDACHECK(status);
    }
    else if (inputDesc[0].dims.nbDims==5){
        cudaError_t status = grid_sampler_3d_cuda(stream,
                                    static_cast<const float*>(inputs[0]), 
                                    static_cast<const float*>(inputs[1]), 
                                    static_cast<float*>(outputs[0]),
                                    mMode,
                                    mPadding_mode,
                                    mAlign_corners,
                                    inputDesc[0].dims.d[0], 
                                    inputDesc[0].dims.d[1],
                                    inputDesc[0].dims.d[2],
                                    inputDesc[0].dims.d[3],
                                    inputDesc[0].dims.d[4],
                                    inputDesc[1].dims.d[1], 
                                    inputDesc[1].dims.d[2],
                                    inputDesc[1].dims.d[3]);
        CUDACHECK(status);
    }
    else 
        ASSERT(false && "Input dimensions should be 4 or 5");

    return 0;
};

// Return the DataType of the plugin output at the requested index
DataType GridSample::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only 2 input and 1 output from the plugin layer
    ASSERT(index == 0 && inputTypes[0] == inputTypes[1]);
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return inputTypes[0];
}


// Configure the layer with input and output data types.

void GridSample::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);
}


// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void GridSample::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void GridSample::detachFromContext() {}
