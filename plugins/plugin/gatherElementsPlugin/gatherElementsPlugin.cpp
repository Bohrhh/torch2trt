/*
Author: kmlee
Date: 20200422
 */

#include <cuda_runtime_api.h>
#include <iostream>
#include "gatherElementsPlugin.h"

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::GatherElements;
using nvinfer1::plugin::GatherElementsPluginCreator;
                        
namespace
{
const char* GATHER_ELEMENTS_PLUGIN_VERSION{"1"};
const char* GATHER_ELEMENTS_PLUGIN_NAME{"GatherElementsPlugins"};
} // namespace

PluginFieldCollection GatherElementsPluginCreator::mFC{};
std::vector<PluginField> GatherElementsPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GatherElementsPluginCreator);

GatherElementsPluginCreator::GatherElementsPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("dim", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GatherElementsPluginCreator::getPluginName() const
{
    return GATHER_ELEMENTS_PLUGIN_NAME;
};

const char* GatherElementsPluginCreator::getPluginVersion() const
{
    return GATHER_ELEMENTS_PLUGIN_VERSION;
};

const PluginFieldCollection* GatherElementsPluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2DynamicExt* GatherElementsPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "dim"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mDim = *(static_cast<const int*>(fields[i].data));
        }
    }
    return new GatherElements(mDim);
};

IPluginV2DynamicExt* GatherElementsPluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new GatherElements(data, length);
};

GatherElements::GatherElements(int dim)
    : mDim(dim)
{   
    ASSERT(mDim > 0);
};


int GatherElements::getNbOutputs() const
{
    return 1;
};

DimsExprs GatherElements::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    ASSERT(nbInputs == 2);
    nvinfer1::DimsExprs output(inputs[1]);
    return output;
};

int GatherElements::initialize()
{
    return 0;
};

void GatherElements::terminate(){

};

void GatherElements::destroy(){

};

size_t GatherElements::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const 
{
    return 0;
}

size_t GatherElements::getSerializationSize() const
{
    return sizeof(int);
};

void GatherElements::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mDim);
    ASSERT(d == a + getSerializationSize());
};

GatherElements::GatherElements(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mDim = read<int>(d);
    ASSERT(d == a + length);
};

const char* GatherElements::getPluginType() const
{
    return GATHER_ELEMENTS_PLUGIN_NAME;
};

const char* GatherElements::getPluginVersion() const
{
    return GATHER_ELEMENTS_PLUGIN_VERSION;
};

IPluginV2DynamicExt* GatherElements::clone() const
{
    return new GatherElements(*this);
};

void GatherElements::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* GatherElements::getPluginNamespace() const
{
    return mNameSpace.c_str();
}


bool GatherElements::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(nbInputs==2 && nbOutputs==1 && pos < (nbInputs + nbOutputs));
    if (pos==0 || pos==2)
        return inOut[pos].type == nvinfer1::DataType::kFLOAT;
    else
        return inOut[pos].type == nvinfer1::DataType::kINT32;
}


int GatherElements::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    int totalElements = 1;
    int nbDims = inputDesc[1].dims.nbDims;
    for (int i=0; i<nbDims; ++i){
        totalElements = totalElements*inputDesc[1].dims.d[i];
    }

    CUDACHECK(gatherElements(
        stream, 
        static_cast<const float *>(inputs[0]), 
        static_cast<float*>(outputs[0]), 
        static_cast<const int *>(inputs[1]),
        inputDesc[0].dims,
        inputDesc[1].dims,
        mDim, 
        nbDims, 
        totalElements));
    return 0;
};

// Return the DataType of the plugin output at the requested index
DataType GatherElements::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only 1 input and 1 output from the plugin layer
    ASSERT(index == 0);

    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}


// Configure the layer with input and output data types.

void GatherElements::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(nbInputs == 2);
    ASSERT(in[0].desc.dims.nbDims==in[1].desc.dims.nbDims && in[0].desc.dims.nbDims==out[0].desc.dims.nbDims);
}


// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void GatherElements::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void GatherElements::detachFromContext() {}
