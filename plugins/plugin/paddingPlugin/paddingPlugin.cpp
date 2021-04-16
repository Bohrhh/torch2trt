/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda_runtime_api.h>
#include <iostream>
#include "common.h"
#include "serialize.hpp"
#include "paddingPlugin.h"


using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::Padding;
using nvinfer1::plugin::PaddingPluginCreator;
                        
namespace
{
const char* PADDING_PLUGIN_VERSION{"1"};
const char* PADDING_PLUGIN_NAME{"PaddingPlugin"};
} // namespace

PluginFieldCollection PaddingPluginCreator::mFC{};
std::vector<PluginField> PaddingPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(PaddingPluginCreator);

PaddingPluginCreator::PaddingPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 16));
    mPluginAttributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* PaddingPluginCreator::getPluginName() const
{
    return PADDING_PLUGIN_NAME;
};

const char* PaddingPluginCreator::getPluginVersion() const
{
    return PADDING_PLUGIN_VERSION;
};

const PluginFieldCollection* PaddingPluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2DynamicExt* PaddingPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    int pads[16];
    int mode;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "padding"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            for (int j=0; j<16; j++)
                pads[j] = static_cast<const int*>(fields[i].data)[j];
        }
        if (!strcmp(attrName, "mode"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mode = *(static_cast<const int*>(fields[i].data));
        }

    }
    return new Padding(pads, mode);
};

IPluginV2DynamicExt* PaddingPluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new Padding(data, length);
};

Padding::Padding(const int* pads, int mode): mMode(mode)
{   
    for (int i=0; i<16; i++)
        mPads[i] = pads[i];
};


int Padding::getNbOutputs() const
{
    return 1;
};

DimsExprs Padding::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{

    ASSERT(nbInputs == 1);
    ASSERT(outputIndex == 0);
    ASSERT(inputs[0].nbDims <= 4 && inputs[0].nbDims>=3);
    nvinfer1::DimsExprs output(inputs[0]);

    if (inputs[0].nbDims==3){
        for (int i=0; i<2; i++){
            auto pad = exprBuilder.constant(mPads[i]);
            output.d[2] = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *output.d[2], *pad);
        }
    }
    else if (inputs[0].nbDims==4){
        for (int i=0; i<2; i++){
            auto pad = exprBuilder.constant(mPads[i]);
            output.d[3] = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *output.d[3], *pad);
        }
        for (int i=2; i<4; i++){
            auto pad = exprBuilder.constant(mPads[i]);
            output.d[2] = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *output.d[2], *pad);
        }
    }

    return output;

};

int Padding::initialize()
{
    return 0;
};

void Padding::terminate(){

};

void Padding::destroy(){

};

size_t Padding::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const 
{
    return 0;
}

size_t Padding::getSerializationSize() const
{
    // mPads, mMode
    return 16*sizeof(int)+sizeof(int);
};

void Padding::serialize(void* buffer) const
{
    serialize_value(&buffer, mPads);
    serialize_value(&buffer, mMode);
};

Padding::Padding(const void* data, size_t length)
{
    deserialize_value(&data, &length, &mPads);
    deserialize_value(&data, &length, &mMode);
};

const char* Padding::getPluginType() const
{
    return PADDING_PLUGIN_NAME;
};

const char* Padding::getPluginVersion() const
{
    return PADDING_PLUGIN_VERSION;
};

IPluginV2DynamicExt* Padding::clone() const
{
    return new Padding(*this);
};

void Padding::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* Padding::getPluginNamespace() const
{
    return mNameSpace.c_str();
}


bool Padding::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT);
}


int Padding::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    CUDACHECK(padding_cuda(stream, 
                           static_cast<const float*>(inputs[0]),
                           static_cast<float*>(outputs[0]),
                           inputDesc[0].dims,
                           mPads,
                           static_cast<PaddingMode>(mMode)));

    return 0;
};

// Return the DataType of the plugin output at the requested index
DataType Padding::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}


// Configure the layer with input and output data types.

void Padding::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);
}


// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Padding::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void Padding::detachFromContext() {}
