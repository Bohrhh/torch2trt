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
#include "correlationPlugin.h"
#include <cuda_runtime_api.h>
#include <iostream>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::Correlation;
using nvinfer1::plugin::CorrelationPluginCreator;
                        
namespace
{
const char* CORRELATION_PLUGIN_VERSION{"1"};
const char* CORRELATION_PLUGIN_NAME{"CorrelationPlugin"};
} // namespace

PluginFieldCollection CorrelationPluginCreator::mFC{};
std::vector<PluginField> CorrelationPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CorrelationPluginCreator);

CorrelationPluginCreator::CorrelationPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("max_disparity", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("is_time", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("is_mean", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* CorrelationPluginCreator::getPluginName() const
{
    return CORRELATION_PLUGIN_NAME;
};

const char* CorrelationPluginCreator::getPluginVersion() const
{
    return CORRELATION_PLUGIN_VERSION;
};

const PluginFieldCollection* CorrelationPluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2DynamicExt* CorrelationPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "max_disparity"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mMaxDisparity = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "stride"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mStride = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "is_time"))
        {
            mIsTime = *(static_cast<const bool*>(fields[i].data));
        }

        if (!strcmp(attrName, "is_mean"))
        {
            mIsMean = *(static_cast<const bool*>(fields[i].data));
        }

    }
    return new Correlation(mMaxDisparity, mStride, mIsTime, mIsMean);
};

IPluginV2DynamicExt* CorrelationPluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new Correlation(data, length);
};

Correlation::Correlation(int maxDisparity, int stride, bool is_time, bool is_mean)
    : mMaxDisparity(maxDisparity), mStride(stride), mIsTime(is_time), mIsMean(is_mean)
{   
    ASSERT(mMaxDisparity > 0);
    ASSERT(mStride > 0);
};


int Correlation::getNbOutputs() const
{
    return 1;
};

DimsExprs Correlation::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{

    ASSERT(nbInputs == 2);
    ASSERT(outputIndex == 0);
    nvinfer1::DimsExprs output(inputs[0]);    
    output.d[1] = exprBuilder.constant(2*mMaxDisparity/mStride+1);

    return output;

};

int Correlation::initialize()
{
    return 0;
};

void Correlation::terminate(){

};

void Correlation::destroy(){

};

size_t Correlation::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const 
{
    return 0;
}

size_t Correlation::getSerializationSize() const
{
    // mMaxDisparity, mStride, mIsTime, mIsMean
    return 2*sizeof(int)+2*sizeof(bool);
};

void Correlation::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mMaxDisparity);
    write(d, mStride);
    write(d, mIsTime);
    write(d, mIsMean);
    ASSERT(d == a + getSerializationSize());
};

Correlation::Correlation(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mMaxDisparity = read<int32_t>(d);
    mStride = read<int32_t>(d);
    mIsTime = read<bool>(d);
    mIsMean = read<bool>(d);
    ASSERT(d == a + length);
};

const char* Correlation::getPluginType() const
{
    return CORRELATION_PLUGIN_NAME;
};

const char* Correlation::getPluginVersion() const
{
    return CORRELATION_PLUGIN_VERSION;
};

IPluginV2DynamicExt* Correlation::clone() const
{
    return new Correlation(*this);
};

void Correlation::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* Correlation::getPluginNamespace() const
{
    return mNameSpace.c_str();
}


bool Correlation::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT);
}


int Correlation::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    int32_t batch = inputDesc[0].dims.d[0];
    int32_t channels = inputDesc[0].dims.d[1];
    int32_t height = inputDesc[0].dims.d[2];
    int32_t width = inputDesc[0].dims.d[3];
    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT){
        const float* left = static_cast<const float*>(inputs[0]);
        const float* right = static_cast<const float*>(inputs[1]);
        float* corr = static_cast<float*>(outputs[0]);
        correlation(stream, left, right, corr, batch, mMaxDisparity, mStride, channels, height, width, mIsTime, mIsMean);
    }
    else{
        ASSERT(false && "now only suport float type!");
    }
    return cudaGetLastError() != cudaSuccess;

};

// Return the DataType of the plugin output at the requested index
DataType Correlation::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only 2 input and 1 output from the plugin layer
    ASSERT(index == 0 && inputTypes[0] == inputTypes[1]);
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return inputTypes[0];
}


// Configure the layer with input and output data types.

void Correlation::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);
}


// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Correlation::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void Correlation::detachFromContext() {}
