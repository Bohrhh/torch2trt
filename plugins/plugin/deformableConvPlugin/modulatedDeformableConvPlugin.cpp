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
#include "serialize.hpp"
#include "common.h"
#include "modulatedDeformableConvPlugin.h"
#include "deform_conv_cuda.h"

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::ModulatedDeformableConv;
using nvinfer1::plugin::ModulatedDeformableConvPluginCreator;
                        
namespace
{
const char* MODULATED_DEFORMABLE_CONV_PLUGIN_VERSION{"1"};
const char* MODULATED_DEFORMABLE_CONV_PLUGIN_NAME{"ModulatedDeformableConvPlugin"};
} // namespace

PluginFieldCollection ModulatedDeformableConvPluginCreator::mFC{};
std::vector<PluginField> ModulatedDeformableConvPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ModulatedDeformableConvPluginCreator);

ModulatedDeformableConvPluginCreator::ModulatedDeformableConvPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("groups", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("deformable_groups", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ModulatedDeformableConvPluginCreator::getPluginName() const
{
    return MODULATED_DEFORMABLE_CONV_PLUGIN_NAME;
};

const char* ModulatedDeformableConvPluginCreator::getPluginVersion() const
{
    return MODULATED_DEFORMABLE_CONV_PLUGIN_VERSION;
};

const PluginFieldCollection* ModulatedDeformableConvPluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2DynamicExt* ModulatedDeformableConvPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    nvinfer1::Dims stride{2, {1, 1}};
    nvinfer1::Dims padding{2, {0, 0}};
    nvinfer1::Dims dilation{2, {1, 1}};

    int deformableGroups = 1;
    int groups = 1;

    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "stride"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            stride.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
            stride.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
        }

        if (!strcmp(attrName, "padding"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            padding.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
            padding.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
        }

        if (!strcmp(attrName, "dilation"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            dilation.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
            dilation.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
        }

        if (!strcmp(attrName, "groups"))
        {   
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            groups = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "deformable_groups"))
        {   
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            deformableGroups = *(static_cast<const int*>(fields[i].data));
        }

    }
    return new ModulatedDeformableConv(stride, padding, dilation, groups, deformableGroups);
};

IPluginV2DynamicExt* ModulatedDeformableConvPluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new ModulatedDeformableConv(data, length);
};

ModulatedDeformableConv::ModulatedDeformableConv(
    const nvinfer1::Dims &stride, 
    const nvinfer1::Dims &padding,
    const nvinfer1::Dims &dilation,
    int groups,
    int deformable_groups)
    : mStride(stride), 
      mPadding(padding), 
      mDilation(dilation),
      mGroups(groups), 
      mDeformableGroups(deformable_groups)
{
    mWithBias = false;
};


int ModulatedDeformableConv::getNbOutputs() const
{
    return 1;
};

DimsExprs ModulatedDeformableConv::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[3].d[0];
    output.d[2] = inputs[1].d[2];
    output.d[3] = inputs[1].d[3];
    return output;
};

int ModulatedDeformableConv::initialize()
{
    cublasCreate(&m_cublas_handle);
    return 0;
};

void ModulatedDeformableConv::terminate(){
    cublasDestroy(m_cublas_handle);
};

void ModulatedDeformableConv::destroy(){

};

size_t ModulatedDeformableConv::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const 
{
    int sizeof_dtype = getDataTypeSize(outputs[0].type);

    int N    = inputs[0].dims.d[0];
    int in_C = inputs[0].dims.d[1];
    int in_H = inputs[0].dims.d[2];
    int in_W = inputs[0].dims.d[3];

    int out_C = outputs[0].dims.d[1];
    int out_H = outputs[0].dims.d[2];
    int out_W = outputs[0].dims.d[3];

    int kW = inputs[3].dims.d[2];
    int kH = inputs[3].dims.d[3];
    int im2col_step = std::min(32, N);

    size_t col_size = getAlignedSize(in_C * kW * kH * out_H * out_W * sizeof_dtype);

    return col_size;
}

size_t ModulatedDeformableConv::getSerializationSize() const
{
  return sizeof(mStride) + sizeof(mPadding) + sizeof(mDilation) +
         sizeof(mDeformableGroups) + sizeof(mGroups);
};

void ModulatedDeformableConv::serialize(void* buffer) const
{
    serialize_value(&buffer, mStride);
    serialize_value(&buffer, mPadding);
    serialize_value(&buffer, mDilation);
    serialize_value(&buffer, mDeformableGroups);
    serialize_value(&buffer, mGroups);
};

ModulatedDeformableConv::ModulatedDeformableConv(const void* data, size_t length)
{
    deserialize_value(&data, &length, &mStride);
    deserialize_value(&data, &length, &mPadding);
    deserialize_value(&data, &length, &mDilation);
    deserialize_value(&data, &length, &mDeformableGroups);
    deserialize_value(&data, &length, &mGroups);
    mWithBias = false;
};

const char* ModulatedDeformableConv::getPluginType() const
{
    return MODULATED_DEFORMABLE_CONV_PLUGIN_NAME;
};

const char* ModulatedDeformableConv::getPluginVersion() const
{
    return MODULATED_DEFORMABLE_CONV_PLUGIN_VERSION;
};

IPluginV2DynamicExt* ModulatedDeformableConv::clone() const
{
    return new ModulatedDeformableConv(*this);
};

void ModulatedDeformableConv::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* ModulatedDeformableConv::getPluginNamespace() const
{
    return mNameSpace.c_str();
}


bool ModulatedDeformableConv::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
  if (pos == 0) {
    return inOut[0].type == DataType::kFLOAT &&
           inOut[0].format == nvinfer1::TensorFormat::kLINEAR;
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}


int ModulatedDeformableConv::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    if (m_cuda_stream != stream) {
        cublasSetStream(m_cublas_handle, stream);
        m_cuda_stream = stream;
    }

    DCN_PARAMS dcn_params;
    dcn_params.cublas_handle = m_cublas_handle;
    dcn_params.batchSize = inputDesc[0].dims.d[0];
    dcn_params.inputChannel = inputDesc[0].dims.d[1];
    dcn_params.inputH = inputDesc[0].dims.d[2];
    dcn_params.inputW = inputDesc[0].dims.d[3];
    dcn_params.outputChannel = inputDesc[3].dims.d[0];
    dcn_params.kernelChannels = inputDesc[3].dims.d[1];
    dcn_params.kernelW = inputDesc[3].dims.d[2];
    dcn_params.kernelH = inputDesc[3].dims.d[3];
    dcn_params.strideW = mStride.d[0];
    dcn_params.strideH = mStride.d[1];
    dcn_params.padW = mPadding.d[0];
    dcn_params.padH = mPadding.d[1];
    dcn_params.dilationW = mDilation.d[0];
    dcn_params.dilationH = mDilation.d[1];
    dcn_params.groups = mGroups;
    dcn_params.deformable_groups = mDeformableGroups;
    dcn_params.im2col_step = std::min(32, dcn_params.batchSize);

    const float *bias = mWithBias ? static_cast<const float*>(inputs[4]) : nullptr;

    CUDACHECK(modulated_deform_conv_cuda_forward(stream, 
                            static_cast<const float*>(inputs[0]),
                            static_cast<const float*>(inputs[3]),
                            bias,
                            static_cast<const float*>(inputs[1]),
                            static_cast<const float*>(inputs[2]),
                            static_cast<float*>(outputs[0]),
                            workspace,
                            dcn_params));
    return 0;
};

// Return the DataType of the plugin output at the requested index
DataType ModulatedDeformableConv::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}


// Configure the layer with input and output data types.

void ModulatedDeformableConv::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
  if (nbInputs == 5) {
    mWithBias = true;
  }
}


// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ModulatedDeformableConv::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{}

// Detach the plugin object from its execution context.
void ModulatedDeformableConv::detachFromContext() {}
