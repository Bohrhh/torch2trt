/*
Author: kmlee
Date: 20200828
 */


#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "dcnV2Plugin.h"
#include "common.h"

#define DEBUG 0

#define CHECK_CUDA(call) do {    \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    return status; \
  } \
} while(0)

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::DCNv2;
using nvinfer1::plugin::DCNv2PluginCreator;

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    if(!init[n]) {
        cublasCreate(&handle[n]);
        init[n] = 1;
    }
    return handle[n];
}
inline bool is_CHW(nvinfer1::Dims const& dims) {
    return (dims.nbDims == 3 &&
            dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
            dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
            dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

namespace
{
const char* DCNv2_PLUGIN_VERSION{"1"};
const char* DCNv2_PLUGIN_NAME{"DCNv2_TRT"};
} // namespace

PluginFieldCollection DCNv2PluginCreator::mFC{};
std::vector<PluginField> DCNv2PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(DCNv2PluginCreator);

DCNv2PluginCreator::DCNv2PluginCreator(){};

const char* DCNv2PluginCreator::getPluginName() const
{
    return DCNv2_PLUGIN_NAME;
};

const char* DCNv2PluginCreator::getPluginVersion() const
{
    return DCNv2_PLUGIN_VERSION;
};

const PluginFieldCollection* DCNv2PluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2DynamicExt* DCNv2PluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    int in_channel,out_channel,kernel_H,kernel_W,deformable_group,dilation,groups,padding,stride;
    std::vector<float> weight;
    std::vector<float> bias;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "in_channel"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            in_channel = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "out_channel"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            out_channel = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "kernel_H"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            kernel_H = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "kernel_W"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            kernel_W = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "deformable_group"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            deformable_group = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "dilation"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            dilation = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "groups"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            groups = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "padding"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            padding = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "stride"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            stride = *(static_cast<const int*>(fields[i].data));
        }

        if (!strcmp(attrName, "weight"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            weight.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                weight.push_back(*w);
                w++;
            }
        }

        if (!strcmp(attrName, "bias"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            if (size!=0){
                bias.reserve(size);
                const auto* w = static_cast<const float*>(fields[i].data);
                for (int j = 0; j < size; j++)
                {
                    bias.push_back(*w);
                    w++;
                }
            }
            else{
                ASSERT(out_channel != 0);
                bias.reserve(out_channel);
                for (int j = 0; j < out_channel; j++)
                {
                    bias.push_back(0.0f);
                }
            }
        }
    }

    Weights convWeight{DataType::kFLOAT, weight.data(), (int64_t) weight.size()};
    Weights convbias{DataType::kFLOAT, bias.data(), (int64_t) bias.size()};

    DCNv2* obj = new DCNv2(in_channel, out_channel, kernel_H, 
                            kernel_W, deformable_group, dilation, 
                            groups, padding, stride, 
                            convWeight, convbias);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
};

IPluginV2DynamicExt* DCNv2PluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new DCNv2(data, length);
};

DCNv2::DCNv2(int in_channel,
             int out_channel,
             int kernel_H,
             int kernel_W,
             int deformable_group,
             int dilation,
             int groups,
             int padding,
             int stride,
             nvinfer1::Weights const &weight, nvinfer1::Weights const &bias):_in_channel(in_channel),
             _out_channel(out_channel),_kernel_H(kernel_H),_kernel_W(kernel_W),_deformable_group(deformable_group),
             _dilation(dilation),_groups(groups),_padding(padding),_stride(stride),_initialized(false){

    if (weight.type == nvinfer1::DataType::kFLOAT)
    {
        _h_weight.assign((float*)weight.values,(float*)weight.values+weight.count);
    } else { throw std::runtime_error("Unsupported  weight dtype");}

    if (bias.type == nvinfer1::DataType::kFLOAT)
    {
        _h_bias.assign((float*)bias.values,(float*)bias.values+bias.count);
    } else { throw std::runtime_error("Unsupported  bias dtype");}

}


int DCNv2::getNbOutputs() const
{
    return 1;
};

DimsExprs DCNv2::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{

    ASSERT(nbInputs == 3);
    ASSERT(outputIndex == 0);
    nvinfer1::DimsExprs output(inputs[0]);
    output.d[1] = exprBuilder.constant(_out_channel);
    output.d[2] = exprBuilder.operation(DimensionOperation::kSUM, *output.d[2], *exprBuilder.constant(2 * _padding - (_dilation * (_kernel_H - 1) + 1)));
    output.d[2] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *output.d[2], *exprBuilder.constant(_stride));
    output.d[2] = exprBuilder.operation(DimensionOperation::kSUM, *output.d[2], *exprBuilder.constant(1));
    output.d[3] = exprBuilder.operation(DimensionOperation::kSUM, *output.d[3], *exprBuilder.constant(2 * _padding - (_dilation * (_kernel_W - 1) + 1)));
    output.d[3] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *output.d[3], *exprBuilder.constant(_stride));
    output.d[3] = exprBuilder.operation(DimensionOperation::kSUM, *output.d[3], *exprBuilder.constant(1));
    return output;
};

int DCNv2::initialize() {
    if(_initialized) return 0;

    size_t ones_size = _outH*_outW* sizeof(float);

    size_t weight_size = _h_weight.size()* sizeof(float);
    size_t bias_size = _h_bias.size()* sizeof(float);
    float *ones_cpu = new float[ones_size/ sizeof(float)];
    for (int i = 0; i < ones_size/ sizeof(float); i++) {
        ones_cpu[i] = 1.0;
    }

    CHECK_CUDA(cudaMalloc((void**)&_d_columns, _in_channel * _kernel_H * _kernel_W * ones_size));
    CHECK_CUDA(cudaMalloc((void**)&_d_ones, ones_size));
    CHECK_CUDA(cudaMalloc((void**)&_d_weight, weight_size));
    CHECK_CUDA(cudaMalloc((void**)&_d_bias, bias_size));
    CHECK_CUDA(cudaMemcpy(_d_ones, ones_cpu, ones_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(_d_weight, _h_weight.data(), weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(_d_bias, _h_bias.data(), bias_size, cudaMemcpyHostToDevice));
    delete[] ones_cpu;
    _initialized = true;

    return 0;
};

void DCNv2::terminate() {
    if (!_initialized) {
        return;
    }
    cudaFree(_d_columns);
    cudaFree(_d_bias);
    cudaFree(_d_weight);
    cudaFree(_d_ones);
    _initialized = false;
};

void DCNv2::destroy(){

};

size_t DCNv2::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const 
{
    return 0;
}

size_t DCNv2::getSerializationSize() const
{
    return (serialized_size(_in_channel) +
            serialized_size(_out_channel) +
            serialized_size(_kernel_H) +
            serialized_size(_kernel_W) +
            serialized_size(_deformable_group) +
            serialized_size(_dilation) +
            serialized_size(_groups) +
            serialized_size(_padding) +
            serialized_size(_stride) +
            serialized_size(_outH) +
            serialized_size(_outW) +
            serialized_size(_h_weight) +
            serialized_size(_h_bias));
};

void DCNv2::serialize(void* buffer) const
{
    serialize_value(&buffer, _in_channel);
    serialize_value(&buffer, _out_channel);
    serialize_value(&buffer, _kernel_H);
    serialize_value(&buffer, _kernel_W);
    serialize_value(&buffer, _deformable_group);
    serialize_value(&buffer, _dilation);
    serialize_value(&buffer, _groups);
    serialize_value(&buffer, _padding);
    serialize_value(&buffer, _stride);
    serialize_value(&buffer, _outH);
    serialize_value(&buffer, _outW);
    serialize_value(&buffer, _h_weight);
    serialize_value(&buffer, _h_bias);

};

DCNv2::DCNv2(const void* serialData, size_t serialLength)
{
    deserialize_value(&serialData, &serialLength, &_in_channel);
    deserialize_value(&serialData, &serialLength, &_out_channel);
    deserialize_value(&serialData, &serialLength, &_kernel_H);
    deserialize_value(&serialData, &serialLength, &_kernel_W);
    deserialize_value(&serialData, &serialLength, &_deformable_group);
    deserialize_value(&serialData, &serialLength, &_dilation);
    deserialize_value(&serialData, &serialLength, &_groups);
    deserialize_value(&serialData, &serialLength, &_padding);
    deserialize_value(&serialData, &serialLength, &_stride);
    deserialize_value(&serialData, &serialLength, &_outH);
    deserialize_value(&serialData, &serialLength, &_outW);
    deserialize_value(&serialData, &serialLength, &_h_weight);
    deserialize_value(&serialData, &serialLength, &_h_bias);
};

const char* DCNv2::getPluginType() const
{
    return DCNv2_PLUGIN_NAME;
};

const char* DCNv2::getPluginVersion() const
{
    return DCNv2_PLUGIN_VERSION;
};

IPluginV2DynamicExt* DCNv2::clone() const
{
    return new DCNv2(*this);
};

void DCNv2::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* DCNv2::getPluginNamespace() const
{
    return mNameSpace.c_str();
}


bool DCNv2::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return (inOut[pos].type == DataType::kFLOAT && inOut[pos].format == PluginFormat::kLINEAR);
}


int DCNv2::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    float alpha ,beta;
    int m, n, k;

    cublasHandle_t handle = blas_handle();
    const float* input = static_cast<const float *>(inputs[0]);
    const float* offset = static_cast<const float *>(inputs[1]);
    const float* mask = static_cast<const float *>(inputs[2]);
    float * output = static_cast<float *>(outputs[0]);
    ASSERT(inputDesc[0].dims.d[0]==1);
    int h = inputDesc[0].dims.d[2];
    int w = inputDesc[0].dims.d[3];
    int height_out = (h + 2 * _padding - (_dilation * (_kernel_H - 1) + 1)) / _stride + 1;
    int width_out = (w + 2 * _padding - (_dilation * (_kernel_W - 1) + 1)) / _stride + 1;
    m = _out_channel;
    n = height_out * width_out;
    k = 1;
    alpha = 1.0;
    beta = 0.0;

    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,&alpha,
                _d_ones, k,
                _d_bias, k,&beta,
                output, n);

    // im2col (offset and mask)
    cudaError_t status = modulated_deformable_im2col_cuda(
        stream,input, offset, mask,
        1, _in_channel, h, w,
        height_out, width_out, _kernel_H, _kernel_W,
        _padding, _padding, _stride, _stride, _dilation, _dilation,
        _deformable_group, _d_columns);

    m = _out_channel;
    n = height_out * width_out;
    k = _in_channel * _kernel_H * _kernel_W;
    alpha = 1.0;
    beta = 1.0;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,&alpha,
                _d_columns, n,
                _d_weight, k,
                &beta,
                output, n);

    assert(status == cudaSuccess);
    return 0;
};


// Return the DataType of the plugin output at the requested index
DataType DCNv2::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only 2 input and 1 output from the plugin layer
    ASSERT(index == 0 && inputTypes[0] == inputTypes[1] && inputTypes[0] == inputTypes[2]);
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return inputTypes[0];
}


// Configure the layer with input and output data types.

void DCNv2::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(nbInputs == 3);
    ASSERT(nbOutputs == 1);
    _outC = out[0].desc.dims.d[1];
    _outH = out[0].desc.dims.d[2];
    _outW = out[0].desc.dims.d[3];
}


// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void DCNv2::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void DCNv2::detachFromContext() {}
