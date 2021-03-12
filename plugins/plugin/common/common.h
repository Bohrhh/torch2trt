#pragma once

#include <iostream>
#include <stdexcept>
#include "NvInfer.h"
#include "NvInferPlugin.h"

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

#define CUDACHECK(status_)                                                                                             \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            std::cerr << __FILE__ << ", " << __LINE__ << ", " << s_ << ", " << cudaGetErrorString(s_) << std::endl;    \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define ASSERT(assertion)                                                                                              \
    {                                                                                                                  \
        if (!(assertion))                                                                                              \
        {                                                                                                              \
            std::cerr << "#assertion" << __FILE__ << "," << __LINE__ << std::endl;                                     \
            abort();                                                                                                   \
        }                                                                                                              \
    }
    
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)


inline int GET_BLOCKS(const int N)
{
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

inline size_t getAlignedSize(size_t origin_size, size_t aligned_number = 16) {
    return size_t((origin_size + aligned_number - 1) / aligned_number) * aligned_number;
}

inline unsigned int getDataTypeSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8:
            return 1;
  }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

namespace nvinfer1
{

class Pads
{
public:
    static const int MAX_DIMS = 16; //!< The maximum number of dimensions supported for a pads.
    int nbPads;                     //!< The number of pads.
    int d[MAX_DIMS];                //!< The extent of each dimension.
};

}
