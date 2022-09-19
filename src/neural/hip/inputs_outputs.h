/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "neural/network.h"

namespace lczero {
namespace hip_backend {

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left,
                size_t tensor_mem_size = 0, size_t scratch_size = 0) {
    ReportHIPErrors(hipHostMalloc(
        (void**)&input_masks_mem_, maxBatchSize * kInputPlanes * sizeof(uint64_t),
        hipHostMallocMapped));
    ReportHIPErrors(
        hipHostGetDevicePointer((void**)&input_masks_mem_gpu_, input_masks_mem_, 0));

    ReportHIPErrors(hipHostMalloc((void**)&input_val_mem_,
                                   maxBatchSize * kInputPlanes * sizeof(float),
                                   hipHostMallocMapped));
    ReportHIPErrors(
        hipHostGetDevicePointer((void**)&input_val_mem_gpu_, input_val_mem_, 0));

    ReportHIPErrors(hipHostMalloc(
        (void**)&op_policy_mem_, maxBatchSize * kNumOutputPolicy * sizeof(float), 0));

    // Seperate device memory copy for policy output.
    // It's faster to write to device memory and then copy to host memory
    // than having the kernel write directly to it.
    ReportHIPErrors(hipMalloc(
        &op_policy_mem_gpu_, maxBatchSize * kNumOutputPolicy * sizeof(float)));

    ReportHIPErrors(hipHostMalloc((void**)&op_value_mem_,
                                   maxBatchSize * (wdl ? 3 : 1) * sizeof(float),
                                   hipHostMallocMapped));
    ReportHIPErrors(
        hipHostGetDevicePointer((void**)&op_value_mem_gpu_, op_value_mem_, 0));
    if (moves_left) {
      ReportHIPErrors(hipHostMalloc((void**)&op_moves_left_mem_,
                                     maxBatchSize * sizeof(float),
                                     hipHostMallocMapped));
      ReportHIPErrors(hipHostGetDevicePointer((void**)&op_moves_left_mem_gpu_,
                                                op_moves_left_mem_, 0));
    }

    // memory for network execution managed inside this structure
    if (tensor_mem_size) {
      multi_stream_ = true;
      ReportHIPErrors(hipStreamCreate(&stream_));
      ReportHIPErrors(hipMalloc(&scratch_mem_, scratch_size));
      for (auto& mem : tensor_mem_) {
        ReportHIPErrors(hipMalloc(&mem, tensor_mem_size));
        ReportHIPErrors(hipMemsetAsync(mem, 0, tensor_mem_size, stream_));
      }
      ReportCUBLASErrors(hipblasCreate(&cublas_));
      // ReportCUBLASErrors(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH));
      ReportCUBLASErrors(hipblasSetStream(cublas_, stream_));
    } else {
      multi_stream_ = false;
    }
  }
  ~InputsOutputs() {
    ReportHIPErrors(hipHostFree(input_masks_mem_));
    ReportHIPErrors(hipHostFree(input_val_mem_));
    ReportHIPErrors(hipHostFree(op_policy_mem_));
    ReportHIPErrors(hipFree(op_policy_mem_gpu_));
    ReportHIPErrors(hipHostFree(op_value_mem_));

    if (multi_stream_) {
      for (auto mem : tensor_mem_) {
        if (mem) ReportHIPErrors(hipFree(mem));
      }
      if (scratch_mem_) ReportHIPErrors(hipFree(scratch_mem_));

      hipStreamDestroy(stream_);
      hipblasDestroy(cublas_);
    }
  
  }
  uint64_t* input_masks_mem_;
  float* input_val_mem_;
  float* op_policy_mem_;
  float* op_value_mem_;
  float* op_moves_left_mem_;

  // GPU pointers for the above allocations.
  uint64_t* input_masks_mem_gpu_;
  float* input_val_mem_gpu_;
  float* op_value_mem_gpu_;
  float* op_moves_left_mem_gpu_;

  // This is a seperate copy.
  float* op_policy_mem_gpu_;

  // memory needed to run the network owned by InputsOutputs when multi_stream
  // is enabled
  bool multi_stream_;
  void* tensor_mem_[3];
  void* scratch_mem_;

  // hip stream used to run the network
  hipStream_t stream_;
  hipblasHandle_t cublas_;

  // cublas handle used to run the network

};

}  // namespace hip_backend
}  // namespace lczero
