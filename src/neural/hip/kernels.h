
#include <hip/hip_runtime.h>
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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
namespace lczero {
namespace hip_backend {

// Adds two vectors (possibly of different sizes), also do optional
// activation (relu, tanh or sigmoid).
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize, bool relu,
                bool use_tanh, bool use_sigmoid, hipStream_t stream);

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W, bool relu,
                  hipStream_t stream);

// Conversion from: fp32 -> fp16 datatype, and NCHW -> NHWC layout.
// Cudnn kernels work best with NCHW layout for fp32, and with NHWC for fp16.
void fp32NCHWtofp16NHWC(half* output_tensor, float* input_tensor, int Nin,
                        int Cin, int Nout, int Cout, int H, int W);

// Plain data-type conversion (no layout conversion).
template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, hipStream_t stream);

// Perform batch normilization.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers, bool relu);

// Unpack planes (input to network).
void expandPlanes_Fp32_NCHW(float* output, const uint64_t* masks,
                            const float* values, int n, hipStream_t stream);

void expandPlanes_Fp16_NHWC(half* output, const uint64_t* masks,
                            const float* values, int n, hipStream_t stream);

void expandPlanes_Fp16_NCHW(half* output, const uint64_t* masks,
                            const float* values, int n, hipStream_t stream);

// Perform global avg pool.
template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc);

// Perform global scale.
template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc);

// Perform Squeeze-and-Excitation (SE) in a single fused kernel.
// Returns false if the fused kernel can't handle the sizes.
bool Se_Fp16_NHWC(int N, int C, int numFc1Out, half* output, const half* skip,
                  const half* input, const half* w1, const half* b1,
                  const half* w2, const half* b2, const half* bPrev);

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize,
               hipStream_t stream);

// Custom winograd helper functions
template <typename T>
void FilterTransform(int N, int C, T* transformedFilter, const T* filter);

template <typename T, bool nhcw>
void InputTransform(int N, int C, T* transformedInput, const T* input,
                    hipStream_t stream);

template <typename T, bool use_se, bool relu, bool use_bias, bool use_skip,
          bool skipInput_nhcw, bool output_nhcw>
void OutputTransform(int N, int C, int se_K, T* output, const T* input,
                     const T* skip, const T* bias, const T* w1, const T* b1,
                     const T* w2, const T* b2, hipStream_t stream);

template <typename T, bool use_se, bool relu, bool use_bias, bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2,
                          hipStream_t stream);

}  // namespace hip_backend
}  // namespace lczero