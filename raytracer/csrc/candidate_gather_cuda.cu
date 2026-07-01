#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kChannels = 13;
constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void gather_candidate_parameters_kernel(
    const scalar_t* parameter_table,
    const int64_t* candidate_ids,
    scalar_t* output,
    int64_t candidate_count) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = candidate_count * kChannels;
  if (linear >= total) {
    return;
  }
  const int64_t candidate = linear / kChannels;
  const int channel = static_cast<int>(linear % kChannels);
  output[linear] = parameter_table[candidate_ids[candidate] * kChannels + channel];
}

__global__ void initialize_positions_kernel(int64_t* positions, int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    positions[index] = index;
  }
}

__device__ int64_t lower_bound_id(
    const int64_t* sorted_ids,
    int64_t count,
    int64_t target) {
  int64_t first = 0;
  int64_t length = count;
  while (length > 0) {
    const int64_t half = length >> 1;
    const int64_t middle = first + half;
    if (sorted_ids[middle] < target) {
      first = middle + 1;
      length -= half + 1;
    } else {
      length = half;
    }
  }
  return first;
}

template <typename scalar_t>
__global__ void segmented_candidate_gradient_kernel(
    const int64_t* sorted_ids,
    const int64_t* sorted_positions,
    const scalar_t* candidate_gradients,
    scalar_t* output,
    int64_t candidate_count,
    int64_t surfel_count) {
  const int64_t surfel = blockIdx.x;
  if (surfel >= surfel_count) {
    return;
  }

  __shared__ int64_t range_begin;
  __shared__ int64_t range_end;
  __shared__ scalar_t partial[kChannels][kThreads];
  if (threadIdx.x == 0) {
    range_begin = lower_bound_id(sorted_ids, candidate_count, surfel);
    range_end = lower_bound_id(sorted_ids, candidate_count, surfel + 1);
  }
  __syncthreads();

#pragma unroll
  for (int channel = 0; channel < kChannels; ++channel) {
    scalar_t sum = scalar_t(0);
    for (int64_t sorted = range_begin + threadIdx.x;
         sorted < range_end;
         sorted += blockDim.x) {
      const int64_t original = sorted_positions[sorted];
      sum += candidate_gradients[original * kChannels + channel];
    }
    partial[channel][threadIdx.x] = sum;
  }
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int channel = 0; channel < kChannels; ++channel) {
        partial[channel][threadIdx.x] += partial[channel][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int channel = 0; channel < kChannels; ++channel) {
      output[surfel * kChannels + channel] = partial[channel][0];
    }
  }
}

}  // namespace

torch::Tensor gather_candidate_parameters_cuda(
    torch::Tensor parameter_table,
    torch::Tensor candidate_ids) {
  const int64_t candidate_count = candidate_ids.numel();
  auto output = torch::empty({candidate_count, kChannels}, parameter_table.options());
  if (candidate_count == 0) {
    return output;
  }
  const int64_t total = candidate_count * kChannels;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  AT_DISPATCH_FLOATING_TYPES(
      parameter_table.scalar_type(), "gather_candidate_parameters_cuda", [&] {
        gather_candidate_parameters_kernel<scalar_t>
            <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
                parameter_table.data_ptr<scalar_t>(),
                candidate_ids.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                candidate_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor reduce_candidate_gradients_cuda(
    torch::Tensor candidate_ids,
    torch::Tensor candidate_gradients,
    int64_t surfel_count) {
  const int64_t candidate_count = candidate_ids.numel();
  auto output = torch::zeros(
      {surfel_count, kChannels}, candidate_gradients.options());
  if (candidate_count == 0) {
    return output;
  }

  auto positions = torch::empty_like(candidate_ids);
  auto sorted_ids = torch::empty_like(candidate_ids);
  auto sorted_positions = torch::empty_like(candidate_ids);
  const int position_blocks = static_cast<int>((candidate_count + kThreads - 1) / kThreads);
  initialize_positions_kernel<<<
      position_blocks, kThreads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
          positions.data_ptr<int64_t>(), candidate_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int end_bit = 1;
  while (end_bit < 63 && (int64_t(1) << end_bit) < surfel_count) {
    ++end_bit;
  }
  size_t temporary_bytes = 0;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr,
      temporary_bytes,
      candidate_ids.data_ptr<int64_t>(),
      sorted_ids.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(),
      sorted_positions.data_ptr<int64_t>(),
      candidate_count,
      0,
      end_bit,
      stream));
  auto temporary = torch::empty(
      {static_cast<int64_t>(temporary_bytes)},
      candidate_ids.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temporary.data_ptr(),
      temporary_bytes,
      candidate_ids.data_ptr<int64_t>(),
      sorted_ids.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(),
      sorted_positions.data_ptr<int64_t>(),
      candidate_count,
      0,
      end_bit,
      stream));

  AT_DISPATCH_FLOATING_TYPES(
      candidate_gradients.scalar_type(), "reduce_candidate_gradients_cuda", [&] {
        segmented_candidate_gradient_kernel<scalar_t>
            <<<static_cast<int>(surfel_count), kThreads, 0, stream>>>(
                sorted_ids.data_ptr<int64_t>(),
                sorted_positions.data_ptr<int64_t>(),
                candidate_gradients.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                candidate_count,
                surfel_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
