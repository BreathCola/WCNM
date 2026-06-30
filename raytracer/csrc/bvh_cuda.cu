#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxStack = 64;

__device__ bool intersects_aabb(
    const float* origin,
    const float* direction,
    const float* minimum,
    const float* maximum) {
  float near_value = 0.0f;
  float far_value = 1.0e30f;
  for (int axis = 0; axis < 3; ++axis) {
    const float d = direction[axis];
    const float o = origin[axis];
    if (fabsf(d) <= 1e-12f) {
      if (o < minimum[axis] || o > maximum[axis]) {
        return false;
      }
      continue;
    }
    const float inv = 1.0f / d;
    float first = (minimum[axis] - o) * inv;
    float second = (maximum[axis] - o) * inv;
    if (first > second) {
      const float temporary = first;
      first = second;
      second = temporary;
    }
    near_value = fmaxf(near_value, first);
    far_value = fminf(far_value, second);
    if (far_value < near_value) {
      return false;
    }
  }
  return far_value >= near_value;
}

template <bool kFill>
__global__ void traverse_kernel(
    const float* origins,
    const float* directions,
    const float* node_min,
    const float* node_max,
    const int64_t* leaf_indices,
    const int64_t* offsets,
    int64_t* candidates,
    int32_t* counts,
    int64_t ray_count,
    int64_t leaf_base) {
  const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ray >= ray_count) {
    return;
  }
  const float* origin = origins + ray * 3;
  const float* direction = directions + ray * 3;
  int stack[kMaxStack];
  int stack_size = 1;
  stack[0] = 1;
  int32_t count = 0;
  int64_t write = kFill ? offsets[ray] : 0;

  while (stack_size > 0) {
    const int node = stack[--stack_size];
    if (!intersects_aabb(origin, direction, node_min + node * 3, node_max + node * 3)) {
      continue;
    }
    if (node >= leaf_base) {
      const int64_t leaf = static_cast<int64_t>(node) - leaf_base;
      const int64_t surfel = leaf_indices[leaf];
      if (surfel >= 0) {
        if (kFill) {
          candidates[write + count] = surfel;
        }
        ++count;
      }
      continue;
    }
    if (stack_size + 2 > kMaxStack) {
      counts[ray] = -1;
      return;
    }
    stack[stack_size++] = node * 2 + 1;
    stack[stack_size++] = node * 2;
  }
  counts[ray] = count;
}

}  // namespace

torch::Tensor count_candidates_cuda(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor node_min,
    torch::Tensor node_max,
    torch::Tensor leaf_indices,
    int64_t leaf_base) {
  const int64_t rays = origins.size(0);
  auto counts = torch::zeros({rays}, origins.options().dtype(torch::kInt32));
  if (rays == 0) {
    return counts;
  }
  const int blocks = static_cast<int>((rays + kThreads - 1) / kThreads);
  traverse_kernel<false><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      origins.data_ptr<float>(), directions.data_ptr<float>(), node_min.data_ptr<float>(),
      node_max.data_ptr<float>(), leaf_indices.data_ptr<int64_t>(), nullptr, nullptr,
      counts.data_ptr<int32_t>(), rays, leaf_base);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return counts;
}

torch::Tensor fill_candidates_cuda(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor node_min,
    torch::Tensor node_max,
    torch::Tensor leaf_indices,
    torch::Tensor offsets,
    int64_t leaf_base) {
  const int64_t rays = origins.size(0);
  const int64_t total = offsets.index({rays}).item<int64_t>();
  auto candidates = torch::empty({total}, offsets.options());
  if (rays == 0 || total == 0) {
    return candidates;
  }
  auto counts = torch::zeros({rays}, origins.options().dtype(torch::kInt32));
  const int blocks = static_cast<int>((rays + kThreads - 1) / kThreads);
  traverse_kernel<true><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      origins.data_ptr<float>(), directions.data_ptr<float>(), node_min.data_ptr<float>(),
      node_max.data_ptr<float>(), leaf_indices.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
      candidates.data_ptr<int64_t>(), counts.data_ptr<int32_t>(), rays, leaf_base);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return candidates;
}
