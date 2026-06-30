#include <torch/extension.h>

#include <vector>

torch::Tensor count_candidates_cuda(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor node_min,
    torch::Tensor node_max,
    torch::Tensor leaf_indices,
    int64_t leaf_base);

torch::Tensor fill_candidates_cuda(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor node_min,
    torch::Tensor node_max,
    torch::Tensor leaf_indices,
    torch::Tensor offsets,
    int64_t leaf_base);

namespace {

void check_inputs(
    const torch::Tensor& origins,
    const torch::Tensor& directions,
    const torch::Tensor& node_min,
    const torch::Tensor& node_max,
    const torch::Tensor& leaf_indices,
    int64_t leaf_base) {
  TORCH_CHECK(origins.is_cuda(), "origins must be CUDA");
  TORCH_CHECK(directions.is_cuda(), "directions must be CUDA");
  TORCH_CHECK(node_min.is_cuda() && node_max.is_cuda(), "BVH nodes must be CUDA");
  TORCH_CHECK(leaf_indices.is_cuda(), "leaf_indices must be CUDA");
  TORCH_CHECK(origins.scalar_type() == torch::kFloat32, "origins must be float32");
  TORCH_CHECK(directions.scalar_type() == torch::kFloat32, "directions must be float32");
  TORCH_CHECK(node_min.scalar_type() == torch::kFloat32 && node_max.scalar_type() == torch::kFloat32,
              "BVH nodes must be float32");
  TORCH_CHECK(leaf_indices.scalar_type() == torch::kInt64, "leaf_indices must be int64");
  TORCH_CHECK(origins.dim() == 2 && origins.size(1) == 3, "origins must have shape [M,3]");
  TORCH_CHECK(directions.sizes() == origins.sizes(), "directions must match origins");
  TORCH_CHECK(node_min.dim() == 2 && node_min.size(1) == 3, "node_min must have shape [K,3]");
  TORCH_CHECK(node_max.sizes() == node_min.sizes(), "node_max must match node_min");
  TORCH_CHECK(leaf_indices.dim() == 1, "leaf_indices must have shape [P]");
  TORCH_CHECK(leaf_base >= 1, "leaf_base must be positive");
  TORCH_CHECK(node_min.size(0) >= 2 * leaf_base, "node arrays are too small");
}

}  // namespace

torch::Tensor count_candidates(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor node_min,
    torch::Tensor node_max,
    torch::Tensor leaf_indices,
    int64_t leaf_base) {
  check_inputs(origins, directions, node_min, node_max, leaf_indices, leaf_base);
  return count_candidates_cuda(
      origins.contiguous(), directions.contiguous(), node_min.contiguous(), node_max.contiguous(),
      leaf_indices.contiguous(), leaf_base);
}

torch::Tensor fill_candidates(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor node_min,
    torch::Tensor node_max,
    torch::Tensor leaf_indices,
    torch::Tensor offsets,
    int64_t leaf_base) {
  check_inputs(origins, directions, node_min, node_max, leaf_indices, leaf_base);
  TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kInt64,
              "offsets must be CUDA int64");
  TORCH_CHECK(offsets.dim() == 1 && offsets.size(0) == origins.size(0) + 1,
              "offsets must have shape [M+1]");
  return fill_candidates_cuda(
      origins.contiguous(), directions.contiguous(), node_min.contiguous(), node_max.contiguous(),
      leaf_indices.contiguous(), offsets.contiguous(), leaf_base);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("count_candidates", &count_candidates, "Count ray/LBVH leaf candidates (CUDA)");
  module.def("fill_candidates", &fill_candidates, "Fill ray/LBVH leaf candidates (CUDA)");
}
