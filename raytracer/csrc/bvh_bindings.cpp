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

torch::Tensor gather_candidate_parameters_cuda(
    torch::Tensor parameter_table,
    torch::Tensor candidate_ids);

torch::Tensor reduce_candidate_gradients_cuda(
    torch::Tensor candidate_ids,
    torch::Tensor candidate_gradients,
    int64_t surfel_count);

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

torch::Tensor gather_candidate_parameters(
    torch::Tensor parameter_table,
    torch::Tensor candidate_ids) {
  TORCH_CHECK(parameter_table.is_cuda(), "parameter_table must be CUDA");
  TORCH_CHECK(candidate_ids.is_cuda(), "candidate_ids must be CUDA");
  TORCH_CHECK(parameter_table.is_floating_point(), "parameter_table must be floating point");
  TORCH_CHECK(candidate_ids.scalar_type() == torch::kInt64, "candidate_ids must be int64");
  TORCH_CHECK(parameter_table.dim() == 2 && parameter_table.size(1) == 13,
              "parameter_table must have shape [N,13]");
  TORCH_CHECK(candidate_ids.dim() == 1, "candidate_ids must have shape [P]");
  TORCH_CHECK(parameter_table.get_device() == candidate_ids.get_device(),
              "parameter_table and candidate_ids must share a CUDA device");
  return gather_candidate_parameters_cuda(
      parameter_table.contiguous(), candidate_ids.contiguous());
}

torch::Tensor reduce_candidate_gradients(
    torch::Tensor candidate_ids,
    torch::Tensor candidate_gradients,
    int64_t surfel_count) {
  TORCH_CHECK(candidate_ids.is_cuda(), "candidate_ids must be CUDA");
  TORCH_CHECK(candidate_gradients.is_cuda(), "candidate_gradients must be CUDA");
  TORCH_CHECK(candidate_ids.scalar_type() == torch::kInt64, "candidate_ids must be int64");
  TORCH_CHECK(candidate_gradients.is_floating_point(), "candidate_gradients must be floating point");
  TORCH_CHECK(candidate_ids.dim() == 1, "candidate_ids must have shape [P]");
  TORCH_CHECK(candidate_gradients.dim() == 2 && candidate_gradients.size(1) == 13,
              "candidate_gradients must have shape [P,13]");
  TORCH_CHECK(candidate_gradients.size(0) == candidate_ids.size(0),
              "candidate gradient count must match candidate IDs");
  TORCH_CHECK(candidate_ids.get_device() == candidate_gradients.get_device(),
              "candidate_ids and candidate_gradients must share a CUDA device");
  TORCH_CHECK(surfel_count > 0, "surfel_count must be positive");
  return reduce_candidate_gradients_cuda(
      candidate_ids.contiguous(), candidate_gradients.contiguous(), surfel_count);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("count_candidates", &count_candidates, "Count ray/LBVH leaf candidates (CUDA)");
  module.def("fill_candidates", &fill_candidates, "Fill ray/LBVH leaf candidates (CUDA)");
  module.def(
      "gather_candidate_parameters", &gather_candidate_parameters,
      "Gather packed Reflection candidate parameters (CUDA)");
  module.def(
      "reduce_candidate_gradients", &reduce_candidate_gradients,
      "Reduce packed candidate gradients by Reflection surfel ID (CUDA)");
}
