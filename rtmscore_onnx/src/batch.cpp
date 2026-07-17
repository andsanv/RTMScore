#include "batch.hpp"

#include <stdexcept>
#include <string>

namespace {

// Concatenate a feature tensor (f32) along dim 0 across poses.
// e.g. [N1, 41] + [N2, 41] = [N1 + N2, 41]
Tensor concat_f32(const std::vector<const Tensor *> &parts, const std::string &name) {
  Tensor out;
  out.name = name;
  out.dtype = "f32";

  int64_t total = 0;
  std::vector<int64_t> trailing;
  for (const Tensor *t : parts) {
    if (t->shape.empty())
      throw std::runtime_error("batch: " + name + " has no shape");
    total += t->shape[0];  // running count of rows across all poses
    std::vector<int64_t> tail(t->shape.begin() + 1, t->shape.end());  // shape past dim 0, must match across poses
    if (trailing.empty())
      trailing = tail;
    else if (trailing != tail)
      throw std::runtime_error("batch: " + name + " trailing shape mismatch across poses");
  }

  out.shape = {total};
  out.shape.insert(out.shape.end(), trailing.begin(), trailing.end());
  out.f32.reserve(static_cast<size_t>(out.numel()));
  for (const Tensor *t : parts)
    out.f32.insert(out.f32.end(), t->f32.begin(), t->f32.end());  // append this pose's rows in order
  return out;
}

// Concatenate edge_index [2,E] tensors, offsetting each pose's node indices by its cumulative node count. Then flats the tensor.
// out: [[src0, src1],
//       [dst0, dst1]]
Tensor concat_edge_index_offset(const std::vector<const Tensor *> &parts,
                                const std::vector<int64_t> &node_offsets,
                                const std::string &name) {
  Tensor out;
  out.name = name;
  out.dtype = "i64";

  std::vector<int64_t> merged_src, merged_dst;
  for (size_t i = 0; i < parts.size(); ++i) {
    const Tensor *t = parts[i];
    if (t->shape.size() != 2 || t->shape[0] != 2)
      throw std::runtime_error("batch: " + name + " expected shape [2,E]");
    const int64_t e = t->shape[1];
    const int64_t offset = node_offsets[i];  // shift this pose's node ids into the merged index space
    merged_src.reserve(merged_src.size() + static_cast<size_t>(e));
    merged_dst.reserve(merged_dst.size() + static_cast<size_t>(e));
    for (int64_t k = 0; k < e; ++k) merged_src.push_back(t->i64[static_cast<size_t>(k)] + offset);  // first half of the buffer is src
    for (int64_t k = 0; k < e; ++k)
      merged_dst.push_back(t->i64[static_cast<size_t>(e + k)] + offset);  // second half is dst
  }

  const int64_t total_e = static_cast<int64_t>(merged_src.size());
  out.shape = {2, total_e};
  out.i64.reserve(static_cast<size_t>(2 * total_e));
  out.i64.insert(out.i64.end(), merged_src.begin(), merged_src.end());
  out.i64.insert(out.i64.end(), merged_dst.begin(), merged_dst.end());
  return out;
}

// Create tensor with node count for every graph.
// Used to recover graph boundaries after tensor concatenation.
Tensor make_batch_num_nodes(const std::vector<int64_t> &node_counts, const std::string &name) {
  Tensor out;
  out.name = name;
  out.dtype = "i64";
  out.shape = {static_cast<int64_t>(node_counts.size())};
  out.i64 = node_counts;
  return out;
}

// per_pose[i] is pose i's 4-tensor vector, ONNX order {ndata, edata, edge_index, pos}.
// per-pose node count comes from ndata's shape[0]. 
std::vector<Tensor> batch_side(const std::vector<std::vector<Tensor>> &per_pose,
                               const std::string &ndata_name, const std::string &edata_name,
                               const std::string &edge_index_name, const std::string &pos_name,
                               const std::string &batch_num_nodes_name) {
  if (per_pose.empty()) throw std::runtime_error("batch: no poses to batch");

  std::vector<int64_t> node_counts, node_offsets;
  int64_t running = 0;
  for (const auto &pose : per_pose) {
    if (pose.size() != 4) throw std::runtime_error("batch: expected 4 tensors per pose");
    const int64_t n = pose[0].shape.empty() ? 0 : pose[0].shape[0];
    node_counts.push_back(n);
    node_offsets.push_back(running);  // where this pose's nodes start in the merged tensor
    running += n;
  }

  std::vector<const Tensor *> ndata, edata, edge_index, pos;
  for (const auto &pose : per_pose) {
    ndata.push_back(&pose[0]);  // just collect pointers, avoid copying the tensors themselves
    edata.push_back(&pose[1]);
    edge_index.push_back(&pose[2]);
    pos.push_back(&pose[3]);
  }

  std::vector<Tensor> out;
  out.push_back(concat_f32(ndata, ndata_name));
  out.push_back(concat_f32(edata, edata_name));
  out.push_back(concat_edge_index_offset(edge_index, node_offsets, edge_index_name));
  out.push_back(concat_f32(pos, pos_name));
  out.push_back(make_batch_num_nodes(node_counts, batch_num_nodes_name));
  return out;
}

}  // namespace

std::vector<Tensor> batch_ligand_tensors(
    const std::vector<std::vector<Tensor>> &per_pose_ligand_tensors) {
  return batch_side(per_pose_ligand_tensors, "l_ndata_atom", "l_edata_bond", "l_edge_index",
                    "l_ndata_pos", "l_batch_num_nodes");
}

std::vector<Tensor> batch_protein_tensors(
    const std::vector<std::vector<Tensor>> &per_pose_protein_tensors) {
  return batch_side(per_pose_protein_tensors, "p_ndata_feats", "p_edata_feats", "p_edge_index",
                    "p_ndata_pos", "p_batch_num_nodes");
}
