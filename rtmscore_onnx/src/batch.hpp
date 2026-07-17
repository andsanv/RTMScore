#ifndef INTERACTION_BATCH_HDR
#define INTERACTION_BATCH_HDR

// Merge single-pose tensor sets into one batched set
// Feature tensors are concatenated along dim 0; edge_index is offset by each pose's cumulative node count before concatenating.

#include <vector>

#include "main.hpp"  // Tensor

// Input: per_pose_ligand_tensors[i] is pose i's 4-tensor vector {l_ndata_atom, l_edata_bond, l_edge_index, l_ndata_pos}
// Output: 5 tensors in ONNX input order, the 4 originals (now batch-merged) plus l_batch_num_nodes.
std::vector<Tensor> batch_ligand_tensors(
    const std::vector<std::vector<Tensor>> &per_pose_ligand_tensors);

std::vector<Tensor> batch_protein_tensors(
    const std::vector<std::vector<Tensor>> &per_pose_protein_tensors);

#endif  // INTERACTION_BATCH_HDR
