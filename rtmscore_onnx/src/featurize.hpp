#ifndef INTERACTION_FEATURIZE_HDR
#define INTERACTION_FEATURIZE_HDR

// Reproduce the RTMScore RDKit/MDAnalysis featurization in C++.
//
// Given a protein pocket (PDB) and a ligand pose (SDF/Mol block), build the 8 ONNX input tensors.

#include <memory>
#include <string>
#include <vector>

#include "main.hpp"  // Tensor

namespace RDKit {
class ROMol;
}  // namespace RDKit

// molecule loading

// Parse one ligand pose from an SDF/MDL Mol block.
std::unique_ptr<RDKit::ROMol> load_ligand_block(const std::string &mol_block);

// Split a multi-molecule .sdf file into per-pose Mol blocks.
std::vector<std::string> split_sdf(const std::string &sdf_path);

// Parse a protein PDB and assign stereo.
// Full-receptor pocket generation passes false so explicit H coordinates can participate in the geometric pocket selection.
std::unique_ptr<RDKit::ROMol> load_protein_pdb(const std::string &pdb_path,
                                              bool remove_hs = true);


// featurization

// Ligand -> {l_ndata_atom, l_edata_bond, l_edge_index, l_ndata_pos}.
std::vector<Tensor> featurize_ligand(const RDKit::ROMol &lig);

// Protein -> {p_ndata_feats, p_edata_feats, p_edge_index, p_ndata_pos}.
std::vector<Tensor> featurize_protein(const RDKit::ROMol &prot, double cutoff);

#endif  // INTERACTION_FEATURIZE_HDR
