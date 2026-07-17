// Ligand featurization

#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/CIPLabeler/CIPLabeler.h>
#include <RDGeneral/types.h>

#include "featurize.hpp"

namespace {

// one_of_k_encoding_unk: unknown folds into the last slot.
template <typename T, size_t N>
void one_hot_unk(std::vector<float> &out, const T &x,
                 const std::array<T, N> &set) {
  size_t hit = N - 1;  // default: last
  for (size_t i = 0; i < N; ++i)
    if (x == set[i]) { hit = i; break; }  // stop at the first matching category
  for (size_t i = 0; i < N; ++i) out.push_back(i == hit ? 1.0f : 0.0f);  // append the one-hot row
}

// one_of_k_encoding: strict, raises on out-of-range.
void one_hot_strict_degree(std::vector<float> &out, unsigned degree) {
  if (degree > 6)
    throw std::runtime_error("atom degree " + std::to_string(degree) +
                             " outside allowable set [0,6]");
  for (unsigned i = 0; i <= 6; ++i) out.push_back(i == degree ? 1.0f : 0.0f);
}

}  // namespace

std::unique_ptr<RDKit::ROMol> load_ligand_block(const std::string &mol_block) {
  RDKit::RWMol *m = RDKit::MolBlockToMol(mol_block, true, true, true);
  if (!m) throw std::runtime_error("failed to parse ligand Mol block");
  return std::unique_ptr<RDKit::ROMol>(m);
}

std::vector<std::string> split_sdf(const std::string &sdf_path) {
  std::ifstream f(sdf_path);
  if (!f) throw std::runtime_error("cannot open " + sdf_path);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string contents = ss.str();

  // VSDataset._sdf_split: split on "$$$$\n", drop the trailing chunk, re-append the delimiter to each block.
  std::vector<std::string> blocks;
  const std::string delim = "$$$$\n";
  size_t pos = 0;
  while (true) {
    const size_t next = contents.find(delim, pos);
    if (next == std::string::npos) break;  // trailing chunk after last "$$$$" dropped
    blocks.push_back(contents.substr(pos, next - pos) + delim);  // keep the delimiter
    pos = next + delim.size();
  }
  return blocks;
}

std::vector<Tensor> featurize_ligand(const RDKit::ROMol &lig) {
  const unsigned n_atoms = lig.getNumAtoms();

  // l_ndata_atom: [N, 41]
  static const std::array<std::string, 17> kElements = {
      "C",  "N",  "O",  "S",  "F",  "P",  "Cl", "Br", "I",
      "B",  "Si", "Fe", "Zn", "Cu", "Mn", "Mo", "other"};
  using Hyb = RDKit::Atom::HybridizationType;
  static const std::array<Hyb, 6> kHyb = {Hyb::SP,   Hyb::SP2,   Hyb::SP3,
                                          Hyb::SP3D, Hyb::SP3D2, Hyb::UNSPECIFIED};
  static const std::array<int, 5> kNumH = {0, 1, 2, 3, 4};

  Tensor t_atom;
  t_atom.name = "l_ndata_atom";
  t_atom.dtype = "f32";
  t_atom.shape = {static_cast<int64_t>(n_atoms), 41};
  t_atom.f32.reserve(static_cast<size_t>(n_atoms) * 41);

  for (const RDKit::Atom *a : lig.atoms()) {
    // one hot encodings
    std::string sym = a->getSymbol(); 
    one_hot_unk<std::string, 17>(t_atom.f32, sym, kElements);
    one_hot_strict_degree(t_atom.f32, a->getDegree());
    
    // other features
    t_atom.f32.push_back(static_cast<float>(a->getFormalCharge())); 
    t_atom.f32.push_back(static_cast<float>(a->getNumRadicalElectrons()));
    one_hot_unk<Hyb, 6>(t_atom.f32, a->getHybridization(), kHyb);
    t_atom.f32.push_back(a->getIsAromatic() ? 1.0f : 0.0f);
    one_hot_unk<int, 5>(t_atom.f32, static_cast<int>(a->getTotalNumHs()), kNumH);
    
    // chirality "one-hot"
    t_atom.f32.push_back(0.0f);  // R, set to 1 later if this atom turns out to be R
    t_atom.f32.push_back(0.0f);  // S
    t_atom.f32.push_back(0.0f);  // unassigned
  }

  // Chirality: Chem.FindMolChiralCenters(force=True, includeUnassigned=True, useLegacyImplementation=False).
  {
    RDKit::RWMol work(lig);  // use a scratch copy because stereo perception mutates the molecule
    RDKit::MolOps::assignStereochemistry(work, true, true, true);
    RDKit::CIPLabeler::assignCIPLabels(work);
    
    for (unsigned i = 0; i < n_atoms; ++i) {
      const RDKit::Atom *a = work.getAtomWithIdx(i);
      const size_t base = static_cast<size_t>(i) * 41 + 38;  // offset of this atom's 3 chirality columns
      std::string cip;
      if (a->getPropIfPresent(RDKit::common_properties::_CIPCode, cip)) {
        if (cip == "R") t_atom.f32[base + 0] = 1.0f;
        else if (cip == "S") t_atom.f32[base + 1] = 1.0f;
        else t_atom.f32[base + 2] = 1.0f;
      } else if (a->hasProp("_ChiralityPossible")) {
        t_atom.f32[base + 2] = 1.0f;  // a stereocenter, but its configuration wasn't determined
      }
    }
  }

  // l_ndata_pos: [N, 3]
  Tensor t_pos;
  t_pos.name = "l_ndata_pos";
  t_pos.dtype = "f32";
  t_pos.shape = {static_cast<int64_t>(n_atoms), 3};
  t_pos.f32.reserve(static_cast<size_t>(n_atoms) * 3);
  
  const RDKit::Conformer &conf = lig.getConformer();
  
  for (unsigned i = 0; i < n_atoms; ++i) {
    const RDGeom::Point3D &p = conf.getAtomPos(i);
    t_pos.f32.push_back(static_cast<float>(p.x));
    t_pos.f32.push_back(static_cast<float>(p.y));
    t_pos.f32.push_back(static_cast<float>(p.z));
  }

  // l_edge_index [2, E] and l_edata_bond [E, 10]
  using BT = RDKit::Bond::BondType;
  static const std::array<BT, 4> kBondTypes = {BT::SINGLE, BT::DOUBLE,
                                               BT::TRIPLE, BT::AROMATIC};
  using BS = RDKit::Bond::BondStereo;
  static const std::array<BS, 4> kStereo = {BS::STEREONONE, BS::STEREOANY,
                                            BS::STEREOZ, BS::STEREOE};

  std::vector<int64_t> src, dst;
  Tensor t_bond;
  t_bond.name = "l_edata_bond";
  t_bond.dtype = "f32";
  const RDKit::RingInfo *ring = lig.getRingInfo();
  const unsigned n_bonds = lig.getNumBonds();
  
  for (unsigned bi = 0; bi < n_bonds; ++bi) {
    const RDKit::Bond *b = lig.getBondWithIdx(bi);
    const int64_t u = b->getBeginAtomIdx();
    const int64_t v = b->getEndAtomIdx();

    std::vector<float> feat;
    feat.reserve(10);
    // bond type one-hot, unknown -> last (AROMATIC); Python uses explicit == so
    // any non-listed type is all-zero, but only the 4 listed occur post-sanitize.
    for (size_t i = 0; i < 4; ++i)
      feat.push_back(b->getBondType() == kBondTypes[i] ? 1.0f : 0.0f);
    feat.push_back(b->getIsConjugated() ? 1.0f : 0.0f);
    feat.push_back(ring->numBondRings(bi) > 0 ? 1.0f : 0.0f);  // is this bond part of any ring
    one_hot_unk<BS, 4>(feat, b->getStereo(), kStereo);

    // two directed edges, duplicated feature row
    src.push_back(u); dst.push_back(v);  // forward direction
    src.push_back(v); dst.push_back(u);  // reverse direction, same features
    t_bond.f32.insert(t_bond.f32.end(), feat.begin(), feat.end());
    t_bond.f32.insert(t_bond.f32.end(), feat.begin(), feat.end());
  }
  const int64_t n_edges = static_cast<int64_t>(src.size());
  t_bond.shape = {n_edges, 10};

  Tensor t_edge;
  t_edge.name = "l_edge_index";
  t_edge.dtype = "i64";
  t_edge.shape = {2, n_edges};
  t_edge.i64.reserve(src.size() + dst.size());
  t_edge.i64.insert(t_edge.i64.end(), src.begin(), src.end());
  t_edge.i64.insert(t_edge.i64.end(), dst.begin(), dst.end());

  return {std::move(t_atom), std::move(t_bond), std::move(t_edge),
          std::move(t_pos)};  // ONNX input order: atom, bond, edge_index, pos
}
