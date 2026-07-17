// Protein featurization. 

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/MonomerInfo.h>

#include "featurize.hpp"
#include "geometry.hpp"

namespace {

constexpr int RES_MAX_NATOMS = 24;

// metal list
const std::set<std::string> &metal_set() {
  static const std::set<std::string> m = {
      "LI", "NA", "K",  "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT",
      "ZN", "CO", "PD", "CR", "FE", "V",  "MN", "HG", "GA", "CD", "YB", "CA",
      "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "Y",  "LA", "CE",
      "PR", "ND", "GD", "TB", "DY", "ER", "TM", "LU", "HF", "ZR", "U",  "PU",
      "TH"};
  return m;
}

std::string strip(const std::string &s) {
  const size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";  // whitespace-only string
  const size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// obtain_resname: metal/CA/FE/CU normalization, then the METAL -> "M" mapping.
std::string obtain_resname(const std::string &raw) {
  std::string resname;
  if (raw.size() >= 2 && raw.compare(0, 2, "CA") == 0) resname = "CA";  // avoid confusing calcium with the C-alpha
  else if (raw.size() >= 2 && raw.compare(0, 2, "FE") == 0) resname = "FE";
  else if (raw.size() >= 2 && raw.compare(0, 2, "CU") == 0) resname = "CU";
  else resname = strip(raw);
  if (metal_set().count(resname)) return "M";  // collapse every other metal into one generic category
  return resname;
}

struct Residue {
  int resid = 0;
  unsigned segnum = 0;
  std::string icode;  // insertion code
  std::string resname_raw;
  std::vector<Vec3> pos;
  std::vector<std::string> names; // atom names
  std::vector<double> masses;     // atomic masses
  std::vector<int> global_idx;    // original RDKit atom indices
};

// Return the single atom position in `res` whose (trimmed) name == name. 
// found=true only when exactly one such atom exists.
const Vec3 *unique_named(const Residue &res, const std::string &name,
                         bool &found) {
  const Vec3 *hit = nullptr;
  int count = 0;
  for (size_t i = 0; i < res.names.size(); ++i)
    if (res.names[i] == name) { hit = &res.pos[i]; ++count; }  // keep scanning to detect duplicates
  found = (count == 1);  // only a unique match counts
  return found ? hit : nullptr;
}

// First atom position whose name == name.
const Vec3 *first_named(const Residue &res, const std::string &name) {
  for (size_t i = 0; i < res.names.size(); ++i)
    if (res.names[i] == name) return &res.pos[i];
  return nullptr;
}

Vec3 centroid(const Residue &res) {
  double x = 0, y = 0, z = 0;
  for (const Vec3 &p : res.pos) {
    x += static_cast<double>(p[0]);  // accumulate and average atoms with double precision
    y += static_cast<double>(p[1]);
    z += static_cast<double>(p[2]);
  }
  const double n = static_cast<double>(res.pos.size());
  return {static_cast<float>(x / n), static_cast<float>(y / n),
          static_cast<float>(z / n)};
}

// obtain_ca_pos
Vec3 obtain_ca_pos(const Residue &res) {
  if (obtain_resname(res.resname_raw) == "M") return res.pos[0];  // a metal "residue" is just the one atom
  if (const Vec3 *ca = first_named(res, "CA")) return *ca;
  return centroid(res);  // some residues lose the CA atom
}

Vec3 center_of_mass(const Residue &res) {
  double x = 0, y = 0, z = 0, m = 0;
  for (size_t i = 0; i < res.pos.size(); ++i) {
    const double w = res.masses[i];  // weight each atom by its mass
    x += w * static_cast<double>(res.pos[i][0]);
    y += w * static_cast<double>(res.pos[i][1]);
    z += w * static_cast<double>(res.pos[i][2]);
    m += w;
  }
  return {static_cast<float>(x / m), static_cast<float>(y / m),
          static_cast<float>(z / m)};  // normalize by total mass
}

// obtain_self_dist -> 5 values.
std::array<float, 5> obtain_self_dist(const Residue &res) {
  const std::array<float, 5> zeros = {0, 0, 0, 0, 0};
  if (res.pos.size() < 2) return zeros;  // no pair of atoms to measure
  bool fca, fc, fn, fo;
  const Vec3 *ca = unique_named(res, "CA", fca);
  const Vec3 *c = unique_named(res, "C", fc);
  const Vec3 *n = unique_named(res, "N", fn);
  const Vec3 *o = unique_named(res, "O", fo);
  if (!(fca && fc && fn && fo)) return zeros;  // distances.dist would raise

  double dmax = -1.0, dmin = std::numeric_limits<double>::max();
  for (size_t i = 0; i < res.pos.size(); ++i)
    for (size_t j = i + 1; j < res.pos.size(); ++j) {  // every atom pair within this residue
      const double d = dist(res.pos[i], res.pos[j]);
      dmax = std::max(dmax, d);
      dmin = std::min(dmin, d);
    }
  return {static_cast<float>(dmax * 0.1), static_cast<float>(dmin * 0.1), // scaling by 0.1 to match original featurizer
          static_cast<float>(dist(*ca, *o) * 0.1),
          static_cast<float>(dist(*o, *n) * 0.1),
          static_cast<float>(dist(*n, *c) * 0.1)};
}

}  // namespace

std::unique_ptr<RDKit::ROMol> load_protein_pdb(const std::string &pdb_path,
                                              bool remove_hs) {
  // Pocket generation temporarily keeps H coordinates and removes them after residue selection.
  RDKit::RWMol *m = RDKit::PDBFileToMol(pdb_path, true, remove_hs);
  if (!m) throw std::runtime_error("failed to parse protein PDB " + pdb_path);
  try {
    RDKit::MolOps::assignStereochemistryFrom3D(*m);
  } catch (...) {
    // stereo tags are unused by the protein features, ignore errors
  }
  return std::unique_ptr<RDKit::ROMol>(m);
}

std::vector<Tensor> featurize_protein(const RDKit::ROMol &prot, double cutoff) {
  // group atoms into residues
  std::vector<Residue> residues;
  const RDKit::Conformer &conf = prot.getConformer(); // molecule conformation (3D positions)
  bool have_prev = false;
  int p_resid = 0;
  unsigned p_seg = 0;
  std::string p_icode, p_rname;

  for (const RDKit::Atom *a : prot.atoms()) {
    const auto *mi = a->getMonomerInfo();
    if (!mi || mi->getMonomerType() != RDKit::AtomMonomerInfo::PDBRESIDUE)
      throw std::runtime_error("protein atom missing PDB residue info");
    const auto *pi = static_cast<const RDKit::AtomPDBResidueInfo *>(mi);
    const int resid = pi->getResidueNumber();
    const unsigned seg = pi->getSegmentNumber();
    const std::string icode = pi->getInsertionCode();
    const std::string rname = pi->getResidueName();

    const bool same = have_prev && resid == p_resid && seg == p_seg &&
                      icode == p_icode && rname == p_rname;
    if (!same) {
      residues.emplace_back();  // atom belongs to a new residue, start a new group
      Residue &r = residues.back();
      r.resid = resid;
      r.segnum = seg;
      r.icode = icode;
      r.resname_raw = rname;
      have_prev = true;
      p_resid = resid; p_seg = seg; p_icode = icode; p_rname = rname;
    }
    Residue &r = residues.back();
    const unsigned idx = a->getIdx();
    const RDGeom::Point3D &p = conf.getAtomPos(idx);
    r.pos.push_back({static_cast<float>(p.x), static_cast<float>(p.y),
                     static_cast<float>(p.z)});
    r.names.push_back(strip(pi->getName()));
    r.masses.push_back(a->getMass());
    r.global_idx.push_back(static_cast<int>(idx));  // remember which atom in the original molecule this is
  }

  const int n_res = static_cast<int>(residues.size());
  if (n_res == 0) throw std::runtime_error("protein has no residues");

  // mapping from atom to residue index, for check_connect
  std::vector<int> atom_res(prot.getNumAtoms(), -1);
  for (int r = 0; r < n_res; ++r)
    for (int gi : residues[r].global_idx) atom_res[gi] = r;  // every atom in residue r maps back to r

  // resid-based neighbour lookup.
  auto find_neighbor = [&](int i, int delta) -> int {
    const int want = residues[i].resid + delta;
    const unsigned seg = residues[i].segnum;
    const int obv = i + delta;  // "obvious candidate" first
    if (obv >= 0 && obv < n_res && residues[obv].resid == want &&
        residues[obv].segnum == seg)
      return obv;  // fast path, residues are almost always stored in sequence
    for (int j = 0; j < n_res; ++j)  // else first residue with matching resid
      if (residues[j].resid == want && residues[j].segnum == seg) return j;
    return -1;  // no such neighbouring residue
  };

  // precompute CA positions and mass centers
  std::vector<Vec3> ca_pos(n_res), com_pos(n_res);
  for (int i = 0; i < n_res; ++i) {
    ca_pos[i] = obtain_ca_pos(residues[i]);
    com_pos[i] = center_of_mass(residues[i]);
  }

  // resname one-hot order (32; unknown -> last "X")
  static const std::array<std::string, 32> kRes = {
      "GLY", "ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TYR", "TRP", "SER", "THR",
      "CYS", "MET", "ASN", "GLN", "ASP", "GLU", "LYS", "ARG", "HIS", "MSE", "CSO",
      "PTR", "TPO", "KCX", "CSD", "SEP", "MLY", "PCA", "LLP", "M",   "X"};

  // p_ndata_feats: [n_res, 41]
  Tensor t_feats;
  t_feats.name = "p_ndata_feats";
  t_feats.dtype = "f32";
  t_feats.shape = {n_res, 41};
  t_feats.f32.reserve(static_cast<size_t>(n_res) * 41);

  for (int i = 0; i < n_res; ++i) {
    const Residue &res = residues[i];
    // resname one-hot
    const std::string rn = obtain_resname(res.resname_raw);
    size_t hit = kRes.size() - 1;  // default: last slot, "X" for unknown resnames
    for (size_t k = 0; k < kRes.size(); ++k)
      if (rn == kRes[k]) { hit = k; break; }
    for (size_t k = 0; k < kRes.size(); ++k)
      t_feats.f32.push_back(k == hit ? 1.0f : 0.0f);

    // self distances (5)
    const std::array<float, 5> sd = obtain_self_dist(res);
    for (float v : sd) t_feats.f32.push_back(v);

    // dihedrals (4): phi, psi, omega, chi1, 0 if undefined
    float phi = 0, psi = 0, omega = 0, chi1 = 0;
    bool fN, fCA, fC;
    const Vec3 *N = unique_named(res, "N", fN);
    const Vec3 *CA = unique_named(res, "CA", fCA);
    const Vec3 *C = unique_named(res, "C", fC);

    const int prev = find_neighbor(i, -1);
    const int next = find_neighbor(i, +1);

    if (prev >= 0 && fN && fCA && fC) {  // phi: C'(prev)-N-CA-C
      bool fCp;
      const Vec3 *Cp = unique_named(residues[prev], "C", fCp);
      if (fCp) phi = static_cast<float>(dihedral_deg(*Cp, *N, *CA, *C));
    }
    if (next >= 0 && fN && fCA && fC) {  // psi: N-CA-C-N'(next)
      bool fNn;
      const Vec3 *Nn = unique_named(residues[next], "N", fNn);
      if (fNn) psi = static_cast<float>(dihedral_deg(*N, *CA, *C, *Nn));
    }
    if (next >= 0 && fCA && fC) {  // omega: CA-C-N'-CA'(next)
      bool fNn, fCAn;
      const Vec3 *Nn = unique_named(residues[next], "N", fNn);
      const Vec3 *CAn = unique_named(residues[next], "CA", fCAn);
      if (fNn && fCAn) omega = static_cast<float>(dihedral_deg(*CA, *C, *Nn, *CAn));
    }
    // chi1: N-CA-CB-*G, gamma is the single heavy atom in {CG,CG1,OG,OG1,SG}
    {
      bool fCB;
      const Vec3 *CB = unique_named(res, "CB", fCB);
      const Vec3 *G = nullptr;
      int gcount = 0;
      static const std::array<std::string, 5> kGamma = {"CG", "CG1", "OG", "OG1", "SG"};
      for (size_t k = 0; k < res.names.size(); ++k)
        for (const std::string &gn : kGamma)
          if (res.names[k] == gn) { G = &res.pos[k]; ++gcount; }  // count matches, must be exactly one
      if (fN && fCA && fCB && gcount == 1)
        chi1 = static_cast<float>(dihedral_deg(*N, *CA, *CB, *G));
    }
    t_feats.f32.push_back(phi * 0.01f);  // scale down to keep dihedral features in a small range
    t_feats.f32.push_back(psi * 0.01f);
    t_feats.f32.push_back(omega * 0.01f);
    t_feats.f32.push_back(chi1 * 0.01f);
  }

  // edges (permutations i!=j, min all-atom distance <= cutoff)
  std::vector<int64_t> src, dst;
  Tensor t_edata;
  t_edata.name = "p_edata_feats";
  t_edata.dtype = "f32";

  for (int i = 0; i < n_res; ++i) {
    for (int j = 0; j < n_res; ++j) {
      if (i == j) continue;  // no self edges
      const Residue &ri = residues[i];
      const Residue &rj = residues[j];
      double dmin = std::numeric_limits<double>::max();
      double dmax = -1.0;
      for (const Vec3 &pi : ri.pos)
        for (const Vec3 &pj : rj.pos) {  // closest and farthest atom pair between the two residues
          const double d = dist(pi, pj);
          dmin = std::min(dmin, d);
          dmax = std::max(dmax, d);
        }
      if (dmin > cutoff) continue;  // residues too far apart, skip the edge

      src.push_back(i);
      dst.push_back(j);

      // check_connect: 1 iff |i-j|==1 and exactly one bond spans the two residues
      float connect = 0.0f;
      if (std::abs(i - j) == 1) {
        const int k = std::min(i, j);
        int spanning = 0;
        for (const RDKit::Bond *b : prot.bonds()) {
          const int ra = atom_res[b->getBeginAtomIdx()];
          const int rb = atom_res[b->getEndAtomIdx()];
          if ((ra == k && rb == k + 1) || (ra == k + 1 && rb == k)) ++spanning;  // bond crosses the residue boundary
        }
        connect = (spanning == 1) ? 1.0f : 0.0f;
      }
      const double cadist = dist(ca_pos[i], ca_pos[j]) * 0.1;  // CA-CA distance between residues
      const double cedist = dist(com_pos[i], com_pos[j]) * 0.1;  // center-of-mass distance
      t_edata.f32.push_back(connect);
      t_edata.f32.push_back(static_cast<float>(cadist));
      t_edata.f32.push_back(static_cast<float>(cedist));
      t_edata.f32.push_back(static_cast<float>(dmin * 0.1));
      t_edata.f32.push_back(static_cast<float>(dmax * 0.1));
    }
  }
  const int64_t n_edges = static_cast<int64_t>(src.size());
  t_edata.shape = {n_edges, 5};

  Tensor t_edge;
  t_edge.name = "p_edge_index";
  t_edge.dtype = "i64";
  t_edge.shape = {2, n_edges};
  t_edge.i64.reserve(src.size() + dst.size());
  t_edge.i64.insert(t_edge.i64.end(), src.begin(), src.end());
  t_edge.i64.insert(t_edge.i64.end(), dst.begin(), dst.end());

  // p_ndata_pos: [n_res, 24, 3], NaN-padded
  Tensor t_pos;
  t_pos.name = "p_ndata_pos";
  t_pos.dtype = "f32";
  t_pos.shape = {n_res, RES_MAX_NATOMS, 3};
  t_pos.f32.reserve(static_cast<size_t>(n_res) * RES_MAX_NATOMS * 3);
  const float nan = std::nanf("");
  for (int i = 0; i < n_res; ++i) {
    const Residue &res = residues[i];
    if (static_cast<int>(res.pos.size()) > RES_MAX_NATOMS)
      throw std::runtime_error("residue exceeds RES_MAX_NATOMS (24) atoms");
    for (int k = 0; k < RES_MAX_NATOMS; ++k) {
      if (k < static_cast<int>(res.pos.size())) {
        t_pos.f32.push_back(res.pos[k][0]);
        t_pos.f32.push_back(res.pos[k][1]);
        t_pos.f32.push_back(res.pos[k][2]);
      } else {
        t_pos.f32.push_back(nan);  // pad out residues with fewer than 24 atoms
        t_pos.f32.push_back(nan);
        t_pos.f32.push_back(nan);
      }
    }
  }

  return {std::move(t_feats), std::move(t_edata), std::move(t_edge),
          std::move(t_pos)};  // ONNX input order: feats, edata, edge_index, pos
}
