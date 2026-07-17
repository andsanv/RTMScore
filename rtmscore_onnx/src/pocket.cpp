// Pocket selection from a full receptor and a reference ligand.
//
// It works directly on RDKit molecules so the generated pocket can flow into featurize_protein()
// without a intermediate PDB dependency.

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/MonomerInfo.h>
#include <GraphMol/RWMol.h>

#include "featurize.hpp"
#include "pocket.hpp"

namespace {

// Identity key of an atom within a residue.
struct ResidueKey {
  int residue_number = 0;
  unsigned segment_number = 0;
  std::string chain_id;
  std::string insertion_code;
};

bool operator==(const ResidueKey &a, const ResidueKey &b) {
  return a.residue_number == b.residue_number &&
         a.segment_number == b.segment_number &&
         a.chain_id == b.chain_id &&
         a.insertion_code == b.insertion_code;
}

ResidueKey residue_key(const RDKit::Atom &atom) {
  const auto *info = atom.getMonomerInfo();
  if (!info || info->getMonomerType() != RDKit::AtomMonomerInfo::PDBRESIDUE)
    throw std::runtime_error("protein atom missing PDB residue info");
  const auto *pdb = static_cast<const RDKit::AtomPDBResidueInfo *>(info);
  return {pdb->getResidueNumber(), pdb->getSegmentNumber(), pdb->getChainId(),
          pdb->getInsertionCode()};
}

double squared_distance(const RDGeom::Point3D &a, const RDGeom::Point3D &b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

}  // namespace

std::unique_ptr<RDKit::ROMol> load_reference_ligand_sdf(
    const std::string &sdf_path) {
  const std::vector<std::string> blocks = split_sdf(sdf_path);  // only the first pose is used as the reference
  if (blocks.empty())
    throw std::runtime_error("no reference ligand in " + sdf_path);

  // keeping file hydrogens (second boolean) here affects pocket geometry only, scoring ligands will remove Hs
  RDKit::RWMol *mol = RDKit::MolBlockToMol(blocks.front(), true, false, true);
  if (!mol)
    throw std::runtime_error("failed to parse reference ligand " + sdf_path);
  if (!mol->getNumConformers())
    throw std::runtime_error("reference ligand has no coordinates: " + sdf_path);
  return std::unique_ptr<RDKit::ROMol>(mol);
}

PocketResult generate_pocket(const RDKit::ROMol &protein,
                             const RDKit::ROMol &reference_ligand,
                             double cutoff) {
  if (!std::isfinite(cutoff) || cutoff <= 0.0)
    throw std::runtime_error("pocket cutoff must be a positive finite number");
  if (!protein.getNumConformers())
    throw std::runtime_error("protein has no coordinates");
  if (!reference_ligand.getNumConformers())
    throw std::runtime_error("reference ligand has no coordinates");

  const RDKit::Conformer &protein_conf = protein.getConformer();
  const RDKit::Conformer &ligand_conf = reference_ligand.getConformer();
  const double cutoff_squared = cutoff * cutoff;  // compare squared distances, skips a sqrt per pair

  // First establish contiguous residue ranges.
  std::vector<std::vector<unsigned>> residues;
  ResidueKey previous;
  bool have_previous = false; // used for first iteration
  for (const RDKit::Atom *atom : protein.atoms()) {
    const ResidueKey key = residue_key(*atom);
    if (!have_previous || !(key == previous)) {
      residues.emplace_back();  // new residue boundary, start a fresh atom group
      previous = key;
      have_previous = true;
    }
    residues.back().push_back(atom->getIdx());
  }

  // For each residue, if any atom is closer to any ref-ligand atom than cutoff, keep it
  std::vector<bool> keep_atom(protein.getNumAtoms(), false);
  std::size_t kept_residues = 0;
  for (const std::vector<unsigned> &residue : residues) {
    bool within_cutoff = false;
    for (unsigned protein_idx : residue) {
      const RDGeom::Point3D &protein_pos =
          protein_conf.getAtomPos(protein_idx);
      for (unsigned ligand_idx = 0;
           ligand_idx < reference_ligand.getNumAtoms(); ++ligand_idx) {
        if (squared_distance(protein_pos,
                             ligand_conf.getAtomPos(ligand_idx)) <=
            cutoff_squared) {
          within_cutoff = true; // any atom pair within range is enough to keep the whole residue
          break;
        }
      }
      if (within_cutoff) break;
    }
    if (!within_cutoff) continue;
    ++kept_residues;
    for (unsigned atom_idx : residue) keep_atom[atom_idx] = true; // mark every atom in this residue as kept
  }

  if (kept_residues == 0)
    throw std::runtime_error(
        "pocket selection found no protein residues within cutoff");

  // duplicate protein removing unused residues
  auto pocket = std::make_unique<RDKit::RWMol>(protein);  // copy to keep caller's protein untouched
  for (std::size_t i = keep_atom.size(); i-- > 0;)  // remove atoms from highest to lowest index sine RDKit updates atom indices at every iter
    if (!keep_atom[i]) pocket->removeAtom(static_cast<unsigned>(i));

  // remove hydrogens before featurization
  RDKit::MolOps::removeHs(*pocket);
  const std::size_t atom_count = pocket->getNumAtoms();
  if (atom_count == 0)
    throw std::runtime_error("generated pocket contains no heavy atoms");

  PocketResult result;
  result.molecule = std::move(pocket);
  result.residue_count = kept_residues;
  result.atom_count = atom_count;
  return result;
}
