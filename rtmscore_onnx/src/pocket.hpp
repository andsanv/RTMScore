#ifndef INTERACTION_POCKET_HDR
#define INTERACTION_POCKET_HDR

// In-memory pocket generation for the full-receptor scoring path.
// Protein residue is retained in full when any of its atoms is within the cutoff of any reference-ligand atom.

#include <cstddef>
#include <memory>
#include <string>

namespace RDKit {
class ROMol;
}  // namespace RDKit

struct PocketResult {
  std::unique_ptr<RDKit::ROMol> molecule;
  std::size_t residue_count = 0;
  std::size_t atom_count = 0;  // after hydrogen removal
};

// Load the first molecule in a reference-ligand SDF, hydrogens are retained
std::unique_ptr<RDKit::ROMol> load_reference_ligand_sdf(
    const std::string &sdf_path);

// select a pocket from a full protein. Returned molecule has hydrogens removed
PocketResult generate_pocket(const RDKit::ROMol &protein,
                             const RDKit::ROMol &reference_ligand,
                             double cutoff);

#endif  // INTERACTION_POCKET_HDR
