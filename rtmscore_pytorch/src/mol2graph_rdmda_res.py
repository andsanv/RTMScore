import re
from itertools import permutations

import MDAnalysis as mda
import numpy as np
import torch as th
from MDAnalysis.analysis import distances
from rdkit import Chem
from scipy.spatial import distance_matrix

# featurizes RDKit/MDAnalysis molecules into the tensor dicts RTMScore expects
# (see mol_to_graph for ligands, prot_to_graph for protein pockets)

# element symbols normalized to "M" (metal) in the protein residue-name one-hot
METAL = [
    "LI", "NA", "K", "RB",
    "CS", "MG", "TL", "CU",
    "AG", "BE", "NI", "PT",
    "ZN", "CO", "PD", "AG",
    "CR", "FE", "V", "MN",
    "HG", "GA", "CD", "YB",
    "CA", "SN", "PB", "EU",
    "SR", "SM", "BA", "RA",
    "AL", "IN", "TL", "Y",
    "LA", "CE", "PR", "ND",
    "GD", "TB", "DY", "ER",
    "TM", "LU", "HF", "ZR",
    "CE", "U", "PU", "TH",
]
# per-residue atom-position padding target (real residues have far fewer atoms)
RES_MAX_NATOMS = 24


def prot_to_graph(prot, cutoff):
    """Build the protein pocket's residue graph: one node per residue, edges between residues within cutoff of each other."""

    universe = mda.Universe(prot)  # load the structure so its atoms and residues can be inspected
    graph = {}
    # add nodes
    num_residues = len(universe.residues)
    graph["num_nodes"] = num_residues

    res_feats = np.array([calc_res_features(res) for res in universe.residues])  # one feature row per residue
    graph["ndata_feats"] = th.tensor(res_feats).float()
    edge_ids, dist_matrix = obtain_edge(universe, cutoff)  # figure out which residues are close enough to connect
    src_list, dst_list = zip(*edge_ids)  # split each edge into its two endpoints
    graph["edge_index"] = th.tensor([src_list, dst_list], dtype=th.long)

    ca_pos = th.tensor(np.array([obtain_ca_pos(res) for res in universe.residues]))  # one reference point per residue
    center_pos = th.tensor(universe.atoms.center_of_mass(compound="residues"))  # another reference point, based on the average position of each residue's atoms
    ca_dist_matrix = distance_matrix(ca_pos, ca_pos)
    ca_edge_dist = th.tensor([ca_dist_matrix[i, j] for i, j in edge_ids]) * 0.1
    center_dist_matrix = distance_matrix(center_pos, center_pos)
    center_edge_dist = th.tensor([center_dist_matrix[i, j] for i, j in edge_ids]) * 0.1
    edge_connect = th.tensor(
        np.array([check_connect(universe, x, y) for x, y in zip(src_list, dst_list)])
    )  # whether each edge is a direct chain connection
    graph["edata_feats"] = th.cat(
        [
            edge_connect.view(-1, 1),
            ca_edge_dist.view(-1, 1),
            center_edge_dist.view(-1, 1),
            th.tensor(dist_matrix),
        ],
        dim=1,
    ).float()  # combine all edge features into one table

    graph["ndata_pos"] = th.tensor(
        np.array(
            [
                np.concatenate(
                    [
                        res.atoms.positions,
                        np.full((RES_MAX_NATOMS - len(res.atoms), 3), np.nan),
                    ],
                    axis=0,
                )
                for res in universe.residues
            ]
        )
    )  # store every residue's atom positions, padded out to the same length
    return graph


def obtain_ca_pos(res):
    """Residue's alpha-carbon position, falling back to the atom centroid if there's no CA (e.g. a bare metal residue, or a missing backbone atom)."""

    if obtain_resname(res) == "M":
        return res.atoms.positions[0]
    else:
        try:
            pos = res.atoms.select_atoms("name CA").positions[0]
            return pos
        except Exception:  ##some residues loss the CA atoms
            return res.atoms.positions.mean(axis=0)


def one_of_k_encoding(x, allowable_set):
    """One-hot encode x against allowable_set; raises if x isn't in the set."""
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_self_dist(res):
    """Residue-local geometry: max/min pairwise atom distance, plus the CA-O, O-N, and N-C backbone distances (0 for all if anything's missing)."""
    try:
        atoms = res.atoms
        dists = distances.self_distance_array(atoms.positions)
        ca_atoms = atoms.select_atoms("name CA")
        c_atoms = atoms.select_atoms("name C")
        n_atoms = atoms.select_atoms("name N")
        o_atoms = atoms.select_atoms("name O")
        return [
            dists.max() * 0.1,
            dists.min() * 0.1,
            distances.dist(ca_atoms, o_atoms)[-1][0] * 0.1,
            distances.dist(o_atoms, n_atoms)[-1][0] * 0.1,
            distances.dist(n_atoms, c_atoms)[-1][0] * 0.1,
        ]
    except Exception:
        return [0, 0, 0, 0, 0]


def _dihedral_or_zero(selection):
    """A dihedral selection's angle in degrees, or 0 if it's undefined (e.g. the residue is missing the atom(s) needed to define it)."""
    return selection.dihedral.value() if selection is not None else 0


def obtain_dihedral_angles(res):
    """Backbone dihedral angles (phi, psi, omega, chi1) in degrees, 0 for any that are undefined; [0, 0, 0, 0] if the lookup itself fails."""
    try:
        phi = _dihedral_or_zero(res.phi_selection())
        psi = _dihedral_or_zero(res.psi_selection())
        omega = _dihedral_or_zero(res.omega_selection())
        chi1 = _dihedral_or_zero(res.chi1_selection())
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except Exception:
        return [0, 0, 0, 0]


def calc_res_features(res):
    """Per-residue feature vector: 32-dim resname one-hot + 5-dim self-distance geometry + 4-dim backbone dihedrals."""
    return np.array(
        one_of_k_encoding_unk(
            obtain_resname(res),
            [
                "GLY", "ALA", "VAL", "LEU",
                "ILE", "PRO", "PHE", "TYR",
                "TRP", "SER", "THR", "CYS",
                "MET", "ASN", "GLN", "ASP",
                "GLU", "LYS", "ARG", "HIS",
                "MSE", "CSO", "PTR", "TPO",
                "KCX", "CSD", "SEP", "MLY",
                "PCA", "LLP", "M", "X",
            ],
        )  # 32  residue type
        + obtain_self_dist(res)  # 5
        + obtain_dihedral_angles(res)  # 4
    )


def obtain_resname(res):
    """Residue name, normalized to a single-element metal symbol or "M" for anything in METAL (so e.g. different metal ions share one-hot slots)."""

    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    if resname in METAL:
        return "M"
    else:
        return resname


def obtain_edge(universe, cutoff=10.0):
    """Residue-residue edges (both directions) for every pair within cutoff of each other, plus their min/max inter-residue atom distances."""

    edge_ids = []
    dist_min = []
    dist_max = []
    for res1, res2 in permutations(universe.residues, 2):  # check every ordered pair of residues
        dist = calc_dist(res1, res2)
        if dist.min() <= cutoff:  # keep the pair only if some part of them is close enough
            edge_ids.append([res1.ix, res2.ix])
            dist_min.append(dist.min() * 0.1)
            dist_max.append(dist.max() * 0.1)

    return edge_ids, np.array([dist_min, dist_max]).T


def check_connect(universe, i, j):
    """Whether residues i and j are peptide-bonded neighbors: true iff they're adjacent in sequence and exactly one bond spans the two (the backbone peptide bond), detected by comparing bond counts before/after merging them."""

    if abs(i - j) != 1:
        return 0
    else:
        if i > j:
            i = j
        n_bonds_i = len(universe.residues[i].get_connections("bonds"))
        n_bonds_j = len(universe.residues[i + 1].get_connections("bonds"))
        n_bonds_ij = len(universe.residues[i : i + 2].get_connections("bonds"))
        if n_bonds_i + n_bonds_j == n_bonds_ij + 1:
            return 1
        else:
            return 0


def calc_dist(res1, res2):
    return distances.distance_array(res1.atoms.positions, res2.atoms.positions)


def calc_atom_features(atom, explicit_H=False):
    """
    atom: rdkit.Chem.rdchem.Atom
    explicit_H: whether to use explicit H
    use_chirality: whether to use chirality
    """

    results = (
        one_of_k_encoding_unk(
            atom.GetSymbol(),  # which element this atom is
            [
                "C", "N", "O", "S",
                "F", "P", "Cl", "Br",
                "I", "B", "Si", "Fe",
                "Zn", "Cu", "Mn", "Mo",
                "other",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])  # how many neighbors it has
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]  # its charge and unpaired electrons
        + one_of_k_encoding_unk(
            atom.GetHybridization(),  # its bonding geometry
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                "other",
            ],
        )
        + [atom.GetIsAromatic()]  # whether it's part of a ring with shared, delocalized bonding
    )

    # in case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.array(results)


def calc_bond_features(bond, use_chirality=True):
    """
    bond: rdkit.Chem.rdchem.Bond
    use_chirality: whether to use chirality
    """
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),  # whether it's part of a chain of alternating bonds
        bond.IsInRing(),
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )  # add the bond's spatial arrangement, if requested
    return np.array(bond_feats).astype(int)


def load_mol(molpath, explicit_H=False, use_chirality=True):
    """Load a .pdb/.mol2/.sdf file into an RDKit Mol, dispatching on extension."""

    if re.search(r".pdb$", molpath):
        mol = Chem.MolFromPDBFile(molpath, removeHs=not explicit_H)
    elif re.search(r".mol2$", molpath):
        mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H)
    elif re.search(r".sdf$", molpath):
        mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H)
    else:
        raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")

    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)  # work out each atom's spatial arrangement from its 3d coordinates
    return mol


def mol_to_graph(mol, explicit_H=False, use_chirality=True):
    """
    Build a ligand's atom graph: one node per atom, two directed edges per bond.

    mol: rdkit.Chem.rdchem.Mol
    explicit_H: whether to use explicit H
    use_chirality: whether to use chirality
    """
    graph = {}
    # add nodes
    num_atoms = mol.GetNumAtoms()
    graph["num_nodes"] = num_atoms

    atom_feats = np.array(
        [calc_atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()]
    )  # one feature row per atom
    if use_chirality:
        chiral_centers = Chem.FindMolChiralCenters(
            mol, force=True, includeUnassigned=True, useLegacyImplementation=False
        )  # find which atoms have a defined spatial arrangement, and which way
        chiral_features = np.zeros([num_atoms, 3])  # one-hot slot per atom: one direction, the other direction, or undefined
        for atom_idx, chirality in chiral_centers:
            if chirality == "R":
                chiral_features[atom_idx, 0] = 1
            elif chirality == "S":
                chiral_features[atom_idx, 1] = 1
            else:
                chiral_features[atom_idx, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_features], axis=1)  # attach the arrangement info to each atom's other features

    graph["ndata_atom"] = th.tensor(atom_feats).float()

    # obtain the positions of the atoms
    atom_coords = mol.GetConformer().GetPositions()
    graph["ndata_pos"] = th.tensor(atom_coords)

    # add edges
    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_feats = calc_bond_features(bond, use_chirality=use_chirality)
        src_list.extend([begin_idx, end_idx])  # record the connection in both directions
        dst_list.extend([end_idx, begin_idx])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)  # duplicate the features to match the two directions above

    if len(src_list) > 0:
        graph["edge_index"] = th.tensor([src_list, dst_list], dtype=th.long)
    else:
        graph["edge_index"] = th.empty((2, 0), dtype=th.long)  # keep an empty placeholder if there are no bonds at all

    if len(bond_feats_all) > 0:
        graph["edata_bond"] = th.tensor(np.array(bond_feats_all)).float()
    else:
        graph["edata_bond"] = th.empty((0, 10)).float()  # keep an empty placeholder if there are no bonds at all
    return graph
