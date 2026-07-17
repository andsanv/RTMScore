import shutil
import tempfile

import numpy as np
import torch as th
from joblib import Parallel, delayed
from rdkit import Chem
from torch.utils.data import Dataset

from .extract_pocket_prody import extract_pocket
from .mol2graph_rdmda_res import load_mol, mol_to_graph, prot_to_graph


class VSDataset(Dataset):
    """
    PyTorch Dataset that featurizes protein and ligand files into RTMScore's graph representation.

    The protein pocket is featurized once (either an already-known pocket, or one generated on the fly from a full protein + reference ligand). Each ligand pose gets its own graph, paired with the shared protein graph (__getitem__).
    """

    def __init__(
        self,
        ids=None,
        ligs=None,
        prot=None,
        gen_pocket=False,
        cutoff=None,
        reflig=None,
        explicit_H=False,
        use_chirality=True,
        parallel=True,
    ):
        self.protein_graph = None
        self.ligand_graphs = None
        self.pocket_dir = None
        self.prot = None
        self.ligs = None
        self.cutoff = cutoff
        self.explicit_H = explicit_H
        self.use_chirality = use_chirality
        self.parallel = parallel

        # protein is an already-parsed Mol, a pocket to extract, or a ready pocket-protein file
        if isinstance(prot, Chem.rdchem.Mol):
            assert not gen_pocket  # already a parsed molecule, so there is no pocket to generate
            self.prot = prot
            self.protein_graph = prot_to_graph(self.prot, cutoff)
        elif gen_pocket:
            if cutoff is None or reflig is None:
                raise ValueError(
                    "If you want to generate the pocket, the cutoff and the reflig should be given"
                )
            self.pocket_dir = tempfile.mkdtemp()  # scratch space for the intermediate pocket file, removed at the end
            extract_pocket(
                prot, reflig, cutoff, protname="temp", workdir=self.pocket_dir
            )
            pocket_path = "%s/temp_pocket_%s.pdb" % (self.pocket_dir, cutoff)  # path where extract_pocket wrote the result
            self._load_protein(pocket_path)
        else:
            self._load_protein(prot)

        # ligands are a list of Mols/graphs already in memory, or a .mol2/.sdf/graph file
        if isinstance(ligs, np.ndarray) or isinstance(ligs, list):
            if isinstance(ligs[0], Chem.rdchem.Mol):
                self.ligs = ligs
                self.ligand_graphs = self._mol_to_graph()
            elif isinstance(ligs[0], dict):
                self.ligand_graphs = ligs  # already featurized graphs, nothing more to do
            else:
                raise ValueError(
                    "Ligands should be a list of rdkit.Chem.rdchem.Mol objects"
                )
        else:
            if ligs.endswith(".mol2"):
                lig_blocks = self._mol2_split(ligs)
                self.ligs = [
                    Chem.MolFromMol2Block(lig_block) for lig_block in lig_blocks
                ]
                self.ligand_graphs = self._mol_to_graph()
            elif ligs.endswith(".sdf"):
                lig_blocks = self._sdf_split(ligs)
                self.ligs = [
                    Chem.MolFromMolBlock(lig_block) for lig_block in lig_blocks
                ]
                self.ligand_graphs = self._mol_to_graph()
            else:
                try:
                    self.ligand_graphs = th.load(ligs)  # fall back to a file of already-featurized graphs
                except Exception:
                    raise ValueError(
                        "Only the ligands with .sdf or .mol2 or a file to genrate graphs will be supported"
                    )

        if ids is None:
            if self.ligs is not None:
                raw_ids = [
                    "%s-%s" % (self.get_ligname(lig), i)
                    for i, lig in enumerate(self.ligs)
                ]  # build a label from each ligand's name and position, since none was given
            else:
                raw_ids = ["lig%s" % i for i in range(len(self.ligand_graphs))]  # no names available, so just number them
        else:
            raw_ids = ids

        # ligands that failed to featurize come back as None from "_mol_to_graph"
        self.ids, self.ligand_graphs = zip(
            *filter(lambda x: x[1] is not None, zip(raw_ids, self.ligand_graphs))
        )
        self.ids = list(self.ids)
        self.ligand_graphs = list(self.ligand_graphs)
        assert len(self.ids) == len(self.ligand_graphs)  # sanity check that the filtering above kept both lists in sync
        if self.pocket_dir is not None:
            shutil.rmtree(self.pocket_dir)  # clean up the scratch space now that the pocket has been featurized

    def _load_protein(self, prot_path_or_mol):
        """Load a protein/pocket file (or Mol) and featurize it, raising a clearer error on failure."""
        try:
            self.prot = load_mol(
                prot_path_or_mol,
                explicit_H=self.explicit_H,
                use_chirality=self.use_chirality,
            )
            self.protein_graph = prot_to_graph(self.prot, self.cutoff)
        except Exception:
            raise ValueError("The graph of pocket cannot be generated")

    def __getitem__(self, idx):
        """Get the ligand graph, protein graph, and id at the given index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (str, dict, dict)
            The pose id, the ligand graph, and the protein graph.
        """
        return self.ids[idx], self.ligand_graphs[idx], self.protein_graph

    def __len__(self):
        """Number of graphs in the dataset."""
        return len(self.ids)

    def _mol2_split(self, infile):
        """Split a multi-molecule .mol2 file into per-molecule blocks."""
        with open(infile, "r") as f:
            contents = f.read()
        return [
            "@<TRIPOS>MOLECULE\n" + block
            for block in contents.split("@<TRIPOS>MOLECULE\n")[1:]
        ]

    def _sdf_split(self, infile):
        """Split a multi-molecule .sdf file into per-molecule blocks."""
        with open(infile, "r") as f:
            contents = f.read()
        return [block + "$$$$\n" for block in contents.split("$$$$\n")[:-1]]

    def _mol_to_graph_single(self, lig):
        try:
            graph = mol_to_graph(
                lig, explicit_H=self.explicit_H, use_chirality=self.use_chirality
            )
        except Exception:
            # skip this one pose instead of failing the whole dataset, it gets
            # filtered out afterward (see the raw_ids/ligand_graphs zip-filter above)
            print("failed to scoring for {} and {}".format(self.protein_graph, lig))
            return None
        return graph

    def _mol_to_graph(self):
        """Featurize every ligand pose in self.ligs, in parallel if requested."""
        if self.parallel:
            return Parallel(n_jobs=-1, backend="threading")(
                delayed(self._mol_to_graph_single)(lig) for lig in self.ligs
            )  # process every pose at once, spread across multiple threads
        else:
            graphs = []
            for lig in self.ligs:  # process one pose at a time
                graphs.append(self._mol_to_graph_single(lig))
            return graphs

    def get_ligname(self, mol):
        """RDKit "_Name" property of a Mol, or None if unset/missing."""
        if mol is None or not mol.HasProp("_Name"):
            return None
        return mol.GetProp("_Name")
