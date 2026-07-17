"""
Extract a protein pocket (the residues near a reference ligand) using ProDy, with OpenBabel used to get the ligand into PDB format first, if needed.
"""

import os
import re

import openbabel
import prody as pr
from openbabel import openbabel as ob

ob_path = os.path.dirname(openbabel.__file__)
# try using package bundle path first, but fallback to homebrew on macOS
_libdir = os.path.join(ob_path, "lib", "openbabel", openbabel.__version__)
if not os.path.exists(_libdir) and os.path.exists("/opt/homebrew/lib/openbabel/3.1.0"):
    os.environ["BABEL_LIBDIR"] = "/opt/homebrew/lib/openbabel/3.1.0"
    os.environ["BABEL_DATADIR"] = "/opt/homebrew/share/openbabel/3.1.0"
else:
    os.environ["BABEL_LIBDIR"] = _libdir
    os.environ["BABEL_DATADIR"] = os.path.join(
        ob_path, "share", "openbabel", openbabel.__version__
    )


def write_file(output_file, outline):
    with open(output_file, "w") as buffer:
        buffer.write(outline)


def lig_rename(infile, outfile):
    """Rewrite every ATOM/HETATM record's residue name to "LIG".

    infile and outfile can be the same path (the file is fully read before any write).

    Some peptides may otherwise impede pocket generation below, so the ligand's residue name is normalized first."""

    with open(infile, "r") as f:
        lines = f.readlines()
    newlines = []
    for line in lines:
        if re.search(r"^HETATM|^ATOM", line):
            newlines.append(line[:17] + "LIG" + line[20:])
        else:
            newlines.append(line)

    write_file(outfile, "".join(newlines))


def check_mol(infile, outfile):
    """Drop any atom record still tagged LIG, as some metals share the ligand's residue ID, which would otherwise pull them into the pocket."""
    os.system(f"cat {infile} | sed '/LIG/d' > {outfile}")


def extract_pocket(
    protpath, ligpath, cutoff=5.0, protname=None, ligname=None, workdir="."
):
    """
    Write the protein residues within cutoff of the ligand to <workdir>/<protname>_pocket_<cutoff>.pdb.

    protpath: the path of protein file (.pdb).
    ligpath: the path of ligand file (.sdf|.mol2|.pdb).
    cutoff: the distance range within the ligand to determine the pocket.
    protname: the name of the protein.
    ligname: the name of the ligand.
    workdir: working directory.
    """

    if protname is None:
        protname = os.path.basename(protpath).split(".")[0]
    if ligname is None:
        ligname = os.path.basename(ligpath).split(".")[0]
    ob_conversion = ob.OBConversion()  # set up the file-format converter, from the ligand's format to pdb
    ob_conversion.SetInAndOutFormats(ligpath.split(".")[-1], "pdb")

    # paths used through the rest of this function
    lig_pdb_path = os.path.join(workdir, f"{ligname}.pdb")
    pocket_temp_path = os.path.join(workdir, f"{protname}_pocket_{cutoff}_temp.pdb")
    pocket_path = os.path.join(workdir, f"{protname}_pocket_{cutoff}.pdb")

    if not re.search(r".pdb$", ligpath):
        # convert ligand to pdb
        ligand = ob.OBMol()
        ob_conversion.ReadFile(ligand, ligpath)
        ob_conversion.WriteFile(ligand, lig_pdb_path)

    protein_structure = pr.parsePDB(protpath)  # load the full protein structure

    lig_rename(lig_pdb_path, lig_pdb_path)  # normalize the ligand's residue name in place
    ligand_structure = pr.parsePDB(lig_pdb_path)  # reload it now that the residue name has been normalized
    ligand_resname = ligand_structure.getResnames()[0]  # the normalized name, used to find the ligand below
    complex_structure = ligand_structure + protein_structure  # combine both structures so they can be searched together

    # select ONLY atoms that belong to the protein
    pocket_selection = complex_structure.select(
        f"same residue as exwithin {cutoff} of resname {ligand_resname}"
    )

    pr.writePDB(pocket_temp_path, pocket_selection)  # write out the selected residues

    check_mol(pocket_temp_path, pocket_path)  # remove any leftover ligand-tagged atoms before finishing
    os.remove(pocket_temp_path)  # clean up the intermediate file
