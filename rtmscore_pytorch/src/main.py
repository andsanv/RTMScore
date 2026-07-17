import argparse
import os
import sys

import MDAnalysis as mda
import numpy as np
import pandas as pd
import torch as th

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # absolute path to the repo root, used for the default file paths below
sys.path.append(ROOT)  # make the package importable regardless of where this script is run from

import torch.multiprocessing
from torch.utils.data import DataLoader

from rtmscore_pytorch.src.data import VSDataset
from rtmscore_pytorch.src.model import (
    GraphTransformer,
    RTMScore,
)
from rtmscore_pytorch.src.utils import collate, run_an_eval_epoch

torch.multiprocessing.set_sharing_strategy("file_system")  # avoid hitting shared-memory limits when loading many samples in parallel

import openbabel

ob_path = os.path.dirname(openbabel.__file__)
_libdir = os.path.join(ob_path, "lib", "openbabel", openbabel.__version__)
if not os.path.exists(_libdir) and os.path.exists("/opt/homebrew/lib/openbabel/3.1.0"):
    # fall back to the homebrew install location if the bundled one isn't found
    os.environ["BABEL_LIBDIR"] = "/opt/homebrew/lib/openbabel/3.1.0"
    os.environ["BABEL_DATADIR"] = "/opt/homebrew/share/openbabel/3.1.0"
else:
    os.environ["BABEL_LIBDIR"] = _libdir
    os.environ["BABEL_DATADIR"] = os.path.join(
        ob_path, "share", "openbabel", openbabel.__version__
    )


def Input():
    """Define the CLI argument parser, parse sys.argv, and validate flag combinations."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prot", required=True, help="Input protein file (.pdb)")
    parser.add_argument(
        "-l", "--lig", required=True, help="Input ligand file (.sdf/.mol2)"
    )
    parser.add_argument(
        "-m",
        "--model",
        default=os.path.join(ROOT, "trained_models", "rtmscore.pth"),
        help='trained model path (default: "trained_models/rtmscore.pth")',
    )
    parser.add_argument(
        "-o",
        "--outprefix",
        default=os.path.join(ROOT, "rtmscore_pytorch", "output", "out"),
        help='the prefix of output file (default: "rtmscore_pytorch/output/out")',
    )
    parser.add_argument(
        "-gen_pocket",
        "--gen_pocket",
        action="store_true",
        default=False,
        help="whether to generate the pocket",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        default=10.0,
        type=float,
        help="the cutoff the define the pocket and interactions within the pocket (default: 10.0)",
    )
    parser.add_argument(
        "-rl",
        "--reflig",
        default=None,
        help="the reference ligand to determine the pocket(.sdf/.mol2)",
    )
    parser.add_argument(
        "-pl",
        "--parallel",
        default=False,
        action="store_true",
        help="whether to obtain the graphs in parallel (When the dataset is too large,\
						 it may be out of memory when conducting the parallel mode).",
    )
    parser.add_argument(
        "-ac",
        "--atom_contribution",
        default=False,
        action="store_true",
        help="whether to obtain the atom contrubution of the score.",
    )
    parser.add_argument(
        "-rc",
        "--res_contribution",
        default=False,
        action="store_true",
        help="whether to obtain the residue contrubution of the score.",
    )
    args = parser.parse_args()
    if args.gen_pocket:
        if args.reflig is None:
            # a pocket can only be built once we know which ligand to build it around
            raise ValueError(
                "if pocket is generated, the reference ligand should be provided."
            )
    if args.atom_contribution and args.res_contribution:
        # these two modes produce differently shaped output, so only one can run at a time
        raise ValueError(
            "only one of the atom_contribution and res_contribution can be supported"
        )
    return args


def _build_graph_transformer(num_node_feats, num_edge_feats, hidden_dim):
    """Build a GraphTransformer encoder with RTMScore's fixed architecture."""

    return GraphTransformer(
        in_channels=num_node_feats,
        edge_features=num_edge_feats,
        num_hidden_channels=hidden_dim,
        activ_fn=th.nn.SiLU(),
        transformer_residual=True,
        num_attention_heads=4,
        norm_to_apply="batch",
        dropout_rate=0.15,
        num_layers=6,
    )


def scoring(
    prot,
    lig,
    modpath,
    cut=10.0,
    gen_pocket=False,
    reflig=None,
    atom_contribution=False,
    res_contribution=False,
    explicit_H=False,
    use_chirality=True,
    parallel=False,
    **kwargs,
):
    """
    The main scoring pipeline.

    Builds a VSDataset from the input files, wraps them into a DataLoader, constructs two GraphTransformer encoders (ligands and proteins), combines them into a RTMScore model, loads the weights and performs inference.

    Parameters
    ---
    prot:
        input protein file ('.pdb')
    lig:
        input ligand file ('.sdf|.mol2', supports multiple ligands)
    modpath:
        path where the pre-trained model is stored
    gen_pocket:
        whether to generate the pocket from the protein file
    reflig:
        reference ligand used to determine the pocket
    cut:
        distance within the reference ligand to determine the pocket
    atom_contribution:
        whether the decompose the score at atom level
    res_contribution:
        whether the decompose the score at residue level
    explicit_H:
        whether to use explicit hydrogen atoms to represent the molecules
    use_chirality:
        whether to adopt the information of chirality to represent the molecules
    parallel:
        whether to generate the graphs in parallel (suitable when there are lots of ligands/poses)
    kwargs:
        other arguments related with model
    """

    # featurize the protein/ligand inputs into graphs and wrap them in a loader.
    data = VSDataset(
        ligs=lig,
        prot=prot,
        cutoff=cut,
        gen_pocket=gen_pocket,
        reflig=reflig,
        explicit_H=explicit_H,
        use_chirality=use_chirality,
        parallel=parallel,
    )

    test_loader = DataLoader(
        dataset=data,
        batch_size=kwargs["batch_size"],
        shuffle=False,  # keep the original order, so ids line up with the results below
        num_workers=kwargs["num_workers"],
        collate_fn=collate,
    )

    # one encoder for ligand atoms, one for protein residues (same architecture, different input feature sizes)
    ligand_model = _build_graph_transformer(
        kwargs["num_node_featsl"], kwargs["num_edge_featsl"], kwargs["hidden_dim0"]
    )
    protein_model = _build_graph_transformer(
        kwargs["num_node_featsp"], kwargs["num_edge_featsp"], kwargs["hidden_dim0"]
    )

    model = RTMScore(
        ligand_model,
        protein_model,
        in_channels=kwargs["hidden_dim0"],
        hidden_dim=kwargs["hidden_dim"],
        n_gaussians=kwargs["n_gaussians"],
        dropout_rate=kwargs["dropout_rate"],
        dist_threshold=kwargs["dist_threshold"],
    ).to(kwargs["device"])

    checkpoint = th.load(modpath, map_location=th.device(kwargs["device"]))  # read the trained weights from disk
    model.load_state_dict(checkpoint["model_state_dict"])  # apply them to the freshly built model

    # run inference; output depends on the contribution mode requested
    if atom_contribution:
        preds, at_contrs, _ = run_an_eval_epoch(
            model,
            test_loader,
            pred=True,
            atom_contribution=True,
            res_contribution=False,
            dist_threshold=kwargs["dist_threshold"],
            device=kwargs["device"],
        )

        atom_ids = [
            "%s%s" % (atom.GetSymbol(), atom.GetIdx())
            for atom in data.ligs[0].GetAtoms()
        ]  # build a readable label for each atom, used as row names in the output
        return data.ids, preds, atom_ids, at_contrs

    elif res_contribution:
        preds, _, res_contrs = run_an_eval_epoch(
            model,
            test_loader,
            pred=True,
            atom_contribution=False,
            res_contribution=True,
            dist_threshold=kwargs["dist_threshold"],
            device=kwargs["device"],
        )
        universe = mda.Universe(data.prot)  # reload the protein to read its residue names/ids
        res_ids = [
            "%s_%s%s" % (chain_id[0], resname, resid)
            for chain_id, resname, resid in zip(
                universe.residues.chainIDs,
                universe.residues.resnames,
                universe.residues.resids,
            )
        ]  # chain_id is per-atom-in-residue here, so take the first (they're all the same); build a readable label for each residue, used as row names in the output
        return data.ids, preds, res_ids, res_contrs

    else:
        preds = run_an_eval_epoch(
            model,
            test_loader,
            pred=True,
            dist_threshold=kwargs["dist_threshold"],
            device=kwargs["device"],
        )
        return data.ids, preds


def _write_contribution_csv(ids, scores, contribs, feature_ids, outprefix, suffix):
    """Write per-atom/per-residue score contributions plus the total score to <outprefix>_<suffix>.csv, sorted best-scoring first."""

    df = pd.DataFrame(contribs).T  # one row per sample, one column per atom/residue
    df.columns = ids
    df.index = feature_ids
    df = df[df.apply(np.sum, axis=1) != 0].T  # drop columns that never contributed anything, then flip back
    scores_df = pd.DataFrame(zip(*(ids, scores)), columns=["id", "score"])
    scores_df.index = scores_df.id
    df = pd.concat([scores_df["score"], df], axis=1)  # attach the total score as the first column
    df.sort_values("score", ascending=False, inplace=True)  # show the best-scoring samples first
    df.to_csv(f"{outprefix}_{suffix}.csv")


def main():
    inargs = Input()
    outdir = os.path.dirname(inargs.outprefix)
    if outdir:
        os.makedirs(outdir, exist_ok=True)  # make sure the output folder exists before anything writes into it

    # fixed architecture/eval hyperparameters (must match trained model dimensions!)
    model_kwargs = {
        "batch_size": 128,
        "dist_threshold": 5,
        "device": "cuda" if th.cuda.is_available() else "cpu",
        "num_workers": 8,
        "num_node_featsp": 41,
        "num_node_featsl": 41,
        "num_edge_featsp": 5,
        "num_edge_featsl": 10,
        "hidden_dim0": 128,
        "hidden_dim": 128,
        "n_gaussians": 10,
        "dropout_rate": 0.10,
    }

    # arguments shared by every scoring() call below, regardless of output mode
    scoring_kwargs = dict(
        prot=inargs.prot,
        lig=inargs.lig,
        modpath=inargs.model,
        cut=inargs.cutoff,
        gen_pocket=inargs.gen_pocket,
        reflig=inargs.reflig,
        explicit_H=False,
        use_chirality=True,
        parallel=inargs.parallel,
        **model_kwargs,
    )

    if inargs.atom_contribution:
        ids, scores, atom_ids, at_contrs = scoring(
            atom_contribution=True, **scoring_kwargs
        )
        _write_contribution_csv(
            ids, scores, at_contrs, atom_ids, inargs.outprefix, "at"
        )
    elif inargs.res_contribution:
        ids, scores, res_ids, res_contrs = scoring(
            res_contribution=True, **scoring_kwargs
        )
        _write_contribution_csv(
            ids, scores, res_contrs, res_ids, inargs.outprefix, "res"
        )
    else:
        # plain scoring, no contribution breakdown
        ids, scores = scoring(**scoring_kwargs)
        df = pd.DataFrame(zip(*(ids, scores)), columns=["id", "score"])
        df.sort_values("score", ascending=False, inplace=True)  # show the best-scoring samples first
        df.to_csv(f"{inargs.outprefix}.csv", index=False)



if __name__ == "__main__":
    main()
