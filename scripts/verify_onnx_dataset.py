import os
import sys
import argparse
import torch as th
import numpy as np
import onnxruntime as ort

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rtmscore_pytorch.src.model import RTMScore, GraphTransformer
from rtmscore_pytorch.src.data import VSDataset
from rtmscore_pytorch.src.utils import collate
from torch.utils.data import DataLoader
from export_onnx import RTMScoreONNX

# Setup OpenBabel environment paths
import openbabel
ob_path = os.path.dirname(openbabel.__file__)
_libdir = os.path.join(ob_path, "lib", "openbabel", openbabel.__version__)
if not os.path.exists(_libdir) and os.path.exists("/opt/homebrew/lib/openbabel/3.1.0"):
    os.environ["BABEL_LIBDIR"] = "/opt/homebrew/lib/openbabel/3.1.0"
    os.environ["BABEL_DATADIR"] = "/opt/homebrew/share/openbabel/3.1.0"
else:
    os.environ["BABEL_LIBDIR"] = _libdir
    os.environ["BABEL_DATADIR"] = os.path.join(ob_path, "share", "openbabel", openbabel.__version__)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_failures = []


def report(name, diff, tol):
    ok = diff <= tol
    print(f"  [{PASS if ok else FAIL}] {name:<48} max|diff| = {diff:.3e}  (tol {tol:.0e})")
    if not ok:
        _failures.append(name)
    return ok


def build_model(device="cpu"):
    ligand_model = GraphTransformer(in_channels=41, edge_features=10, num_hidden_channels=128,
                                    activ_fn=th.nn.SiLU(), transformer_residual=True,
                                    num_attention_heads=4, norm_to_apply='batch',
                                    dropout_rate=0.15, num_layers=6)
    protein_model = GraphTransformer(in_channels=41, edge_features=5, num_hidden_channels=128,
                                     activ_fn=th.nn.SiLU(), transformer_residual=True,
                                     num_attention_heads=4, norm_to_apply='batch',
                                     dropout_rate=0.15, num_layers=6)
    model = RTMScore(ligand_model, protein_model, in_channels=128, hidden_dim=128, n_gaussians=10,
                      dropout_rate=0.10, dist_threshold=5.0).to(device)
    ckpt = th.load(os.path.join(ROOT, "trained_models/rtmscore.pth"), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAMES = ['score', 'prob', 'pi', 'sigma', 'mu', 'dist', 'atom_types', 'bond_types']


def to_ort_inputs(ligand_batch, protein_batch):
    return {
        'l_ndata_atom': ligand_batch["ndata_atom"].float().numpy(),
        'l_edata_bond': ligand_batch["edata_bond"].float().numpy(),
        'l_edge_index': ligand_batch["edge_index"].numpy(),
        'l_ndata_pos': ligand_batch["ndata_pos"].float().numpy(),
        'l_batch_num_nodes': ligand_batch["batch_num_nodes"].numpy(),
        'p_ndata_feats': protein_batch["ndata_feats"].float().numpy(),
        'p_edata_feats': protein_batch["edata_feats"].float().numpy(),
        'p_edge_index': protein_batch["edge_index"].numpy(),
        'p_ndata_pos': protein_batch["ndata_pos"].float().numpy(),
        'p_batch_num_nodes': protein_batch["batch_num_nodes"].numpy(),
    }


def check_eager_vs_onnx(onnx_wrapper, session, batch_size, n_batches, dataset):
    """Per-tensor eager-PyTorch-wrapper vs ONNX-Runtime comparison, at a given batch_size."""
    print(f"\nCheck: eager wrapper vs ONNX Runtime, batch_size={batch_size}")
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, collate_fn=collate)
    max_diffs = {name: 0.0 for name in NAMES}
    for idx, (pdbids, ligand_batch, protein_batch) in enumerate(loader):
        if idx >= n_batches:
            break
        ort_inputs = to_ort_inputs(ligand_batch, protein_batch)
        with th.no_grad():
            py_outputs = onnx_wrapper(
                th.from_numpy(ort_inputs['l_ndata_atom']), th.from_numpy(ort_inputs['l_edata_bond']),
                ligand_batch["edge_index"], th.from_numpy(ort_inputs['l_ndata_pos']), ligand_batch["batch_num_nodes"],
                th.from_numpy(ort_inputs['p_ndata_feats']), th.from_numpy(ort_inputs['p_edata_feats']),
                protein_batch["edge_index"], th.from_numpy(ort_inputs['p_ndata_pos']), protein_batch["batch_num_nodes"],
            )
        ort_outputs = session.run(NAMES, ort_inputs)
        for i, name in enumerate(NAMES):
            py_val = py_outputs[i]
            ort_val = ort_outputs[i]
            if th.is_tensor(py_val):
                py_val = py_val.numpy()
            diff = float(np.abs(py_val - ort_val).max())
            max_diffs[name] = max(max_diffs[name], diff)
        print(f"  batch {idx} ({', '.join(str(p) for p in pdbids)}): "
              f"score shape {ort_outputs[0].shape}, "
              f"score diff {np.abs(np.asarray(py_outputs[0]) - ort_outputs[0]).max():.3e}")
    for name in NAMES:
        report(f"{name} (eager vs ONNX, batch_size={batch_size})", max_diffs[name], 1e-2)


def check_batch_invariance_onnx(session, dataset, batch_size, n_poses):
    """ONNX-exported graph's batch invariance test to check if scoring each real pose alone (batch=1) matches its slot in a batch_size=N run.
    Tests the scatter_add score-aggregation"""

    print(f"\nCheck: ONNX-level batch invariance (batch=1 vs batch_size={batch_size}), real data")
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, collate_fn=collate)
    pdbids, ligand_batch, protein_batch = next(iter(loader))
    n = min(n_poses, len(pdbids))
    batch_scores = session.run(['score'], to_ort_inputs(ligand_batch, protein_batch))[0]

    loader1 = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                          num_workers=0, collate_fn=collate)
    single_scores = []
    for i, (pdbid1, ligand_batch1, protein_batch1) in enumerate(loader1):
        if i >= n:
            break
        s = session.run(['score'], to_ort_inputs(ligand_batch1, protein_batch1))[0]
        single_scores.append(float(np.asarray(s).reshape(-1)[0]))

    diff = max(abs(batch_scores[i] - single_scores[i]) for i in range(n))
    print(f"    batch scores : {[round(float(batch_scores[i]), 5) for i in range(n)]}")
    print(f"    single scores: {[round(s, 5) for s in single_scores]}")
    report(f"per-complex score (batch_size={batch_size} vs batch=1), real data", diff, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prot", default=os.path.join(ROOT, "example/1qkt_p_pocket_10.0.pdb"))
    ap.add_argument("--lig", default=os.path.join(ROOT, "example/1qkt_decoys.sdf"))
    ap.add_argument("--model", default=os.path.join(ROOT, "trained_models/rtmscore.onnx"))
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8],
                     help="batch sizes to exercise (default: 1 and 8)")
    args = ap.parse_args()

    device = 'cpu'
    model = build_model(device)
    onnx_wrapper = RTMScoreONNX(model, dist_threshold=5.0)
    onnx_wrapper.eval()

    print(f"Loading ONNX model from {args.model}...")
    session = ort.InferenceSession(args.model)

    print(f"Loading real dataset from {args.prot} and {args.lig}...")
    dataset = VSDataset(ligs=args.lig, prot=args.prot, cutoff=10.0,
                         gen_pocket=False, explicit_H=False, use_chirality=True, parallel=False)
    print(f"Loaded {len(dataset)} examples.")

    for bs in args.batch_sizes:
        check_eager_vs_onnx(onnx_wrapper, session, batch_size=bs, n_batches=3, dataset=dataset)

    for bs in args.batch_sizes:
        if bs > 1:
            check_batch_invariance_onnx(session, dataset, batch_size=bs, n_poses=min(bs, 8))

    print()
    if _failures:
        print(f"{FAIL}: {len(_failures)} check(s) failed -> {_failures}")
        sys.exit(1)
    print(f"{PASS}: all checks passed, the exported ONNX model matches eager PyTorch and is batch-invariant on real data.")


if __name__ == "__main__":
    main()
