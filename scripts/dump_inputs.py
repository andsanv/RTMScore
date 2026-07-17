"""
Featurize protein+ligand complexes in Python and dump the ONNX input tensors to disk, so a C++ program can just load them and run ONNX Runtime (path "a").

This script performs that featurization (via VSDataset) and serialises each batch's inputs into a binary bundle that main.cpp consumes.

Bundle layout (one directory per batch of `--batch-size` poses):

    <out>/<id>/manifest.txt        # one line per tensor, in ONNX input order:
                                    #     <name> <dtype:f32|i64> <ndim> <dim0> [dim1 ...]
    <out>/<id>/<name>.bin          # raw little-endian, row-major tensor data
    <out>/<id>/pose_ids.txt        # one pose id per line, in batch_num_nodes order
    <out>/<id>/expected.txt        # reference scores from ONNX Runtime, one per line,
                                    # in the same order as pose_ids.txt

With --batch-size 1 (the default), each bundle holds exactly one pose, this is the same use case dump_inputs.py always supported, just with l_batch_num_nodes/p_batch_num_nodes added, since the current ONNX model requires them.

All float inputs are written as float32, edge indices and batch_num_nodes as int64.  Positions are cast from float64 to float32.

Usage:
    .venv/bin/python scripts/dump_inputs.py \
        -p example/1qkt_p_pocket_10.0.pdb -l example/1qkt_decoys.sdf \
        -o rtmscore_onnx/fixtures --limit 5

    .venv/bin/python scripts/dump_inputs.py \
        -p example/1qkt_p_pocket_10.0.pdb -l example/1qkt_decoys.sdf \
        -o rtmscore_onnx/fixtures_batched --batch-size 8 --limit 16
"""
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

# OpenBabel env (same shim the other scripts use)
import openbabel
_ob = os.path.dirname(openbabel.__file__)
_libdir = os.path.join(_ob, "lib", "openbabel", openbabel.__version__)
if not os.path.exists(_libdir) and os.path.exists("/opt/homebrew/lib/openbabel/3.1.0"):
    os.environ["BABEL_LIBDIR"] = "/opt/homebrew/lib/openbabel/3.1.0"
    os.environ["BABEL_DATADIR"] = "/opt/homebrew/share/openbabel/3.1.0"
else:
    os.environ["BABEL_LIBDIR"] = _libdir
    os.environ["BABEL_DATADIR"] = os.path.join(_ob, "share", "openbabel", openbabel.__version__)

import onnxruntime as ort
from torch.utils.data import DataLoader, Subset
from rtmscore_pytorch.src.data import VSDataset
from rtmscore_pytorch.src.utils import collate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# (onnx input name, source dict, key, dtype) in the exact order ONNX expects.
SPEC = [
    ("l_ndata_atom",       "l", "ndata_atom",       "f32"),
    ("l_edata_bond",       "l", "edata_bond",       "f32"),
    ("l_edge_index",       "l", "edge_index",       "i64"),
    ("l_ndata_pos",        "l", "ndata_pos",        "f32"),
    ("l_batch_num_nodes",  "l", "batch_num_nodes",  "i64"),
    ("p_ndata_feats",      "p", "ndata_feats",      "f32"),
    ("p_edata_feats",      "p", "edata_feats",      "f32"),
    ("p_edge_index",       "p", "edge_index",       "i64"),
    ("p_ndata_pos",        "p", "ndata_pos",        "f32"),
    ("p_batch_num_nodes",  "p", "batch_num_nodes",  "i64"),
]


def to_numpy(t, dtype):
    a = t.detach().cpu().numpy()
    return a.astype(np.float32) if dtype == "f32" else a.astype(np.int64)


def dump_bundle(out_dir, arrays):
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for name, _, _, dtype in SPEC:
        a = arrays[name]
        a = np.ascontiguousarray(a) # row-major, little-endian
        a.tofile(os.path.join(out_dir, f"{name}.bin"))
        lines.append(f"{name} {dtype} {a.ndim} " + " ".join(str(d) for d in a.shape))
    with open(os.path.join(out_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prot", default=os.path.join(ROOT, "example/1qkt_p_pocket_10.0.pdb"))
    ap.add_argument("-l", "--lig", default=os.path.join(ROOT, "example/1qkt_decoys.sdf"))
    ap.add_argument("-m", "--model", default=os.path.join(ROOT, "trained_models/rtmscore.onnx"))
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "rtmscore_onnx/fixtures"))
    ap.add_argument("-c", "--cutoff", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=5, help="max number of poses to dump")
    ap.add_argument("--batch-size", type=int, default=1,
                     help="poses per bundle (default 1: one pose per bundle, as before)")
    args = ap.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")

    print(f"Featurizing {args.lig} against {args.prot} ...")
    ds = VSDataset(ligs=args.lig, prot=args.prot, cutoff=args.cutoff,
                   gen_pocket=False, explicit_H=False, use_chirality=True, parallel=False)
    # Limit samples before collation so a partial final batch is built from exactly the remaining poses
    limited_ds = Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(limited_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=0, collate_fn=collate)

    sess = ort.InferenceSession(args.model)
    os.makedirs(args.out, exist_ok=True)

    n_dumped = 0
    n_batches = 0
    index_lines = []
    for batch_idx, (pdbids, ligand_batch, protein_batch) in enumerate(loader):
        src = {"l": ligand_batch, "p": protein_batch}
        arrays = {name: to_numpy(src[which][key], dtype)
                  for name, which, key, dtype in SPEC}

        if args.batch_size == 1:
            bundle_id = str(pdbids[0])
        else:
            bundle_id = f"batch{args.batch_size}_{batch_idx}"
        out_dir = os.path.join(args.out, bundle_id)
        dump_bundle(out_dir, arrays)

        with open(os.path.join(out_dir, "pose_ids.txt"), "w") as f:
            f.write("\n".join(str(p) for p in pdbids) + "\n")

        # reference scores from ONNX Runtime, for the C++ program to validate against
        feed = {name: arrays[name] for name, *_ in SPEC}
        scores = np.asarray(sess.run(["score"], feed)[0]).reshape(-1)
        with open(os.path.join(out_dir, "expected.txt"), "w") as f:
            f.write("\n".join(f"{s:.10f}" for s in scores) + "\n")

        n_l_total = arrays["l_ndata_atom"].shape[0]
        n_p_total = arrays["p_ndata_feats"].shape[0]
        print(f"  [batch {batch_idx}] {bundle_id:20} poses={len(pdbids):3} "
              f"N_l_total={n_l_total:4} N_p_total={n_p_total:5} "
              f"scores={np.round(scores, 4)} -> {out_dir}")
        index_lines.append(bundle_id)
        n_dumped += len(pdbids)
        n_batches += 1

    with open(os.path.join(args.out, "index.txt"), "w") as f:
        f.write("\n".join(index_lines) + "\n")
    print(f"\nWrote {n_batches} bundle(s) ({n_dumped} poses) to {args.out}")
    print(f"Run in C++:  ./interaction {os.path.relpath(args.model, ROOT)} "
          f"{os.path.relpath(os.path.join(args.out, index_lines[0]), ROOT)}")


if __name__ == "__main__":
    main()
