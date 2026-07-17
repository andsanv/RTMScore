"""
Benchmark pure-PyTorch inference latency on the 1qkt example (61 poses), scores one pose at a time (batch_size=1) to match ONNX/C++ benchmarking.

Run:  .venv/bin/python scripts/benchmark_inference.py [cpu|cuda|both]
"""
import os
import sys
import time
import statistics

import torch as th
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rtmscore_pytorch.src.model import RTMScore, GraphTransformer
from rtmscore_pytorch.src.data import VSDataset
from rtmscore_pytorch.src.utils import collate

import openbabel
ob_path = os.path.dirname(openbabel.__file__)
_libdir = os.path.join(ob_path, "lib", "openbabel", openbabel.__version__)
os.environ["BABEL_LIBDIR"] = _libdir
os.environ["BABEL_DATADIR"] = os.path.join(ob_path, "share", "openbabel", openbabel.__version__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_model(device):
    ligand_model = GraphTransformer(in_channels=41, edge_features=10, num_hidden_channels=128,
                                   activ_fn=th.nn.SiLU(), transformer_residual=True,
                                   num_attention_heads=4, norm_to_apply="batch",
                                   dropout_rate=0.15, num_layers=6)
    protein_model = GraphTransformer(in_channels=41, edge_features=5, num_hidden_channels=128,
                                    activ_fn=th.nn.SiLU(), transformer_residual=True,
                                    num_attention_heads=4, norm_to_apply="batch",
                                    dropout_rate=0.15, num_layers=6)
    model = RTMScore(ligand_model, protein_model, in_channels=128, hidden_dim=128, n_gaussians=10,
                     dropout_rate=0.10, dist_threshold=5.0).to(device)
    ckpt = th.load(os.path.join(ROOT, "trained_models/rtmscore.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def to_device(bg, device):
    return {k: (v.to(device) if th.is_tensor(v) else v) for k, v in bg.items()}


def report_timing(ms):
    first, steady = ms[0], ms[1:]
    print(f"\nInference timing (ms), n={len(ms)} (first call reported separately as warm-up)")
    print(f"  first (warm-up) = {first:.4f}")
    print(f"  mean   = {statistics.mean(steady):.4f}")
    print(f"  median = {statistics.median(steady):.4f}")
    print(f"  min    = {min(steady):.4f}")
    print(f"  max    = {max(steady):.4f}")


def benchmark(device):
    print(f"\n=== device: {device} ===")
    model = build_model(device)

    dataset = VSDataset(ligs=os.path.join(ROOT, "example/1qkt_decoys.sdf"),
                        prot=os.path.join(ROOT, "example/1qkt_p_pocket_10.0.pdb"),
                        cutoff=10.0, gen_pocket=False, explicit_H=False,
                        use_chirality=True, parallel=False)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate)

    ms = []
    with th.no_grad():
        for _, ligand_batch, protein_batch in loader:
            ligand_batch = to_device(ligand_batch, device)
            protein_batch = to_device(protein_batch, device)
            if device == "cuda":
                th.cuda.synchronize()
            t0 = time.perf_counter()
            model(protein_batch, ligand_batch)
            if device == "cuda":
                th.cuda.synchronize()
            t1 = time.perf_counter()
            ms.append((t1 - t0) * 1000.0)

    report_timing(ms)
    return ms


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    devices = ["cpu", "cuda"] if which == "both" else [which]
    for d in devices:
        if d == "cuda" and not th.cuda.is_available():
            print("\n=== device: cuda === (skipped, no CUDA available)")
            continue
        benchmark(d)
