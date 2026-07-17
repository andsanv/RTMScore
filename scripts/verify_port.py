"""
Verify the pytorch port of RTMScore.

In MultiHeadAttentionLayer the vectorised scatter/gather forward is compared against reference that does what DGL's apply_edges + update_all(u_mul_e, sum) computed.
The function to_dense_batch_dgl is compared against a naive loop reference and the batch_size==1 path is checked against the general path.
End-to-end batch invariance with the trained checkpoint is checked.

Run:  .venv/bin/python scripts/verify_port.py
"""

import os
import sys
import math
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch as th
from rdkit import Chem

from rtmscore_pytorch.src.model import (
    RTMScore,
    GraphTransformer,
    MultiHeadAttentionLayer,
    to_dense_batch_dgl,
)
from rtmscore_pytorch.src.utils import batch_graphs
from rtmscore_pytorch.src.mol2graph_rdmda_res import mol_to_graph

th.manual_seed(0)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_failures = []


def report(name, diff, tol):
    ok = diff <= tol
    print(f"  [{PASS if ok else FAIL}] {name:<48} max|diff| = {diff:.3e}  (tol {tol:.0e})")
    if not ok:
        _failures.append(name)
    return ok


# ttention layer vs. independent edge-loop reference
def check_attention():
    print("\nCheck 1: MultiHeadAttentionLayer vs. DGL-semantics reference")
    N, E, Fin, H, d = 13, 37, 32, 4, 8
    layer = MultiHeadAttentionLayer(Fin, d, H, using_bias=False, update_edge_feats=True).eval()

    node_feats = th.randn(N, Fin, dtype=th.double)
    edge_feats = th.randn(E, Fin, dtype=th.double)
    layer.double()
    # random directed edges (allow self loops / duplicates, like a real graph)
    edge_index = th.randint(0, N, (2, E), dtype=th.long)
    g = {"edge_index": edge_index}

    with th.no_grad():
        h_out, e_out = layer(g, node_feats, edge_feats)

        # reproduce apply_edges + update_all
        Q = layer.Q(node_feats).view(N, H, d)
        K = layer.K(node_feats).view(N, H, d)
        V = layer.V(node_feats).view(N, H, d)
        pe = layer.edge_feats_projection(edge_feats).view(E, H, d)

        h_ref = th.zeros(N, H, d, dtype=th.double)
        z_ref = th.zeros(N, H, 1, dtype=th.double)
        e_ref = th.zeros(E, H, d, dtype=th.double)
        scale = math.sqrt(d)
        for e in range(E):
            u, v = int(edge_index[0, e]), int(edge_index[1, e])
            s = (K[u] * Q[v] / scale).clamp(-5.0, 5.0) * pe[e] # src K, dst Q
            e_ref[e] = s
            se = th.exp(s.sum(-1, keepdim=True).clamp(-5.0, 5.0)) # [H,1]
            h_ref[v] += V[u] * se  # msg from src, agg at dst
            z_ref[v] += se
        h_ref = h_ref / (z_ref + 1e-6)

    report("node output (h_out)", (h_out - h_ref).abs().max().item(), 1e-10)
    report("edge output (e_out)", (e_out - e_ref).abs().max().item(), 1e-10)


# to_dense_batch_dgl correctness + fast-path equivalence
def _dense_reference(feats, num_nodes_list, fill_value=0):
    """Straightforward loop implementation of dense batching."""
    max_n = max(num_nodes_list)
    B = len(num_nodes_list)
    out = feats.new_full([B, max_n] + list(feats.size())[1:], fill_value)
    mask = th.zeros(B, max_n, dtype=th.bool)
    off = 0
    for b, n in enumerate(num_nodes_list):
        out[b, :n] = feats[off:off + n]
        mask[b, :n] = True
        off += n
    return out, mask


def check_dense_batch():
    print("\nCheck 2: to_dense_batch_dgl (general path, fast path, 3-D pos)")
    num_nodes = [3, 5, 2, 4]
    total = sum(num_nodes)

    for shape_tail, label in [((7,), "2-D node feats"), ((24, 3), "3-D residue pos")]:
        feats = th.randn(total, *shape_tail, dtype=th.double)
        bg = {"batch_size": len(num_nodes),
              "batch_num_nodes": th.tensor(num_nodes, dtype=th.long)}
        out, mask = to_dense_batch_dgl(bg, feats)
        ref_out, ref_mask = _dense_reference(feats, num_nodes)
        report(f"general path values  [{label}]", (out - ref_out).abs().max().item(), 0.0)
        report(f"general path mask    [{label}]",
               (mask.int() - ref_mask.int()).abs().max().item(), 0.0)

        # fast path (batch_size==1) must equal general path on the same single graph
        single = feats[:num_nodes[0]]
        fast_bg = {"batch_size": 1, "batch_num_nodes": th.tensor([num_nodes[0]], dtype=th.long)}
        fast_out, fast_mask = to_dense_batch_dgl(fast_bg, single)
        gen_out, gen_mask = _dense_reference(single, [num_nodes[0]])
        report(f"fast path vs general [{label}]", (fast_out - gen_out).abs().max().item(), 0.0)
        report(f"fast path mask       [{label}]",
               (fast_mask.int() - gen_mask.int()).abs().max().item(), 0.0)


# end-to-end batch invariance with the real trained model
def build_model(device="cpu"):
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
    ckpt = th.load(os.path.join(ROOT, "trained_models/rtmscore.pth"),
                   map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def synth_prot_graph(n_res, n_edges, seed, center):
    """A protein-graph dict with the correct feature dims.
    Residue atoms are placed near center (the ligand centroid) so some ligand-residue distances fall under the threshold and the MDN/score path is taken."""
    gen = th.Generator().manual_seed(seed)
    pos = center.view(1, 1, 3) + 4.0 * th.randn(n_res, 24, 3, generator=gen)
    # emulate ragged residues: pad a few atom slots with NaN, as the real builder does
    pos[:, 18:, :] = float("nan")
    ei = th.randint(0, n_res, (2, n_edges), generator=gen, dtype=th.long)
    return {
        "num_nodes": n_res,
        "ndata_feats": th.randn(n_res, 41, generator=gen),
        "edata_feats": th.randn(n_edges, 5, generator=gen),
        "edge_index": ei,
        "ndata_pos": pos,
    }


def pair_prob(model, protein_batch, ligand_batch):
    """Per ligand-atom x residue MDN probability + per-complex score, copying the inference path (calculate_probablity / run_an_eval_epoch)."""
    pi, sigma, mu, dist, atom_types, bond_types, batch = model(protein_batch, ligand_batch)
    val = dist.expand_as(mu)
    logprob = (-0.5 * ((val - mu) / sigma) ** 2 - th.log(sigma) - 0.9189385332046727) + th.log(pi)
    prob = logprob.exp().sum(dim=1)
    prob = th.where(dist.squeeze(1) > model.dist_threshold, th.zeros_like(prob), prob)
    B = int(batch.max()) + 1
    score = th.zeros(B, dtype=prob.dtype).scatter_add_(0, batch, prob)
    return score, prob, atom_types, bond_types, batch


def check_end_to_end():
    print("\nCheck 3: end-to-end batch invariance with real trained weights")
    model = build_model()

    # real ligand poses from the shipped example
    supp = Chem.SDMolSupplier(os.path.join(ROOT, "example/1qkt_l.sdf"), removeHs=True)
    lig = next(m for m in supp if m is not None)
    lig_graphs = [mol_to_graph(lig, explicit_H=False, use_chirality=True) for _ in range(3)]
    center = lig_graphs[0]["ndata_pos"].mean(dim=0)
    # give each pose a distinct protein so batching genuinely mixes different sizes
    prot_graphs = [synth_prot_graph(60 + 7 * i, 300 + 11 * i, seed=100 + i, center=center)
                   for i in range(3)]

    # individual scoring (batch_size==1, fast path)
    single_scores = []
    with th.no_grad():
        for gl, gp in zip(lig_graphs, prot_graphs):
            ligand_batch = batch_graphs([gl])
            protein_batch = batch_graphs([gp])
            s, _, _, _, _ = pair_prob(model, protein_batch, ligand_batch)
            single_scores.append(s.item())

    # batched scoring (general path)
    with th.no_grad():
        ligand_batch = batch_graphs(lig_graphs)
        protein_batch = batch_graphs(prot_graphs)
        batch_scores, _, _, _, _ = pair_prob(model, protein_batch, ligand_batch)

    diff = max(abs(batch_scores[i].item() - single_scores[i]) for i in range(3))
    rel = diff / max(1e-9, max(abs(s) for s in single_scores))
    print(f"    single scores : {[round(s, 5) for s in single_scores]}")
    print(f"    batched scores: {[round(batch_scores[i].item(), 5) for i in range(3)]}")
    report("per-complex score (batched vs single)", rel, 1e-5)


if __name__ == "__main__":
    check_attention()
    check_dense_batch()
    check_end_to_end()
    print()
    if _failures:
        print(f"{FAIL}: {len(_failures)} check(s) failed -> {_failures}")
        sys.exit(1)
    print(f"{PASS}: all checks passed - the pure-PyTorch port matches the reference semantics.")
