import random

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Normal


def mdn_loss_fn(pi, sigma, mu, y, eps=1e-10):
    """Negative log-likelihood of y under the predicted Gaussian mixture (pi, sigma, mu).
    Training-only, used by run_an_eval_epoch's loss path."""

    normal = Normal(mu, sigma)  # build a distribution around the predicted center and spread
    loglik = normal.log_prob(y.expand_as(normal.loc))  # measure how likely the true value is under each distribution
    loss = -th.logsumexp(th.log(pi + eps) + loglik, dim=1)  # combine them by weight, then flip the sign so lower is better

    return loss


def run_an_eval_epoch(
    model,
    data_loader,
    pred=False,
    atom_contribution=False,
    res_contribution=False,
    dist_threshold=None,
    aux_weight=0.001,
    device="cpu",
):
    """
    Run model over every batch in data_loader (no gradients).

    If pred (or atom_contribution/res_contribution) is set, returns predicted scores, optionally with per-atom/per-residue contribution breakdowns. Otherwise computes the training losses (MDN and auxiliary atom/bond cross-entropy) and returns their averages.
    """

    model.eval()  # switch to evaluation mode, so results stay consistent between runs
    total_loss = 0
    mdn_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    at_contrs = []
    res_contrs = []

    with th.no_grad():  # nothing here needs to track operations for backpropagation
        for batch_id, batch_data in enumerate(data_loader):
            pdbids, ligand_batch, protein_batch = batch_data
            ligand_batch = to_device(ligand_batch, device)
            protein_batch = to_device(protein_batch, device)
            batch_size = ligand_batch["batch_size"]
            atom_labels = th.argmax(
                ligand_batch["ndata_atom"][:, :17], dim=1, keepdim=False
            )  # turn the one-hot atom encoding back into a single label per atom
            bond_labels = th.argmax(
                ligand_batch["edata_bond"][:, :4], dim=1, keepdim=False
            )  # turn the one-hot bond encoding back into a single label per bond

            pi, sigma, mu, dist, atom_types, bond_types, batch = model(
                protein_batch, ligand_batch
            )

            if pred or atom_contribution or res_contribution:
                prob = calculate_probability(pi, sigma, mu, dist)
                if dist_threshold is not None:
                    prob[th.where(dist > dist_threshold)[0]] = 0.0  # zero out pairs that are farther apart than allowed

                batch = batch.to(device)
                if pred:
                    complex_scores = th.zeros(
                        batch_size, dtype=prob.dtype, device=device
                    ).scatter_add_(0, batch, prob)  # add up every pair's contribution that belongs to the same sample
                    probs.append(complex_scores)
                if atom_contribution or res_contribution:
                    contribs = [
                        prob[batch == i].reshape(
                            (
                                ligand_batch["batch_num_nodes"][i].item(),
                                protein_batch["batch_num_nodes"][i].item(),
                            )
                        )
                        for i in range(batch_size)
                    ]  # reshape each sample's flat list of pairs back into a grid
                    if atom_contribution:
                        at_contrs.extend(
                            [
                                contribs[i].sum(1).cpu().detach().numpy()  # add along one side to get each atom's total
                                for i in range(batch_size)
                            ]
                        )
                    if res_contribution:
                        res_contrs.extend(
                            [
                                contribs[i].sum(0).cpu().detach().numpy()  # add along the other side to get each residue's total
                                for i in range(batch_size)
                            ]
                        )

            else:
                mdn = mdn_loss_fn(pi, sigma, mu, dist)
                mdn = mdn[th.where(dist <= model.dist_threshold)[0]]  # only keep the pairs that are close enough to count
                mdn = mdn.mean()
                atom = F.cross_entropy(atom_types, atom_labels)  # measure how well the predicted atom types match the real ones
                bond = F.cross_entropy(bond_types, bond_labels)  # measure how well the predicted bond types match the real ones
                loss = mdn + (atom * aux_weight) + (bond * aux_weight)  # combine the main objective with the two smaller side ones

                total_loss += loss.item() * batch_size
                mdn_loss += mdn.item() * batch_size
                atom_loss += atom.item() * batch_size
                bond_loss += bond.item() * batch_size

            del (
                ligand_batch,
                protein_batch,
                atom_labels,
                bond_labels,
                pi,
                sigma,
                mu,
                dist,
                atom_types,
                bond_types,
                batch,
            )  # clear these out to free up memory before the next round
            th.cuda.empty_cache()

    if atom_contribution or res_contribution:
        if pred:
            preds = th.cat(probs)
            return [preds.cpu().detach().numpy(), at_contrs, res_contrs]
        else:
            return [None, at_contrs, res_contrs]
    else:
        if pred:
            preds = th.cat(probs)
            return preds.cpu().detach().numpy()
        else:
            return (
                total_loss / len(data_loader.dataset),
                mdn_loss / len(data_loader.dataset),
                atom_loss / len(data_loader.dataset),
                bond_loss / len(data_loader.dataset),
            )  # turn each running total into an average over the full dataset


def calculate_probability(pi, sigma, mu, y):
    """Mixture-density probability of y under the predicted Gaussian mixture."""

    normal = Normal(mu, sigma)  # build a distribution around the predicted center and spread
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += th.log(pi)  # weight each distribution by how likely it is to be the right one
    prob = logprob.exp().sum(1)  # undo the earlier log step and add everything together

    return prob


def batch_graphs(graphs):
    """
    Disjoint-union batch a list of per-graph feature dicts into one dict.

    Concatenate every tensor, offsetting edge_index by each graph's cumulative node count so indices stay valid in the merged node ordering.
    """

    batch = {}
    batch_size = len(graphs)
    batch["batch_size"] = batch_size

    batch_num_nodes = []
    node_offsets = []
    current_node_offset = 0

    # compute node counts and offsets
    for graph in graphs:
        num_nodes = graph["num_nodes"]
        batch_num_nodes.append(num_nodes)
        node_offsets.append(current_node_offset)  # remember where this item's nodes begin once everything is merged
        current_node_offset += num_nodes

    batch["batch_num_nodes"] = th.tensor(batch_num_nodes, dtype=th.long)

    # concatenate all properties
    keys = list(graphs[0].keys())
    for key in keys:
        if key == "num_nodes" or key == "batch_size" or key == "batch_num_nodes":
            continue  # already handled above
        elif key == "edge_index":
            edge_indices = []
            for i, graph in enumerate(graphs):
                if graph["edge_index"].size(1) > 0:
                    edge_indices.append(graph["edge_index"] + node_offsets[i])  # shift indices so they still point to the right place after merging
            if len(edge_indices) > 0:
                batch[key] = th.cat(edge_indices, dim=1)
            else:
                batch[key] = th.empty((2, 0), dtype=th.long)  # nothing to add, keep an empty placeholder instead
        else:
            # regular tensors
            tensors = [graph[key] for graph in graphs if graph[key].size(0) > 0]
            if len(tensors) > 0:
                batch[key] = th.cat(tensors, dim=0)
            else:
                batch[key] = th.empty(
                    (0,) + graphs[0][key].size()[1:], dtype=graphs[0][key].dtype
                )  # nothing to add here either, keep an empty placeholder with a matching shape
    return batch


def to_device(graph_dict, device):
    """Move every tensor value in graph_dict to device, in place."""

    for key, value in graph_dict.items():
        if th.is_tensor(value):  # only move actual tensors, leave plain values untouched
            graph_dict[key] = value.to(device)
    return graph_dict


def collate(data):
    """DataLoader collate_fn: batch a list of (id, ligand_graph, protein_graph) samples into (ids, ligand_batch, protein_batch)."""

    pdbids, ligand_graphs, protein_graphs = map(list, zip(*data))  # split the list of samples into separate lists for each part
    ligand_batch = batch_graphs(ligand_graphs)
    protein_batch = batch_graphs(protein_graphs)
    return pdbids, ligand_batch, protein_batch


def set_random_seed(seed=10):
    """Seed Python's random, numpy, and torch (CPU + CUDA) for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)  # cover every gpu if more than one is being used
