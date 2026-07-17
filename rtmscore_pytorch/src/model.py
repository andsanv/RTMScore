import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn


def glorot_orthogonal(tensor, scale):
    """Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""

    if tensor is not None:
        th.nn.init.orthogonal_(tensor.data)  # start from a random but well-spread-out set of values
        scale /= (tensor.size(-2) + tensor.size(-1)) * tensor.var()  # adjust the scale based on the tensor's shape and current spread
        tensor.data *= scale.sqrt()  # apply the adjusted scale


class MultiHeadAttentionLayer(nn.Module):
    """Compute attention scores from a graph's node and edge (geometric) features."""

    def __init__(
        self,
        num_input_feats,
        num_output_feats,
        num_heads,
        using_bias=False,
        update_edge_feats=True,
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        # declare shared variables
        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats

        # define node features' query, key, and value tensors, and define edge features' projection tensors
        self.Q = nn.Linear(
            num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias
        )
        self.K = nn.Linear(
            num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias
        )
        self.V = nn.Linear(
            num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias
        )
        self.edge_feats_projection = nn.Linear(
            num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        if self.using_bias:
            glorot_orthogonal(self.Q.weight, scale=scale)
            self.Q.bias.data.fill_(0)

            glorot_orthogonal(self.K.weight, scale=scale)
            self.K.bias.data.fill_(0)

            glorot_orthogonal(self.V.weight, scale=scale)
            self.V.bias.data.fill_(0)

            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
            self.edge_feats_projection.bias.data.fill_(0)
        else:
            glorot_orthogonal(self.Q.weight, scale=scale)
            glorot_orthogonal(self.K.weight, scale=scale)
            glorot_orthogonal(self.V.weight, scale=scale)
            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)

    def forward(self, graph, node_feats, edge_feats):
        edge_index = graph["edge_index"]  # pairs of connected node indices
        num_nodes = node_feats.size(0)

        node_feats_q = self.Q(node_feats)  # project into "query" space
        node_feats_k = self.K(node_feats)  # project into "key" space
        node_feats_v = self.V(node_feats)  # project into "value" space
        edge_feats_projection = self.edge_feats_projection(edge_feats)  # project edge features into the same space

        # reshape tensors into [num_nodes, num_heads, feat_dim] to get projections for multi-head attention
        Q_h = node_feats_q.view(-1, self.num_heads, self.num_output_feats)
        K_h = node_feats_k.view(-1, self.num_heads, self.num_output_feats)
        V_h = node_feats_v.view(-1, self.num_heads, self.num_output_feats)
        proj_e = edge_feats_projection.view(-1, self.num_heads, self.num_output_feats)

        src = edge_index[0]  # index of each edge's starting node
        dst = edge_index[1]  # index of each edge's ending node

        # gather source/destination features
        K_h_src = K_h[src]
        Q_h_dst = Q_h[dst]

        # compute attention scores
        score = K_h_src * Q_h_dst
        # scale and clip attention scores
        score = (score / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)
        # use available edge features to modify the attention scores
        score = score * proj_e

        e_out = None
        if self.update_edge_feats:
            e_out = score  # only keep the edge scores if a later layer will need them

        # apply softmax to attention scores, followed by clipping
        score_sum = score.sum(-1, keepdim=True)
        score_exp = th.exp(score_sum.clamp(-5.0, 5.0))

        # send weighted values to target nodes
        dst_index_wV = (
            dst.view(-1, 1, 1)
            .expand(-1, self.num_heads, self.num_output_feats)
            .contiguous()
        )
        wV = th.zeros(
            num_nodes,
            self.num_heads,
            self.num_output_feats,
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        wV.scatter_add_(0, dst_index_wV, V_h[src] * score_exp)  # accumulate weighted values at each destination node

        dst_index_z = dst.view(-1, 1, 1).expand(-1, self.num_heads, 1).contiguous()
        z = th.zeros(
            num_nodes,
            self.num_heads,
            1,
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        z.scatter_add_(0, dst_index_z, score_exp)  # accumulate the total weight at each destination node, used to normalize below

        h_out = wV / (z + 1e-6)  # normalize so the weights add up to (roughly) one

        return h_out, e_out


class GraphTransformerModule(nn.Module):
    """
    One graph-transformer layer: multi-head attention over node/edge features, each followed by a pre-norm + MLP + residual block (transformer-style).

    Set update_edge_feats=False for the last layer in a stack: edge representations aren't needed anymore once no further layer will consume them, so that output path (and its parameters) is skipped entirely.
    """

    def __init__(
        self,
        num_hidden_channels,
        activ_fn=nn.SiLU(),
        residual=True,
        num_attention_heads=4,
        norm_to_apply="batch",
        dropout_rate=0.1,
        num_layers=4,
        update_edge_feats=True,
    ):
        super(GraphTransformerModule, self).__init__()

        # record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.update_edge_feats = update_edge_feats

        self.apply_layer_norm = "layer" in self.norm_to_apply.lower()  # decide which kind of normalization to use, based on the requested scheme

        self.num_hidden_channels, self.num_output_feats = (
            num_hidden_channels,
            num_hidden_channels,
        )
        # first-round normalization: edge_feats is always normalized here since it's fed into mha_module regardless of update_edge_feats
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels
            != self.num_output_feats,  # only use bias if a Linear() has to change sizes
            update_edge_feats=self.update_edge_feats,
        )

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = (
            nn.Dropout(p=self.dropout_rate)
            if self.dropout_rate > 0.0
            else nn.Identity()
        )
        self.node_feats_MLP = nn.ModuleList(
            [
                nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
                self.activ_fn,
                dropout,
                nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False),
            ]
        )

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        else:  # default to using batch normalization
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)

        # edge-feature output path, only needed on layers whose edge representations are consumed downstream (i.e. not the final layer)
        if self.update_edge_feats:
            self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
            self.edge_feats_MLP = nn.ModuleList(
                [
                    nn.Linear(
                        self.num_output_feats, self.num_output_feats * 2, bias=False
                    ),
                    self.activ_fn,
                    dropout,
                    nn.Linear(
                        self.num_output_feats * 2, self.num_output_feats, bias=False
                    ),
                ]
            )
            if self.apply_layer_norm:
                self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
            else:  #  default to using batch normalization
                self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)

        if self.update_edge_feats:
            glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
            self.O_edge_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, "weight"):  # skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

        if self.update_edge_feats:
            for layer in self.edge_feats_MLP:
                if hasattr(layer, "weight"):
                    glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, graph, node_feats, edge_feats):
        """Perform a forward pass of graph attention using a multi-head attention (MHA) module."""

        node_feats_in1 = (
            node_feats  # cache node representations for first residual connection
        )
        edge_feats_in1 = (
            edge_feats  # cache edge representations for first residual connection
        )

        # apply first round of normalization before applying graph attention, for performance enhancement
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  # default to using batch normalization
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        # get MHA output using provided node and edge representations
        node_attn_out, edge_attn_out = self.mha_module(graph, node_feats, edge_feats)

        # node path
        node_feats = node_attn_out.view(-1, self.num_output_feats)
        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)  # randomly zero out some values during training, to reduce overfitting
        node_feats = self.O_node_feats(node_feats)  # project back to the model's working size

        if self.residual:
            node_feats = (
                node_feats_in1 + node_feats
            )  # make first node residual connection

        node_feats_in2 = (
            node_feats  # cache node representations for second residual connection
        )

        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
        else:  # default to using batch normalization
            node_feats = self.batch_norm2_node_feats(node_feats)

        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        if self.residual:
            node_feats = (
                node_feats_in2 + node_feats
            )  # make second node residual connection

        if not self.update_edge_feats:
            return node_feats, None

        # edge path
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)
        edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)  # randomly zero out some values during training, to reduce overfitting
        edge_feats = self.O_edge_feats(edge_feats)  # project back to the model's working size

        if self.residual:
            edge_feats = (
                edge_feats_in1 + edge_feats
            )  # make first edge residual connection

        edge_feats_in2 = (
            edge_feats  # cache edge representations for second residual connection
        )

        if self.apply_layer_norm:
            edge_feats = self.layer_norm2_edge_feats(edge_feats)
        else:  # default to using batch normalization
            edge_feats = self.batch_norm2_edge_feats(edge_feats)

        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)

        if self.residual:
            edge_feats = (
                edge_feats_in2 + edge_feats
            )  # make second edge residual connection

        return node_feats, edge_feats

    def forward(self, graph, node_feats, edge_feats):
        """Perform a forward pass of a Graph Transformer layer. Returns (node_feats, edge_feats) if update_edge_feats, else just node_feats."""

        node_feats, edge_feats = self.run_gt_layer(graph, node_feats, edge_feats)
        if self.update_edge_feats:
            return node_feats, edge_feats

        return node_feats


class GraphTransformer(nn.Module):
    """Encodes a batch of graphs (nodes + edges) into node representations via a stack of GraphTransformerModule layers."""

    def __init__(
        self,
        in_channels,
        edge_features=10,
        num_hidden_channels=128,
        activ_fn=nn.SiLU(),
        transformer_residual=True,
        num_attention_heads=4,
        norm_to_apply="batch",
        dropout_rate=0.1,
        num_layers=4,
        **kwargs,
    ):
        """Graph Transformer Layer

        Parameters
        ----------
        in_channels : int
                Input channel size for nodes.
        edge_features : int
                Input channel size for edges.
        num_hidden_channels : int
                Hidden channel size for both nodes and edges.
        activ_fn : Module
                Activation function to apply in MLPs.
        transformer_residual : bool
                Whether to use a transformer-residual update strategy for node features.
        num_attention_heads : int
                How many attention heads to apply to the input node features in parallel.
        norm_to_apply : str
                Which normalization scheme to apply to node and edge representations (i.e. 'batch' or 'layer').
        dropout_rate : float
                How much dropout (i.e. forget rate) to apply before activation functions.
        num_layers : int
                How many layers of geometric attention to apply.
        """
        super(GraphTransformer, self).__init__()

        # initialize model parameters
        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # initializer modules
        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)

        # transformer module
        num_intermediate_layers = max(0, num_layers - 1)  # every layer except the last one
        gt_block_modules = [
            GraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=transformer_residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate,
                num_layers=num_layers,
                update_edge_feats=True,
            )
            for _ in range(num_intermediate_layers)
        ]
        if num_layers > 0:
            # the last layer only needs to update node features, not edges
            gt_block_modules.append(
                GraphTransformerModule(
                    num_hidden_channels=num_hidden_channels,
                    activ_fn=activ_fn,
                    residual=transformer_residual,
                    num_attention_heads=num_attention_heads,
                    norm_to_apply=norm_to_apply,
                    dropout_rate=dropout_rate,
                    num_layers=num_layers,
                    update_edge_feats=False,
                )
            )
        self.gt_block = nn.ModuleList(gt_block_modules)

    def forward(self, graph, node_feats, edge_feats):
        node_feats = self.node_encoder(node_feats)
        edge_feats = self.edge_encoder(edge_feats)

        # apply a given number of intermediate graph attention layers to the node and edge features given
        for gt_layer in self.gt_block[:-1]:
            node_feats, edge_feats = gt_layer(graph, node_feats, edge_feats)

        # apply final layer to update node representations by merging current node and edge representations
        node_feats = self.gt_block[-1](graph, node_feats, edge_feats)
        return node_feats


def to_dense_batch_dgl(graph_batch, feats, fill_value=0):
    """Turn a disjoint-union-batched graph's flat features into a dense tensor, padded with fill_value, plus a boolean mask marking which entries are real nodes."""
    batch_size = graph_batch["batch_size"]

    if batch_size == 1:
        # nothing to pad, a single graph is already "dense"
        out = feats.unsqueeze(0)
        mask = th.ones(1, feats.size(0), dtype=th.bool, device=feats.device)
        return out, mask

    batch_num_nodes = graph_batch["batch_num_nodes"]
    device = feats.device

    max_num_nodes = int(batch_num_nodes.max())  # the size every graph gets padded up to

    # batch indices for each node. batch_num_nodes is e.g. [3, 4] -> batch is [0, 0, 0, 1, 1, 1, 1]
    batch = th.repeat_interleave(
        th.arange(batch_num_nodes.shape[0], dtype=th.long, device=device),
        batch_num_nodes,
    )

    # where each node lands in the padded [batch_size * max_num_nodes] layout: its slot within its own graph, offset by that graph's position in the batch
    cum_nodes = th.cat([batch.new_zeros(1), batch_num_nodes.cumsum(dim=0)])
    idx = th.arange(feats.size(0), dtype=th.long, device=device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    feat_dims = list(feats.size())[1:]
    out = feats.new_full([batch_size * max_num_nodes] + feat_dims, fill_value)  # start with an all-padding canvas the right size
    out[idx] = feats  # drop the real values into their padded positions
    out = out.view([batch_size, max_num_nodes] + feat_dims)

    mask = th.zeros(batch_size * max_num_nodes, dtype=th.bool, device=device)
    mask[idx] = 1  # mark those same positions as real (not padding)
    mask = mask.view(batch_size, max_num_nodes)
    return out, mask


class RTMScore(nn.Module):
    """Predicts a protein-ligand binding score from a pair of graph encoders.

    Encodes the ligand (atoms) and protein pocket (residues) separately via GraphTransformer, combines every ligand-atom/protein-residue pair, and feeds each pair through a mixture density network (MDN) that predicts a distribution over their distance.

    The final score aggregates these per-pair probabilities (see utils.run_an_eval_epoch)."""

    def __init__(
        self,
        lig_model,
        prot_model,
        in_channels,
        hidden_dim,
        n_gaussians,
        dropout_rate=0.15,
        dist_threshold=1000,
    ):
        super(RTMScore, self).__init__()

        self.lig_model = lig_model
        self.prot_model = prot_model
        self.MLP = nn.Sequential(  # combines each ligand-atom/protein-residue pair's features before the prediction heads
            nn.Linear(in_channels * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
        )
        # MDN head: for each ligand-atom/protein-residue pair, predicts the mixture weights (pi), std devs (sigma), and means (mu) of a Gaussian mixture over their distance
        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)
        # auxiliary heads for the training-time cross-entropy losses
        self.atom_types = nn.Linear(in_channels, 17)
        self.bond_types = nn.Linear(in_channels * 2, 4)

        self.dist_threshold = dist_threshold

    def forward(self, protein_batch, ligand_batch):
        h_l = self.lig_model(  # encode the ligand graph into per-atom representations
            ligand_batch,
            ligand_batch["ndata_atom"].float(),
            ligand_batch["edata_bond"].float(),
        )
        h_p = self.prot_model(  # encode the protein graph into per-residue representations
            protein_batch,
            protein_batch["ndata_feats"].float(),
            protein_batch["edata_feats"].float(),
        )

        # dense-pad both batches so ligand atoms and protein residues can be combined pairwise below
        h_l_x, l_mask = to_dense_batch_dgl(ligand_batch, h_l)
        h_p_x, p_mask = to_dense_batch_dgl(protein_batch, h_p)
        h_l_pos, _ = to_dense_batch_dgl(ligand_batch, ligand_batch["ndata_pos"])
        h_p_pos, _ = to_dense_batch_dgl(protein_batch, protein_batch["ndata_pos"])

        (B, N_l, _), N_p = h_l_x.size(), h_p_x.size(1)
        self.B = B  # stash for later use in compute_euclidean_distances_matrix
        self.N_l = N_l  # stash for later use in compute_euclidean_distances_matrix

        # combine and mask: broadcast into [B, N_l, N_p, C_out] so every ligand atom is paired with every protein residue in the same graph.
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_p, 1)  # [B, N_l, N_t, C_out]

        h_p_x = h_p_x.unsqueeze(-3)
        h_p_x = h_p_x.repeat(1, N_l, 1, 1)  # [B, N_l, N_t, C_out]

        C = th.cat((h_l_x, h_p_x), -1)
        C_mask = l_mask.view(B, N_l, 1) & p_mask.view(B, 1, N_p)
        C = C[C_mask]  # drop the padding positions, keeping only real atom/residue pairs
        C = self.MLP(C)

        # get batch indexes for ligand-target combined features
        C_batch = th.arange(B, device=C_mask.device).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.repeat(1, N_l, N_p)[C_mask]

        # outputs
        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C)) + 1.1
        mu = F.elu(self.z_mu(C)) + 1
        atom_types = self.atom_types(h_l)
        bond_types = self.bond_types(
            th.cat(
                [
                    h_l[ligand_batch["edge_index"][0]],
                    h_l[ligand_batch["edge_index"][1]],
                ],
                axis=1,
            )
        )

        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_p_pos.view(B, -1, 3))[
            C_mask
        ]  # distance between every kept ligand-atom/protein-residue pair
        return (
            pi,
            sigma,
            mu,
            dist.unsqueeze(1).detach(),
            atom_types,
            bond_types,
            C_batch,
        )

    def compute_euclidean_distances_matrix(self, X, Y):
        """Pairwise L2 distance between every X row and every Y row (per batch), collapsed to the minimum over each residue's up-to-24 padded atom slots."""
        X = X.double()  # use higher precision to avoid rounding errors below
        Y = Y.double()  # use higher precision to avoid rounding errors below

        dists = (
            -2 * th.bmm(X, Y.permute(0, 2, 1))
            + th.sum(Y**2, axis=-1).unsqueeze(1)
            + th.sum(X**2, axis=-1).unsqueeze(-1)
        )  # expand the squared-distance formula instead of subtracting directly, for speed
        return th.nan_to_num((dists**0.5).view(self.B, self.N_l, -1, 24), 10000).min(
            axis=-1
        )[0]  # replace missing atoms with a large number so they never win, then keep the closest atom per residue
