from typing import Callable, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Parameter as Param
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul, masked_select_nnz
from torch_geometric.typing import OptTensor, Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, glorot, zeros
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.data import Batch

from data import graph_from_tensor_torch


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')


def masked_edge_attrs(edge_attrs, edge_mask):
    return edge_attrs[edge_mask, :]


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    .. note::
        This implementation is as memory-efficient as possible by iterating
        over each individual relation type.
        Therefore, it may result in low GPU utilization in case the graph has a
        large number of relations.
        As an alternative approach, :class:`FastRGCNConv` does not iterate over
        each individual type, but may consume a large amount of memory to
        compensate.
        We advise to check out both implementations to see which one fits your
        needs.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
            In case no input features are given, this argument should
            correspond to the number of nodes in your graph.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        num_bases (int, optional): If set to not :obj:`None`, this layer will
            use the basis-decomposition regularization scheme where
            :obj:`num_bases` denotes the number of bases to use.
            (default: :obj:`None`)
        num_blocks (int, optional): If set to not :obj:`None`, this layer will
            use the block-diagonal-decomposition regularization scheme where
            :obj:`num_blocks` denotes the number of blocks to use.
            (default: :obj:`None`)
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        nn: Callable,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        dropout: Optional[float] = 0.1,
        aggr: str = 'mean',
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.dropout = dropout
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        if num_bases is not None:
            self.weight = Parameter(
                torch.Tensor(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))

        elif num_blocks is not None:
            assert (in_channels[0] % num_blocks == 0
                    and out_channels % num_blocks == 0)
            self.weight = Parameter(
                torch.Tensor(num_relations, num_blocks,
                             in_channels[0] // num_blocks,
                             out_channels // num_blocks))
            self.register_parameter('comp', None)

        else:
            self.weight = Parameter(
                torch.Tensor(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)

        if root_weight:
            self.root = Param(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        reset(self.nn)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None,
                edge_attr: OptTensor = None):
        r"""
        Args:
            x: The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or an optional
                one-dimensional node index tensor (in which case input features
                are treated as trainable node embeddings).
                Furthermore, :obj:`x` can be of type :obj:`tuple` denoting
                source and destination node features.
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
        """

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        weight = self.weight

        # Basis-decomposition
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        # Block-diagonal-decomposition
        if self.num_blocks is not None:

            if x_l.dtype == torch.long and self.num_blocks is not None:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out += h.contiguous().view(-1, self.out_channels)

        else:
            # No regularization/Basis-decomposition
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                attr = masked_edge_attrs(edge_attr, edge_type == i)

                if x_l.dtype == torch.long:
                    out += self.propagate(tmp, x=weight[i, x_l], size=size)
                else:
                    h = self.propagate(tmp, x=x_l, size=size,
                                       edge_attr=attr)
                    out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        weights = self.nn(edge_attr)
        weights = weights[..., :self.in_channels_l]
        weights = weights.view(-1, self.in_channels_l)
        ret = x_j * weights
        ret = F.relu(ret)
        ret = F.dropout(ret, p=self.dropout, training=self.training)
        return ret

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')


class MLP(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=256, output_dim=256,
                 num_layers=2, act=True, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))

            for i in range(num_layers-2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.act = act
        self.p = dropout

    def forward(self, x):

        for layer in self.layers:
            x = F.dropout(x, p=self.p, training=self.training)
            x = layer(x)
            if self.act:
                x = F.relu(x)

        return x


class GCN(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=256, n_layers=3,
                 num_relations=3, num_dists=32, batch_norm=False, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        edge_nn = nn.Linear(num_dists, input_dim)
        self.batch_norm = batch_norm

        self.layers.append(RGCNConv(input_dim, hidden_dim,
                                    num_relations, edge_nn))
        if self.batch_norm:
            self.norm_layers.append(BatchNorm(hidden_dim))

        for i in range(n_layers-1):
            self.layers.append(RGCNConv(hidden_dim, hidden_dim,
                                        num_relations, edge_nn))
            if self.batch_norm:
                self.norm_layers.append(BatchNorm(hidden_dim))

        self.p = dropout

    def forward(self, data):
        x, edge_index, edge_attrs = data.x, data.edge_index, data.edge_attrs
        edge_type = edge_attrs[:, 0]
        edge_attr = edge_attrs[:, 1:]

        for i in range(len(self.layers)):
            residual = x
            x = F.dropout(x, p=self.p, training=self.training)
            x = self.layers[i](x, edge_index, edge_type, edge_attr)
            if self.batch_norm:
                x = self.norm_layers[i](x)
            x = F.relu(x)
            x = residual + x

        return x


class CNNEncoder(nn.Module):

    def __init__(self, output_dim=256, dense_dim=256, batch_norm=False,
                 dropout=0.1):
        super().__init__()

        # Convolutional layers
        if batch_norm:
            self.conv = nn.Sequential(
                # From 4x32 to 8x4x32
                nn.Conv2d(1, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                # From 8x4x32 to 8x4x8
                nn.MaxPool2d((1, 4), stride=(1, 4)),
                # From 8x4x8 to 16x4x8
                nn.Conv2d(8, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            )
        else:
            self.conv = nn.Sequential(
                # From 4x32 to 8x4x32
                nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(True),
                # From 8x4x32 to 8x4x8
                nn.MaxPool2d((1, 4), stride=(1, 4)),
                # From 8x4x8 to 16x4x8
                nn.Conv2d(8, 16, 3, padding=1),
                nn.ReLU(True)
            )

        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        self.lin = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(16*4*8, dense_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, output_dim)
        )

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.lin(x)

        return x


class CNNDecoder(nn.Module):

    def __init__(self, input_dim=256, dense_dim=256, batch_norm=False,
                 dropout=0.1):
        super().__init__()

        # Linear decompressors
        self.lin = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, dense_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 16*4*8),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(16, 4, 8))

        # Upsample and convolutional layers
        if batch_norm:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=(1, 4), mode='nearest'),
                nn.Conv2d(16, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 1, 3, padding=1)
            )
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=(1, 4), mode='nearest'),
                nn.Conv2d(16, 8, 3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(8, 1, 3, padding=1)
            )

    def forward(self, x):

        x = self.lin(x)
        x = self.unflatten(x)
        x = self.conv(x)

        return x.unsqueeze(1)


class Encoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Pitch and duration embedding layers (separate layers for drums
        # and non drums)
        self.notes_pitch_emb = nn.Linear(self.d_token_pitches, self.d//2)
        self.drums_pitch_emb = nn.Linear(self.d_token_pitches, self.d//2)
        self.dur_emb = nn.Linear(self.d_token_dur, self.d//2)

        # Batch norm layers
        self.bn_npe = nn.BatchNorm1d(num_features=self.d//2)
        self.bn_dpe = nn.BatchNorm1d(num_features=self.d//2)
        self.bn_de = nn.BatchNorm1d(num_features=self.d//2)

        self.chord_encoder = nn.Linear(
            self.d * (self.max_simu_notes-1), self.d)

        # Graph encoder
        self.graph_encoder = GCN(
            dropout=self.dropout,
            input_dim=self.d,
            hidden_dim=self.d,
            n_layers=self.gnn_n_layers,
            num_relations=self.n_relations,
            batch_norm=self.batch_norm
        )

        # Soft attention node-aggregation layer
        gate_nn = nn.Sequential(
            MLP(input_dim=self.d, output_dim=1, num_layers=1,
                act=False, dropout=self.dropout),
            nn.BatchNorm1d(1)
        )
        self.graph_attention = GlobalAttention(gate_nn)

        self.bars_encoder_attr = nn.Linear(self.n_bars*self.d, self.d)

        self.cnn_encoder = CNNEncoder(
            dense_dim=self.d,
            output_dim=self.d,
            dropout=self.dropout,
            batch_norm=self.batch_norm
        )
        self.bars_encoder_struct = nn.Linear(self.n_bars*self.d, self.d)

        self.linear_merge = nn.Linear(2*self.d, self.d)
        self.bn_lm = nn.BatchNorm1d(num_features=self.d)

        # Linear layers that compute the final mu and log_var
        self.linear_mu = nn.Linear(self.d, self.d)
        self.linear_log_var = nn.Linear(self.d, self.d)

    def forward(self, x_seq, x_acts, x_graph):

        # No start of seq token
        x_seq = x_seq[:, 1:, :]

        # Get drums and non drums tensors
        drums = x_seq[x_graph.is_drum]
        non_drums = x_seq[torch.logical_not(x_graph.is_drum)]

        # Compute note/drums embeddings
        s = drums.size()
        drums_pitch = self.drums_pitch_emb(drums[..., :self.d_token_pitches])
        drums_pitch = self.bn_dpe(drums_pitch.view(-1, self.d//2))
        drums_pitch = drums_pitch.view(s[0], s[1], self.d//2)
        drums_dur = self.dur_emb(drums[..., self.d_token_pitches:])
        drums_dur = self.bn_de(drums_dur.view(-1, self.d//2))
        drums_dur = drums_dur.view(s[0], s[1], self.d//2)
        drums = torch.cat((drums_pitch, drums_dur), dim=-1)
        # [n_nodes x max_simu_notes x d]

        s = non_drums.size()
        non_drums_pitch = self.notes_pitch_emb(
            non_drums[..., :self.d_token_pitches]
        )
        non_drums_pitch = self.bn_npe(non_drums_pitch.view(-1, self.d//2))
        non_drums_pitch = non_drums_pitch.view(s[0], s[1], self.d//2)
        non_drums_dur = self.dur_emb(non_drums[..., self.d_token_pitches:])
        non_drums_dur = self.bn_de(non_drums_dur.view(-1, self.d//2))
        non_drums_dur = non_drums_dur.view(s[0], s[1], self.d//2)
        non_drums = torch.cat((non_drums_pitch, non_drums_dur), dim=-1)
        # [n_nodes x max_simu_notes x d]

        # Compute chord embeddings both for drums and non drums
        drums = self.chord_encoder(
            drums.view(-1, self.d * (self.max_simu_notes-1))
        )
        non_drums = self.chord_encoder(
            non_drums.view(-1, self.d * (self.max_simu_notes-1))
        )
        drums = F.relu(drums)
        non_drums = F.relu(non_drums)
        drums = self.dropout_layer(drums)
        non_drums = self.dropout_layer(non_drums)
        # [n_nodes x d]

        # Merge drums and non-drums
        out = torch.zeros((x_seq.size(0), self.d), device=self.device,
                          dtype=torch.half)
        out[x_graph.is_drum] = drums
        out[torch.logical_not(x_graph.is_drum)] = non_drums
        # [n_nodes x d]

        x_graph.x = out
        x_graph.distinct_bars = x_graph.bars + self.n_bars*x_graph.batch
        out = self.graph_encoder(x_graph)
        # [n_nodes x d]

        with torch.cuda.amp.autocast(enabled=False):
            out = self.graph_attention(out, batch=x_graph.distinct_bars)
            # [bs x n_bars x d]

        out = out.view(-1, self.n_bars * self.d)
        # [bs x n_bars * d]
        out_attr = self.bars_encoder_attr(out)
        # [bs x d]

        # Process structure
        out = self.cnn_encoder(x_acts.view(-1, self.n_tracks,
                                           self.resolution * 4))
        # [bs * n_bars x d]
        out = out.view(-1, self.n_bars * self.d)
        # [bs x n_bars * d]
        out_struct = self.bars_encoder_struct(out)
        # [bs x d]

        # Merge attr state and struct state
        out = torch.cat((out_attr, out_struct), dim=1)
        out = self.dropout_layer(out)
        out = self.linear_merge(out)
        out = self.bn_lm(out)
        out = F.relu(out)

        # Compute mu and log(std^2)
        out = self.dropout_layer(out)
        mu = self.linear_mu(out)
        log_var = self.linear_log_var(out)

        return mu, log_var


class StructureDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.bars_decoder = nn.Linear(self.d, self.d*self.n_bars)
        self.cnn_decoder = CNNDecoder(
            input_dim=self.d,
            dense_dim=self.d,
            dropout=self.dropout,
            batch_norm=self.batch_norm
        )

    def forward(self, z_s):
        # z_s: bs x d
        out = self.bars_decoder(z_s)  # bs x (n_bars*d)
        out = self.cnn_decoder(out.reshape(-1, self.d))
        out = out.view(z_s.size(0), self.n_bars, self.n_tracks, -1)
        return out


class ContentDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.bars_decoder = nn.Linear(self.d, self.d * self.n_bars)

        self.graph_decoder = GCN(
            dropout=self.dropout,
            input_dim=self.d,
            hidden_dim=self.d,
            n_layers=self.gnn_n_layers,
            num_relations=self.n_relations,
            batch_norm=self.batch_norm
        )

        self.chord_decoder = nn.Linear(self.d, self.d*(self.max_simu_notes-1))

        # Pitch and duration (un)embedding linear layers
        self.drums_pitch_emb = nn.Linear(self.d//2, self.d_token_pitches)
        self.non_drums_pitch_emb = nn.Linear(self.d//2, self.d_token_pitches)
        self.dur_emb = nn.Linear(self.d//2, self.d_token_dur)

        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, z_c, structure):

        out = self.bars_decoder(z_c)  # bs x (n_bars*d)

        # Initialize node features with corresponding z_bar
        # and propagate with GNN
        structure.distinct_bars = structure.bars + self.n_bars*structure.batch
        _, counts = torch.unique(structure.distinct_bars, return_counts=True)
        out = out.view(-1, self.d)
        out = torch.repeat_interleave(out, counts, axis=0)  # n_nodes x d
        structure.x = out
        out = self.graph_decoder(structure)  # n_nodes x d

        out = self.chord_decoder(out)  # n_nodes x (max_simu_notes*d)
        out = out.view(-1, self.max_simu_notes-1, self.d)

        drums = out[structure.is_drum]  # n_nodes_drums x max_simu_notes x d
        non_drums = out[torch.logical_not(structure.is_drum)]
        # n_nodes_non_drums x max_simu_notes x d

        # Obtain final pitch and dur decodings
        # (softmax to be applied after forward)
        non_drums = self.dropout_layer(non_drums)
        drums = self.dropout_layer(drums)

        drums_pitch = self.drums_pitch_emb(drums[..., :self.d//2])
        drums_dur = self.dur_emb(drums[..., self.d//2:])
        drums = torch.cat((drums_pitch, drums_dur), dim=-1)
        # n_nodes_drums x max_simu_notes x d_token
        non_drums_pitch = self.non_drums_pitch_emb(non_drums[..., :self.d//2])
        non_drums_dur = self.dur_emb(non_drums[..., self.d//2:])
        non_drums = torch.cat((non_drums_pitch, non_drums_dur), dim=-1)
        # n_nodes_non_drums x max_simu_notes x d_token

        # Merge drums and non-drums
        out = torch.zeros((structure.num_nodes, self.max_simu_notes-1,
                           self.d_token), device=self.device)
        out[structure.is_drum] = drums
        out[torch.logical_not(structure.is_drum)] = non_drums

        return out


class Decoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.lin_decoder = nn.Linear(self.d, 2 * self.d)
        self.batch_norm = nn.BatchNorm1d(num_features=2*self.d)
        self.dropout = nn.Dropout(p=self.dropout)

        self.s_decoder = StructureDecoder(**kwargs)
        self.c_decoder = ContentDecoder(**kwargs)

        self.sigmoid_thresh = 0.5

    def _structure_from_binary(self, s_tensor):

        # Create graph structures for each batch
        s = []
        for i in range(s_tensor.size(0)):
            s.append(graph_from_tensor_torch(s_tensor[i]))

        # Create batch of graphs from single graphs
        s = Batch.from_data_list(s, exclude_keys=['batch'])
        s = s.to(next(self.parameters()).device)

        return s

    def _binary_from_logits(self, s_logits):

        # Hard threshold instead of sampling gives more pleasant results
        s_tensor = torch.sigmoid(s_logits)
        s_tensor[s_tensor >= self.sigmoid_thresh] = 1
        s_tensor[s_tensor < self.sigmoid_thresh] = 0
        s_tensor = s_tensor.bool()

        return s_tensor

    def _structure_from_logits(self, s_logits):

        # Compute binary structure tensor from logits and build torch geometric
        # structure from binary tensor
        s_tensor = self._binary_from_logits(s_logits)
        s = self._structure_from_binary(s_tensor)

        return s, s_tensor

    def forward(self, z, s=None):

        # Obtain z_s and z_c from z
        z = self.lin_decoder(z)
        z = self.batch_norm(z)
        z = F.relu(z)
        z = self.dropout(z)  # bs x (2*d)
        z_s, z_c = z[:, :self.d], z[:, self.d:]

        # Obtain the tensor containing structure logits
        s_logits = self.s_decoder(z_s)

        s_tensor = None
        if s is None:
            # Build torch geometric graph structure from structure logits.
            # This step involves non differentiable operations.
            # No gradients pass through here.
            s, s_tensor = self._structure_from_logits(s_logits.detach())

        # Obtain the tensor containing content logits
        c_logits = self.c_decoder(z_c, s)

        return s_logits, c_logits, s_tensor


class VAE(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)

    def forward(self, x_seq, x_acts, x_graph):

        # Encoder pass
        mu, log_var = self.encoder(x_seq, x_acts, x_graph)

        # Reparameterization trick
        z = torch.exp(0.5 * log_var)
        z = z * torch.randn_like(z)
        z = z + mu

        # Decoder pass
        out = self.decoder(z, x_acts, x_graph)

        return out, mu, log_var
