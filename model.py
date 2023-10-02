from typing import Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_sparse import SparseTensor, masked_select_nnz
from torch_geometric.typing import OptTensor, Adj
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.data import Batch
from torch_geometric.nn.conv import RGCNConv

import constants
from data import graph_from_tensor


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


class GCL(RGCNConv):

    def __init__(self, in_channels, out_channels, num_relations, nn,
                 dropout=0.1, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         num_relations=num_relations, **kwargs)
        self.nn = nn
        self.dropout = dropout

        self.reset_edge_nn()

    def reset_edge_nn(self):
        reset(self.nn)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None,
                edge_attr: OptTensor = None):

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

        # Use edge nn to compute weight tensor from edge attributes
        # (=onehot timestep distances between nodes)
        weights = self.nn(edge_attr)
        weights = weights[..., :self.in_channels_l]
        weights = weights.view(-1, self.in_channels_l)

        out = x_j * weights
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out


class MLP(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=256, output_dim=256,
                 num_layers=2, activation=True, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Input layer (1) + Intermediate layers (n-2) + Output layer (1)
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.activation = activation
        self.p = dropout

    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(x, p=self.p, training=self.training)
            x = layer(x)
            if self.activation:
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

        self.layers.append(GCL(input_dim, hidden_dim, num_relations, edge_nn))
        if self.batch_norm:
            self.norm_layers.append(BatchNorm(hidden_dim))

        for i in range(n_layers-1):
            self.layers.append(GCL(hidden_dim, hidden_dim,
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
                # From (4 x 32) to (8 x 4 x 32)
                nn.Conv2d(1, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                # From (8 x 4 x 32) to (8 x 4 x 8)
                nn.MaxPool2d((1, 4), stride=(1, 4)),
                # From (8 x 4 x 8) to (16 x 4 x 8)
                nn.Conv2d(8, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d((1, 4), stride=(1, 4)),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.ReLU(True)
            )

        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        self.lin = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(16 * 4 * 8, dense_dim),
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
            nn.Linear(dense_dim, 16 * 4 * 8),
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
        x = x.unsqueeze(1)
        return x


class ContentEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Pitch and duration embedding layers (separate layers for drums
        # and non drums)
        self.non_drums_pitch_emb = nn.Linear(constants.N_PITCH_TOKENS, 
                                             self.d//2)
        self.drums_pitch_emb = nn.Linear(constants.N_PITCH_TOKENS, self.d//2)
        self.dur_emb = nn.Linear(constants.N_DUR_TOKENS, self.d//2)

        # Batch norm layers
        self.bn_non_drums = nn.BatchNorm1d(num_features=self.d//2)
        self.bn_drums = nn.BatchNorm1d(num_features=self.d//2)
        self.bn_dur = nn.BatchNorm1d(num_features=self.d//2)

        self.chord_encoder = nn.Linear(
            self.d * (constants.MAX_SIMU_TOKENS-1), self.d)

        self.graph_encoder = GCN(
            dropout=self.dropout,
            input_dim=self.d,
            hidden_dim=self.d,
            n_layers=self.gnn_n_layers,
            num_relations=constants.N_EDGE_TYPES,
            batch_norm=self.batch_norm
        )

        # Soft attention node-aggregation layer
        gate_nn = nn.Sequential(
            MLP(input_dim=self.d, output_dim=1, num_layers=1,
                activation=False, dropout=self.dropout),
            nn.BatchNorm1d(1)
        )
        self.graph_attention = GlobalAttention(gate_nn)

        self.bars_encoder = nn.Linear(self.n_bars * self.d, self.d)
    
    def forward(self, graph):
        
        c_tensor = graph.c_tensor

        # Discard SOS token
        c_tensor = c_tensor[:, 1:, :]

        # Get drums and non drums tensors
        drums = c_tensor[graph.is_drum]
        non_drums = c_tensor[torch.logical_not(graph.is_drum)]

        # Compute drums embeddings
        sz = drums.size()
        drums_pitch = self.drums_pitch_emb(
            drums[..., :constants.N_PITCH_TOKENS])
        drums_pitch = self.bn_drums(drums_pitch.view(-1, self.d//2))
        drums_pitch = drums_pitch.view(sz[0], sz[1], self.d//2)
        drums_dur = self.dur_emb(drums[..., constants.N_PITCH_TOKENS:])
        drums_dur = self.bn_dur(drums_dur.view(-1, self.d//2))
        drums_dur = drums_dur.view(sz[0], sz[1], self.d//2)
        drums = torch.cat((drums_pitch, drums_dur), dim=-1)
        # n_nodes x MAX_SIMU_TOKENS x d

        # Compute non drums embeddings
        sz = non_drums.size()
        non_drums_pitch = self.non_drums_pitch_emb(
            non_drums[..., :constants.N_PITCH_TOKENS]
        )
        non_drums_pitch = self.bn_non_drums(non_drums_pitch.view(-1, self.d//2))
        non_drums_pitch = non_drums_pitch.view(sz[0], sz[1], self.d//2)
        non_drums_dur = self.dur_emb(non_drums[..., constants.N_PITCH_TOKENS:])
        non_drums_dur = self.bn_dur(non_drums_dur.view(-1, self.d//2))
        non_drums_dur = non_drums_dur.view(sz[0], sz[1], self.d//2)
        non_drums = torch.cat((non_drums_pitch, non_drums_dur), dim=-1)
        # n_nodes x MAX_SIMU_TOKENS x d

        # Compute chord embeddings (drums and non drums)
        drums = self.chord_encoder(
            drums.view(-1, self.d * (constants.MAX_SIMU_TOKENS-1))
        )
        non_drums = self.chord_encoder(
            non_drums.view(-1, self.d * (constants.MAX_SIMU_TOKENS-1))
        )
        drums = F.relu(drums)
        non_drums = F.relu(non_drums)
        drums = self.dropout_layer(drums)
        non_drums = self.dropout_layer(non_drums)
        # n_nodes x d

        # Merge drums and non drums
        out = torch.zeros((c_tensor.size(0), self.d), device=self.device,
                          dtype=drums.dtype)
        out[graph.is_drum] = drums
        out[torch.logical_not(graph.is_drum)] = non_drums
        # n_nodes x d

        # Set initial graph node states to intermediate chord representations 
        # and pass through GCN
        graph.x = out
        graph.distinct_bars = graph.bars + self.n_bars*graph.batch
        out = self.graph_encoder(graph)
        # n_nodes x d

        # Aggregate final node states into bar encodings with soft attention
        with torch.cuda.amp.autocast(enabled=False):
            out = self.graph_attention(out, batch=graph.distinct_bars)
        # bs x n_bars x d

        out = out.view(-1, self.n_bars * self.d)
        # bs x (n_bars*d)
        z_c = self.bars_encoder(out)
        # bs x d
        
        return z_c


class StructureEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.cnn_encoder = CNNEncoder(
            dense_dim=self.d,
            output_dim=self.d,
            dropout=self.dropout,
            batch_norm=self.batch_norm
        )
        self.bars_encoder = nn.Linear(self.n_bars * self.d, self.d)
    
    def forward(self, graph):
        
        s_tensor = graph.s_tensor
        out = self.cnn_encoder(s_tensor.view(-1, constants.N_TRACKS,
                                             self.resolution * 4))
        # (bs*n_bars) x d
        out = out.view(-1, self.n_bars * self.d)
        # bs x (n_bars*d)
        z_s = self.bars_encoder(out)
        # bs x d

        return z_s
    

class Encoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.s_encoder = StructureEncoder(**kwargs)
        self.c_encoder = ContentEncoder(**kwargs)

        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Linear layer that merges content and structure representations
        self.linear_merge = nn.Linear(2*self.d, self.d)
        self.bn_linear_merge = nn.BatchNorm1d(num_features=self.d)

        self.linear_mu = nn.Linear(self.d, self.d)
        self.linear_log_var = nn.Linear(self.d, self.d)

    def forward(self, graph):
        
        z_s = self.s_encoder(graph)
        z_c = self.c_encoder(graph)
        
        # Merge content and structure representations
        z_g = torch.cat((z_c, z_s), dim=1)
        z_g = self.dropout_layer(z_g)
        z_g = self.linear_merge(z_g)
        z_g = self.bn_linear_merge(z_g)
        z_g = F.relu(z_g)

        # Compute mu and log(std^2)
        z_g = self.dropout_layer(z_g)
        mu = self.linear_mu(z_g)
        log_var = self.linear_log_var(z_g)

        return mu, log_var


class StructureDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.bars_decoder = nn.Linear(self.d, self.d * self.n_bars)
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
        out = out.view(z_s.size(0), self.n_bars, constants.N_TRACKS, -1)
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
            num_relations=constants.N_EDGE_TYPES,
            batch_norm=self.batch_norm
        )

        self.chord_decoder = nn.Linear(
            self.d, self.d*(constants.MAX_SIMU_TOKENS-1))

        # Pitch and duration (un)embedding linear layers
        self.drums_pitch_emb = nn.Linear(self.d//2, constants.N_PITCH_TOKENS)
        self.non_drums_pitch_emb = nn.Linear(
            self.d//2, constants.N_PITCH_TOKENS)
        self.dur_emb = nn.Linear(self.d//2, constants.N_DUR_TOKENS)

        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, z_c, s):

        out = self.bars_decoder(z_c)  # bs x (n_bars*d)

        # Initialize node features with corresponding z_bar
        # and propagate with GNN
        s.distinct_bars = s.bars + self.n_bars*s.batch
        _, counts = torch.unique(s.distinct_bars, return_counts=True)
        out = out.view(-1, self.d)
        out = torch.repeat_interleave(out, counts, axis=0)  # n_nodes x d
        s.x = out
        out = self.graph_decoder(s)  # n_nodes x d

        out = self.chord_decoder(out)  # n_nodes x (MAX_SIMU_TOKENS*d)
        out = out.view(-1, constants.MAX_SIMU_TOKENS-1, self.d)

        drums = out[s.is_drum]  # n_nodes_drums x MAX_SIMU_TOKENS x d
        non_drums = out[torch.logical_not(s.is_drum)]
        # n_nodes_non_drums x MAX_SIMU_TOKENS x d

        # Obtain final pitch and dur logits (softmax will be applied
        # outside forward)
        non_drums = self.dropout_layer(non_drums)
        drums = self.dropout_layer(drums)

        drums_pitch = self.drums_pitch_emb(drums[..., :self.d//2])
        drums_dur = self.dur_emb(drums[..., self.d//2:])
        drums = torch.cat((drums_pitch, drums_dur), dim=-1)
        # n_nodes_drums x MAX_SIMU_TOKENS x d_token

        non_drums_pitch = self.non_drums_pitch_emb(non_drums[..., :self.d//2])
        non_drums_dur = self.dur_emb(non_drums[..., self.d//2:])
        non_drums = torch.cat((non_drums_pitch, non_drums_dur), dim=-1)
        # n_nodes_non_drums x MAX_SIMU_TOKENS x d_token

        # Merge drums and non-drums in the final output tensor
        d_token = constants.D_TOKEN_PAIR
        out = torch.zeros((s.num_nodes, constants.MAX_SIMU_TOKENS-1, d_token),
                          device=self.device, dtype=drums.dtype)
        out[s.is_drum] = drums
        out[torch.logical_not(s.is_drum)] = non_drums

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
            s.append(graph_from_tensor(s_tensor[i]))

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
        
        # Avoid empty bars by creating a fake activation for each empty
        # (n_tracks x n_timesteps) bar matrix in position [0, 0]
        empty_mask = ~s_tensor.any(dim=-1).any(dim=-1)
        idxs = torch.nonzero(empty_mask, as_tuple=True)
        s_tensor[idxs + (0, 0)] = True

        return s_tensor

    def _structure_from_logits(self, s_logits):

        # Compute binary structure tensor from logits and build torch geometric
        # structure from binary tensor
        s_tensor = self._binary_from_logits(s_logits)
        s = self._structure_from_binary(s_tensor)

        return s

    def forward(self, z, s=None):

        # Obtain z_s and z_c from z
        z = self.lin_decoder(z)
        z = self.batch_norm(z)
        z = F.relu(z)
        z = self.dropout(z)  # bs x (2*d)
        z_s, z_c = z[:, :self.d], z[:, self.d:]

        # Obtain the tensor containing structure logits
        s_logits = self.s_decoder(z_s)

        if s is None:
            # Build torch geometric graph structure from structure logits.
            # This step involves non differentiable operations.
            # No gradients pass through here.
            s = self._structure_from_logits(s_logits.detach())

        # Obtain the tensor containing content logits
        c_logits = self.c_decoder(z_c, s)

        return s_logits, c_logits


class VAE(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)

    def forward(self, graph):

        # Encoder pass
        mu, log_var = self.encoder(graph)

        # Reparameterization trick
        z = torch.exp(0.5 * log_var)
        z = z * torch.randn_like(z)
        z = z + mu

        # Decoder pass
        out = self.decoder(z, graph)

        return out, mu, log_var
