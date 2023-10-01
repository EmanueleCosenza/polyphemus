import itertools
import os

import torch
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data.collate import collate

import constants
from constants import EdgeTypes


def get_node_labels(s_tensor, ones_idxs):
    # Build a tensor which has node labels in place of each activation in the
    # stucture tensor
    labels = torch.zeros_like(s_tensor, dtype=torch.long, 
                              device=s_tensor.device)
    n_nodes = len(ones_idxs[0])
    labels[ones_idxs] = torch.arange(n_nodes, device=s_tensor.device)
    return labels


def get_track_edges(s_tensor, ones_idxs=None, node_labels=None):

    track_edges = []

    if ones_idxs is None:
        # Indices where the binary structure tensor is active
        ones_idxs = torch.nonzero(s_tensor, as_tuple=True)

    if node_labels is None:
        node_labels = get_node_labels(s_tensor, ones_idxs)

    # For each track, add direct and inverse edges between consecutive nodes
    for track in range(s_tensor.size(0)):
        # List of active timesteps in the current track
        tss = list(ones_idxs[1][ones_idxs[0] == track])
        edge_type = EdgeTypes.TRACK.value + track
        edges = [
            # Edge tuple: (u, v, type, ts_distance). Zip is used to obtain
            # consecutive active timesteps. Edges in different tracks have
            # different types.
            (node_labels[track, t1],
             node_labels[track, t2], edge_type, t2 - t1)
            for t1, t2 in zip(tss[:-1], tss[1:])
        ]
        inverse_edges = [(u, v, t, d) for (v, u, t, d) in edges]
        track_edges.extend(edges + inverse_edges)

    return torch.tensor(track_edges, dtype=torch.long)


def get_onset_edges(s_tensor, ones_idxs=None, node_labels=None):

    onset_edges = []
    edge_type = EdgeTypes.ONSET.value

    if ones_idxs is None:
        # Indices where the binary structure tensor is active
        ones_idxs = torch.nonzero(s_tensor, as_tuple=True)

    if node_labels is None:
        node_labels = get_node_labels(s_tensor, ones_idxs)

    # Add direct and inverse edges between nodes played in the same timestep
    for ts in range(s_tensor.size(1)):
        # List of active tracks in the current timestep
        tracks = list(ones_idxs[0][ones_idxs[1] == ts])
        # Obtain all possible pairwise combinations of active tracks
        combinations = list(itertools.combinations(tracks, 2))
        edges = [
            # Edge tuple: (u, v, type, ts_distance(=0)).
            (node_labels[track1, ts], node_labels[track2, ts], edge_type, 0)
            for track1, track2 in combinations
        ]
        inverse_edges = [(u, v, t, d) for (v, u, t, d) in edges]
        onset_edges.extend(edges + inverse_edges)

    return torch.tensor(onset_edges, dtype=torch.long)


def get_next_edges(s_tensor, ones_idxs=None, node_labels=None):

    next_edges = []
    edge_type = EdgeTypes.NEXT.value

    if ones_idxs is None:
        # Indices where the binary structure tensor is active
        ones_idxs = torch.nonzero(s_tensor, as_tuple=True)

    if node_labels is None:
        node_labels = get_node_labels(s_tensor, ones_idxs)

    # List of active timesteps
    tss = torch.nonzero(torch.any(s_tensor.bool(), dim=0)).squeeze()
    if tss.dim() == 0:
        return torch.tensor([], dtype=torch.long)

    for i in range(tss.size(0)-1):
        # Get consecutive active timesteps
        t1, t2 = tss[i], tss[i+1]
        # Get all the active tracks in the two timesteps
        t1_tracks = ones_idxs[0][ones_idxs[1] == t1]
        t2_tracks = ones_idxs[0][ones_idxs[1] == t2]

        # Combine the source and destination tracks, removing combinations with
        # the same source and destination track (since these represent track
        # edges).
        tracks_product = list(itertools.product(t1_tracks, t2_tracks))
        tracks_product = [(track1, track2)
                          for (track1, track2) in tracks_product
                          if track1 != track2]
        # Edge tuple: (u, v, type, ts_distance).
        edges = [(node_labels[track1, t1], node_labels[track2, t2],
                  edge_type, t2 - t1)
                 for track1, track2 in tracks_product]

        next_edges.extend(edges)

    return torch.tensor(next_edges, dtype=torch.long)


def get_track_features(s_tensor):

    # Indices where the binary structure tensor is active
    ones_idxs = torch.nonzero(s_tensor)

    n_nodes = len(ones_idxs)
    tracks = ones_idxs[:, 0]
    n_tracks = s_tensor.size(0)

    # The feature n_nodes x n_tracks tensor contains one-hot tracks
    # representations for each node
    features = torch.zeros((n_nodes, n_tracks))
    features[torch.arange(n_nodes), tracks] = 1

    return features


def graph_from_tensor(s_tensor):

    bars = []

    # Iterate over bars and construct a graph for each bar
    for i in range(s_tensor.size(0)):

        bar = s_tensor[i]

        # If the bar contains no activations, add a fake one to avoid having 
        # to deal with empty graphs
        if not torch.any(bar):
            bar[0, 0] = 1

        # Get edges from boolean activations
        track_edges = get_track_edges(bar)
        onset_edges = get_onset_edges(bar)
        next_edges = get_next_edges(bar)
        edges = [track_edges, onset_edges, next_edges]

        # Concatenate edge tensors (N x 4) (if any)
        is_edgeless = (len(track_edges) == 0 and
                       len(onset_edges) == 0 and
                       len(next_edges) == 0)
        if not is_edgeless:
            edge_list = torch.cat([x for x in edges
                                   if torch.numel(x) > 0])

        # Adapt tensor to torch_geometric's Data
        # If no edges, add fake self-edge
        # edge_list[:, :2] contains source and destination node labels
        # edge_list[:, 2:] contains edge types and timestep distances
        edge_index = (edge_list[:, :2].t().contiguous() if not is_edgeless else
                      torch.LongTensor([[0], [0]]))
        attrs = (edge_list[:, 2:] if not is_edgeless else
                 torch.Tensor([[0, 0]]))

        # Add one hot timestep distance to edge attributes
        edge_attrs = torch.zeros(attrs.size(0), s_tensor.shape[-1] + 1)
        edge_attrs[:, 0] = attrs[:, 0]
        edge_attrs[torch.arange(edge_attrs.size(0)),
                   attrs.long()[:, 1] + 1] = 1

        node_features = get_track_features(bar)
        is_drum = node_features[:, 0].bool()
        num_nodes = torch.sum(bar, dtype=torch.long)

        bars.append(Data(edge_index=edge_index, edge_attrs=edge_attrs,
                         num_nodes=num_nodes, node_features=node_features,
                         is_drum=is_drum).to(s_tensor.device))

    # Merge the graphs corresponding to different bars into a single big graph
    graph, _, _ = collate(
        Data,
        data_list=bars,
        increment=True,
        add_batch=True
    )

    # Change bars assignment vector name (otherwise, Dataloader's collate
    # would overwrite graphs.batch)
    graph.bars = graph.batch

    return graph


class PolyphemusDataset(Dataset):

    def __init__(self, dir, n_bars=2):
        self.dir = dir
        self.files = list(os.scandir(self.dir))
        self.len = len(self.files)
        self.n_bars = n_bars

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        # Load tensors
        sample_path = os.path.join(self.dir, self.files[idx].name)
        data = np.load(sample_path)
        c_tensor = torch.tensor(data["c_tensor"], dtype=torch.long)
        s_tensor = torch.tensor(data["s_tensor"], dtype=torch.bool)

        # From (n_tracks x n_timesteps x ...)
        # to (n_bars x n_tracks x n_timesteps x ...)
        c_tensor = c_tensor.reshape(c_tensor.shape[0], self.n_bars, -1,
                                    c_tensor.shape[2], c_tensor.shape[3])
        c_tensor = c_tensor.permute(1, 0, 2, 3, 4)
        s_tensor = s_tensor.reshape(s_tensor.shape[0], self.n_bars, -1)
        s_tensor = s_tensor.permute(1, 0, 2)

        # From decimals to onehot (pitches)
        pitches = c_tensor[..., 0]
        onehot_p = torch.zeros(
            (pitches.shape[0]*pitches.shape[1]*pitches.shape[2]*pitches.shape[3],
             constants.N_PITCH_TOKENS),
            dtype=torch.float32
        )
        onehot_p[torch.arange(0, onehot_p.shape[0]), pitches.reshape(-1)] = 1.
        onehot_p = onehot_p.reshape(pitches.shape[0], pitches.shape[1],
                                    pitches.shape[2], pitches.shape[3],
                                    constants.N_PITCH_TOKENS)

        # From decimals to onehot (durations)
        durs = c_tensor[..., 1]
        onehot_d = torch.zeros(
            (durs.shape[0]*durs.shape[1]*durs.shape[2]*durs.shape[3],
             constants.N_DUR_TOKENS),
            dtype=torch.float32
        )
        onehot_d[torch.arange(0, onehot_d.shape[0]), durs.reshape(-1)] = 1.
        onehot_d = onehot_d.reshape(durs.shape[0], durs.shape[1],
                                    durs.shape[2], durs.shape[3],
                                    constants.N_DUR_TOKENS)

        # Concatenate pitches and durations
        c_tensor = torch.cat((onehot_p, onehot_d), dim=-1)

        # Build graph structure from structure tensor
        graph = graph_from_tensor(s_tensor)

        # Filter silences in order to get a sparse representation
        c_tensor = c_tensor.reshape(-1, c_tensor.shape[-2], c_tensor.shape[-1])
        c_tensor = c_tensor[s_tensor.reshape(-1).bool()]

        graph.c_tensor = c_tensor
        graph.s_tensor = s_tensor.float()

        return graph
