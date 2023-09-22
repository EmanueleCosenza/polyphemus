import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import itertools
from torch_geometric.data.collate import collate
import os
import numpy as np


def get_track_edges(acts, edge_type_ind=0):

    a_t = acts.transpose()
    inds = np.stack(np.where(a_t == 1)).transpose()

    # Create node labels
    labels = np.zeros(acts.shape)
    acts_inds = np.where(acts == 1)
    num_nodes = len(acts_inds[0])
    labels[acts_inds] = np.arange(num_nodes)
    labels = labels.transpose()

    track_edges = []

    for track in range(a_t.shape[1]):
        tr_inds = list(inds[inds[:,1] == track])
        e_inds = [(tr_inds[i],
                tr_inds[i+1]) for i in range(len(tr_inds)-1)]
        edges = [(labels[tuple(e[0])], labels[tuple(e[1])],
                  edge_type_ind+track, e[1][0]-e[0][0]) for e in e_inds]
        inv_edges = [(e[1], e[0], *e[2:]) for e in edges]
        track_edges.extend(edges)
        track_edges.extend(inv_edges)

    return np.array(track_edges, dtype='long')


def get_track_edges_torch(acts, edge_type_ind=0):

    a_t = acts.t()
    inds = torch.stack(torch.where(a_t == 1)).t()

    # Create node labels
    labels = torch.zeros(acts.size())
    acts_inds = torch.where(acts == 1)
    num_nodes = len(acts_inds[0])
    labels[acts_inds] = torch.arange(num_nodes, dtype=torch.float)
    labels = labels.t()

    track_edges = []

    for track in range(a_t.size(1)):
        tr_inds = list(inds[inds[:, 1] == track])
        e_inds = [(tr_inds[i],
                tr_inds[i+1]) for i in range(len(tr_inds)-1)]
        edges = [(labels[tuple(e[0])], labels[tuple(e[1])],
                  edge_type_ind+track, e[1][0]-e[0][0]) for e in e_inds]
        inv_edges = [(e[1], e[0], *e[2:]) for e in edges]
        track_edges.extend(edges)
        track_edges.extend(inv_edges)

    return torch.tensor(track_edges, dtype=torch.long)


def get_onset_edges(acts, edge_type_ind=4):

    a_t = acts.transpose()
    inds = np.stack(np.where(a_t == 1)).transpose()
    ts_acts = np.any(a_t, axis=1)
    ts_inds = np.where(ts_acts)[0]

    # Create node labels
    labels = np.zeros(acts.shape)
    acts_inds = np.where(acts == 1)
    num_nodes = len(acts_inds[0])
    labels[acts_inds] = np.arange(num_nodes)
    labels = labels.transpose()

    onset_edges = []

    for i in ts_inds:
        ts_acts_inds = list(inds[inds[:,0] == i])
        if len(ts_acts_inds) < 2:
            continue
        e_inds = list(itertools.combinations(ts_acts_inds, 2))
        edges = [(labels[tuple(e[0])], labels[tuple(e[1])],
                  edge_type_ind, 0) for e in e_inds]
        inv_edges = [(e[1], e[0], *e[2:]) for e in edges]
        onset_edges.extend(edges)
        onset_edges.extend(inv_edges)

    return np.array(onset_edges, dtype='long')


def get_onset_edges_torch(acts, edge_type_ind=4):

    a_t = acts.t()
    inds = torch.stack(torch.where(a_t == 1)).t()
    ts_acts = torch.any(a_t, dim=1)
    ts_inds = torch.where(ts_acts)[0]

    # Create node labels
    labels = torch.zeros(acts.shape)
    acts_inds = torch.where(acts == 1)
    num_nodes = len(acts_inds[0])
    labels[acts_inds] = torch.arange(num_nodes, dtype=torch.float)
    labels = labels.t()

    onset_edges = []

    for i in ts_inds:
        ts_acts_inds = list(inds[inds[:,0] == i])
        if len(ts_acts_inds) < 2:
            continue
        e_inds = list(itertools.combinations(ts_acts_inds, 2))
        edges = [(labels[tuple(e[0])], labels[tuple(e[1])],
                  edge_type_ind, 0) for e in e_inds]
        inv_edges = [(e[1], e[0], *e[2:]) for e in edges]
        onset_edges.extend(edges)
        onset_edges.extend(inv_edges)

    return torch.tensor(onset_edges, dtype=torch.long)


def get_next_edges(acts, edge_type_ind=5):

    a_t = acts.transpose()
    inds = np.stack(np.where(a_t == 1)).transpose()
    ts_acts = np.any(a_t, axis=1)
    ts_inds = np.where(ts_acts)[0]

    # Create node labels
    labels = np.zeros(acts.shape)
    acts_inds = np.where(acts == 1)
    num_nodes = len(acts_inds[0])
    labels[acts_inds] = np.arange(num_nodes)
    labels = labels.transpose()

    next_edges = []

    for i in range(len(ts_inds)-1):

        ind_s = ts_inds[i]
        ind_e = ts_inds[i+1]
        s = inds[inds[:,0] == ind_s]
        e = inds[inds[:,0] == ind_e]

        e_inds = [t for t in list(itertools.product(s, e)) if t[0][1] != t[1][1]]
        edges = [(labels[tuple(e[0])], labels[tuple(e[1])],
                  edge_type_ind, ind_e-ind_s) for e in e_inds]
        inv_edges = [(e[1], e[0], *e[2:]) for e in edges]

        next_edges.extend(edges)
        next_edges.extend(inv_edges)

    return np.array(next_edges, dtype='long')
    

def get_next_edges_torch(acts, edge_type_ind=5):

    a_t = acts.t()
    inds = torch.stack(torch.where(a_t == 1)).t()
    ts_acts = torch.any(a_t, dim=1)
    ts_inds = torch.where(ts_acts)[0]

    # Create node labels
    labels = torch.zeros(acts.shape)
    acts_inds = torch.where(acts == 1)
    num_nodes = len(acts_inds[0])
    labels[acts_inds] = torch.arange(num_nodes, dtype=torch.float)
    labels = labels.t()

    next_edges = []

    for i in range(len(ts_inds)-1):

        ind_s = ts_inds[i]
        ind_e = ts_inds[i+1]
        s = inds[inds[:,0] == ind_s]
        e = inds[inds[:,0] == ind_e]

        e_inds = [t for t in list(itertools.product(s, e)) if t[0][1] != t[1][1]]
        edges = [(labels[tuple(e[0])], labels[tuple(e[1])],
                  edge_type_ind, ind_e-ind_s) for e in e_inds]
        inv_edges = [(e[1], e[0], *e[2:]) for e in edges]

        next_edges.extend(edges)
        next_edges.extend(inv_edges)

    return torch.tensor(next_edges, dtype=torch.long)
    

def get_node_features(acts, num_nodes):
        
    num_tracks = acts.shape[0]
    features = torch.zeros((num_nodes, num_tracks), dtype=torch.float)
    features[np.arange(num_nodes), np.stack(np.where(acts))[0]] = 1.

    return features


def get_node_features_torch(acts, num_nodes):
        
    num_tracks = acts.size(0)
    features = torch.zeros((num_nodes, num_tracks), dtype=torch.float)
    features[torch.arange(num_nodes), torch.stack(torch.where(acts))[0]] = 1.

    return features
    

def graph_from_tensor(s, force_no_empty=True):
    
    bars = []
        
    # Iterate over bars and construct a graph for each bar
    for i in range(s.shape[0]):
        
        bar = s[i]
        
        if force_no_empty:
            if not np.any(bar):
                bar[0][0] = 1

        # Number of nodes
        n = torch.sum(torch.Tensor(bar), dtype=torch.long)

        # Get edges from boolean activations
        # Todo: optimize and refactor
        track_edges = get_track_edges(bar)
        onset_edges = get_onset_edges(bar)
        next_edges = get_next_edges(bar)
        edges = [track_edges, onset_edges, next_edges]

        # Concatenate edge tensors (N x 4) (if any)
        # First two columns -> source and dest nodes
        # Third column -> edge_type, Fourth column -> timestep distance
        no_edges = (len(track_edges) == 0 and 
                    len(onset_edges) == 0 and len(next_edges) == 0)
        if not no_edges:
            edge_list = np.concatenate([x for x in edges
                                          if x.size > 0])
            edge_list = torch.from_numpy(edge_list)

        # Adapt tensor to torch_geometric's Data
        # No edges: add fictitious self-edge
        edge_index = (torch.LongTensor([[0], [0]]) if no_edges else
                               edge_list[:, :2].t().contiguous())
        attrs = (torch.Tensor([[0, 0]]) if no_edges else
                                       edge_list[:, 2:])

        # One hot timestep distance concatenated to edge type
        edge_attrs = torch.zeros(attrs.size(0), 1+s.shape[-1])
        edge_attrs[:, 0] = attrs[:, 0]
        edge_attrs[np.arange(edge_attrs.size(0)), attrs.long()[:, 1]+1] = 1
        #edge_attrs = torch.Tensor(attrs.float())

        node_features = get_node_features(bar, n)
        is_drum = node_features[:, 0].bool()

        bars.append(Data(edge_index=edge_index, edge_attrs=edge_attrs,
                         num_nodes=n, node_features=node_features,
                         is_drum=is_drum))


    # Merge the graphs corresponding to different bars into a single big graph
    graph, _, inc_dict = collate(
        Data,
        data_list=bars,
        increment=True,
        add_batch=True
    )
        
    # Change bars assignment vector name (otherwise, Dataloader's collate
    # would overwrite graphs.batch)
    graph.bars = graph.batch
    
    return graph
    

def graph_from_tensor_torch(s, force_no_empty=True):
    
    bars = []
        
    # Iterate over bars and construct a graph for each bar
    for i in range(s.size(0)):
        
        bar = s[i]
        
        if force_no_empty:
            if not torch.any(bar):
                bar[0][0] = 1

        # Number of nodes
        n = torch.sum(bar, dtype=torch.long)

        # Get edges from boolean activations
        # Todo: optimize and refactor
        track_edges = get_track_edges_torch(bar)
        onset_edges = get_onset_edges_torch(bar)
        next_edges = get_next_edges_torch(bar)
        edges = [track_edges, onset_edges, next_edges]

        # Concatenate edge tensors (N x 4) (if any)
        # First two columns -> source and dest nodes
        # Third column -> edge_type, Fourth column -> timestep distance
        no_edges = (len(track_edges) == 0 and 
                    len(onset_edges) == 0 and len(next_edges) == 0)
        if not no_edges:
            edge_list = torch.cat([x for x in edges
                                      if torch.numel(x) > 0])

        # Adapt tensor to torch_geometric's Data
        # No edges: add fictitious self-edge
        edge_index = (torch.LongTensor([[0], [0]]) if no_edges else
                               edge_list[:, :2].t().contiguous())
        attrs = (torch.Tensor([[0, 0]]) if no_edges else
                                       edge_list[:, 2:])

        # One hot timestep distance concatenated to edge type
        edge_attrs = torch.zeros(attrs.size(0), 1+s.shape[-1])
        edge_attrs[:, 0] = attrs[:, 0]
        edge_attrs[torch.arange(edge_attrs.size(0)), attrs.long()[:, 1]+1] = 1

        node_features = get_node_features_torch(bar, n)
        is_drum = node_features[:, 0].bool()

        bars.append(Data(edge_index=edge_index, edge_attrs=edge_attrs,
                         num_nodes=n, node_features=node_features,
                         is_drum=is_drum).to(s.device))


    # Merge the graphs corresponding to different bars into a single big graph
    graph, _, inc_dict = collate(
        Data,
        data_list=bars,
        increment=True,
        add_batch=True
    )
        
    # Change bars assignment vector name (otherwise, Dataloader's collate
    # would overwrite graphs.batch)
    graph.bars = graph.batch
    
    return graph



class MIDIDataset(Dataset):

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
        seq_tensor = data["seq_tensor"]
        seq_acts = data["seq_acts"]
        
        # From (#tracks x #timesteps x ...) to (#bars x #tracks x #timesteps x ...)
        seq_tensor = seq_tensor.reshape(seq_tensor.shape[0], self.n_bars, -1,
                                        seq_tensor.shape[2], seq_tensor.shape[3])
        seq_tensor = seq_tensor.transpose(1, 0, 2, 3, 4)
        seq_acts = seq_acts.reshape(seq_acts.shape[0], self.n_bars, -1)
        seq_acts = seq_acts.transpose(1, 0, 2)
        
        # Construct src_key_padding_mask (PAD = 130)
        src_mask = torch.from_numpy((seq_tensor[..., 0] == 130))

        # From decimals to one-hot (pitch)
        pitches = seq_tensor[..., 0]
        onehot_p = np.zeros(
            (pitches.shape[0]*pitches.shape[1]*pitches.shape[2]*pitches.shape[3],
             131), 
            dtype=float
        )
        onehot_p[np.arange(0, onehot_p.shape[0]), pitches.reshape(-1)] = 1.
        onehot_p = onehot_p.reshape(pitches.shape[0], pitches.shape[1], 
                                    pitches.shape[2], pitches.shape[3], 131)
        
        # From decimals to one-hot (dur)
        durs = seq_tensor[..., 1]
        onehot_d = np.zeros(
            (durs.shape[0]*durs.shape[1]*durs.shape[2]*durs.shape[3],
             99),
            dtype=float
        )
        onehot_d[np.arange(0, onehot_d.shape[0]), durs.reshape(-1)] = 1.
        onehot_d = onehot_d.reshape(durs.shape[0], durs.shape[1], 
                                    durs.shape[2], durs.shape[3], 99)
        
        # Concatenate pitches and durations
        new_seq_tensor = np.concatenate((onehot_p, onehot_d),
                                        axis=-1)
        
        # Construct the graph representing the whole piece from structure tensor
        graph = graph_from_tensor(seq_acts)
        
        # Filter silences in order to get a sparse representation
        new_seq_tensor = new_seq_tensor.reshape(-1, new_seq_tensor.shape[-2],
                                                new_seq_tensor.shape[-1])
        src_mask = src_mask.reshape(-1, src_mask.shape[-1])
        new_seq_tensor = new_seq_tensor[seq_acts.reshape(-1).astype(bool)]
        src_mask = src_mask[seq_acts.reshape(-1).astype(bool)]
        
        new_seq_tensor = torch.Tensor(new_seq_tensor)
        seq_acts = torch.Tensor(seq_acts)
        graph.x_seq = new_seq_tensor
        graph.x_acts = seq_acts
        graph.src_mask = src_mask
        
        # Todo: start with torch at mount
        #return torch.Tensor(new_seq_tensor), torch.Tensor(seq_acts), graphs, src_mask
        return graph
