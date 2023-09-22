from matplotlib import pyplot as plt
import numpy as np
import copy
import muspy
import os
from scipy.special import softmax
import torch
    

# Constructs dense representation (with silences) from sparse representation
def dense_from_sparse(content, structure):
    
    dense = np.zeros((structure.shape[0], structure.shape[1], structure.shape[2],
                      structure.shape[3], content.shape[-2], content.shape[-1]))

    size = dense.shape

    dense = dense.reshape(-1, dense.shape[-2], dense.shape[-1])

    silence = np.zeros((dense.shape[-2], dense.shape[-1]))
    silence[0, 129] = 1. # EOS token
    silence[1:, 130] = 1. # PAD token

    dense[structure.astype(bool).reshape(-1)] = content
    dense[np.logical_not(structure.astype(bool).reshape(-1))] = silence

    dense = dense.reshape(size)
    
    return dense


# Constructs dense representation (with silences) from sparse representation
def dense_from_sparse_torch(content, structure):
    
    dense = torch.zeros((structure.shape[0], structure.shape[1], structure.shape[2], 
                         structure.shape[3], content.shape[-2], content.shape[-1]),
                         device=content.device, dtype=content.dtype)

    size = dense.size()

    dense = dense.view(-1, dense.size(-2), dense.size(-1))

    silence = torch.zeros((dense.size(-2), dense.size(-1)),
                          device=content.device, dtype=content.dtype)
    silence[0, 129] = 1. # Pitch EOS token
    silence[1:, 130] = 1. # Pitch PAD token
    silence[0, 228] = 1. # Dur EOS token
    silence[1:, 229] = 1. # Dur PAD token

    dense[structure.bool().view(-1)] = content
    dense[torch.logical_not(structure.bool().view(-1))] = silence

    dense = dense.view(size)
    
    return dense


def muspy_from_dense(dense, track_data, resolution):
    
    tracks = []
    
    for tr in range(dense.shape[0]):
        
        notes = []
        
        for ts in range(dense.shape[1]):
            for note in range(dense.shape[2]):
                
                pitch = dense[tr, ts, note, :131]
                pitch = np.argmax(pitch)

                # EOS or PAD
                if pitch == 129 or pitch == 130:
                    break
                
                if pitch != 128:
                    dur = dense[tr, ts, note, 131:]
                    dur = np.argmax(dur) + 1
                    
                    if dur == 97 or dur == 98 or dur == 99:
                        dur = 4
                        continue
                    
                    dur = min(dur, dense.shape[1]-ts-1)
                    
                    notes.append(muspy.Note(ts, pitch.item(), dur, 64))
        
        if track_data[tr][0] == 'Drums':
            track = muspy.Track(name='Drums', is_drum=True, notes=copy.deepcopy(notes))
        else:
            track = muspy.Track(name=track_data[tr][0], 
                                program=track_data[tr][1],
                                notes=copy.deepcopy(notes))
        tracks.append(track)
    
    meta = muspy.Metadata(title='prova')
    music = muspy.Music(tracks=tracks, metadata=meta, resolution=resolution)
    
    return music


def plot_pianoroll(music, save_dir=None, name=None, figsize=(10, 10),
                   fformat="png", xticklabel='on', preset='full', **kwargs):

    fig, axs_ = plt.subplots(4, sharex=True, figsize=figsize)
    fig.subplots_adjust(hspace=0)
    axs = axs_.tolist()
    muspy.show_pianoroll(music=music, yticklabel='off',
                         xticklabel=xticklabel, grid_axis='off',
                         axs=axs, preset=preset, **kwargs)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, name+"."+fformat), format=fformat, dpi=200)
        
        
def plot_struct(s, save_dir=None, name=None, figsize=(10, 10), fformat="svg", n_bars=2):
    
    plt.figure(figsize=figsize)
    plt.pcolormesh(s, edgecolors='k', linewidth=1)
    ax = plt.gca()
    
    plt.xticks(range(0, s.shape[1], 8), range(1, n_bars*4+1))
    plt.yticks(range(0, 4), ['Drums', 'Bass', 'Guitar', 'Strings'])
    
    ax.invert_yaxis()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, name+"."+fformat), format=fformat, dpi=200)
        

def midi_from_muspy(music, save_dir, name):
    muspy.write_midi(os.path.join(save_dir, name+".mid"), music)
    