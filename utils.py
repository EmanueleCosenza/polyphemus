from matplotlib import pyplot as plt
import numpy as np
import copy
import muspy
import os


def plot_struct(s):
    
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(s, edgecolors='k', linewidth=1)
    ax = plt.gca()
    ax.set_aspect('equal')
    
    plt.xticks(range(0, s.shape[1], 8), range(1, 5))
    plt.yticks(range(0, 4), range(1, 5))
    
    ax.invert_yaxis()
    

# Constructs dense representation (with silences) from sparse representation
def dense_from_sparse(content, structure):
    
    #acts = acts.view(-1, bars, acts.size(-2), acts.size(-1))
    #dtype = content.dtype
    dense = np.zeros((structure.shape[0], structure.shape[1], structure.shape[2],
                      structure.shape[3], content.shape[-2], content.shape[-1]))

    size = dense.shape

    dense = dense.reshape(-1, dense.shape[-2], dense.shape[-1])

    silence = np.zeros((dense.shape[-2], dense.shape[-1]))
    silence[:, 129] = 1. # eos token

    dense[structure.astype(bool).reshape(-1)] = content
    dense[np.logical_not(structure.astype(bool).reshape(-1))] = silence

    dense = dense.reshape(size)
    
    return dense


def muspy_from_dense(dense, track_data, resolution):
    
    tracks = []
    
    for tr in range(dense.shape[0]):
        
        notes = []
        
        for ts in range(dense.shape[1]):
            for note in range(dense.shape[2]):
                
                pitch = dense[tr, ts, note, :131]
                pitch = np.argmax(pitch)

                if pitch == 129:
                    break
                
                if pitch != 128:
                    dur = dense[tr, ts, note, 131:]
                    dur = np.argmax(dur) + 1
                    
                    if dur == 97 or dur == 98 or dur == 99:
                        dur = 4
                        continue
                    
                    notes.append(muspy.Note(ts, pitch.item(), dur.item(), 64))
        
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


def plot_pianoroll(music, save_dir=None, name=None):

    fig, axs_ = plt.subplots(4, sharex=True, figsize=(10,10))
    fig.subplots_adjust(hspace=0)
    axs = axs_.tolist()
    muspy.show_pianoroll(music=music, yticklabel='off', grid_axis='off', axs=axs)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, name+".png"), dpi=200)
        

def midi_from_muspy(music, save_dir, name):
    muspy.write_midi(os.path.join(save_dir, name+".mid"), music)
    