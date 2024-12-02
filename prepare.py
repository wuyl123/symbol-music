import os
import pretty_midi
import pickle
import numpy as np
from intervaltree import IntervalTree

midi_folder = 'midis'
midi_list = os.listdir(midi_folder)

data = []
durations = []
basic_beat = 0.25

def read_note_sequence(midi_list):
    for i,dir in enumerate(midi_list):
        midi_path = os.path.join(midi_folder,dir)

        midi = pretty_midi.PrettyMIDI(midi_path)
        assert len(midi.instruments)==1
        notes = midi.instruments[0].notes
        segments = []
        for note in notes:
            segments.append((note.start,note.end,note.pitch))
        tree = IntervalTree.from_tuples(segments)
        tree.split_overlaps()
        print(f"Processing {i}/{len(midi_list)}")
        print(len(tree))
        for interval in sorted(tree):
            start = interval.begin
            end = interval.end
            pitch = interval.data
            dur,flag = close_dur(durations,end-start)
            d = dur
            if not flag:
                durations.append(dur)
            if len(tree.overlap(interval)) > 1:
                chord = ''
                for p in tree[start:end]:
                    chord += str(p.data)+'|'
                    tree.remove(p)
                data.append(chord+'_'+str(d))
            else:
                data.append(str(pitch)+'_'+str(d))

def close_dur(durations,dur):
    flag = False
    for duration in durations:
        if abs(duration - dur)<=0.25:
            flag = True
            return duration,flag   
    return dur,flag

# for i,dir in enumerate(midi_list):
#     midi_path = os.path.join(midi_folder,dir)

#     midi = pretty_midi.PrettyMIDI(midi_path)
#     assert len(midi.instruments)==1
#     notes = midi.instruments[0].notes
#     segments = []
#     for note in notes:
#         segments.append((note.start,note.end,note.pitch))
#     tree = IntervalTree.from_tuples(segments)
#     tree.split_overlaps()
#     print(f"Processing {i}/{len(midi_list)}")
#     print(len(tree))
#     for interval in sorted(tree):
#         start = interval.begin
#         end = interval.end
#         pitch = interval.data
#         dur,flag = close_dur(durations,end-start)
#         d = dur
#         if not flag:
#             durations.append(dur)
#         if len(tree.overlap(interval)) > 1:
#             chord = ''
#             for p in tree[start:end]:
#                 chord += str(p.data)+'|'
#                 tree.remove(p)
#             data.append(chord+'_'+str(d))
#         else:
#             data.append(str(pitch)+'_'+str(d))

chars = sorted(list(set(data)))
print("vocab size is {}".format(len(chars)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a list of string, output a list of integers
def decode(l):
    return [itos[i] for i in l] # decoder: take a list of integers, output list of string

# print(decode([8,5]),encode(["100_0.005208333333335702","100_0.005208333333333329"]))

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# # export to bin files
# train_ids = np.array(train_ids).astype(np.uint16)
# val_ids = np.array(val_ids).astype(np.uint16)
# train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
# val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# # save the meta information as well, to help us encode/decode later
# meta = {
#     'vocab_size': vocab_size,
#     'itos': itos,
#     'stoi': stoi,
# }
# with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
#     pickle.dump(meta, f)