import torch
import librosa
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
import numpy as np
import itertools
import logging

class AudioIterableDataset(torch.utils.data.IterableDataset):
    """
    Iterable Dataloader
    """

    def __init__(self, file_paths, segment_length, sampling_rate, hop_size, transform=None):
        super().__init__()

        self.transform = transform
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.hop_size = hop_size
        self.file_paths = file_paths  

        if segment_length % hop_size != 0:
            raise ValueError("segment_length {} is not a multiple of hop_size {}".format(segment_length, hop_size))
        
        # Calculate total segments across all files
        self.total_segments = 0
        for file_path in self.file_paths:
            audio, _ = librosa.load(file_path, sr=self.sampling_rate)
            self.total_segments += (len(audio) // self.hop_size) - (self.segment_length // self.hop_size) + 1

    def __iter__(self):

        # Infinite iterator over filepaths
        for file_path in itertools.cycle(self.file_paths):
            print(f'Parsing audio file: {file_path}')  # Print the current audio file being parsed
            audio, _ = librosa.load(file_path, sr=self.sampling_rate)

            # Pad audio if necessary
            if len(audio) % self.hop_size != 0:
                num_zeros = self.hop_size - (len(audio) % self.hop_size)
                audio = np.pad(audio, (0, num_zeros), 'constant', constant_values=(0, 0))

                # Yield segments from this file
                num_segments = (len(audio) // self.hop_size) - (self.segment_length // self.hop_size) + 1
                for seg_idx in range(num_segments):
                    seg_start = seg_idx * self.hop_size
                    seg_end = seg_start + self.segment_length
                    sample = audio[seg_start:seg_end]

                    if self.transform:
                        sample = self.transform(sample)

                    yield sample

    def __len__(self):
        return self.total_segments


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)


class TestDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, audio_np, segment_length, sampling_rate, transform=None):
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length

        if len(audio_np) % segment_length != 0:
            num_zeros = segment_length - (len(audio_np) % segment_length)
            audio_np = np.pad(audio_np, (0, num_zeros), 'constant', constant_values=(0, 0))

        self.audio_np = audio_np

    def __getitem__(self, index):
        # Take segment
        seg_start = index * self.segment_length
        seg_end = (index * self.segment_length) + self.segment_length
        sample = self.audio_np[seg_start:seg_end]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.audio_np) // self.segment_length
