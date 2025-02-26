# UPDATED FOR FILE PATH ACCEPTANCE.
# ORIGINAL DATASET.PY IN KELSEY'S LOCAL

import torch
import librosa
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
import numpy as np

class AudioIterableDataset(torch.utils.data.IterableDataset):
    """
    Stream audio files in chunks for memory-efficient training.
    """

    def __init__(self, file_paths, segment_length, sampling_rate, hop_size, transform=None):
        super().__init__()

        self.transform = transform
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.hop_size = hop_size
        self.file_paths = file_paths  # List of file paths

        if segment_length % hop_size != 0:
            raise ValueError("segment_length {} is not a multiple of hop_size {}".format(segment_length, hop_size))

        # Calculate total segments across all files
        self.total_segments = 0
        for file_path in self.file_paths:
            audio, _ = librosa.load(file_path, sr=self.sampling_rate)
            self.total_segments += (len(audio) // self.hop_size) - (self.segment_length // self.hop_size) + 1

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # Single-process loading
            file_indices = range(len(self.file_paths))
        else:  # Multi-process loading
            per_worker = int(np.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.file_paths))
            file_indices = range(start_idx, end_idx)

        # Iterate over files assigned to this worker
        for file_idx in file_indices:
            file_path = self.file_paths[file_idx]
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


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.

    Streaming audio data in chunks
    """

    def __init__(self, audio_np, segment_length, sampling_rate, hop_size, transform=None, cache_monitor=None, cache_size=10):
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.hop_size = hop_size

        if segment_length % hop_size != 0:
            raise ValueError("segment_length {} is not a multiple of hop_size {}".format(segment_length, hop_size))

        if len(audio_np) % hop_size != 0:
            num_zeros = hop_size - (len(audio_np) % hop_size)
            audio_np = np.pad(audio_np, (0, num_zeros), 'constant', constant_values=(0, 0))

        self.audio_np = audio_np

    def __getitem__(self, index):
        # Take segment
        seg_start = index * self.hop_size
        seg_end = (index * self.hop_size) + self.segment_length
        sample = self.audio_np[seg_start:seg_end]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return (len(self.audio_np) // self.hop_size) - (self.segment_length // self.hop_size) + 1


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
