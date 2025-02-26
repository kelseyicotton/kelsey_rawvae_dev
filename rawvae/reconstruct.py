"""

This file will handle loading the dataset, performing the reconstruction, and saving the output audio file.

For shits and giggles we also plot the spectrograms of the reconstructions against the original audio spectrograms. 

All file locations have been formatted for Kelsey's computer, make sure to read this script, make your changes, and then you're ready to go.

"""
import time

print("WARNING: DIRECTORIES HAVE BEEN ASSIGNED BASED ON LOCATIONS ON KELSEY'S COMPUTER")
time.sleep(10)
print("ABORT NOW IF YOU AREN'T KELSEY OTHERWISE YOU WILL F*** 💩 UP OR GET CONFUSED AF")
time.sleep(5)
print("Read the full reconstruct.py script, make your changes, then come back later")
time.sleep(5)
print("I'm not kidding...CTRL + C NOW!")
time.sleep(5)

import torch
import librosa
import numpy as np
from dataset import TestDataset  # Import our dataset class
from torch.utils.data import Dataset, TensorDataset, DataLoader
import soundfile as sf
import matplotlib.pyplot as plt
import os

# Load our audio file 
audio_file_path = r"D:\kelse\03_Repositories\rawaudiovae_kelsey\rawvae\01-1_burps_ah-200106_1520-008.wav"
audio_np, sampling_rate = librosa.load(audio_file_path, sr=None)

# Define segment and hop size
segment_lengths = [512, 1024, 2048, 4096, 8192] # Previously tested values [512, 1024, 2048, 4096, 8192]
hop_sizes = [100, 200, 300, 600, 512, 1024, 2048] # Previously tested values [100, 200, 300, 600, 512, 1024, 2048] 

# Create a directory to save spectrogram plots
plot_dir = r"D:\kelse\03_Repositories\rawaudiovae_kelsey\rawvae\reconstructions\plots"
os.makedirs(plot_dir, exist_ok=True)

results = {} # 

# Function to reconstruct audio with given parameters
def reconstruct_audio_with_params(audio_np, segment_length, hop_size, sampling_rate):
    test_dataset = TestDataset(audio_np, segment_length, hop_size, sampling_rate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    reconstructed_audio = []
    for segment in test_loader:
        reconstructed_audio.append(segment.numpy().flatten())

    return np.concatenate(reconstructed_audio)

# Function to plot spectrograms
def plot_spectrogram(original_audio, reconstructed_audio, sampling_rate, segment_length, hop_size, fixed_segment_length, fixed_hop_size):
    # Compute the spectrograms
    original_spectrogram = librosa.stft(original_audio, n_fft=fixed_segment_length, hop_length=fixed_hop_size)
    reconstructed_spectrogram = librosa.stft(reconstructed_audio, n_fft=segment_length, hop_length=hop_size)

    # Convert to decibels
    original_db = librosa.amplitude_to_db(np.abs(original_spectrogram), ref=np.max)
    reconstructed_db = librosa.amplitude_to_db(np.abs(reconstructed_spectrogram), ref=np.max)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(original_db, sr=sampling_rate, x_axis='time', y_axis='log', cmap='coolwarm')
    plt.title(f'Original Audio Spectrogram\nComputed with Fixed Segment Length: {fixed_segment_length}, Hop Size: {fixed_hop_size}')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(reconstructed_db, sr=sampling_rate, x_axis='time', y_axis='log', cmap='coolwarm')
    plt.title(f'Reconstructed Audio Spectrogram\nComputed with Segment Length: {segment_length}, Hop Size: {hop_size}')
    plt.colorbar(format='%+2.0f dB')

    # Save the plot
    plot_file_path = os.path.join(plot_dir, f"spectrogram_segment_{segment_length}_hop_{hop_size}.png")
    plt.savefig(plot_file_path)
    plt.close()  # Close the plot to free memory

# Define fixed parameters for the original audio file's spectrogram
fixed_segment_length = 2048 # value
fixed_hop_size = 512 # value

# Compute and plot our OG audio file BEFORE the loop
original_spectrogram = librosa.stft(audio_np, n_fft=fixed_segment_length, hop_length=fixed_hop_size)
original_db = librosa.amplitude_to_db(np.abs(original_spectrogram), ref=np.max)

# PLOT IT BABY!
original_spectrogram_file_path = os.path.join(plot_dir, "original_audio_spectrogram.png")
plt.figure(figsize=(12, 6))
librosa.display.specshow(original_db, sr=sampling_rate, x_axis='time', y_axis='log', cmap='coolwarm')
plt.title('Original Audio Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.savefig(original_spectrogram_file_path)
plt.close()  # Close the plot to free memory
print(f"Original audio spectrogram saved to {original_spectrogram_file_path}.")

# Loop through the defined ranges of segment lengths and hop sizes
for segment_length in segment_lengths:
    for hop_size in hop_sizes:
        reconstructed_audio = reconstruct_audio_with_params(audio_np, segment_length, hop_size, sampling_rate)
        results[(segment_length, hop_size)] = reconstructed_audio
        
        # Save the reconstructed audio for listening
        output_file_path = f"D:\kelse\03_Repositories\rawaudiovae_kelsey\rawvae\reconstructions\audio{segment_length}_{hop_size}.wav"
        sf.write(output_file_path, reconstructed_audio, sampling_rate)
        print(f"Reconstructed audio saved to {output_file_path} with segment length {segment_length} and hop size {hop_size}.")

        # Plot and save the spectrograms of our reconstructed VS original audio
        plot_spectrogram(audio_np, reconstructed_audio, sampling_rate, segment_length, hop_size, fixed_segment_length, fixed_hop_size)
        print(f"Spectrogram plots saved to {plot_dir}")