import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from pathlib import Path
import configparser
import argparse

from rawvae.model import VAE
print("Imports done! ✅")

def load_config(config_path):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_path)
    return config

def read_config(config_path):
    """
    Read and parse arguments from configuration file
    """

    print(f"\nReading configuration file from: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)
    print("Params loaded")

    # Extract model parameters from config
    model_params = {
        'segment_length': config.getint('audio', 'segment_length'),
        'sampling_rate': config.getint('audio', 'sampling_rate'),
        'n_units': config.getint('VAE', 'n_units'),
        'latent_dim': config.getint('VAE', 'latent_dim'),
        'batch_size': config.getint('training', 'batch_size'),
        'learning_rate': config.getfloat('training', 'learning_rate'),
        'epochs': config.getint('training', 'epochs'),
        'audio_dir': config['dataset'].get('test_dataset'),
        'checkpoint_dir': config['dataset'].get()
    }

    return model_params

def get_random_audio(audio_dir):
    """
    Random pick from directory
    """

    audio_files = list(Path(audio_dir).glob('*.wav'))

    if len(audio_files) < 2:
        raise ValueError(f"Not enough files in {audio_dir} to facilitate random choice")
    
    audio1 = np.random.choice(audio_files)

    remaining_files = [f for f in audio_files if f != audio1]
    audio2 = np.random.choice(remaining_files)

    print(f"File 1: {audio1.name}\nFile2: {audio2.name}")

    return str(audio1), str(audio2)

def match_audio_lengths(audio1, audio2):
    """
    Match audio lengths
    Just concatenate shorter audio until length of longer file is met
    """

    len_audio1, len_audio2 = len(audio1), len(audio2)

    if len_audio1 == len_audio2:
        return audio1, audio2

    if len_audio1 < len_audio2:

        # Calculate how many full repeats needed
        repeats = len_audio2 // len_audio1
        remainder = len_audio2 % len_audio1

        # Concatenate
        matched_audio = np.tile(audio1, repeats)

        if remainder > 0:
            matched_audio = np.concatenate([matched_audio, audio1[:remainder]])
        return matched_audio, audio2
    else:

        repeats = len_audio1 // len_audio2
        remainder = len_audio1 % len_audio2

        matched_audio = np.tile(audio2, repeats)
        if remainder > 0:
            matched_audio = np.concatenate([matched_audio, audio2[:remainder]])
        return matched_audio, audio1
    
def process_full_audio(audio, segment_length):
    """
    Process full audios into segments and return tensor
    """

    n_segments = len(audio) // segment_length
    audio = audio[:n_segments * segment_length]
    segments = audio.reshape(n_segments, segment_length)

    return torch.FloatTensor(segments)

def load_process_audio(file_path, segment_length, sr=44100):
    """
    Load and process audio file (mainly normalization and shit)
    """

    print(f"\nProcessing audio file: {file_path}")
    print(f"Sample rate: {sr}")

    audio, _ = librosa.load(file_path, sr=sr)
    audio = audio / np.max(np.abs(audio))

    return audio

# GET LATENT VECTOR
def get_latent_vector(model, audio_tensor, device):
    """
    Encode audio into latent vector
    """
    audio_tensor = audio_tensor.to(device)
    with torch.no_grad():
        mu, _ = model.encode(audio_tensor)
        print(f"Latent vector shape: {mu.shape}")
        print(f"Latent vector mean: {mu.mean().item():.3f}")
        print(f"Latent vector std: {mu.std().item():.3f}")
        return mu # using mean distibution as latent vector!
        
# INTERPOLATE VECTORS

def interpolate_vectors(z1, z2, alpha):
    """ 
    Interpolate between 2 latent factors 
    aplha is our interpolation factor (0 = z1, 1 = z2)
    """
    interp = (1 - alpha) * z1 + alpha * z2
    print(f"Interpolation at alpha={alpha:.1f}")
    print(f"z1 mean: {z1.mean().item():.3f}, z2 mean: {z2.mean().item():.3f}")
    print(f"Interpolated mean: {interp.mean().item():.3f}")
    return interp

def create_interpolations(model, z1, z2, sampling_rate, output_dir):
    """
    Create interpolations between 2 latent vectors.
    Save audio, save spectrogram and waveform 
    """

    # Create output dirs
    output_dir = Path(output_dir)
    os.makedirs(output_dir / "audio", exist_ok=True)
    os.makedirs(output_dir / "spectrograms", exist_ok=True)
    os.makedirs(output_dir / "waveforms", exist_ok=True)

    # Interpolation stations
    alphas = np.arange(0.1, 1.0, 0.1)

    for alpha in alphas:
        # Interpolate baby
        print(f"\nProcessing interpolation α={alpha:.1f}")
        z_interp = interpolate_vectors(z1, z2, alpha)

        # Decode
        with torch.no_grad():
            audio_interp = model.decode(z_interp)
            audio_np = audio_interp.cpu().numpy().flatten()
            print(f"Decoded audio range: [{audio_np.min():.3f}, {audio_np.max():.3f}]")
            print(f"Decoded audio mean: {audio_np.mean():.3f}")
            print(f"Decoded audio std: {audio_np.std():.3f}")

        # Save audio
        audio_path = output_dir / "audio" / f"interpolation_{alpha:.1f}.wav"
        sf.write(audio_path, audio_np, sampling_rate)
        print(f"Saved audio in: {audio_path}")

        # Create and save spectrogram
        spec_path = output_dir / "spectrograms" / f"spectrogram_{alpha:.1f}.png"
        plt.figure(figsize=(10,4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram ((α={alpha:.1f})')
        plt.tight_layout()
        plt.savefig(output_dir / "spectrograms" / f"spectrogram_{alpha:.1f}.png")
        plt.close()
        print(f"Saved spectrogram: {spec_path}")

        # Create and save waveform
        wave_path = output_dir / "waveforms" / f"waveform_{alpha:.1f}.png"
        plt.figure(figsize=(10,4))
        plt.plot(audio_np)
        plt.title(f'Waveform (α={alpha:.1f})')
        plt.tight_layout()
        plt.savefig(output_dir / "waveforms" / f"waveform_{alpha:.1f}.png")
        plt.close()
        print(f"Saved waveform: {wave_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./default-alvis.ini', help='Path to the config file')
    args = parser.parse_args()

    model_params = read_config(args.config)
    
    # Extract parameters
    segment_length = model_params['segment_length']
    n_units = model_params['n_units']
    latent_dim = model_params['latent_dim']
    sampling_rate = model_params['sampling_rate']
    audio_dir = model_params['audio_dir']
    checkpoint_path = model_params['checkpoint_dir']

    # Load model and perform other operations as needed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Checkpoint located at: {checkpoint_path}")

    print("\n ↗️=== Starting Audio Interpolation Process ===↙️")

    # Load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directories at the start
    output_dir = Path("./interpolations")
    os.makedirs(output_dir / "audio", exist_ok=True)
    os.makedirs(output_dir / "spectrograms", exist_ok=True)
    os.makedirs(output_dir / "waveforms", exist_ok=True)
    print("Created output directories")

    # Read configuration file for our model
    config_path = input("Please paste path to your config file")
    # r"D:\kelse\03_Repositories\RAWAUDIOVAE_PROJECT\KELSEY_DEV\rawaudiovae_kelsey\rawaudiovae_BASE\content\kelsey\config.ini"
    # REPLACE THIS EVENTUALLY WITH READING BACK FROM CONFIG FILE

    model_params = read_config(config_path)

    # EXTRACT AND ASSIGN PARAMS
    segment_length = model_params['segment_length']
    n_units = model_params['n_units']
    latent_dim = model_params['latent_dim']
    sampling_rate = model_params['sampling_rate']
    print("Loaded model parameters from config:")
    print(f"Segment Length: {segment_length}")
    print(f"Number of Units: {n_units}")
    print(f"Latent Dimension: {latent_dim}")
    print(f"Sampling Rate: {sampling_rate}")

    # LOAD MODEL, CHECK PATH CORRECT
    print(f"\nLoading model checkpoint... ⌛")
    checkpoint_path = input("Please paste path to your checkpoint file")
    # r'D:\kelse\03_Repositories\RAWAUDIOVAE_PROJECT\KELSEY_DEV\rawaudiovae_kelsey\rawaudiovae_BASE\content\kelsey\checkpoints\ckpt_00150'
    # REPLACE THIS EVENTUALLY WITH READING BACK FROM LAST MODEL RUN LOG

    print(f"Checkpoint located at: {checkpoint_path}")
    
    state = torch.load(checkpoint_path, map_location=device)
    
    # INITIALISE
    print("Initialising model 🚀")
    model = VAE(segment_length, n_units, latent_dim).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    print("Locked and loaded ⛑️")

    # AUDIO DIR
    audio_dir = input("Please paste path to your audio directory")
    # r"D:\kelse\03_Repositories\RAWAUDIOVAE_PROJECT\KELSEY_DEV\rawaudiovae_kelsey\rawaudiovae_BASE\audio_experiments"
    # REPLACE THIS EVENTUALLY WITH READING BACK TEST_AUDIO FROM CONFIG FILE
    audio_file1, audio_file2 = get_random_audio(audio_dir)
    
    # PROCESS
    audio1 = load_process_audio(audio_file1, segment_length, sampling_rate)
    audio2 = load_process_audio(audio_file2, segment_length, sampling_rate)

    # MATCH
    audio1_matched, audio2_matched = match_audio_lengths(audio1, audio2)
    print(f"Matched lengths: {len(audio1_matched)} samples")

    # TENSOR BUSINESS
    print("\nProcessing input audio files...!")
    audio_tensor1 = process_full_audio(audio1_matched, segment_length)
    audio_tensor2 = process_full_audio(audio2_matched, segment_length)
    print(f"Created tensors of shape: {audio_tensor1.shape}")

    # GET LATENT VECTORS
    print("\nGenerating latent vectors 👩‍🍳")
    z1 = get_latent_vector(model, audio_tensor1, device)
    z2 = get_latent_vector(model, audio_tensor2, device)

    # Added this section to test encode-decode of original files
    print("\nTesting encode-decode of original files...")
    with torch.no_grad():
        # First file
        print("\nProcessing file 1: ")
        decoded1 = model.decode(z1)
        print(f"Decoded tensor shape: {decoded1.shape}")
        audio_np1 = decoded1.cpu().numpy().flatten()

        # AUDIO
        sf.write("./interpolations/audio/original1_encoded_decoded.wav", audio_np1, sampling_rate)
        print("Saved encoded-decoded version of first file")
        
        # SPECTROGRAM
        plt.figure(figsize=(10,4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np1)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Original 1 Encoded-Decoded Spectrogram')
        plt.tight_layout()
        plt.savefig("./interpolations/spectrograms/original1_encoded_decoded.png")
        plt.close()
        print("Saved spectrogram of encoded-decoded first file")
        
        # WAVEFORM
        plt.figure(figsize=(10,4))
        plt.plot(audio_np1)
        plt.title('Original 1 Encoded-Decoded Waveform')
        plt.tight_layout()
        plt.savefig("./interpolations/waveforms/original1_encoded_decoded.png")
        plt.close()
        print("Saved waveform of encoded-decoded first file")

        # Second file
        print("\nProcessing file 2: ")
        decoded2 = model.decode(z2)
        print(f"Decoded tensor shape: {decoded2.shape}")
        audio_np2 = decoded2.cpu().numpy().flatten()

        # AUDIO
        sf.write("./interpolations/audio/original2_encoded_decoded.wav", audio_np2, sampling_rate)
        print("Saved encoded-decoded version of second file")

        # SPECTROGRAM
        plt.figure(figsize=(10,4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np2)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Original 2 Encoded-Decoded Spectrogram')
        plt.tight_layout()
        plt.savefig("./interpolations/spectrograms/original2_encoded_decoded.png")
        plt.close()
        print("Saved spectrogram of encoded-decoded second file")
        
        # WAVEFORM
        plt.figure(figsize=(10,4))
        plt.plot(audio_np2)
        plt.title('Original 2 Encoded-Decoded Waveform')
        plt.tight_layout()
        plt.savefig("./interpolations/waveforms/original2_encoded_decoded.png")
        plt.close()
        print("Saved waveform of encoded-decoded second file")


    # INTERPOLATE BABY
    create_interpolations(model, z1, z2, sampling_rate, "./interpolations")
    print(f"Interpolation complete! Check folder for results.")

if __name__ == "__main__":
    main()        