import librosa
import numpy as np

def load_audio(file_path, sr=16000):
    audio, sample_rate = librosa.load(file_path,sr=sr,mono=True)
    return audio, sample_rate

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    max_amplitude = np.max(np.abs(audio))

    if max_amplitude == 0:
        return audio

    return audio / max_amplitude

def compute_stft(audio: np.ndarray, n_fft: int = 1024, hop_length: int = 512):
    return librosa.stft(audio,n_fft=n_fft,hop_length=hop_length)

def compute_mel_spectrogram(
        audio: np.ndarray,
        sr: int,
        n_fft: int = 1024,
        hop_length: int =512,
        n_mels: int = 128
):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=80
    )

    return mel_db

def resize_spectrogram(
        spectrogram: np.ndarray,
        target_frames: int = 320, #313
) -> np.ndarray:
    n_mels, n_frames = spectrogram.shape

    if n_frames < target_frames:
        pad_width = target_frames - n_frames

        spectrogram = np.pad(
            spectrogram,
            (
                (0,0),
                (0,pad_width)
            ),
            mode='constant'
        )
    else:
        spectrogram = spectrogram[
            :,
            :target_frames
        ]

    return spectrogram