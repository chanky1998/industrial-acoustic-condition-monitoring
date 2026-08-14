from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import sys

def extract_acoustic_feature(
        y: np.ndarray,
        sr: int
) -> dict:
    features = {}

    features["rms"] = float(np.mean(librosa.feature.rms(y=y)))

    features["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    features["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y,sr=sr)))

    features["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y,sr=sr)))

    features["spectral_flatness"] = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    return features