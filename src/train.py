from pathlib import Path

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .models import CNNAutoencoder

from .preprocessing import (
    load_audio,
    normalize_audio,
    compute_mel_spectrogram,
    resize_spectrogram
)
from .evaluation import evaluate_predictions

from .utils import (
    collect_files,
    get_labels,
    get_device,
    compute_frequency_weights,
    compute_frequency_weight_errors,
    compute_reconstruction_errors,
    determine_threshold,
    scores_to_labels,
)

SR = 16000
N_MELS = 128
TARGET_FRAMES = 320

BATCH_SIZE = 32

EPOCHS = 20
ADAPTATION_EPOCHS = 10

LEARNING_RATE = 1E-3
ADAPTATION_LR = 1E-4

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class AcousticDataset(Dataset):

    def __init__(
            self,
            file_paths,
            sr=SR,
            n_mels=N_MELS,
            target_frames=TARGET_FRAMES
    ):
        self.file_paths = file_paths
        self.sr = sr
        self.n_mels = n_mels
        self.target_frames=target_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]

        audio, sr = load_audio(file_path, sr=self.sr)

        #audio = normalize_audio(audio)

        mel = compute_mel_spectrogram(
            audio,
            sr,
            n_mels=self.n_mels
        )

        mel = resize_spectrogram(mel, target_frames=self.target_frames)

        mel = (mel + 80.0) / 80.0
        mel = np.clip(mel, 0.0, 1.0)

        tensor = torch.tensor(mel, dtype=torch.float32)

        tensor = tensor.unsqueeze(0)

        return tensor

def create_dataloader(
        file_paths,
        batch_size=BATCH_SIZE,
        shuffle=True
):
    dataset = AcousticDataset(file_paths)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    return dataloader

def train_autoencoder(
    model,
    dataloader,
    device,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE
):
    """
    Train CNN Autoencoder using normal sounds.
    """

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model.to(device)

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{epochs} "
            f"- Loss: {epoch_loss:.6f}"
        )

    return model

def adapt_model(
    model,
    target_dataloader,
    device,
    epochs=ADAPTATION_EPOCHS,
    learning_rate=ADAPTATION_LR
):
    """
    Fine-tune a source-trained model using the 3 targert-domain normal samples.
    """

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in target_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(target_dataloader)

        print(
            f"Adaptation Epoch "
            f"{epoch +1:02d}/{epochs} "
            f"- Loss: {epoch_loss:.6f}"
        )

    return model

def run_experiment(
        section,
        data_dir,
        results_dir,
        device
):
    """
    Run all three experiments for one section:
    1. Source -> Source
    2. Source -> Target
    3. Source + 3 Target -> Target
    """

    print()
    print("=" * 60)
    print(f"Section {section}")
    print("=" * 60)

    source_train = collect_files(
        data_dir,
        section=section,
        domain="source",
        split="train",
        label="normal"
    )

    target_train = collect_files(
        data_dir,
        section=section,
        domain="target",
        split="train",
        label="normal"
    )

    source_test_normal = collect_files(
        data_dir,
        section=section,
        domain="source",
        split="test",
        label="normal"
    )
    source_test_anomaly = collect_files(
        data_dir,
        section=section,
        domain="source",
        split="test",
        label="anomaly"
    )
    source_test_files = source_test_normal + source_test_anomaly

    target_test_files =collect_files(
        data_dir,
        section=section,
        domain="target",
        split="test"
    )

    source_test_labels = get_labels(source_test_files)
    target_test_labels = get_labels(target_test_files)

    source_train_loader = create_dataloader(source_train, shuffle=True)
    source_train_eval_loader = create_dataloader(source_train, shuffle=False)
    source_test_loader = create_dataloader(source_test_files, shuffle=False)

    #test
    source_normal_loader = create_dataloader(source_test_normal, shuffle=False)
    source_anomaly_loader = create_dataloader(source_test_anomaly, shuffle=False)

    target_test_loader = create_dataloader(target_test_files, shuffle=False)

    target_train_loader = create_dataloader(target_train, shuffle=True)

    print()
    print("--- Source -> Source")

    model = CNNAutoencoder()
    model = train_autoencoder(model, source_train_loader, device)

    #save model
    torch.save(model.state_dict(), results_dir/f"cnn_ae_section_{section}_source.pth")

    frequency_weights = compute_frequency_weights(model, source_train_eval_loader, device)
    #np.save(results_dir/ f"section_{section}_frequency_weights.npy", frequency_weights)
    train_errors = compute_frequency_weight_errors(
        model, source_train_eval_loader, device, frequency_weights
    )
    #train_errors = compute_reconstruction_errors(model, source_train_eval_loader, device)

    threshold = determine_threshold(train_errors)
    print(f"Threshold: {threshold:.6f}")

    source_errors = compute_reconstruction_errors(model, source_test_loader, device)
    source_prediction = scores_to_labels(source_errors, threshold)

    #test
    source_normal_errors = compute_reconstruction_errors(model, source_normal_loader, device)
    source_anomaly_errors = compute_reconstruction_errors(model, source_anomaly_loader, device)
    np.save(results_dir/ f"section_{section}_source_normal_errors.npy", source_normal_errors)
    np.save(results_dir/ f"section_{section}_source_anomaly_errors.npy", source_anomaly_errors)
    np.save(results_dir/ f"section_{section}_threshold.npy", np.array(threshold))

    results_source = evaluate_predictions(
        source_test_labels,
        source_prediction,
        model_name="CNN-AE Source to Source",
        section=section,
        save_path=(
            results_dir / f"cnn_ae_section_{section}_source_source.png"
        )
    )
    #print(results_source)

    print()
    print("--- Source -> Target")

    target_errors = compute_frequency_weight_errors(
        model, target_test_loader, device, frequency_weights
    )
    #target_errors = compute_reconstruction_errors(model, target_test_loader, device)
    target_prediction = scores_to_labels(target_errors, threshold)

    results_target = evaluate_predictions(
        target_test_labels,
        target_prediction,
        model_name="CNN-AE Source to Target",
        section=section,
        save_path=(
            results_dir / f"cnn_ae_section_{section}_source_target.png"
        )
    )
    #print(results_target)

    print()
    print("--- Source + 3 Target -> Target")

    adapted_model = copy.deepcopy(model)

    adapted_model = adapt_model(
        adapted_model,
        target_train_loader,
        device
    )
    adapted_frequency_weights = compute_frequency_weights(adapted_model, source_train_eval_loader, device)

    target_train_errors = compute_frequency_weight_errors(
        adapted_model, target_train_loader, device, adapted_frequency_weights
    )

    adapted_source_errors = compute_frequency_weight_errors(
        adapted_model, source_train_eval_loader, device, adapted_frequency_weights
    )
    #target_train_errors = compute_reconstruction_errors(adapted_model,target_train_loader,device)
    #adapted_source_errors = compute_reconstruction_errors(adapted_model,source_train_eval_loader,device)

    adaptation_train_errors = np.concatenate(
        [
            adapted_source_errors,
            target_train_errors
        ]
    )

    adapted_threshold = determine_threshold(adaptation_train_errors)
    print(f"Adapted threshold: {adapted_threshold:.6f}")

    adapted_target_errors = compute_frequency_weight_errors(
        adapted_model, target_test_loader, device, adapted_frequency_weights
    )
    #adapted_target_errors = compute_reconstruction_errors(adapted_model,target_test_loader,device)

    adapted_perdictions = scores_to_labels(adapted_target_errors, adapted_threshold)

    results_adapted = evaluate_predictions(
        target_test_labels,
        adapted_perdictions,
        model_name="CNN-AE Source and 3 Target to Target",
        section=section,
        save_path=(
            results_dir/f"cnn_ae_section_{section}_source_target_target"
        )
    )

    #print(results_adapted)

    return{
        "Source_Source": results_source,
        "Source_Target": results_target,
        "Source+Target_Target": results_adapted
    }

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT/"data"/"dev_data_fan"
    RESULTS_DIR = PROJECT_ROOT/"results"

    RESULTS_DIR.mkdir(exist_ok=True)

    device = get_device()

    all_results = []

    for section in ["00","01","02"]:
        results = run_experiment(
            section=section,
            data_dir=DATA_DIR,
            results_dir=RESULTS_DIR,
            device=device
        )

        for experiment, metrics in results.items():
            all_results.append(
                {
                    "section": section,
                    "experiment": experiment,
                    **metrics
                }
            )

    import pandas as pd

    results_df = pd.DataFrame(
        all_results
    )

    results_path = RESULTS_DIR/"cnn_ae_results.csv"

    results_df.to_csv(results_path, index=False)

    print()
    print("=" * 60)
    print("Final CNN Autoencoder Results")
    print("=" * 60)

    print(results_df)

    print()
    print(f"Results saved to: {results_path}")