from pathlib import Path
import numpy as np
import torch


def parse_filename(file_path):
    filename = Path(file_path).name
    parts = filename.split("_")

    if len(parts) <5:
        raise ValueError(f"Unexpected filename: {filename}")
 
    return {
        "section": parts[1],
        "domain": parts[2],
        "split": parts[3],
        "label": parts[4]
    }

def collect_files(
        data_dir,
        section,
        domain=None,
        split=None,
        label=None
):
    """
    Collect files matching the requested section/domain/spilt/label.
    """

    files = []

    for file_path in Path(data_dir).rglob("*.wav"):
        metadata = parse_filename(file_path)

        if metadata["section"] != section:
            continue

        if domain is not None:
            if metadata["domain"] != domain:
                continue

        if split is not None:
            if metadata["split"] != split:
                continue

        if label is not None:
            if metadata["label"] != label:
                continue

        files.append(file_path)

    return sorted(files)

def get_labels(file_paths):
    return[
        parse_filename(file_path)["label"]
        for file_path in file_paths
    ]

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    return device

def compute_frequency_weights(
        model,
        dataloader,
        device,
        epsilon=1e-6
):
    """
    Compute frequency weights from normal training reconstruction errors.
    Frequency bands with lower normal reconstruction variability receive higher weights.
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstructed = model(batch)

            error_map = torch.abs(batch - reconstructed).squeeze(1)

            errors.append(error_map.detach().cpu())

    errors = torch.cat(errors, dim=0)
    #frequency_mean = errors.mean(dim=(0,2))
    frequency_std = errors.std(dim=(0,2))
    weights = 1.0 / (frequency_std + epsilon)

    weights = weights / weights.mean()

    return weights.numpy()

def compute_frequency_weight_errors(
        model,
        dataloader,
        device,
        frequency_weights
):
    """
    Compute ferquency-weighted reconstrution error.
    """
    model.eval()
    errors = []

    weights = torch.tensor(frequency_weights, dtype=torch.float32, device=device)
    weights = weights.view(1, -1, 1) # [128] -> [1, 128, 1]

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstructed = model(batch)

            error_map = (batch - reconstructed) ** 2

            weighted_error = (error_map * weights)

            batch_errors = torch.quantile(
                weighted_error.reshape(weighted_error.shape[0], -1), 0.95, dim=1
            )

            errors.extend(batch_errors.detach().cpu().numpy())

    return np.asarray(errors)

def compute_reconstruction_errors(
        model,
        dataloader,
        device
):
    """
    Compute one reconstruction error for each audio sample.
    """

    model.eval()

    errors = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstructed = model(batch)

            #error = torch.mean((batch - reconstructed) ** 2,dim=(1,2,3))
            pixel_error = (batch - reconstructed) ** 2
            error_map = pixel_error.squeeze(1)
            #Top 5% reconstruction errors
            batch_errors = torch.quantile(error_map.reshape(error_map.shape[0],-1),0.95,dim=1)
            error = batch_errors

            errors.extend(error.detach().cpu().numpy())

    return np.asarray(errors)

def determine_threshold(
        train_errors,
        percentile=90
):
    """
    Determine anomaly threshold from normal training reconstruction errors.
    """

    threshold = np.percentile(train_errors, percentile)

    return threshold

def scores_to_labels(errors, threshold):
    """
    Convert reconstruction errors to normal/anomaly labels.
    """

    predictions = np.where(errors > threshold, "anomaly", "normal")

    return predictions