"""
PyTorch convolutional autoencoder for direct Raman spectral unmixing.
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.utils import ID_MOLECULES
from src.utils import digital_mix_unmixing
from src.utils import spectra_normalization


AE_UNMIXING_FILENAME = "ae_unmixing.pt"
AE_UNMIXING_BEST_FILENAME = "ae_unmixing_best.pt"
AE_UNMIXING_LAST_FILENAME = "ae_unmixing_last.pt"
AE_UNMIXING_METRICS_FILENAME = "ae_unmixing_metrics.joblib"


def spectral_angle_distance(y_pred, y_true, eps=1e-8):
    """Mean spectral angle distance in radians."""
    y_pred_norm = F.normalize(y_pred, dim=-1, eps=eps)
    y_true_norm = F.normalize(y_true, dim=-1, eps=eps)
    cosine = torch.sum(y_pred_norm * y_true_norm, dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cosine).mean()


class MultiKernelConvBlock(nn.Module):
    def __init__(self, kernel_sizes=(3, 5, 7), num_filters=16):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.GELU(),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.project = nn.Sequential(
            nn.Conv1d(num_filters * len(kernel_sizes), 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        features = [conv(x) for conv in self.convs]
        merged = torch.cat(features, dim=1)
        return self.project(merged)


class RamanUnmixingAE(nn.Module):
    """
    Direct unmixing AE.

    Latent abundance vector contains four channels: DA, E, NE, BA.
    The decoder is a non-negative linear mixture model.
    """

    def __init__(self, input_dim, latent_dim=4):
        super().__init__()
        self.feature_extractor = MultiKernelConvBlock()
        self.encoder_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * 16, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.abundance_logits = nn.Linear(128, latent_dim)
        self.decoder_weight_raw = nn.Parameter(torch.zeros(latent_dim, input_dim))

    def initialize_decoder(self, reference_endmembers):
        reference = torch.as_tensor(reference_endmembers, dtype=self.decoder_weight_raw.dtype)
        reference = torch.clamp(reference, min=1e-4)
        with torch.no_grad():
            self.decoder_weight_raw.copy_(torch.log(torch.expm1(reference)))

    def decoder_weight(self):
        return F.softplus(self.decoder_weight_raw)

    def forward(self, x):
        encoded = self.feature_extractor(x)
        encoded = self.encoder_mlp(encoded)
        abundances = torch.softmax(self.abundance_logits(encoded), dim=-1)
        endmembers = self.decoder_weight()
        reconstruction = abundances @ endmembers
        return {
            "abundances": abundances,
            "reconstruction": reconstruction,
            "endmembers": endmembers,
        }


@dataclass
class UnmixingTrainingConfig:
    samples_per_combination: int = 2000
    batch_size: int = 128
    epochs: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_size: float = 0.2
    random_state: int = 42
    reconstruction_weight: float = 1.0
    abundance_weight: float = 0.30
    endmember_weight: float = 0.20
    patience: int = 50
    concentration_scale: float = 10.0


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_output_dirs():
    os.makedirs("visualization", exist_ok=True)
    os.makedirs("log", exist_ok=True)


def _build_training_logger(timestamp):
    logger_name = f"ae_unmixing_train_{timestamp}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    log_path = os.path.join("log", f"ae_unmixing_train_{timestamp}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    return logger, log_path


def _close_logger(logger):
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _save_training_history(history, timestamp):
    history_path = os.path.join("log", f"ae_unmixing_history_{timestamp}.csv")
    with open(history_path, "w", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,val_loss,val_concentration_rmse,val_ratio_rmse\n")
        for idx in range(len(history["epoch"])):
            handle.write(
                f"{history['epoch'][idx]},{history['train_loss'][idx]:.6f},{history['val_loss'][idx]:.6f},"
                f"{history['val_concentration_rmse'][idx]:.6f},{history['val_ratio_rmse'][idx]:.6f}\n"
            )
    return history_path


def _save_loss_curve(history, timestamp):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["epoch"], history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("AE Unmixing Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join("visualization", f"AE_Unmixing_Loss_Curve.png")
    fig.savefig(plot_path, dpi=600)
    plt.close(fig)
    return plot_path


def _compute_reference_endmembers(Raman_Shift, Intensity, Category, Concentration):
    category = np.asarray(Category)
    concentration = np.asarray(Concentration)

    reference_spectra = []
    for molecule in ["DA", "E", "NE"]:
        indices = np.where((category == molecule) & (concentration == 10))[0]
        if indices.size == 0:
            raise ValueError(f"Missing 10 uM spectra for {molecule}.")
        reference_spectra.append(np.mean(Intensity[indices], axis=0))

    ba_indices = np.where((category == "BA") & (concentration == 0))[0]
    if ba_indices.size == 0:
        raise ValueError("Missing BA background spectra.")
    reference_spectra.append(np.mean(Intensity[ba_indices], axis=0))

    reference_spectra = np.asarray(reference_spectra, dtype=np.float32)
    reference_spectra = spectra_normalization(
        Raman_Shift,
        reference_spectra,
        peak_position=920,
        peak_range=20,
        plot=False,
        mode="ae_unmixing_reference",
    ).astype(np.float32)
    return reference_spectra


def _prepare_mix_dataset(Raman_Shift, Intensity, Category, Concentration, config):
    selected = np.where((Concentration == 10) | (Concentration == 0))[0]
    intensity_selected = np.asarray(Intensity[selected], dtype=np.float32)
    category_selected = np.asarray(Category[selected])

    X_mix, abundance_mix, concentration_mix, combo_labels = digital_mix_unmixing(
        Raman_Shift,
        intensity_selected,
        category_selected,
        data_concentration=config.concentration_scale,
        samples_per_combination=config.samples_per_combination,
        Range=(0.5, config.concentration_scale),
        seed=config.random_state,
    )
    X_mix = spectra_normalization(
        Raman_Shift,
        X_mix,
        peak_position=920,
        peak_range=20,
        plot=False,
        mode="ae_unmixing_mix",
    ).astype(np.float32)
    return X_mix, abundance_mix, concentration_mix, combo_labels


def _build_dataloaders(X, abundances, concentrations, combo_labels, config):
    unique_classes = np.unique(combo_labels)
    requested_val_count = int(np.ceil(len(X) * config.validation_size))
    stratify_labels = combo_labels if requested_val_count >= len(unique_classes) else None

    split = train_test_split(
        X,
        abundances,
        concentrations,
        combo_labels,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=stratify_labels,
    )
    X_train, X_val, abund_train, abund_val, conc_train, conc_val, combo_train, combo_val = split

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(abund_train),
        torch.from_numpy(conc_train),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(abund_val),
        torch.from_numpy(conc_val),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, combo_train, combo_val, X_val, abund_val, conc_val


def _compute_ratio(concentrations, eps=1e-8):
    totals = np.sum(concentrations, axis=1, keepdims=True)
    return np.divide(concentrations, totals + eps, out=np.zeros_like(concentrations), where=totals > eps)


def _evaluate_model(model, data_loader, device, config, reference_endmembers):
    model.eval()
    losses = []
    pred_abundances = []
    true_abundances = []
    pred_concentrations = []
    true_concentrations = []

    reference_tensor = torch.from_numpy(reference_endmembers).to(device)

    with torch.no_grad():
        for batch_x, batch_abundance, batch_concentration in data_loader:
            batch_x = batch_x.to(device)
            batch_abundance = batch_abundance.to(device)
            batch_concentration = batch_concentration.to(device)

            output = model(batch_x)
            recon_loss = F.mse_loss(output["reconstruction"], batch_x) + spectral_angle_distance(
                output["reconstruction"], batch_x
            )
            abundance_loss = F.mse_loss(output["abundances"], batch_abundance)
            endmember_loss = F.mse_loss(output["endmembers"], reference_tensor)
            loss = (
                config.reconstruction_weight * recon_loss
                + config.abundance_weight * abundance_loss
                + config.endmember_weight * endmember_loss
            )
            losses.append(float(loss.detach().cpu()))

            predicted_abundance = output["abundances"].detach().cpu().numpy()
            pred_abundances.append(predicted_abundance)
            true_abundances.append(batch_abundance.detach().cpu().numpy())

            predicted_concentration = predicted_abundance[:, :3] * config.concentration_scale
            pred_concentrations.append(predicted_concentration)
            true_concentrations.append(batch_concentration.detach().cpu().numpy())

    pred_abundances = np.concatenate(pred_abundances, axis=0)
    true_abundances = np.concatenate(true_abundances, axis=0)
    pred_concentrations = np.concatenate(pred_concentrations, axis=0)
    true_concentrations = np.concatenate(true_concentrations, axis=0)
    pred_ratios = _compute_ratio(pred_concentrations)
    true_ratios = _compute_ratio(true_concentrations)

    return {
        "loss": float(np.mean(losses)),
        "pred_abundances": pred_abundances,
        "true_abundances": true_abundances,
        "pred_concentrations": pred_concentrations,
        "true_concentrations": true_concentrations,
        "concentration_rmse": float(np.sqrt(np.mean((pred_concentrations - true_concentrations) ** 2))),
        "ratio_rmse": float(np.sqrt(np.mean((pred_ratios - true_ratios) ** 2))),
    }


def _save_endmember_plot(Raman_Shift, reference_endmembers, learned_endmembers, timestamp):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for idx, molecule in enumerate(["DA", "E", "NE", "BA"]):
        ax = axes[idx // 2, idx % 2]
        ax.plot(Raman_Shift, reference_endmembers[idx], label=f"Reference {molecule}", linewidth=2)
        ax.plot(Raman_Shift, learned_endmembers[idx], label=f"Learned {molecule}", linewidth=1.5, linestyle="--")
        ax.set_title(molecule)
        ax.legend()
    fig.tight_layout()
    plot_path = os.path.join("visualization", f"AE_Unmixing_Endmembers.png")
    fig.savefig(plot_path, dpi=600)
    plt.close(fig)
    return plot_path


def AE_Unmixing_Train(Raman_Shift, Intensity, Category, Concentration, model_dir, plot=False, config=None):
    """
    Train the direct AE unmixing model.
    """
    if config is None:
        config = UnmixingTrainingConfig()

    _ensure_output_dirs()
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = _build_training_logger(timestamp)
    device = _default_device()
    print(f"Training AE unmixing model on device: {device}")
    logger.info("Training AE unmixing model on device: %s", device)
    logger.info("Training config: %s", config)

    reference_endmembers = _compute_reference_endmembers(Raman_Shift, Intensity, Category, Concentration)
    X_mix, abundance_mix, concentration_mix, combo_labels = _prepare_mix_dataset(
        Raman_Shift, Intensity, Category, Concentration, config
    )
    logger.info(
        "Synthetic unmixing dataset prepared. X_mix=%s, abundance_mix=%s, concentration_mix=%s",
        X_mix.shape,
        abundance_mix.shape,
        concentration_mix.shape,
    )
    train_loader, val_loader, _combo_train, combo_val, X_val, abund_val, conc_val = _build_dataloaders(
        X_mix, abundance_mix, concentration_mix, combo_labels, config
    )

    model = RamanUnmixingAE(input_dim=X_mix.shape[1], latent_dim=4).to(device)
    model.initialize_decoder(reference_endmembers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    reference_tensor = torch.from_numpy(reference_endmembers).to(device)

    best_state = None
    best_metrics = None
    best_payload = None
    best_score = np.inf
    best_epoch = 0
    patience_counter = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_concentration_rmse": [],
        "val_ratio_rmse": [],
    }

    try:
        for epoch in range(config.epochs):
            model.train()
            train_losses = []
            for batch_x, batch_abundance, _batch_concentration in train_loader:
                batch_x = batch_x.to(device)
                batch_abundance = batch_abundance.to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    output = model(batch_x)
                    recon_loss = F.mse_loss(output["reconstruction"], batch_x) + spectral_angle_distance(
                        output["reconstruction"], batch_x
                    )
                    abundance_loss = F.mse_loss(output["abundances"], batch_abundance)
                    endmember_loss = F.mse_loss(output["endmembers"], reference_tensor)
                    loss = (
                        config.reconstruction_weight * recon_loss
                        + config.abundance_weight * abundance_loss
                        + config.endmember_weight * endmember_loss
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.detach().cpu()))

            validation_metrics = _evaluate_model(model, val_loader, device, config, reference_endmembers)
            train_loss_mean = float(np.mean(train_losses))

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss_mean)
            history["val_loss"].append(validation_metrics["loss"])
            history["val_concentration_rmse"].append(validation_metrics["concentration_rmse"])
            history["val_ratio_rmse"].append(validation_metrics["ratio_rmse"])

            epoch_message = (
                f"AE Unmixing Epoch {epoch + 1:03d}/{config.epochs} - "
                f"Train Loss: {train_loss_mean:.4f}, Val Loss: {validation_metrics['loss']:.4f}, "
                f"Val Conc RMSE: {validation_metrics['concentration_rmse']:.4f}, "
                f"Val Ratio RMSE: {validation_metrics['ratio_rmse']:.4f}"
            )
            print(epoch_message)
            logger.info(epoch_message)

            score = validation_metrics["concentration_rmse"] + validation_metrics["ratio_rmse"]
            if score < best_score:
                best_score = score
                best_epoch = epoch + 1
                patience_counter = 0
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                best_metrics = validation_metrics
                best_payload = {
                    "model_state_dict": best_state,
                    "input_dim": int(X_mix.shape[1]),
                    "latent_dim": 4,
                    "molecules": ["DA", "E", "NE", "BA"],
                    "concentration_scale": float(config.concentration_scale),
                    "reference_endmembers": torch.from_numpy(reference_endmembers.copy()),
                    "best_epoch": best_epoch,
                    "timestamp": timestamp,
                    "log_path": log_path,
                }
                torch.save(best_payload, os.path.join(model_dir, AE_UNMIXING_FILENAME))
                torch.save(best_payload, os.path.join(model_dir, AE_UNMIXING_BEST_FILENAME))
                logger.info("Best checkpoint updated at epoch %d with score %.6f", best_epoch, best_score)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    stop_message = f"Early stopping triggered at epoch {epoch + 1}."
                    print(stop_message)
                    logger.info(stop_message)
                    break

        if best_state is None:
            raise RuntimeError("AE unmixing training did not produce a valid checkpoint.")

        last_payload = {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "input_dim": int(X_mix.shape[1]),
            "latent_dim": 4,
            "molecules": ["DA", "E", "NE", "BA"],
            "concentration_scale": float(config.concentration_scale),
            "reference_endmembers": torch.from_numpy(reference_endmembers.copy()),
            "last_epoch": history["epoch"][-1],
            "timestamp": timestamp,
            "log_path": log_path,
        }
        torch.save(last_payload, os.path.join(model_dir, AE_UNMIXING_LAST_FILENAME))

        learned_endmembers = F.softplus(model.decoder_weight_raw).detach().cpu().numpy()
        loss_curve_path = _save_loss_curve(history, timestamp)
        endmember_plot_path = _save_endmember_plot(Raman_Shift, reference_endmembers, learned_endmembers, timestamp)
        history_path = _save_training_history(history, timestamp)

        metrics_payload = {
            "best_epoch": best_epoch,
            "validation_loss": float(best_metrics["loss"]),
            "validation_concentration_rmse": float(best_metrics["concentration_rmse"]),
            "validation_ratio_rmse": float(best_metrics["ratio_rmse"]),
            "validation_true_concentrations": conc_val,
            "validation_predicted_concentrations": best_metrics["pred_concentrations"],
            "validation_true_abundances": abund_val,
            "validation_predicted_abundances": best_metrics["pred_abundances"],
            "combo_validation_labels": combo_val,
            "history": history,
            "loss_curve_path": loss_curve_path,
            "endmember_plot_path": endmember_plot_path,
            "history_path": history_path,
            "log_path": log_path,
        }
        joblib.dump(metrics_payload, os.path.join(model_dir, AE_UNMIXING_METRICS_FILENAME))
        logger.info("Loss curve saved to %s", loss_curve_path)
        logger.info("Endmember plot saved to %s", endmember_plot_path)
        logger.info("Training history saved to %s", history_path)

        if plot:
            print(f"Loss curve saved to {loss_curve_path}")
            print(f"Endmember plot saved to {endmember_plot_path}")

        return best_payload
    finally:
        _close_logger(logger)


def _load_trained_model(model_dir, map_location=None):
    if map_location is None:
        map_location = _default_device()
    checkpoint_path = os.path.join(model_dir, AE_UNMIXING_FILENAME)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    model = RamanUnmixingAE(input_dim=payload["input_dim"], latent_dim=payload["latent_dim"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, payload


def AE_Unmixing_Predict(Intensity, model_dir):
    """
    Predict direct abundances, concentrations, and ratios for DA/E/NE/BA.
    """
    device = _default_device()
    model, payload = _load_trained_model(model_dir, map_location=device)
    intensity = np.asarray(Intensity, dtype=np.float32)

    with torch.no_grad():
        batch = torch.from_numpy(intensity).to(device)
        output = model(batch)
        abundances = output["abundances"].cpu().numpy()
        reconstructions = output["reconstruction"].cpu().numpy()
        endmembers = output["endmembers"].cpu().numpy()

    concentrations = abundances[:, :3] * float(payload["concentration_scale"])
    ratios = _compute_ratio(concentrations)

    return {
        "abundances": abundances,
        "concentrations": concentrations,
        "ratios": ratios,
        "reconstructions": reconstructions,
        "endmembers": endmembers,
        "payload": payload,
    }
