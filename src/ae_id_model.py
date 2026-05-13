"""
PyTorch autoencoder-based identification model for spectral unmixing.
"""
import logging
import os
from datetime import datetime
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.utils import DA_PROB_THRESHOLD, E_PROB_THRESHOLD, NE_PROB_THRESHOLD
from src.utils import ID_MOLECULES
from src.utils import digital_mix_ID_multilabel
from src.utils import spectra_normalization


MOLECULE_TO_INDEX = {'DA': 0, 'E': 1, 'NE': 2}
AE_ID_FILENAME = "ae_identification_multilabel.pt"
AE_ID_METRICS_FILENAME = "ae_identification_multilabel.joblib"
COMBINATION_TO_INDEX = {
    (0, 0, 0): 0,
    (1, 0, 0): 1,
    (0, 1, 0): 2,
    (0, 0, 1): 3,
    (1, 1, 0): 4,
    (1, 0, 1): 5,
    (0, 1, 1): 6,
    (1, 1, 1): 7,
}


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


class RamanMultiLabelAE(nn.Module):
    """
    Convolutional encoder + abundance decoder + multi-label classifier.

    The latent abundance vector has four entries: DA, E, NE, BA.
    """

    def __init__(self, input_dim, latent_dim=4, classifier_dim=3):
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
        self.classifier = nn.Linear(latent_dim, classifier_dim)
        self.decoder_weight_raw = nn.Parameter(torch.randn(latent_dim, input_dim) * 0.02)

    def decoder_weight(self):
        return F.softplus(self.decoder_weight_raw)

    def forward(self, x):
        encoded = self.feature_extractor(x)
        encoded = self.encoder_mlp(encoded)
        abundance = torch.softmax(self.abundance_logits(encoded), dim=-1)
        reconstruction = abundance @ self.decoder_weight()
        logits = self.classifier(abundance)
        return {
            "abundance": abundance,
            "reconstruction": reconstruction,
            "logits": logits,
        }


@dataclass
class TrainingConfig:
    samples_per_combination: int = 2000
    batch_size: int = 128
    epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_size: float = 0.2
    random_state: int = 42
    reconstruction_weight: float = 0.6
    abundance_weight: float = 0.4
    classification_weight: float = 1.0
    patience: int = 50


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_output_dirs():
    os.makedirs("visualization", exist_ok=True)
    os.makedirs("log", exist_ok=True)


def _build_training_logger(timestamp):
    logger_name = f"ae_id_train_{timestamp}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    log_path = os.path.join("log", f"ae_identification_train_{timestamp}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    return logger, log_path


def _close_logger(logger):
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _save_training_history(history, timestamp):
    history_path = os.path.join("log", f"ae_identification_history_{timestamp}.csv")
    with open(history_path, "w", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,val_loss,macro_f1,exact_match_accuracy\n")
        for epoch_idx in range(len(history["epoch"])):
            handle.write(
                f"{history['epoch'][epoch_idx]},{history['train_loss'][epoch_idx]:.6f},"
                f"{history['val_loss'][epoch_idx]:.6f},{history['macro_f1'][epoch_idx]:.6f},"
                f"{history['exact_match_accuracy'][epoch_idx]:.6f}\n"
            )
    return history_path


def _save_loss_curve(history, timestamp):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["epoch"], history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("AE Identification Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join("visualization", f"AE_Identification_Loss_Curve_{timestamp}.png")
    fig.savefig(plot_path, dpi=600)
    plt.close(fig)
    return plot_path


def _prepare_mix_dataset(Raman_Shift, Intensity, Category, Concentration, config):
    X_mix, y_binary, y_abundance, combo_labels = digital_mix_ID_multilabel(
        Raman_Shift,
        Intensity,
        Category,
        Concentration,
        samples_per_combination=config.samples_per_combination,
        Range=(0.3, 10.0),
        seed=config.random_state,
    )
    X_mix = spectra_normalization(
        Raman_Shift,
        X_mix,
        peak_position=920,
        peak_range=20,
        plot=False,
        mode="ae_id_mix",
    ).astype(np.float32)
    y_binary = np.asarray(y_binary, dtype=np.float32)
    y_abundance = np.asarray(y_abundance, dtype=np.float32)
    return X_mix, y_binary, y_abundance, combo_labels

def _prepare_real_dataset(Raman_Shift, Intensity, Category, Concentration, config):
    combo_labels = np.array(
        [COMBINATION_TO_INDEX[tuple(row)] for row in zip(Category == 'DA', Category == 'E', Category == 'NE')],
        dtype=np.int64,
    )
    y_binary = np.column_stack([(Category == molecule).astype(np.float32) for molecule in ID_MOLECULES])
    category_to_abundance_index = {"DA": 0, "E": 1, "NE": 2, "BA": 3}
    y_abundance = np.zeros((len(Category), 4), dtype=np.float32)
    for idx, molecule in enumerate(Category):
        abundance_index = category_to_abundance_index.get(str(molecule))
        if abundance_index is not None:
            y_abundance[idx, abundance_index] = 1.0
    X_mix = spectra_normalization(
        Raman_Shift,
        Intensity,
        peak_position=920,
        peak_range=20,
        plot=False,
        mode="ae_id_real",
    ).astype(np.float32)
    return X_mix, y_binary, y_abundance, combo_labels


def _build_dataloaders(X, y_binary, y_abundance, combo_labels, config):
    unique_classes = np.unique(combo_labels)
    requested_val_count = int(np.ceil(len(X) * config.validation_size))
    stratify_labels = combo_labels if requested_val_count >= len(unique_classes) else None

    split = train_test_split(
        X,
        y_binary,
        y_abundance,
        combo_labels,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=stratify_labels,
    )
    X_train, X_val, y_train, y_val, a_train, a_val, combo_train, combo_val = split

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(a_train).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
        torch.from_numpy(a_val).float(),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, combo_train, combo_val, X_val, y_val


def _evaluate_model(model, data_loader, device, config):
    model.eval()
    losses = []
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for batch_x, batch_labels, batch_abundance in data_loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_labels = batch_labels.to(device=device, dtype=torch.float32)
            batch_abundance = batch_abundance.to(device=device, dtype=torch.float32)

            output = model(batch_x)
            cls_loss = F.binary_cross_entropy_with_logits(output["logits"], batch_labels)
            abundance_loss = F.mse_loss(output["abundance"], batch_abundance)
            recon_loss = F.mse_loss(output["reconstruction"], batch_x) + spectral_angle_distance(
                output["reconstruction"], batch_x
            )
            loss = (
                config.classification_weight * cls_loss
                + config.abundance_weight * abundance_loss
                + config.reconstruction_weight * recon_loss
            )
            losses.append(float(loss.detach().cpu()))
            logits_all.append(output["logits"].detach().cpu().numpy())
            labels_all.append(batch_labels.detach().cpu().numpy())

    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(np.int64)
    molecule_f1 = {
        molecule: float(f1_score(labels[:, idx], predictions[:, idx], zero_division=0))
        for idx, molecule in enumerate(ID_MOLECULES)
    }
    exact_match_accuracy = float(np.mean(np.all(predictions == labels, axis=1)))
    return {
        "loss": float(np.mean(losses)),
        "probabilities": probabilities,
        "predictions": predictions,
        "labels": labels,
        "molecule_f1": molecule_f1,
        "exact_match_accuracy": exact_match_accuracy,
    }


def AE_Identification_Train(Raman_Shift, Intensity, Category, Concentration, model_dir, plot=False, config=None, digital_mix=True):
    """
    Train a shared multi-label autoencoder identification model.
    """
    if config is None:
        config = TrainingConfig()

    _ensure_output_dirs()
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = _build_training_logger(timestamp)
    device = _default_device()
    print(f"Training AE identification model on device: {device}")
    logger.info("Training AE identification model on device: %s", device)
    logger.info("Training config: %s", config)

    if digital_mix:
        X_mix, y_binary, y_abundance, combo_labels = _prepare_mix_dataset(
            Raman_Shift, Intensity, Category, Concentration, config
        )
    else:
        X_mix, y_binary, y_abundance, combo_labels = _prepare_real_dataset(
            Raman_Shift, Intensity, Category, Concentration, config
        )
    logger.info(
        "Synthetic dataset prepared. X_mix=%s, y_binary=%s, y_abundance=%s",
        X_mix.shape,
        y_binary.shape,
        y_abundance.shape,
    )
    train_loader, val_loader, _combo_train, combo_val, X_val, y_val = _build_dataloaders(
        X_mix, y_binary, y_abundance, combo_labels, config
    )

    model = RamanMultiLabelAE(input_dim=X_mix.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_state = None
    best_metrics = None
    best_score = -np.inf
    patience_counter = 0
    best_epoch = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "macro_f1": [],
        "exact_match_accuracy": [],
    }

    try:
        for epoch in range(config.epochs):
            model.train()
            train_losses = []
            for batch_x, batch_labels, batch_abundance in train_loader:
                batch_x = batch_x.to(device=device, dtype=torch.float32)
                batch_labels = batch_labels.to(device=device, dtype=torch.float32)
                batch_abundance = batch_abundance.to(device=device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    output = model(batch_x)
                    cls_loss = F.binary_cross_entropy_with_logits(output["logits"], batch_labels)
                    abundance_loss = F.mse_loss(output["abundance"], batch_abundance)
                    recon_loss = F.mse_loss(output["reconstruction"], batch_x) + spectral_angle_distance(
                        output["reconstruction"], batch_x
                    )
                    loss = (
                        config.classification_weight * cls_loss
                        + config.abundance_weight * abundance_loss
                        + config.reconstruction_weight * recon_loss
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.detach().cpu()))

            validation_metrics = _evaluate_model(model, val_loader, device, config)
            macro_f1 = float(np.mean(list(validation_metrics["molecule_f1"].values())))
            score = 0.7 * macro_f1 + 0.3 * validation_metrics["exact_match_accuracy"]
            train_loss_mean = float(np.mean(train_losses))

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss_mean)
            history["val_loss"].append(validation_metrics["loss"])
            history["macro_f1"].append(macro_f1)
            history["exact_match_accuracy"].append(validation_metrics["exact_match_accuracy"])

            epoch_message = (
                f"AE IdModel Epoch {epoch + 1:03d}/{config.epochs} - "
                f"Train Loss: {train_loss_mean:.4f}, Val Loss: {validation_metrics['loss']:.4f}, "
                f"Macro F1: {macro_f1:.4f}, Exact Match: {validation_metrics['exact_match_accuracy']:.4f}"
            )
            print(epoch_message)
            logger.info(epoch_message)

            if score > best_score:
                best_score = score
                patience_counter = 0
                best_epoch = epoch + 1
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                best_metrics = validation_metrics
                best_payload = {
                    "model_state_dict": best_state,
                    "input_dim": int(X_mix.shape[1]),
                    "latent_dim": 4,
                    "classifier_dim": 3,
                    "molecules": ID_MOLECULES,
                    "thresholds": {
                        "DA": float(DA_PROB_THRESHOLD),
                        "E": float(E_PROB_THRESHOLD),
                        "NE": float(NE_PROB_THRESHOLD),
                    },
                    "validation_exact_match_accuracy": float(best_metrics["exact_match_accuracy"]),
                    "validation_molecule_f1": best_metrics["molecule_f1"],
                    "device_used_for_training": str(device),
                    "best_epoch": best_epoch,
                    "timestamp": timestamp,
                    "log_path": log_path,
                }
                torch.save(best_payload, os.path.join(model_dir, AE_ID_FILENAME))
                torch.save(best_payload, os.path.join(model_dir, "ae_identification_multilabel_best.pt"))
                logger.info("Best checkpoint updated at epoch %d with score %.6f", best_epoch, best_score)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    stop_message = f"Early stopping triggered at epoch {epoch + 1}."
                    print(stop_message)
                    logger.info(stop_message)
                    break

        if best_state is None:
            raise RuntimeError("AE identification training did not produce a valid checkpoint.")

        last_payload = {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "input_dim": int(X_mix.shape[1]),
            "latent_dim": 4,
            "classifier_dim": 3,
            "molecules": ID_MOLECULES,
            "thresholds": {
                "DA": float(DA_PROB_THRESHOLD),
                "E": float(E_PROB_THRESHOLD),
                "NE": float(NE_PROB_THRESHOLD),
            },
            "device_used_for_training": str(device),
            "last_epoch": history["epoch"][-1],
            "timestamp": timestamp,
            "log_path": log_path,
        }
        torch.save(last_payload, os.path.join(model_dir, "ae_identification_multilabel_last.pt"))

        metrics_payload = {
            "validation_exact_match_accuracy": float(best_metrics["exact_match_accuracy"]),
            "validation_molecule_f1": best_metrics["molecule_f1"],
            "combo_validation_accuracy": float(
                accuracy_score(
                    combo_val,
                    np.asarray(
                        [
                            COMBINATION_TO_INDEX[tuple(row.tolist())]
                            for row in best_metrics["predictions"].astype(int)
                        ],
                        dtype=np.int64,
                    ),
                )
            ),
            "validation_labels": y_val,
            "validation_probabilities": best_metrics["probabilities"],
            "validation_predictions": best_metrics["predictions"],
            "best_epoch": best_epoch,
            "history": history,
            "loss_curve_path": _save_loss_curve(history, timestamp),
            "log_path": log_path,
        }
        joblib.dump(metrics_payload, os.path.join(model_dir, AE_ID_METRICS_FILENAME))
        history_path = _save_training_history(history, timestamp)
        logger.info("Training history saved to %s", history_path)
        logger.info("Loss curve saved to %s", metrics_payload["loss_curve_path"])
        logger.info("Best checkpoint saved to %s", os.path.join(model_dir, "ae_identification_multilabel_best.pt"))
        logger.info("Last checkpoint saved to %s", os.path.join(model_dir, "ae_identification_multilabel_last.pt"))

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for idx, molecule in enumerate(ID_MOLECULES):
                cm = confusion_matrix(
                    best_metrics["labels"][:, idx],
                    best_metrics["predictions"][:, idx],
                    labels=[0, 1],
                )
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Not {molecule}", molecule])
                disp.plot(ax=axes[idx], cmap=plt.cm.Blues, colorbar=False)
                axes[idx].set_title(f"AE Identification {molecule}")
            fig.tight_layout()
            fig.savefig("visualization/AE_Identification_Confusion_Matrices.png", dpi=600)
            plt.show(block=False)
            plt.pause(5)
            plt.close(fig)

        return best_payload
    finally:
        _close_logger(logger)


def _load_trained_model(model_dir, map_location=None):
    if map_location is None:
        map_location = _default_device()
    checkpoint_path = os.path.join(model_dir, AE_ID_FILENAME)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    model = RamanMultiLabelAE(
        input_dim=payload["input_dim"],
        latent_dim=payload["latent_dim"],
        classifier_dim=payload["classifier_dim"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, payload


def AE_Identification_Predict_Multi(Intensity, model_dir):
    """
    Predict DA/E/NE presence probabilities with the shared AE model.
    """
    device = _default_device()
    model, payload = _load_trained_model(model_dir, map_location=device)
    intensity = np.asarray(Intensity, dtype=np.float32)

    with torch.no_grad():
        batch = torch.from_numpy(intensity).to(device)
        output = model(batch)
        probabilities = torch.sigmoid(output["logits"]).cpu().numpy()

    thresholds = payload["thresholds"]
    predictions = np.column_stack(
        [
            (probabilities[:, idx] >= thresholds[molecule]).astype(int)
            for idx, molecule in enumerate(ID_MOLECULES)
        ]
    )
    return predictions, probabilities


def AE_Identification_Predict(Intensity, CA, model_dir, plot=False, labels=None):
    """
    Predict one molecule's presence using the shared AE identification model.
    """
    predictions, probabilities = AE_Identification_Predict_Multi(Intensity, model_dir)
    if CA not in MOLECULE_TO_INDEX:
        raise ValueError(f"Unknown molecule '{CA}'. Expected one of {list(MOLECULE_TO_INDEX)}")

    molecule_index = MOLECULE_TO_INDEX[CA]
    molecule_predictions = predictions[:, molecule_index]
    molecule_probabilities = probabilities[:, molecule_index]

    if plot and labels is not None:
        cm = confusion_matrix(labels, molecule_predictions, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Not {CA}", CA])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {CA} Identification (AE)")
        plt.savefig(f"visualization/AE_Identification_{CA}_Confusion_Matrix.png", dpi=600)
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    return molecule_predictions, molecule_probabilities
