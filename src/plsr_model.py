"""
PLSR model for total analyte concentration quantification from SERS spectra.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_squared_error

PLSR_FILENAME = "plsr_concentration.joblib"


def PLSR_Train(Raman_Shift, Intensity, Concentration, Category, model_dir,
               max_components=15, n_splits=5, plot=True):
    """
    Train PLSR model to predict total analyte concentration from SERS spectra.

    Uses only pure analyte spectra (DA/E/NE at non-zero concentrations).
    BA (background) samples are excluded.

    Args:
        Raman_Shift (np.ndarray): Raman shift values.
        Intensity (np.ndarray): 2D array of intensity values (samples x features).
        Concentration (np.ndarray): Concentration values for each sample.
        Category (np.ndarray): Category labels (DA/E/NE/BA).
        model_dir (str): Directory to save the trained model.
        max_components (int): Maximum number of PLS components to test.
        n_splits (int): Number of cross-validation folds.
        plot (bool): Whether to plot diagnostic figures.

    Returns:
        dict: Best model payload with metrics.
    """
    analyte_mask = (Category != 'BA') & (Concentration > 0)
    X = Intensity[analyte_mask].copy()
    y = Concentration[analyte_mask].copy()

    print(f"PLSR training samples: {X.shape[0]}, features: {X.shape[1]}")
    print(f"Concentration range: {y.min():.2f} - {y.max():.2f} uM")

    n_max = min(max_components, X.shape[0] - 1, X.shape[1])
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_n_comp = 1
    best_rmse = np.inf
    cv_results = []

    for n_comp in range(1, n_max + 1):
        pls = PLSRegression(n_components=n_comp, scale=False)
        y_pred = cross_val_predict(pls, X, y, cv=cv)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        cv_results.append((n_comp, rmse, r2))
        print(f"  PLSR n_comp={n_comp:2d}: RMSE={rmse:.4f} uM, R2={r2:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_n_comp = n_comp

    print(f"Best n_components: {best_n_comp} (RMSE={best_rmse:.4f} uM)")

    final_model = PLSRegression(n_components=best_n_comp, scale=False)
    final_model.fit(X, y)
    y_train_pred = final_model.predict(X).ravel()
    train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
    train_r2 = r2_score(y, y_train_pred)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, PLSR_FILENAME)
    payload = {
        "model": final_model,
        "n_components": best_n_comp,
        "cv_rmse": best_rmse,
        "cv_r2": r2_score(y, cross_val_predict(
            PLSRegression(n_components=best_n_comp, scale=False), X, y, cv=cv)),
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "concentration_range": (float(y.min()), float(y.max())),
        "cv_results": cv_results,
    }
    joblib.dump(payload, model_path)
    print(f"PLSR model saved to {model_path}")
    print(f"  Train RMSE: {train_rmse:.4f} uM, R2: {train_r2:.4f}")

    if plot:
        _plot_plsr_diagnostics(y, y_train_pred, cv_results, best_n_comp)

    return payload


def PLSR_Predict(Intensity, model_dir, plot=False, true_concentrations=None):
    """
    Predict total analyte concentration using trained PLSR model.

    Args:
        Intensity (np.ndarray): 2D array of intensity values (samples x features).
        model_dir (str): Path to the trained model directory.
        plot (bool): Whether to plot predicted vs true concentrations.
        true_concentrations (np.ndarray): True total concentrations for validation.

    Returns:
        np.ndarray: Predicted total concentrations.
    """
    model_path = os.path.join(model_dir, PLSR_FILENAME)
    payload = joblib.load(model_path)
    model = payload['model']

    predictions = model.predict(Intensity).ravel()
    predictions = np.maximum(predictions, 0)

    if plot and true_concentrations is not None:
        _plot_prediction_scatter(true_concentrations, predictions, payload)

    return predictions


def _plot_plsr_diagnostics(y_true, y_pred, cv_results, best_n_comp):
    os.makedirs("visualization", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', linewidths=0.5)
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel('True Concentration (uM)')
    ax.set_ylabel('Predicted Concentration (uM)')
    ax.set_title(f'PLSR Training Fit\nRMSE={rmse:.3f} uM, R2={r2:.3f}')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    n_comps, rmses, r2s = zip(*cv_results)
    ax2 = ax.twinx()
    ax.plot(n_comps, rmses, 'b-o', linewidth=2, label='RMSE')
    ax2.plot(n_comps, r2s, 'r-s', linewidth=2, label='R2')
    ax.axvline(best_n_comp, color='gray', linestyle='--', alpha=0.7,
               label=f'Best n_comp={best_n_comp}')
    ax.set_xlabel('Number of PLS Components')
    ax.set_ylabel('RMSE (uM)', color='b')
    ax2.set_ylabel('R2', color='r')
    ax.set_title('PLSR Cross-Validation Results')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('visualization/PLSR_Training_Diagnostics.png', dpi=600)
    plt.show(block=False)
    plt.pause(5)
    plt.close(fig)


def _plot_prediction_scatter(y_true, y_pred, payload):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', linewidths=0.5)
    max_val = max(y_true.max(), y_pred.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel('True Total Concentration (uM)')
    ax.set_ylabel('Predicted Total Concentration (uM)')
    ax.set_title(f'PLSR Test Set Prediction\nRMSE={rmse:.3f} uM, R2={r2:.3f}')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('visualization/PLSR_Test_Prediction.png', dpi=600)
    plt.show(block=False)
    plt.pause(5)
    plt.close(fig)
