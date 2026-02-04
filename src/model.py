"""
Model definitions and training functions for Transfer Learning
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import joblib

def RF_Identification_Train(Raman_Shift, Intensity, Category, CA, model_dir, plot = False):
    """
    Train Random Forest model to identify pure CA from blank SERS spectra.
    Args:
        Raman_Shift (np.ndarray): Array of Raman shift values.
        Intensity (np.ndarray): 2D array of intensity values (samples x features).
        Category (np.ndarray): Array of category labels for each sample.
        CA (str): Target Molecule.
        model_dir (str): Directory to save the trained model.
        plot (bool): Whether to plot feature importances.
    Returns:
        dict: A dictionary with keys: model, feature_indices, avg_accuracy, avg_f1,
              and avg_feature_importances.
    """    
    
    # filter data for category == CA or 'BA'
    filter_indices = (Category == CA) | (Category == 'BA')
    Intensity_filtered = Intensity[filter_indices]
    Category_filtered = Category[filter_indices]
    Labels = (Category_filtered == CA).astype(int)  # 1 for CA, 0 for BA

    n_features = Intensity_filtered.shape[1]
    n_iterations = 100
    test_size = 0.25

    accuracies = []
    f1_scores = []
    feature_importances = []

    for i in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(
            Intensity_filtered,
            Labels,
            test_size=test_size,
            random_state=42 + i,
            stratify=Labels
        )

        rf_model = RandomForestClassifier(
            n_estimators=1000,
            max_features="sqrt",
            max_depth=10,
            min_samples_split=5,
            random_state=42 + i,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        feature_importances.append(rf_model.feature_importances_)

        if (i+1) % 10 == 0 or i == n_iterations - 1:
            print(f"IdModel of {CA} Iteration {i+1}/{n_iterations} - Accuracy: {accuracies[-1]:.4f}, F1 Score: {f1_scores[-1]:.4f}")

    avg_accuracy = float(np.mean(accuracies))
    avg_f1 = float(np.mean(f1_scores))
    avg_feature_importances = np.mean(np.vstack(feature_importances), axis=0)

    top_k = min(50, n_features)
    top_feature_indices = np.argsort(avg_feature_importances)[::-1][:top_k]

    X_selected = Intensity_filtered[:, top_feature_indices]

    final_model = RandomForestClassifier(
        n_estimators=1000,
        max_features="sqrt",
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_selected, Labels)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"rf_identification_{CA}.joblib")
    payload = {
        "model": final_model,
        "feature_indices": top_feature_indices,
        "avg_accuracy": avg_accuracy,
        "avg_f1": avg_f1,
        "avg_feature_importances": avg_feature_importances,
    }
    joblib.dump(payload, model_path)

    if plot:
        plt.figure(figsize=(10, 6))
        # plot filtered spectra with feature importance bars
        # select one spectrum wich is not BA
        sample_index = np.where(Category_filtered == CA)[0][0]
        plt.plot(Raman_Shift, Intensity_filtered[sample_index, :], label=f'normalized spectum of {CA}')
        # select one spectrum which is Background
        sample_index_ba = np.where(Category_filtered == 'BA')[0][0]
        plt.plot(Raman_Shift, Intensity_filtered[sample_index_ba, :], label='normalized spectum of Background')
        # scale feature importance to 0-1
        scaled_importances = avg_feature_importances / np.max(avg_feature_importances)
        plt.bar(Raman_Shift[top_feature_indices], 
                scaled_importances[top_feature_indices], # scale to 0-1
                width=5, color='red', alpha=0.7, label='Top Features')
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.title(f'Filtered SERS Spectra and Top Features for {CA} Identification')
        plt.legend()
        plt.show()

    return payload

