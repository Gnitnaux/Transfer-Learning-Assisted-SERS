"""
Model definitions and training functions for Transfer Learning
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
        f1_scores.append(f1_score(y_test, y_pred, average='binary'))
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

def RF_Identification_Predict(Intensity, CA, model_dir, plot = False, labels = None):
    """
    Predict using trained Random Forest model to identify pure CA from blank SERS spectra.
    Args:
        Raman_Shift (np.ndarray): Array of Raman shift values.
        Intensity (np.ndarray): 2D array of intensity values (samples x features).
        CA (str): Target Molecule.
        model_dir (str): Path to the trained model.
        plot (bool): Whether to plot confusion matrix.
        lables (np.ndarray): True labels for the samples (optional, for plotting confusion matrix).
    Returns:
        np.ndarray: Predictions (0 or 1) for each sample.
        np.ndarray: Probabilities for class 1 (CA) for each sample.
    """
    model_path = os.path.join(model_dir, f"rf_identification_{CA}.joblib")
    payload = joblib.load(model_path)
    model = payload['model']
    feature_indices = payload['feature_indices']

    X_selected = Intensity[:, feature_indices]
    predictions = model.predict(X_selected)
    probabilities = model.predict_proba(X_selected)[:, 1]  # Probability of class 1 (CA)

    if plot and labels is not None:
        cm = confusion_matrix(labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Not {CA}', CA])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {CA} Identification')
        plt.show()

    return predictions, probabilities

def RF_Ratio_Train(Raman_Shift, Intensity, Category, CAs, model_dir, plot = False):
    """
    Train Random Forest model to predict concentration ratios between two CAs.
    Args:
        Raman_Shift (np.ndarray): Array of Raman shift values.
        Intensity (np.ndarray): 2D array of intensity values (samples x features).
        Category (np.ndarray): Array of category labels for each sample.
        CAs (list): List of st least two target Molecules [CA1, CA2, ...].
        model_dir (str): Directory to save the trained model.
        plot (bool): Whether to plot feature importances.
    Returns:
        dict: A dictionary with keys: model, feature_indices, avg_rmse,
              and avg_feature_importances.
    """    
    CA_num = len(CAs)
    if CA_num < 2:
        raise ValueError("At least two CAs are required for ratio prediction.")
    
    # filter data for categories in CAs
    filter_indices = np.isin(Category, CAs)
    Intensity_filtered = Intensity[filter_indices]
    Category_filtered = Category[filter_indices]
    Labels = np.array([CAs.index(cat) for cat in Category_filtered])  # 0 for CA1, 1 for CA2, ...

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
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        feature_importances.append(rf_model.feature_importances_)

        if (i+1) % 10 == 0 or i == n_iterations - 1:
            print(f"RatioModel of {CAs} Iteration {i+1}/{n_iterations} - Accuracy: {accuracies[-1]:.4f}, F1 Score: {f1_scores[-1]:.4f}")

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
    model_filename = 'rf_ratio'
    # ca in CAs may be in any order, but model filename is fixed order [DA, E, NE]
    for ca in ['DA', 'E', 'NE']:
        if ca in CAs:
            model_filename += f"_{ca}"
    model_path = os.path.join(model_dir, f"{model_filename}.joblib")
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
        # select one spectrum for each category
        for ca in CAs:
            sample_index = np.where(Category_filtered == ca)[0][0]
            plt.plot(Raman_Shift, Intensity_filtered[sample_index, :], label=f'normalized spectrum of {ca}')

        # scale feature importance to 0-1
        scaled_importances = avg_feature_importances / np.max(avg_feature_importances)
        plt.bar(Raman_Shift[top_feature_indices], 
                scaled_importances[top_feature_indices], # scale to 0-1
                width=5, color='red', alpha=0.7, label='Top Features')
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.title(f'Filtered SERS Spectra and Top Features for {CAs} Identification')
        plt.legend()
        plt.show()

    return payload   


def RF_Ratio_Predict(Intensity, CAs, model_dir, plot = False, labels = None):
    """
    Predict using trained Random Forest model to predict concentration ratios between two CAs.
    Args:
        Intensity (np.ndarray): 2D array of intensity values (samples x features).
        CAs (list): List of at least two target Molecules [CA1, CA2, ...].
        model_dir (str): Path to the trained model.
        plot (bool): Whether to plot confusion matrix.
        labels (np.ndarray): True labels for the samples (optional, for plotting confusion matrix).
    Returns:
        np.ndarray: Predictions for each sample.
        np.ndarray: Probabilities for each class for each sample.
    """
    model_filename = 'rf_ratio'
    # ca in CAs may be in any order, but model filename is fixed order [DA, E, NE]
    for ca in ['DA', 'E', 'NE']:
        if ca in CAs:
            model_filename += f"_{ca}"
    model_path = os.path.join(model_dir, f"{model_filename}.joblib")
    payload = joblib.load(model_path)
    model = payload['model']
    feature_indices = payload['feature_indices']

    X_selected = Intensity[:, feature_indices]
    predictions = model.predict(X_selected)
    probabilities = model.predict_proba(X_selected)  # Probabilities for each class 

    Labels = None
    if labels is not None:
        Labels = np.array([CAs.index(cat) for cat in labels])  # 0 for CA1, 1 for CA2, ...

    if plot and Labels is not None:
        cm = confusion_matrix(Labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CAs)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {"-".join(CAs)} Ratio Prediction')
        plt.show()

    return predictions, probabilities


