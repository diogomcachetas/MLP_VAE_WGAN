from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SMOTENC
from generator.paper.supervised_model_sep import *

# Set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# VAE-WGAN samples
X_generated = np.load("./MLP-VAE-WGAN/data/ML_DL_Dataset/generated_samples.npy")
y_generated = np.load("./MLP-VAE-WGAN/data/ML_DL_Dataset/y_train.npy")

X_train = np.load("./MLP-VAE-WGAN/data/ML_DL_Dataset/X_train.npy")
y_train = np.load("./MLP-VAE-WGAN/data/ML_DL_Dataset/y_train.npy")

X_test =np.load("./MLP-VAE-WGAN/data/ML_DL_Dataset/X_test.npy")
Y_test =np.load("./MLP-VAE-WGAN/data/ML_DL_Dataset/y_test.npy")

X_train_vae = np.concatenate((X_train, X_generated))
y_train_vae = np.concatenate((y_train, y_generated))

augmented_datasets = {
    "Original": (X_train, y_train),
    "VAE-WGAN Augmented": (X_train_vae, y_train_vae)
}

# Desired class counts
original_class_counts = Counter(y_train)
target_counts = {cls: cnt * 2 for cls, cnt in original_class_counts.items()}

# Oversamplers (excluding ADASYN for now)
oversamplers = {
    "SMOTE": SMOTE(sampling_strategy=target_counts, random_state=42),
    "RandomOverSampler": RandomOverSampler(sampling_strategy=target_counts, random_state=42),
    "BorderlineSMOTE": BorderlineSMOTE(sampling_strategy=target_counts, random_state=42)
}

# Add SMOTENC if needed
categorical_indices = []  # Update if needed
if categorical_indices:
    oversamplers["SMOTENC"] = SMOTENC(categorical_features=categorical_indices,
                                      sampling_strategy=target_counts, random_state=42)

# Apply oversamplers
for name, sampler in oversamplers.items():
    try:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        n_synthetic = len(X_resampled) - len(X_train)
        if n_synthetic > len(X_train):
            X_resampled = np.concatenate([X_train, X_resampled[-len(X_train):]])
            y_resampled = np.concatenate([y_train, y_resampled[-len(X_train):]])

        augmented_datasets[name] = (X_resampled, y_resampled)
        print(f"✔ {name} created {n_synthetic} synthetic samples. Final shape: {X_resampled.shape}")
    except Exception as e:
        print(f"⚠ {name} skipped: {e}")

# Classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

# Evaluate
metrics_list = []

for dataset_name, (X_train, y_train) in augmented_datasets.items():
    print(f"\n--- Dataset: {dataset_name} ---")
    print(f"Label distribution: {Counter(y_train)}")

    for model_name, model in classifiers.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        metrics_list.append({
            "Model": f"{dataset_name} - {model_name}",
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{precision * 100:.2f}%",
            "Recall": f"{recall * 100:.2f}%",
            "F1-score": f"{f1 * 100:.2f}%"
        })

# Save metrics
metrics_df = pd.DataFrame(metrics_list)
#metrics_df.to_csv("ml_classifiers_metrics.csv", index=False)
#print("\nSaved metrics to 'ml_classifiers_metrics.csv'")
print(metrics_df)
