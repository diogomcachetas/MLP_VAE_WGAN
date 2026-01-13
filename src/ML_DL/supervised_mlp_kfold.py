import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch.optim import Adam, LBFGS
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from generator.paper.supervised_model_sep import *
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import ConfusionMatrixDisplay

# Set environment variable to specify GPU (if you have multiple GPUs)
#os.environ['CUDA_VISIBLE_DEVICES'] = '7'  # Use GPU 0

# Disable cuDNN benchmark
torch.backends.cudnn.benchmark = False

def calculate_accuracy(model, test_loader):
    model.eval()  
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            classification = model(inputs)
            #classification = model(inputs.unsqueeze(1))
            predictions = torch.argmax(classification, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

def calculate_accuracy_1(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            classification = model(inputs)
            predictions = torch.argmax(classification, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy, all_labels, all_predictions

def perform_pca(latent_vectors, n_components=3):
    pca = PCA(n_components=n_components)
    latent_pca = pca.fit_transform(latent_vectors)
    explained_variance = pca.explained_variance_ratio_
    return latent_pca, explained_variance

def plot_3d_pca(ax, latent_pca, labels, explained_variance, title):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)
        ax.scatter(latent_pca[indices, 0], latent_pca[indices, 1], latent_pca[indices, 2], label=f'Label {label}')
    ax.legend()
    ax.set_xlabel(f'PCA Component 1 ({explained_variance[0]:.2f} variance)')
    ax.set_ylabel(f'PCA Component 2 ({explained_variance[1]:.2f} variance)')
    ax.set_zlabel(f'PCA Component 3 ({explained_variance[2]:.2f} variance)')
    ax.set_title(title)

def batch_normalization_with_train_stats(S_train, S_test):
    mean = S_train.mean(axis=0, keepdims=True)
    std = S_train.std(axis=0, keepdims=True)
    S_train_bn = (S_train - mean) / std
    S_test_bn = (S_test - mean) / std  # use same mean/std!
    return S_train_bn, S_test_bn

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load and preprocess data
df = pd.read_excel("./output/smartg/window_feature/rs_feat.xlsx", index_col=0)
#df = pd.read_excel("C:/Users/diogo/Desktop/groundwater/data/dataset/preprocessed_data.xlsx", index_col=0)

features, labels, wavenumbers, concentration = preprocess_data(df)

# Split the data into training and test sets
X_train_full, X_test, y_train_full, y_test, conc_train, conc_test = train_test_split(features, labels, concentration, test_size=0.2, random_state=99, stratify=labels, shuffle=True)
print(f"Train set shape: {X_train_full.shape}, Test set shape: {X_test.shape}")

#X_train_full, X_test = batch_normalization_with_train_stats(X_train_full, X_test)

X_generated = np.load("C:/Users/diogo/Desktop/final_models/mlp_vae_wgan/generated_samples.npy")
y_generated = np.load("C:/Users/diogo/Desktop/final_models/mlp_vae_wgan/train_labels.npy")

X_train_augmented = np.concatenate((X_train_full, X_generated))
y_train_augmented = np.concatenate((y_train_full, y_generated))

'''X_train_full = np.load("C:/Users/diogo/Desktop/benchmark/X_train.npy")
y_train_full = np.load("C:/Users/diogo/Desktop/benchmark/y_train.npy")

X_test = np.load("C:/Users/diogo/Desktop/benchmark/X_test.npy")
y_test = np.load("C:/Users/diogo/Desktop/benchmark/y_test.npy")'''

# Combine the original training data with the generated samples
#X_combined = np.concatenate((X_train_full, X_generated), axis=0)
#y_combined = np.concatenate((y_train_full, y_generated), axis=0)

#print(y_train_full, y_generated)

# Track best accuracy and average accuracy for each random state
std_accuracies = [] #

# Initialize StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)

# Set parameters
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
num_epochs = 200
weights = [0.5]
learning_rate = [0.0005]

best_epoch_per_seed = []  

best_val_epoch_per_seed = []
best_train_epoch_per_seed = []

# Initialize lists to store per-seed averages
seeds_avg_train_loss = []
seeds_avg_val_loss = []

avg_train_total_seeds = []
avg_val_total_seeds = []

# Split data
for r_s in range(0, 50):
    print(r_s)
    X_train, y_train = X_train_full, y_train_full
    print(X_train.shape)

    fold_accuracies = []

    all_train_classification_losses = []
    all_val_classification_losses = []
    all_train_accuracies = []
    all_val_accuracies = []

    # Iterate through each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_augmented, y_train_augmented)): #X_train_augmented, y_train_augmented #X_train, y_train

        train_accuracies_per_fold = []
        val_accuracies_per_fold = []

        train_classification_losses = []
        val_classification_losses = []

        print(f"Training fold {fold + 1}/{n_splits}...")

        X_train_fold, X_val_fold = X_train_augmented[train_idx], X_train_augmented[val_idx] #X_train
        y_train_fold, y_val_fold = y_train_augmented[train_idx], y_train_augmented[val_idx] #y_train

        #X_traingen_fold, X_valgen_fold = X_generated[train_idx], X_generated[val_idx]
        #y_traingen_fold, y_valgen_fold = y_generated[train_idx], y_generated[val_idx]

        # Combine the original training data with the generated samples
        #X_train_c = np.concatenate((X_train_fold, X_traingen_fold), axis=0)
        #y_train_c = np.concatenate((y_train_fold, y_traingen_fold), axis=0)
        
        # Combine the original training data with the generated samples
        #X_val_c = np.concatenate((X_val_fold, X_valgen_fold), axis=0)
        #y_val_c = np.concatenate((y_val_fold, y_valgen_fold), axis=0)

        # Prepare the validation data
        val_tensor = torch.tensor(X_val_fold, dtype=torch.float32) #X_val_fold
        val_labels_tensor = torch.tensor(y_val_fold, dtype=torch.long) #y_val_fold
        val_dataset = TensorDataset(val_tensor, val_labels_tensor)
        val_loader = DataLoader(val_dataset, batch_size=len(val_labels_tensor), shuffle=False)

        # Instantiate Models
        mlp = Classifier(input_dim=features.shape[1], num_classes=5).to(device)
        #mlp = Classifier(input_dim=1000, num_classes=30).to(device)

        # Train Datal
        X_tl = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_tl = torch.tensor(y_train_fold, dtype=torch.float32).to(device)

        #X_tl = torch.tensor(X_train, dtype=torch.float32).to(device)
        #y_tl = torch.tensor(y_train, dtype=torch.float32).to(device)

        #X_tl = torch.tensor(np.concatenate((X_train, X_generated), axis=0), dtype=torch.float32).to(device)
        #y_tl = torch.tensor(np.concatenate((y_train, y_generated), axis=0), dtype=torch.float32).to(device)

        data = DataLoader(TensorDataset(X_tl, y_tl), batch_size=128, shuffle=True, drop_last=False)

        for l_r in learning_rate:
            for weight in weights:
                mlp.apply(lambda m: weights_init(m, r_s))

                optimizer_mlp = Adam(mlp.parameters(), lr=l_r, betas=(0.5, 0.9)) #betas=(0.6, 0.9))

                # Initialize variables for tracking accuracy
                best_accuracy = 0 
                # Loss values
                mlp_losses = []
                accuracies = []

                for epoch in range(num_epochs):
                    mlp.train()                    
                    total_mlp_loss = 0

                    for batch in data:
                        optimizer_mlp.zero_grad()
                        x_batch = batch[0].to(device)
                        y_batch = batch[1].to(device).long()

                        #Adam
                        optimizer_mlp.zero_grad()

                        classification = mlp(x_batch)
                        loss = criterion(classification, y_batch)
                        #Backward pass and optimize
                        loss.backward()
                        torch_utils.clip_grad_norm_(mlp.parameters(), max_norm=1.5, foreach=True)
                        optimizer_mlp.step()
                        #scheduler.step()

                        # Store loss values regarding the number of batches
                        total_mlp_loss += loss.item() 
                        
                    # stop criteria
                    accuracy = calculate_accuracy(mlp, val_loader)
                    accuracies.append(accuracy)

                    # Validation Accuracy
                    val_accuracy = calculate_accuracy(mlp, val_loader)
                    val_accuracies_per_fold.append(val_accuracy)

                    # Training Accuracy
                    train_loader = DataLoader(TensorDataset(X_tl, y_tl.long()), batch_size=128, shuffle=False) #128
                    train_accuracy = calculate_accuracy(mlp, train_loader)
                    train_accuracies_per_fold.append(train_accuracy)

                    avg_train_loss = total_mlp_loss / len(X_tl)
                    train_classification_losses.append(avg_train_loss)

                    # Calculate val loss
                    mlp.eval()
                    with torch.no_grad():
                        val_loss = 0
                        total_samples = 0
                        for x_val_batch, y_val_batch in val_loader:
                            x_val_batch = x_val_batch.to(device)
                            y_val_batch = y_val_batch.to(device)
                            outputs = mlp(x_val_batch)
                            loss = criterion(outputs, y_val_batch)
                            val_loss += loss.item()

                    val_classification_losses.append(val_loss / len(y_val_batch))

                    all_val_classification_losses.append(val_classification_losses)
                    all_train_classification_losses.append(train_classification_losses)

                    # Check if this is the best accuracy for this random state
                    if accuracy > best_accuracy: #
                        best_accuracy = accuracy
                        best_epoch = epoch + 1 #
                        #torch.save(mlp.state_dict(), './output/generator/mlp_model.pth')
                    
                    if epoch + 1 == 48:
                        #torch.save(mlp.state_dict(), './final_models/augmented/model_1.pth')
                        

    all_train_accuracies.append(train_accuracies_per_fold)
    all_val_accuracies.append(val_accuracies_per_fold)
    avg_val_acc = np.mean(val_accuracies_per_fold, axis=0)
    best_epoch = np.argmax(avg_val_acc) + 1  # +1 because epoch index starts from 0

    # Compute average accuracy across folds per epoch
    avg_val_accuracies = np.mean(all_val_accuracies, axis=0)  # shape: (epochs,)
    avg_train_accuracies = np.mean(all_train_accuracies, axis=0)

    # Identify best epoch based on avg validation accuracy
    best_epoch_idx = int(np.argmax(avg_val_accuracies))  # 0-based
    best_val_acc = float(avg_val_accuracies[best_epoch_idx])
    corresponding_train_acc = float(avg_train_accuracies[best_epoch_idx])

    # Append to DataFrame collection
    best_epoch_per_seed.append(best_epoch_idx + 1)  # +1 for 1-based indexing
    best_val_epoch_per_seed.append(best_val_acc)
    best_train_epoch_per_seed.append(corresponding_train_acc)
    std_accuracies.append(np.std(val_accuracies_per_fold))

    # Compute average loss curves across folds
    avg_train_classification_loss = np.mean(all_train_classification_losses, axis=0)
    std_train_classification_loss = np.std(all_train_classification_losses, axis=0)
    avg_val_classification_loss = np.mean(all_val_classification_losses, axis=0)
    std_val_classification_loss = np.std(all_val_classification_losses, axis=0)

    seeds_avg_train_loss.append(avg_train_classification_loss)
    seeds_avg_val_loss.append(avg_val_classification_loss)

    # Font size settings
    title_font = 24
    label_font = 20
    tick_font = 18
    legend_font = 18

    # Plot average classification loss curves
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, num_epochs + 1)

    plt.plot(epochs, avg_train_classification_loss, label='Train Loss', color='#4CC9F0')
    plt.fill_between(epochs,
                    avg_train_classification_loss - std_train_classification_loss,
                    avg_train_classification_loss + std_train_classification_loss,
                    color='#4CC9F0', alpha=0.2)

    plt.plot(epochs, avg_val_classification_loss, label='Validation Loss', color='#7209B7')
    plt.fill_between(epochs,
                    avg_val_classification_loss - std_val_classification_loss,
                    avg_val_classification_loss + std_val_classification_loss,
                    color='#7209B7', alpha=0.2)

    #plt.title('Average Cross Entropy Loss', fontsize=title_font)
    plt.xlabel('Epoch', fontsize=label_font)
    plt.ylabel('Loss', fontsize=label_font)
    plt.xticks(fontsize=tick_font)
    plt.yticks(fontsize=tick_font)
    plt.ylim(0, 2.6)
    plt.legend(fontsize=legend_font, loc='lower right')
    plt.grid(True)
    plt.axvline(x=48, color='gray', linestyle='--', linewidth=1.5)      
    plt.tight_layout()
    plt.show()
    plt.close()

    # Compute average acc curves across folds
    avg_train_accuracies = np.mean(all_train_accuracies, axis=0)
    std_train_accuracies = np.std(all_train_accuracies, axis=0)
    avg_train_total_seeds.append(avg_train_accuracies)

    avg_val_accuracies = np.mean(all_val_accuracies, axis=0)
    std_val_accuracies = np.std(all_val_accuracies, axis=0)
    avg_val_total_seeds.append(avg_val_accuracies)

    # Plot average train and val accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_train_accuracies, label='Train Accuracy', color='#4CC9F0')
    plt.fill_between(epochs,
                    avg_train_accuracies - std_train_accuracies,
                    avg_train_accuracies + std_train_accuracies,
                    color='#4CC9F0', alpha=0.2)

    plt.plot(epochs, avg_val_accuracies, label='Validation Accuracy', color='#7209B7')
    plt.fill_between(epochs,
                    avg_val_accuracies - std_val_accuracies,
                    avg_val_accuracies + std_val_accuracies,
                    color='#7209B7', alpha=0.2)

    #plt.title('Average Accuracy', fontsize=title_font)
    plt.xlabel('Epoch', fontsize=label_font)
    plt.ylabel('Accuracy', fontsize=label_font)
    plt.xticks(fontsize=tick_font)
    plt.yticks(fontsize=tick_font)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=legend_font, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.axvline(x=48, color='gray', linestyle='--', linewidth=1.5)
    plt.show()
    plt.tight_layout()
    plt.close()

# Convert to numpy arrays
seeds_avg_train_loss = np.array(seeds_avg_train_loss)
seeds_avg_val_loss = np.array(seeds_avg_val_loss)

# Compute grand averages across seeds
final_avg_train_loss = np.mean(seeds_avg_train_loss, axis=0)
final_avg_val_loss = np.mean(seeds_avg_val_loss, axis=0)

final_avg_train_acc = np.mean(avg_train_total_seeds, axis=0)
final_avg_val_acc = np.mean(avg_val_total_seeds, axis=0)

# Save to CSV
import pandas as pd

epochs = np.arange(1, len(final_avg_train_loss) + 1)
df = pd.DataFrame({
    'Epoch': epochs,
    'Train_CrossEntropy': final_avg_train_loss,
    'Val_CrossEntropy': final_avg_val_loss,
    'Train_Accuracy': final_avg_train_acc,
    'Val_Accuracy': final_avg_val_acc
})
#df.to_csv('./output/mlp_vae_wgan/mlp_loss/cnn_average_curves_across_seeds.csv', index=False)

#Save seed metrics
accuracy_df = pd.DataFrame({
    'Random_State': list(range(0, 1)),  # update if more seeds used
    'Std_Accuracy': std_accuracies,
    'Best_Epoch': best_epoch_per_seed,
    'Avg_Val_Acc_at_Best_Epoch': best_val_epoch_per_seed,
    'Avg_Train_Acc_at_Best_Epoch': best_train_epoch_per_seed
})
#accuracy_df.to_csv('./output/mlp_vae_wgan/mlp_loss/cnn_accuracies.csv', index=False)

# The remaining code
print("Training completed.")

# Load the best model
best_model = Classifier(input_dim=features.shape[1], num_classes=5).to(device)
#best_model.load_state_dict(torch.load('./output/generator/mlp_model.pth'))

original_state_dict = torch.load("C:/Users/diogo/Desktop/final_models/augmented/model_1.pth")
new_state_dict = {}
for k, v in original_state_dict.items():
    new_key = f"model.{k}"  # Prepend 'model.' to each key
    new_state_dict[new_key] = v
best_model.load_state_dict(original_state_dict)

# Prepare the validation data
test_tensor = torch.tensor(X_test, dtype=torch.float32) #X_val_fold
test_labels_tensor = torch.tensor(y_test, dtype=torch.long) #y_val_fold
test_dataset = TensorDataset(test_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=len(test_labels_tensor), shuffle=False)

# Get predictions and plot confusion matrix
accuracy, y_true, y_pred = calculate_accuracy_1(best_model, test_loader)
print(f"Final Test Accuracy: {accuracy:.4f}")

labels_mapping = {0: 'Control', 1: 'Mixture', 2: 'Sulfamethoxazole', 3: 'Sulfapyridine', 4: 'Sulfathiazole'}

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Create display object with custom labels
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
                              display_labels=[labels_mapping[i] for i in np.unique(y_true)])

from matplotlib.colors import LinearSegmentedColormap

custom_cmap = LinearSegmentedColormap.from_list("custom_blue_purple", ["#ffffff", "#5F69D4"])

# Plot confusion matrix using same parameters
plt.figure(figsize=(8, 8))
disp.plot(cmap=custom_cmap, ax=plt.gca(), colorbar=False)

# Format axis and labels
ax = plt.gca()
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# Set axis titles with larger font
ax.set_xlabel(ax.get_xlabel(), fontsize=16)
ax.set_ylabel(ax.get_ylabel(), fontsize=16)

# Make text annotations bold and larger
for text in ax.texts:
    text.set_fontsize(20)
    text.set_fontweight('bold')

# Title
plt.title("MLP Confusion Matrix", fontsize=20)
plt.tight_layout()
plt.show()

# Print results
print("Confusion Matrix:")
print(conf_matrix)

df = pd.DataFrame({
    "True Label": y_true,
    "Predicted Label": y_pred,
    "Concentration": conc_test  # <- needs to be aligned with test set
})

df["Concentration"] = df["Concentration"].astype(str)  # or pd.Categorical(df["Concentration"])
df["Prediction"] = df["True Label"] == df["Predicted Label"]

# Sort the DataFrame by Concentration
df_sorted = df.sort_values(by="Concentration")

# Get the unique hue values (e.g., True/False or 0/1)
unique_hues = df_sorted["Prediction"].unique()
# Map colors from the "cool" colormap
colors = [plt.cm.cool(i / len(unique_hues)) for i in range(len(unique_hues))]

plt.figure(figsize=(10, 5))
sns.histplot(
    data=df_sorted,
    x="Concentration",
    hue="Prediction",
    multiple="stack",
    bins=20,
    palette=colors
)
#plt.title("Correct vs Incorrect Predictions by Concentration")

plt.xlabel("Concentration [mg/L]")
plt.ylabel("Number of Spectra")
plt.show()

# Create DataFrame for training set
train_df = pd.DataFrame({
    "Label": y_train,
    "Concentration": conc_train
})

# Ensure concentration is categorical for grouping
train_df["Concentration"] = train_df["Concentration"].astype(str)

# Plot
plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x="Label", hue="Concentration")

plt.title("Number of Samples per Label and Concentration (Training Set)")
plt.xlabel("Label")
plt.ylabel("Number of Samples")
plt.legend(title="Concentration")
plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print classification report (includes all above per class)
report = classification_report(y_true, y_pred, target_names=[labels_mapping[i] for i in np.unique(y_true)])
print("Classification Report:\n", report)

# If you want to log these as individual lines too
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
