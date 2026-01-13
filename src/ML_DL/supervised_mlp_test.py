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
            #classification = model.classifier(inputs.unsqueeze(1))
            predictions = torch.argmax(classification, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load and preprocess data
df = pd.read_excel("./output/smartg/window_feature/rs_feat.xlsx", index_col=0)

features, labels, wavenumbers = preprocess_data(df)

# Split the data into training and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(features, labels, test_size=0.2, random_state=99, stratify=labels, shuffle=True)
print(f"Train set shape: {X_train_full.shape}, Test set shape: {X_test.shape}")

# Track best accuracy and average accuracy for each random state
average_accuracies = [] #
std_accuracies = [] #

# Initialize StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)

# Set parameters
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 200
weights = [0.5]
learning_rate = [0.0005]

# Split data
for r_s in range(0, 50):
    #X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=2/10, random_state=29, stratify=labels, shuffle=True)
    X_train, y_train = X_train_full, y_train_full
    print(X_train.shape)

    #X_generated = np.load('./output/generator/generated_samples.npy')
    #y_generated = np.argmax(np.load('./output/generator/generated_labels.npy'), axis=-1)

    # Combine the original training data with the generated samples
    #X_train = np.concatenate((X_train, X_generated), axis=0)
    #y_train = np.concatenate((y_train, y_generated), axis=0)

    fold_accuracies = []

    # Iterate through each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Training fold {fold + 1}/{n_splits}...")

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Prepare the validation data
        val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        val_labels_tensor = torch.tensor(y_val_fold, dtype=torch.long)
        val_dataset = TensorDataset(val_tensor, val_labels_tensor)
        val_loader = DataLoader(val_dataset, batch_size=len(val_labels_tensor), shuffle=False)

        # Prepare Test Data
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_labels_tensor = torch.tensor(y_test, dtype=torch.long)
        test_dataset = TensorDataset(test_tensor, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=len(val_labels_tensor), shuffle=False)

        # Instantiate Models
        mlp = Classifier(input_dim=features.shape[1], num_classes=5).to(device)

        # Train Data
        X_tl = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_tl = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
        data = DataLoader(TensorDataset(X_tl, y_tl), batch_size=128, shuffle=True, drop_last=False)

        # full data
        #data = DataLoader(TensorDataset(torch.tensor(X_train_full, dtype=torch.float32).to(device), torch.tensor(y_train_full, dtype=torch.float32).to(device)), batch_size=128, shuffle=True, drop_last=False)

        for l_r in learning_rate:
            for weight in weights:
                mlp.apply(lambda m: weights_init(m, r_s))

                optimizer_mlp = Adam(mlp.parameters(), lr=l_r, betas=(0.6, 0.9))  

                # Initialize variables for tracking accuracy
                best_accuracy = 0 

                # Loss values
                mlp_losses = []
                accuracies = []

                num_batches = len(data)
                best_accuracy = 0.0

                train_losses = []
                test_losses = []
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

                        # Store loss values regarding the number of batches
                        total_mlp_loss += loss.item() / len(x_batch)
                        
                    # Check stopping criteria
                    accuracy = calculate_accuracy(mlp, test_loader)
                    accuracies.append(accuracy)

                    avg_train_loss = total_mlp_loss / num_batches
                    train_losses.append(avg_train_loss)

                    # Calculate test loss
                    mlp.eval()
                    with torch.no_grad():
                        test_loss = 0
                        total_samples = 0
                        for x_test_batch, y_test_batch in test_loader:
                            x_test_batch = x_test_batch.to(device)
                            y_test_batch = y_test_batch.to(device)
                            outputs = mlp(x_test_batch)
                            loss = criterion(outputs, y_test_batch)
                            test_loss += loss.item() #* x_test_batch.size(0)
                        test_losses.append(test_loss)

                    # Check if this is the best accuracy for this random state
                    if accuracy > best_accuracy: #
                        best_accuracy = accuracy
                        best_epoch = epoch + 1 #
                        torch.save(mlp.state_dict(), './output/generator/mlp_model.pth')

                    #if (epoch + 1) % 1000 == 0: 
                    #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_mlp_loss / num_batches}, Test Accuracy: {accuracy:.4f}')

                plt.figure(figsize=(10, 6))
                plt.plot(train_losses, label='Training Loss', color='blue')
                plt.plot(test_losses, label='Test Loss', color='orange')
                plt.xlabel('Epoch')
                plt.ylabel('Cross-Entropy Loss')
                plt.title('Training and Test Loss over Epochs')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

        print(f"Best accuracy for fold {fold + 1}: {best_accuracy} at epoch n.º: {best_epoch}")
        fold_accuracies.append(best_accuracy)

    # Compute the mean accuracy across all folds
    average_accuracies.append(np.mean(fold_accuracies))
    std_accuracies.append(np.std(fold_accuracies))
    print(f"Mean accuracy across all folds: {np.mean(fold_accuracies)}")

# Save the best and average accuracies to a file
#accuracy_df = pd.DataFrame({'Random_State': range(5), 'Std_Accuracy': std_accuracies, 'Average_Accuracy': average_accuracies})
#accuracy_df.to_csv('./output/window_feature/mlp_accuracies.csv', index=False)

# The remaining code
print("Training completed.")

# Load the best model
best_model = Classifier(input_dim=features.shape[1], num_classes=5).to(device)
best_model.load_state_dict(torch.load('./output/generator/mlp_model.pth'))

# Get predictions and plot confusion matrix
accuracy, y_true, y_pred = calculate_accuracy(best_model, test_loader)
print(f"Final Test Accuracy: {accuracy:.4f}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
# Visualize the confusion matrix with enhanced clarity
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_true),
            yticklabels=np.unique(y_true),
            cbar_kws={'label': 'Number of predictions'}, 
            linewidths=.5, linecolor='black',
            annot_kws={"size": 16, "color": "black"})  # Set annotation size and color
# Setting the color of text based on the cell background
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        text_color = 'white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black'
        plt.text(j + 0.5, i + 0.5, f'{conf_matrix[i, j]}', 
                ha='center', va='center', color=text_color, fontsize=16)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# Print results
print("Confusion Matrix:")
print(conf_matrix)