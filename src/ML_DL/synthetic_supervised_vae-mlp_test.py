import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import random

from generator.paper.supervised_model import *

def calculate_accuracy(model, test_loader):
    model.eval()  
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            classification = model.classifier(inputs)
            #classification = model.module.classifier(inputs)
            #classification = model.classifier(inputs.unsqueeze(1))
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
            #classification = model(inputs.unsqueeze(1))
            predictions = torch.argmax(classification, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
print(device)

# Load and preprocess data
df = pd.read_excel("./output/smartg/window_feature/rs_feat.xlsx", index_col=0)
features, labels, wavenumbers = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=99)

# Track best accuracy and average accuracy for each random state
average_accuracies = [] #
std_accuracies = [] #

# Initialize StratifiedKFold

n_splits = 5 #ATTENTION

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)

# Split data
#X_train, y_train = features, labels
#print(labels.shape)

#criterion = torch.nn.CrossEntropyLoss()

# Set parameters
num_epochs = 800
weights = [0.5]
learning_rate = [0.0005]

random_state_acc_ = []

for r_s in range(0, 50):
    print(r_s)
    # Iterate through each fold
    five_fold_acc = []

    all_train_classification_losses = []
    all_test_classification_losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # Set seed for reproducibility
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

        train_total_losses = []
        train_classification_losses = []
        test_total_losses = []
        test_classification_losses = []

        print(f"Training fold {fold + 1}/{n_splits}...")

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        #np.save(os.path.join('./output/mlp_vae_wgan_30000/', f'X_val_fold{fold + 1}.npy'), X_val_fold)
        #np.save(os.path.join('./output/mlp_vae_wgan_30000/', f'y_val_fold{fold + 1}.npy'), y_val_fold)

        #np.save(os.path.join('./output/', f'X_fold{fold + 1}.npy'), X_train_fold)
        #np.save(os.path.join('./output/', f'y_fold{fold + 1}.npy'), y_train_fold)

        # Prepare the validation data
        val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        val_labels_tensor = torch.tensor(y_val_fold, dtype=torch.long)
        val_dataset = TensorDataset(val_tensor, val_labels_tensor)
        val_loader = DataLoader(val_dataset, batch_size=len(val_labels_tensor), shuffle=False)

        # Prepare the test data
        '''val_tensor = torch.tensor(X_test, dtype=torch.float32)
        val_labels_tensor = torch.tensor(y_test, dtype=torch.long)
        val_dataset = TensorDataset(val_tensor, val_labels_tensor)
        val_loader = DataLoader(val_dataset, batch_size=len(val_labels_tensor), shuffle=False)'''

        # Instantiate Models
        vae = VAE(input_dim=features.shape[1], latent_dim=32).to(device) #latent_dim=32
        discriminator = Discriminator(input_dim=features.shape[1]).to(device)
        
        # Train Data
        X_tl = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_tl = torch.tensor(y_train_fold, dtype=torch.float32).to(device)

        # Full data
        #X_tl = torch.tensor(X_train, dtype=torch.float32).to(device)
        #y_tl = torch.tensor(y_train, dtype=torch.float32).to(device)

        data = DataLoader(TensorDataset(X_tl, y_tl), batch_size=128, shuffle=True, drop_last=False)

        # weights
        vae.apply(lambda m: weights_init(m, r_s))
        discriminator.apply(lambda m: weights_init(m, r_s))

        # Adam 
        optimizer_vae = Adam(vae.parameters(), lr=0.0005, betas=(0.6, 0.9)) #lr=0.0005, betas=(0.6, 0.9)
        optimizer_discriminator = Adam(discriminator.parameters(), lr=0.0005, betas=(0.6, 0.9)) 

        # Define weights
        wasserstein_weight = 0.5
        normal_weight = 1 - 0.5

        # Initialize variables for tracking accuracy
        best_accuracy = 0.0

        # train loop
        for epoch in range(num_epochs):

            epoch_train_total_loss = 0
            epoch_train_classification_loss = 0
            num_train_batches = 0

            vae.train()   
            discriminator.train()
            epoch_generated_samples = []
            epoch_generated_labels = []
            
            # batch loop
            for batch in data:
                optimizer_discriminator.zero_grad()
                x_batch = batch[0].to(device)
                y_batch = batch[1].to(device).long()

                x_decoded, z_mean, z_log_var, z, classification = vae(x_batch)
                vae_loss_value, kl_loss, mae_loss, mse_loss, classification_loss = vae_loss(x_batch, x_decoded, z_mean, z_log_var, classification, y_batch)

                # Train discriminator
                real_samples_output = discriminator(x_batch)
                fake_samples_output = discriminator(x_decoded.detach())
                discriminator_loss = wasserstein_loss(real_samples_output, fake_samples_output)
                gradient_penalty = compute_gradient_penalty(discriminator, x_batch, x_decoded.detach())
                discriminator_loss = discriminator_loss + gradient_penalty * 10
                # Update discriminator
                discriminator_loss.backward()
                optimizer_discriminator.step()

                # Train VAE
                optimizer_vae.zero_grad()
                fake_samples_output = discriminator(x_decoded)#.detach()) #better not to use detach
                discriminator_loss_g = wasserstein_loss_g(fake_samples_output)
                # Combine losses with weights
                total_loss = (normal_weight * vae_loss_value) + (wasserstein_weight * discriminator_loss_g)
                # Update VAE
                total_loss.backward()
                torch_utils.clip_grad_norm_(vae.parameters(), max_norm=1.5, foreach=True)
                optimizer_vae.step()

                epoch_train_total_loss += total_loss.item()
                epoch_train_classification_loss += classification_loss.item()
                num_train_batches += 1

                # Store generated samples, labels, mean, and logvar
                epoch_generated_samples.append(x_decoded.detach().cpu().numpy())
                epoch_generated_labels.append(classification.detach().cpu().numpy())

            train_total_losses.append(epoch_train_total_loss / len(X_tl))
            train_classification_losses.append(epoch_train_classification_loss / len(X_tl))

            vae.eval()
            test_total_loss = 0
            test_classification_loss = 0
            num_test_batches = 0

            with torch.no_grad():
                for x_test_batch, y_test_batch in val_loader:
                    x_test_batch = x_test_batch.to(device)
                    y_test_batch = y_test_batch.to(device).long()

                    x_decoded, z_mean, z_log_var, z, classification = vae(x_test_batch)
                    classification_loss___ = F.cross_entropy(classification, y_test_batch, reduction='sum')
                    vae_loss_value, _, _, _, classification_loss_val = vae_loss(x_test_batch, x_decoded, z_mean, z_log_var, classification, y_test_batch)

                    fake_samples_output = discriminator(x_decoded)
                    discriminator_loss_g = wasserstein_loss_g(fake_samples_output)
                    total_loss_val = (normal_weight * vae_loss_value) + (wasserstein_weight * discriminator_loss_g)

                    test_total_loss += total_loss_val.item()
                    #test_classification_loss += classification_loss_val.item()
                    test_classification_loss += classification_loss___.item()

                    #print(classification_loss___)
                    num_test_batches += 1

            test_total_losses.append(test_total_loss/ len(x_test_batch))# / num_test_batches)
            test_classification_losses.append(test_classification_loss / len(x_test_batch))# / num_test_batches)

            all_test_classification_losses.append(test_classification_losses)
            all_train_classification_losses.append(train_classification_losses)

            # Check stopping criteria
            accuracy = calculate_accuracy(vae, val_loader)
            #print(accuracy)
            # Check if this is the best accuracy for this random state
            #if accuracy >= best_accuracy: 
            if accuracy > best_accuracy: 
                best_accuracy = accuracy
                epoch_number = epoch
                X_generated = np.concatenate(epoch_generated_samples)
                y_generated = np.argmax(np.concatenate(epoch_generated_labels), axis=-1)
                #np.save(f'./output/mlp_vae_wgan_30000/fold{fold+1}_generated_samples.npy', X_generated)
                #np.save(f'./output/mlp_vae_wgan_30000/fold{fold+1}_generated_labels.npy', y_generated)
                #torch.save(vae.classifier.state_dict(), f'./output/mlp_vae_wgan_30000/classifier_fold{fold+1}.pth')

        print(f'VAE ACC: {best_accuracy} at epoch {epoch_number}')
        five_fold_acc.append(best_accuracy)
       
        results_df = pd.DataFrame({
            'Epoch': list(range(1, num_epochs + 1)),
            'Train_Total_Loss': train_total_losses,
            'Train_Classification_Loss': train_classification_losses,
            'Test_Total_Loss': test_total_losses,
            'Test_Classification_Loss': test_classification_losses,
        })
        #results_df.to_csv(f'./output/mlp_vae_wgan_30000/losses_fold{fold+1}_seed{r_s}.csv', index=False)

        '''plt.figure(figsize=(10, 6))
        #plt.plot(results_df['Train_Total_Loss'], label='Train Total Loss', color='blue')
        #plt.plot(results_df['Test_Total_Loss'], label='Test Total Loss', color='orange')
        plt.plot(results_df['Train_Classification_Loss'], label='Train Classification Loss', color='green', linestyle='--')
        plt.plot(results_df['Test_Classification_Loss'], label='Test Classification Loss', color='red', linestyle='--')
        plt.title(f'Loss Curves (Fold {fold+1}, Seed {r_s})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./output/mlp_vae_wgan_30000/loss_plot_fold{fold+1}_seed{r_s}.png')
        plt.close()'''

    random_state_acc_.append(np.mean(five_fold_acc))

# Compute average loss curves across folds
avg_train_classification_loss = np.mean(all_train_classification_losses, axis=0)
std_train_classification_loss = np.std(all_train_classification_losses, axis=0)

avg_test_classification_loss = np.mean(all_test_classification_losses, axis=0)
std_test_classification_loss = np.std(all_test_classification_losses, axis=0)

# Plot average classification loss curves
plt.figure(figsize=(10, 6))
epochs = np.arange(1, num_epochs + 1)

plt.plot(epochs, avg_train_classification_loss, label='Avg Train Classification Loss', color='green', linestyle='--')
plt.fill_between(epochs,
                 avg_train_classification_loss - std_train_classification_loss,
                 avg_train_classification_loss + std_train_classification_loss,
                 color='green', alpha=0.2)

plt.plot(epochs, avg_test_classification_loss, label='Avg Test Classification Loss', color='red', linestyle='--')
plt.fill_between(epochs,
                 avg_test_classification_loss - std_test_classification_loss,
                 avg_test_classification_loss + std_test_classification_loss,
                 color='red', alpha=0.2)

plt.title('Average Classification Loss Across 5 Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('./output/mlp_vae_wgan_30000/avg_train_test_classification_loss.png')
plt.close()

print(f"#Mean accuracy across all folds: {np.mean(random_state_acc_)}")

# Save the best and average accuracies to a file
#accuracy_df = pd.DataFrame({'Random_State': range(45, 50), 'Average_Accuracy': (random_state_acc_)})
#accuracy_df.to_csv('./output/artigo/45_50_cnn_vae_wgan_accuracies_50000.csv', index=False)

# The remaining code
print("Training completed.")