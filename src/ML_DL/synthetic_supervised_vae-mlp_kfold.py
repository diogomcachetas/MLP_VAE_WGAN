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

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import random

from generator.paper.supervised_model import *

from torch.optim.lr_scheduler import CosineAnnealingLR

def calculate_accuracy(model, test_loader):
    model.eval()  
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            classification = model.classifier(inputs)

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

print(torch.__version__)

# Load and preprocess data
df = pd.read_excel("./output/smartg/window_feature/rs_feat.xlsx", index_col=0)

features, labels, wavenumbers, concentration = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=99)

#np.save(f'./final_models/mlp_vae_wgan/train_features.npy', X_train)

# Track best accuracy and average accuracy for each random state
average_accuracies = [] 
std_accuracies = [] 

# Initialize StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)

# Set epochs
num_epochs = 800 

random_state_acc_ = []

best_val_epoch_per_seed = []
best_train_epoch_per_seed = []

seeds_avg_train_loss = []
seeds_avg_val_loss = []
avg_train_total_seeds = []
avg_val_total_seeds = []

for r_s in range(0, 50):
    print(r_s)
    # Iterate through each fold
    five_fold_acc = []

    all_train_total_losses = []
    all_val_total_losses = []
    all_train_classification_losses = []
    all_val_classification_losses = []

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
        val_total_losses = []
        val_classification_losses = []

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
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_labels_tensor = torch.tensor(y_test, dtype=torch.long)
        test_dataset = TensorDataset(test_tensor, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=len(test_labels_tensor), shuffle=False)

        # Instantiate Models
        vae = VAE(input_dim=features.shape[1], latent_dim=32).to(device) #latent_dim=32
        discriminator = Discriminator(input_dim=features.shape[1]).to(device)
        
        # Train Data
        #X_tl = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        #y_tl = torch.tensor(y_train_fold, dtype=torch.float32).to(device)

        X_tl = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_tl = torch.tensor(y_train, dtype=torch.float32).to(device)
        
        indexes = torch.arange(len(X_tl)) 
        index_tl = indexes.to(device) 

        data = DataLoader(TensorDataset(X_tl, y_tl, index_tl), batch_size=128, shuffle=True, drop_last=False) 
        train_loader = DataLoader(TensorDataset(X_tl, y_tl), batch_size=len(X_tl), shuffle=True, drop_last=False)

        # weights
        vae.apply(lambda m: weights_init(m, r_s))
        discriminator.apply(lambda m: weights_init(m, r_s))

        # Adam 
        optimizer_vae = Adam(vae.parameters(), lr=0.0005, betas=(0.6, 0.9)) #lr=0.0005, betas=(0.6, 0.9)
        optimizer_discriminator = Adam(discriminator.parameters(), lr=0.0005, betas=(0.0, 0.9)) #0.0001
        #scheduler_vae = CosineAnnealingLR(optimizer_vae, T_max=num_epochs, eta_min=1e-5) #1e-5 #5e-6

        # Define weights
        wasserstein_weight = 0.5
        normal_weight = 1 - wasserstein_weight

        # Initialize variables for tracking accuracy
        best_accuracy = 0.0
        best_test_accuracy = 0.0
        best_train_accuracy = 0.0

        # Add global lists for accuracy tracking
        all_val_accuracies = []
        all_test_accuracies = []
        all_train_accuracies = []

        # train loop
        for epoch in range(num_epochs):

            epoch_train_total_loss = 0
            epoch_train_classification_loss = 0

            vae.train()   
            discriminator.train()
            epoch_generated_samples = []
            epoch_generated_labels = []
            epoch_generated_indices = []

            #epoch_generated_samples = np.zeros((len(X_tl), features.shape[1]))  # new
            #epoch_generated_labels = np.zeros(len(X_tl), dtype=int) # new
            
            # batch loop
            for batch in data:
                #optimizer_discriminator.zero_grad()
                
                x_batch = batch[0].to(device)
                y_batch = batch[1].to(device).long()
                idx = batch[2]

                x_decoded, z_mean, z_log_var, z, classification = vae(x_batch)
                vae_loss_value, kl_loss, mae_loss, mse_loss, classification_loss = vae_loss(x_batch, x_decoded, z_mean, z_log_var, classification, y_batch, epoch+1, num_epochs)

                #for _ in range(5):
                # Train discriminator
                optimizer_discriminator.zero_grad() 
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
                torch_utils.clip_grad_norm_(vae.parameters(), max_norm=1.5, foreach=True) #max_norm=0.5 #max_norm=1.5
                optimizer_vae.step()
                #scheduler_vae.step()   

                epoch_train_total_loss += total_loss.item()
                epoch_train_classification_loss += classification_loss.item()

                epoch_generated_samples.append(x_decoded.detach().cpu().numpy())
                epoch_generated_labels.append(classification.detach().cpu().numpy())
                epoch_generated_indices.append(idx.cpu().numpy())

            train_total_losses.append(epoch_train_total_loss / len(X_tl))
            
            train_classification_losses.append(epoch_train_classification_loss / len(X_tl))

            vae.eval()
            test_total_loss = 0
            test_classification_loss = 0
            num_test_batches = 0

            with torch.no_grad():
                for x_val_batch, y_val_batch in val_loader:
                    x_val_batch = x_val_batch.to(device)
                    y_val_batch = y_val_batch.to(device).long()

                    x_decoded, z_mean, z_log_var, z, classification = vae(x_val_batch)
                    classification_loss___ = F.cross_entropy(classification, y_val_batch, reduction='sum')#, reduction='sum')
                    vae_loss_value, _, _, _, _ = vae_loss(x_val_batch, x_decoded, z_mean, z_log_var, classification, y_val_batch, epoch+1, num_epochs)

                    fake_samples_output = discriminator(x_decoded)
                    discriminator_loss_g = wasserstein_loss_g(fake_samples_output)
                    total_loss_val = (normal_weight * vae_loss_value) + (wasserstein_weight * discriminator_loss_g)

                    test_total_loss += total_loss_val.item()
                    test_classification_loss += classification_loss___.item()

            val_total_losses.append(test_total_loss/ len(x_val_batch))
            val_classification_losses.append(test_classification_loss / len(x_val_batch))

            all_train_total_losses.append(train_total_losses)
            all_val_classification_losses.append(val_classification_losses)
            all_train_classification_losses.append(train_classification_losses)

            # Check stopping criteria
            accuracy = calculate_accuracy(vae, val_loader)
            accuracy_test = calculate_accuracy(vae, test_loader)
            accuracy_train = calculate_accuracy(vae, train_loader)
            #print(accuracy)

            if epoch == 0:
                val_accuracies = []
                test_accuracies = []
                train_accuracies = []

            val_accuracies.append(accuracy)
            test_accuracies.append(accuracy_test)
            train_accuracies.append(accuracy_train)

            # Check if this is the best accuracy for this random state
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_test_accuracy = accuracy_test
                best_train_accuracy = accuracy_train
                epoch_number = epoch
                #X_generated = np.concatenate(epoch_generated_samples)
                #y_generated = np.argmax(np.concatenate(epoch_generated_labels), axis=-1)
                
                #torch.save(vae.classifier.state_dict(), f'./output/mlp_vae_wgan/classifier_fold{fold+1}.pth')

                #print(f'VAE ACC at epoch {epoch_number}:\nTrain: {best_train_accuracy}\nVal: {best_accuracy}\nTest: {best_test_accuracy}')

        five_fold_acc.append(best_accuracy)
        all_val_accuracies.append(val_accuracies)
        all_test_accuracies.append(test_accuracies)
        all_train_accuracies.append(train_accuracies)

    avg_train_loss = np.mean(all_train_classification_losses, axis=0)
    avg_val_loss = np.mean(all_val_classification_losses, axis=0)
    avg_train_acc = np.mean(train_accuracies, axis=0)
    avg_val_acc = np.mean(val_accuracies, axis=0)

    seeds_avg_train_loss.append(avg_train_loss)
    seeds_avg_val_loss.append(avg_val_loss)

    random_state_acc_.append(np.mean(five_fold_acc))

    average_accuracies.append(np.mean(five_fold_acc))
    std_accuracies.append(np.std(five_fold_acc))

    best_val_epoch_per_seed.append({'epoch': epoch_number, 'val_accuracy': best_accuracy})
    #best_train_epoch_per_seed.append(np.argmax(train_accuracies)) 
    best_train_epoch_per_seed.append(train_accuracies[epoch_number])

    # Compute and plot average accuracy
    avg_val_accuracy = np.mean(all_val_accuracies, axis=0)
    avg_val_total_seeds.append(avg_val_accuracy)
    std_val_accuracy = np.std(all_val_accuracies, axis=0)
    avg_test_accuracy = np.mean(all_test_accuracies, axis=0)
    std_test_accuracy = np.std(all_test_accuracies, axis=0)
    avg_train_accuracy = np.mean(all_train_accuracies, axis=0)
    avg_train_total_seeds.append(avg_train_accuracy)
    std_train_accuracy = np.std(all_train_accuracies, axis=0)
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, num_epochs + 1)
    plt.plot(epochs, avg_train_accuracy, label='Avg Train Accuracy', color='blue')
    plt.fill_between(epochs,
                    avg_train_accuracy - std_train_accuracy,
                    avg_train_accuracy + std_train_accuracy,
                    color='blue', alpha=0.2)
    plt.plot(epochs, avg_val_accuracy, label='Avg Validation Accuracy', color='orange')
    plt.fill_between(epochs,
                    avg_val_accuracy - std_val_accuracy,
                    avg_val_accuracy + std_val_accuracy,
                    color='orange', alpha=0.2)
    plt.plot(epochs, avg_test_accuracy, label='Avg Test Accuracy', color='purple')
    plt.fill_between(epochs,
                    avg_test_accuracy - std_test_accuracy,
                    avg_test_accuracy + std_test_accuracy,
                    color='purple', alpha=0.2)
    plt.title('Average Accuracy Across 5 Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig(f'./output/mlp_vae_wgan/mlp_vae_wgan_loss/{r_s}mlp_vae_wgan_accuracy_curve_s.png')
    plt.show()
    plt.close()

    # Compute average loss curves across folds
    avg_train_classification_loss = np.mean(all_train_classification_losses, axis=0)
    std_train_classification_loss = np.std(all_train_classification_losses, axis=0)
    avg_val_classification_loss = np.mean(all_val_classification_losses, axis=0)
    std_val_classification_loss = np.std(all_val_classification_losses, axis=0)
    # Plot average classification (cross-entropy) loss
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, num_epochs + 1)
    plt.plot(epochs, avg_train_classification_loss, label='Avg Train Classification Loss', color='green', linestyle='--')
    plt.fill_between(epochs,
                    avg_train_classification_loss - std_train_classification_loss,
                    avg_train_classification_loss + std_train_classification_loss,
                    color='green', alpha=0.2)
    plt.plot(epochs, avg_val_classification_loss, label='Avg Val Classification Loss', color='red', linestyle='--')
    plt.fill_between(epochs,
                    avg_val_classification_loss - std_val_classification_loss,
                    avg_val_classification_loss + std_val_classification_loss,
                    color='red', alpha=0.2)
    plt.title('Average Cross-Entropy Loss Across 5 Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig(f'./output/mlp_vae_wgan/mlp_vae_wgan_loss/{r_s}classification_loss_plot_s.png')
    plt.show()
    plt.close()

    # Compute average total loss curves across folds
    avg_train_total_loss = np.mean(all_train_total_losses, axis=0)
    std_train_total_loss = np.std(all_train_total_losses, axis=0)
    # Compute average total loss curves across folds
    avg_val_total_loss = np.mean(all_val_total_losses, axis=0)
    std_val_total_loss = np.std(all_val_total_losses, axis=0)
    # Plot average total loss
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, num_epochs + 1)
    plt.plot(epochs, avg_train_total_loss, label='Avg Train Total Loss', color='blue', linestyle='-')
    plt.fill_between(epochs,
                    avg_train_total_loss - std_train_total_loss,
                    avg_train_total_loss + std_train_total_loss,
                    color='blue', alpha=0.2)
    plt.plot(epochs, avg_val_total_loss, label='Avg Val Total Loss', color='orange', linestyle='-')
    plt.fill_between(epochs,
                    avg_val_total_loss - std_val_total_loss,
                    avg_val_total_loss + std_val_total_loss,
                    color='orange', alpha=0.2)
    plt.title('Average Total Loss Across 5 Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig(f'./output/mlp_vae_wgan/mlp_vae_wgan_loss/{r_s}total_loss_plot_s.png')
    plt.show()
    plt.close()

final_avg_train_loss = np.mean(seeds_avg_train_loss, axis=0)
final_avg_val_loss = np.mean(seeds_avg_val_loss, axis=0)

final_avg_train_acc = np.mean(avg_train_total_seeds, axis=0)
final_avg_val_acc = np.mean(avg_val_total_seeds, axis=0)

base_dir = "C:/Users/diogo/Desktop/final_models/new"

epochs = np.arange(1, len(final_avg_train_loss) + 1)
df = pd.DataFrame({
    'Epoch': epochs,
    'Train_CrossEntropy': final_avg_train_loss,
    'Val_CrossEntropy': final_avg_val_loss,
    'Train_Accuracy': final_avg_train_acc,
    'Val_Accuracy': final_avg_val_acc
})
#df.to_csv('./output/mlp_vae_wgan/mlp_vae_wgan_loss/cnn_average_curves_across_seeds.csv', index=False)
#df.to_csv(f'{base_dir}/curves_across_seeds.csv', index=False)

accuracy_df = pd.DataFrame({
    'Random_State': list(range(0, 50)),
    'Best_Epoch':  [b['epoch'] for b in best_val_epoch_per_seed],
    'Avg_Val_Acc_at_Best_Epoch': [b['val_accuracy'] for b in best_val_epoch_per_seed],
    'Train_Acc_at_Best_Epoch': best_train_epoch_per_seed
})
#accuracy_df.to_csv('./output/mlp_vae_wgan/mlp_vae_wgan_loss/cnn_accuracies.csv', index=False)
#accuracy_df.to_csv(f'{base_dir}/accuracies.csv', index=False)

# The remaining code
print(f"Mean accuracy across all folds: {np.mean(random_state_acc_)}")
print("Training completed.")