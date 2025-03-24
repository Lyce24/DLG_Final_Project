import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import umap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from prediction_model import MLP


def extract_patientids(df):
    """
    Extracts unique patient IDs from the DataFrame.
    Assumes that the first column of the DataFrame contains patient IDs.
    """
    return df['Patient'].values

def cal_pos_weight(X):
    """
    Calculates the positive weight for the loss function based on the class distribution.
    Assumes that the last column of the DataFrame is the target variable.
    """
    num_pos = np.sum(X, axis=0)
    num_neg = X.shape[0] - num_pos
    pos_weight = num_neg / (num_pos + 1e-6)  # Avoid division by zero
    return pos_weight

class MutationDataset(Dataset):
    def __init__(self, X, y):
        
        # Convert to torch tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)
            
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        """
        Returns:
          x_in      : the input features to the autoencoder
          x_target  : same as x_in for reconstruction
          y_label   : the binary label for classification
        """
        x_in = self.X[idx]
        x_target = self.X[idx]
        y_label = self.y[idx]
        
        return x_in, x_target, y_label

def create_dataset(X, y, batch_size=64):
    """
    Creates a PyTorch Dataset from the input features and target labels.
    """
    dataset = MutationDataset(X, y)
    n = len(dataset)
    n_train = int(0.85 * n)
    n_val = n - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def training_loop(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, pos_weight=None, device='cpu', task ='recon', alpha = 0.5, l2_lambda = 1e-4):
    """
    Training loop for the autoencoder and classifier.
    """
    recon_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    class_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for x_in, x_target, y_label in train_loader:
            x_in, x_target, y_label = x_in.to(device), x_target.to(device), y_label.to(device)
            
            if task == 'recon':
                x_recon_logits, _, _ = model(x_in)  # Get reconstruction logits
                loss = recon_criterion(x_recon_logits, x_target)
            elif task == 'joint':
                x_recon_logits, class_logits, _ = model(x_in)  # Get reconstruction and classification logits
                loss_recon = recon_criterion(x_recon_logits, x_target)
                loss_class = class_criterion(class_logits, y_label)
                loss = alpha * loss_recon + (1 - alpha) * loss_class
            else:
                raise ValueError("Invalid task. Choose 'recon' or 'joint'.")
            
            # compute L2 regularization
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param,2) ** 2
            loss += l2_lambda * l2_reg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_in, x_target, y_label in val_loader:
                x_in, x_target, y_label = x_in.to(device), x_target.to(device), y_label.to(device)
                
                if task == 'recon':
                    x_recon_logits, _, _ = model(x_in)
                    loss = recon_criterion(x_recon_logits, x_target)
                elif task == 'joint':
                    x_recon_logits, class_logits, _ = model(x_in)
                    loss_recon = recon_criterion(x_recon_logits, x_target)
                    loss_class = class_criterion(class_logits, y_label)
                    loss = alpha * loss_recon + (1 - alpha) * loss_class
                
                # compute L2 regularization
                l2_reg = 0.0
                for param in model.parameters():
                    l2_reg += torch.norm(param,2) ** 2
                loss += l2_lambda * l2_reg
                
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        if epoch+1 % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    return train_losses, val_losses

def plot_losses(train_losses, val_losses, save_path=None):
    """
    Plots the training and validation losses over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Losses over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def generate_patient_embeddings(model, X, y, device='cpu', batch_size=64, latent_dim=128, patient_ids=None, save_path=None):
    full_dataset = MutationDataset(X, y)
    model.eval()
    all_latent_embeddings = []
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for x_in, _, _ in full_loader:
            x_in = x_in.to(device)
            z = model.encode(x_in)
            all_latent_embeddings.append(z.cpu().numpy())
            
    all_latent_embeddings = np.concatenate(all_latent_embeddings, axis=0)
    
    # save the latent embeddings to a CSV file
    latent_cols = [f'latent_{i}' for i in range(latent_dim)]
    latent_df = pd.DataFrame(all_latent_embeddings, columns=latent_cols)
    latent_df.insert(0, 'Patient', patient_ids)  # Insert patient IDs as the first column
    if save_path:
        latent_df.to_csv(save_path, index=False)
        
    return all_latent_embeddings, latent_df
        
def plot_umap_by_targets(latent_embeddings, y, target_names=None, save_path_prefix=None):
    """
    Plots UMAP projections of latent embeddings, coloring the points by each target column.
    
    Args:
      latent_embeddings (np.ndarray): Array of shape [n_samples, latent_dim].
      y (np.ndarray): Target values of shape [n_samples, n_targets].
      target_names (list of str, optional): Names of the targets (one per column in y).
      save_path_prefix (str, optional): If provided, each plot is saved with this prefix plus the target index.
    """
    n_targets = y.shape[1]
    y_np = y if isinstance(y, np.ndarray) else y.numpy()
    
    for i in range(n_targets):
        # Create a new UMAP reducer instance
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_embeddings = reducer.fit_transform(latent_embeddings)
        
        # Get the values for the current target column
        target_values = y_np[:, i]
        # Use a target name if provided, else default to "Target i"
        target_name = target_names[i] if target_names is not None else f"Target {i}"
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1],
                              c=target_values, cmap='viridis', s=10, alpha=0.7)
        plt.title(f"UMAP Projection colored by {target_name}")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        cbar = plt.colorbar(scatter)
        cbar.set_label(target_name)
        
        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_{i}.png")
        plt.show()



def downstream_performance(model, val_loader, device='cpu', task='recon'):
    """
    Evaluates the downstream performance of the model on the validation set.
    Returns the average loss and accuracy for the classification task.
    """
    class_criterion = nn.BCEWithLogitsLoss()
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_in, _, y_label in val_loader:
            x_in, y_label = x_in.to(device), y_label.to(device)
            
            if task == 'recon':
                x_recon_logits, _, _ = model(x_in)
                loss = class_criterion(x_recon_logits, x_in)  # Reconstruction loss
            elif task == 'joint':
                _, class_logits, _ =