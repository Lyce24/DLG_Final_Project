import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import umap
from sklearn.model_selection import train_test_split
from utils.prediction_model import MLP, SelfAttentionSurvival
from lifelines.utils import concordance_index
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from captum.attr import IntegratedGradients

class MultimodalDataset(Dataset):
    def __init__(self, gmp, cd_binary, patient_ids=None):
        """
        Args:
            gmp (np.ndarray): Gene mutation profile data, shape [n_samples, num_gmp_features]
            cd_binary (np.ndarray): Binary clinical features, shape [n_samples, 11]
            cd_numeric (np.ndarray): Continuous clinical features, shape [n_samples, 3]
            risk_target (np.ndarray): Target values for prediction, shape [n_samples] or [n_samples, 1]
            patient_ids (np.ndarray, optional): Patient identifiers.
        """
        self.gmp = gmp
        self.cd_binary = cd_binary
        self.patient_ids = patient_ids
    
    def __len__(self):
        return self.gmp.shape[0]
    
    def __getitem__(self, idx):
        sample = {
            'x_gmp': torch.tensor(self.gmp[idx], dtype=torch.float32),
            'cd_binary': torch.tensor(self.cd_binary[idx], dtype=torch.float32),
        }
        # Optionally include patient IDs.
        if self.patient_ids is not None:
            sample['patient_id'] = self.patient_ids[idx]
        return sample
    
# Custom Dataset for Survival Data
class SurvivalDataset(Dataset):
    def __init__(self, X, time, event):
        self.X = X
        self.time = time
        self.event = event
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.time[idx], self.event[idx]

def prepare_data(gene_mutations, data_labels, cd_bin, device='cpu'):
    # Extract patient IDs
    patient_ids = gene_mutations['Patient'].values

    # Get binary mutation features; convert to float32 for PyTorch
    GMP = gene_mutations.drop(columns=['Patient'])

    CD_BINARY = data_labels[cd_bin]
    
    # Get the column names for each modality.
    gmp_columns = GMP.columns.tolist()
    cd_columns = CD_BINARY.columns.tolist()
    all_columns = gmp_columns + cd_columns  # final feature order in X

    GMP = GMP.values.astype(np.float32)
    CD_BINARY = CD_BINARY.values.astype(np.float32)
    
    X = np.hstack([GMP, CD_BINARY])
    X = torch.tensor(X, dtype=torch.float32).to(device)

    return GMP, CD_BINARY, patient_ids, all_columns, X
        

def create_dataset(gmp, cd_binary, patient_ids=None, batch_size=64, train_split = 0.85):
    """
    Creates a PyTorch Dataset from the input features and target labels.
    """
    dataset = MultimodalDataset(
        gmp=gmp,
        cd_binary=cd_binary,
        patient_ids=patient_ids
    )

    n = len(dataset)
    n_train = int(train_split * n)  # Use 85% of the data for training
    n_val = n - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def info_nce_loss(h1, h2, temperature=0.2):
    """
    h1, h2: (B, D) two “views” of the same batch
    returns scalar NT-Xent loss
    """
    B, _ = h1.shape
    # 1) normalize
    z = torch.cat([h1, h2], dim=0)                 # (2B, D)
    z = F.normalize(z, dim=1)
    
    # 2) similarity matrix
    sim = torch.matmul(z, z.T) / temperature       # (2B, 2B)
    
    # 3) mask out self‐sims
    diag = torch.eye(2*B, device=sim.device, dtype=torch.bool)
    mask = ~diag
    
    sim = sim.masked_fill(~mask, float('-inf'))
    
    # 4) for each i, the positive is at index i+B (or i-B)
    positives = torch.arange(B, device=sim.device)
    positives = torch.cat([positives + B, positives], dim=0)  # (2B,)
    
    loss = F.cross_entropy(sim, positives)
    return loss

def combined_ae_training_loop(model,
                            train_loader,
                            val_loader,
                            num_epochs=10,
                            learning_rate=0.001,
                            pos_weight=None,
                            device='cpu',
                            method ='masked',
                            l2_lambda = 1e-4,
                            mask_ratio = 0.3,
                            beta = 0.5,
                            save_path=None,
                            verbose=True):
    
    if pos_weight is not None:
        recon_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # For gene mutation profile.
        recon_criterion_none = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')  # For binary part (GMP + CD_binary)
    else:
        recon_criterion = nn.BCEWithLogitsLoss()
        recon_criterion_none = nn.BCEWithLogitsLoss(reduction='none')
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    train_losses = []
    val_losses = []
    
    best_model = None
    best_val_loss = float('inf')
        
    for epoch in range(num_epochs):
        # training phase
        model.train()
        tot_loss = 0.0
        n = 0
        
        for batch in train_loader:
            x_gmp = batch['x_gmp'].to(device)               # Gene mutation profile.
            x_cd = batch['cd_binary'].to(device)       # [batch, 11]
            x_target = torch.cat((x_gmp, x_cd), dim=1)
        
            if method == "masked":
                m_gmp = (torch.rand_like(x_gmp) < mask_ratio).float()
                m_cd  = (torch.rand_like(x_cd)  < mask_ratio).float()
                
                m_target = torch.cat((m_gmp, m_cd), dim=1)
                
                recon, _, _, _ = model(x_gmp, x_cd, m_gmp, m_cd)
                
                loss = (recon_criterion_none(recon, x_target) * m_target).sum() / (m_target.sum() + 1e-8)
                
            elif method == "contrastive":
                m_gmp_1 = (torch.rand_like(x_gmp) < mask_ratio).float()
                m_cd_1  = (torch.rand_like(x_cd)  < mask_ratio).float()
                m_target_1 = torch.cat((m_gmp_1, m_cd_1), dim=1)
  
                m_gmp_2 = (torch.rand_like(x_gmp) < mask_ratio).float()
                m_cd_2  = (torch.rand_like(x_cd)  < mask_ratio).float()
                m_target_2 = torch.cat((m_gmp_2, m_cd_2), dim=1)

                recon_1, _, h_1, _ = model(x_gmp, x_cd, m_gmp_1, m_cd_1)
                recon_2, _, h_2, _ = model(x_gmp, x_cd, m_gmp_2, m_cd_2)
                
                recon_loss_1 = (recon_criterion_none(recon_1, x_target) * m_target_1).sum() / (m_target_1.sum() + 1e-8)
                recon_loss_2 = (recon_criterion_none(recon_2, x_target) * m_target_2).sum() / (m_target_2.sum() + 1e-8)
                
                recon_loss = (recon_loss_1 + recon_loss_2) * 0.5
                
                contrastive_loss = info_nce_loss(h_1, h_2)
                loss = recon_loss + contrastive_loss * beta
        
            else:  # "normal" training phase
                recon, _, _, _ = model(x_gmp, x_cd)
                loss = recon_criterion(recon, x_target)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tot_loss += loss.item()
            n += 1
        mean_loss = tot_loss / n
        train_losses.append(mean_loss)
        
        # validation phase
        model.eval()
        tot_loss = 0.0
        n = 0
        for batch in val_loader:
            x_gmp = batch['x_gmp'].to(device)               # Gene mutation profile.
            x_cd = batch['cd_binary'].to(device)       # [batch, 11]
            x_target = torch.cat((x_gmp, x_cd), dim=1)  # [batch, 11+G]
            
            with torch.no_grad():
                if method == "masked":
                    m_gmp = (torch.rand_like(x_gmp) < mask_ratio).float()
                    m_cd  = (torch.rand_like(x_cd)  < mask_ratio).float()
                    
                    m_target = torch.cat((m_gmp, m_cd), dim=1)
                    
                    recon, _, _, _ = model(x_gmp, x_cd, m_gmp, m_cd)
                    
                    loss = (recon_criterion_none(recon, x_target) * m_target).sum() / (m_target.sum() + 1e-8)
                    
                elif method == "contrastive":
                    m_gmp_1 = (torch.rand_like(x_gmp) < mask_ratio).float()
                    m_cd_1  = (torch.rand_like(x_cd)  < mask_ratio).float()
                    m_target_1 = torch.cat((m_gmp_1, m_cd_1), dim=1)
    
                    m_gmp_2 = (torch.rand_like(x_gmp) < mask_ratio).float()
                    m_cd_2  = (torch.rand_like(x_cd)  < mask_ratio).float()
                    m_target_2 = torch.cat((m_gmp_2, m_cd_2), dim=1)

                    recon_1, _, h_1, _ = model(x_gmp, x_cd, m_gmp_1, m_cd_1)
                    recon_2, _, h_2, _ = model(x_gmp, x_cd, m_gmp_2, m_cd_2)
                    
                    recon_loss_1 = (recon_criterion_none(recon_1, x_target) * m_target_1).sum() / (m_target_1.sum() + 1e-8)
                    recon_loss_2 = (recon_criterion_none(recon_2, x_target) * m_target_2).sum() / (m_target_2.sum() + 1e-8)
                    
                    recon_loss = (recon_loss_1 + recon_loss_2) * 0.5
                    
                    contrastive_loss = info_nce_loss(h_1, h_2)
                    loss = recon_loss + contrastive_loss * beta
            
                else:  # "normal" training phase
                    recon, _, _, _ = model(x_gmp, x_cd)
                    loss = recon_criterion(recon, x_target)
                # Note: In the validation phase, we do not update the model parameters.
                # We only compute the loss and accumulate it for validation.
                
                tot_loss += loss.item()
                n += 1
        mean_loss = tot_loss / n
        val_losses.append(mean_loss)
        
        if mean_loss < best_val_loss:
            best_val_loss, best_model = mean_loss, model.state_dict()
            if save_path is not None:
                torch.save(best_model, save_path)

        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: train {train_losses[-1]:.4f}  val {val_losses[-1]:.4f}")

    return train_losses, val_losses, best_model

def multimodal_ae_training_loop(model,
                                train_loader,
                                val_loader,
                                num_epochs=10,
                                learning_rate=0.001,
                                pos_weight=None,
                                device='cpu',
                                method ='masked',
                                alpha = 1,
                                l2_lambda = 1e-4,
                                mask_ratio = 0.3,
                                beta = 0.5,
                                gamma = 1,
                                patience = 5,
                                verbose=True,
                                save_path=None):
    
    if pos_weight is not None:
        recon_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # For gene mutation profile.
        bce_loss_fn     = nn.BCEWithLogitsLoss()  # For binary clinical features.

        recon_criterion_none = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')  # For binary part (GMP + CD_binary)
        bce_loss_none = nn.BCEWithLogitsLoss(reduction='none')  # For binary clinical features
    else:
        recon_criterion = nn.BCEWithLogitsLoss()
        bce_loss_fn = nn.BCEWithLogitsLoss()
        
        recon_criterion_none = nn.BCEWithLogitsLoss(reduction='none')
        bce_loss_none = nn.BCEWithLogitsLoss(reduction='none')
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    train_losses = []
    val_losses = []
    
    best_model = None
    best_val_loss = float('inf')
        
    for epoch in range(num_epochs):
        # training phase
        model.train()
        tot_loss = 0.0
        n = 0
        
        for batch in train_loader:
            x_gmp = batch['x_gmp'].to(device)               # Gene mutation profile.
            x_cd = batch['cd_binary'].to(device)       # [batch, 11]
        
            if method == "masked":
                m_gmp = (torch.rand_like(x_gmp) < mask_ratio).float()
                m_cd  = (torch.rand_like(x_cd)  < mask_ratio).float()
                
                recon_g, recon_c, _, _ = model(x_gmp, x_cd, m_gmp, m_cd)
                
                l_g = (recon_criterion_none(recon_g, x_gmp) * m_gmp).sum() / (m_gmp.sum() + 1e-8)
                l_c = (bce_loss_none(recon_c, x_cd)  * m_cd ).sum() / (m_cd.sum() + 1e-8)
                loss = l_g + alpha * l_c
                
            elif method == "contrastive":
                m_gmp_1 = (torch.rand_like(x_gmp) < mask_ratio).float()
                m_cd_1  = (torch.rand_like(x_cd)  < mask_ratio).float()
  
                m_gmp_2 = (torch.rand_like(x_gmp) < mask_ratio).float()
                m_cd_2  = (torch.rand_like(x_cd)  < mask_ratio).float()

                recon_g_1, recon_c_1, h_1, _ = model(x_gmp, x_cd, m_gmp_1, m_cd_1)
                recon_g_2, recon_c_2, h_2, _ = model(x_gmp, x_cd, m_gmp_2, m_cd_2)
                
                l_g_1 = (recon_criterion_none(recon_g_1, x_gmp) * m_gmp_1).sum() / (m_gmp_1.sum() + 1e-8)
                l_c_1 = (bce_loss_none(recon_c_1, x_cd)  * m_cd_1 ).sum() / (m_cd_1.sum() + 1e-8)
                l_g_2 = (recon_criterion_none(recon_g_2, x_gmp) * m_gmp_2).sum() / (m_gmp_2.sum() + 1e-8)
                l_c_2 = (bce_loss_none(recon_c_2, x_cd)  * m_cd_2 ).sum() / (m_cd_2.sum() + 1e-8)
                
                recon_loss = (l_g_1 + alpha * l_c_1 + l_g_2 + alpha * l_c_2) * 0.5
                
                contrastive_loss = info_nce_loss(h_1, h_2)
                loss = gamma * recon_loss + contrastive_loss * beta
        
            else:  # "normal" training phase
                recon_g, recon_c, _, _ = model(x_gmp, x_cd)
                l_g = recon_criterion(recon_g, x_gmp)
                l_c = bce_loss_fn(recon_c, x_cd)
                loss = l_g + alpha * l_c
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tot_loss += loss.item()
            n += 1
        mean_loss = tot_loss / n
        train_losses.append(mean_loss)
        
        # validation phase
        model.eval()
        tot_loss = 0.0
        n = 0
        for batch in val_loader:
            x_gmp = batch['x_gmp'].to(device)               # Gene mutation profile.
            x_cd = batch['cd_binary'].to(device)       # [batch, 11]
            
            with torch.no_grad():
                if method == "masked":
                    m_gmp = (torch.rand_like(x_gmp) < mask_ratio).float()
                    m_cd  = (torch.rand_like(x_cd)  < mask_ratio).float()
                
                    recon_g, recon_c, _, _ = model(x_gmp, x_cd, m_gmp, m_cd)
                    l_g = (recon_criterion_none(recon_g, x_gmp) * m_gmp).sum() / (m_gmp.sum() + 1e-8)
                    l_c = (bce_loss_none(recon_c, x_cd)  * m_cd ).sum() / (m_cd.sum() + 1e-8)
                    loss = l_g + alpha * l_c
                    
                elif method == "contrastive":
                    m_gmp_1 = (torch.rand_like(x_gmp) < mask_ratio).float()
                    m_cd_1  = (torch.rand_like(x_cd)  < mask_ratio).float()

                    m_gmp_2 = (torch.rand_like(x_gmp) < mask_ratio).float()
                    m_cd_2  = (torch.rand_like(x_cd)  < mask_ratio).float()

                    recon_g_1, recon_c_1, h_1, _ = model(x_gmp, x_cd, m_gmp_1, m_cd_1)
                    recon_g_2, recon_c_2, h_2, _ = model(x_gmp, x_cd, m_gmp_2, m_cd_2)

                    l_g_1 = (recon_criterion_none(recon_g_1, x_gmp) * m_gmp_1).sum() / (m_gmp_1.sum() + 1e-8)
                    l_c_1 = (bce_loss_none(recon_c_1, x_cd)  * m_cd_1 ).sum() / (m_cd_1.sum() + 1e-8)
                    l_g_2 = (recon_criterion_none(recon_g_2, x_gmp) * m_gmp_2).sum() / (m_gmp_2.sum() + 1e-8)
                    l_c_2 = (bce_loss_none(recon_c_2, x_cd)  * m_cd_2 ).sum() / (m_cd_2.sum() + 1e-8)

                    recon_loss = (l_g_1 + alpha * l_c_1 + l_g_2 + alpha * l_c_2) * 0.5
                    contrastive_loss = info_nce_loss(h_1, h_2)
                    loss = gamma * recon_loss + contrastive_loss * beta
        
                else:  # "normal" validation phase
                    recon_g, recon_c, _, _ = model(x_gmp, x_cd)
                    l_g = recon_criterion(recon_g, x_gmp)
                    l_c = bce_loss_fn(recon_c, x_cd)
                    loss = l_g + alpha * l_c
                # Note: In the validation phase, we do not update the model parameters.
                # We only compute the loss and accumulate it for validation.
                
                tot_loss += loss.item()
                n += 1
        mean_loss = tot_loss / n
        val_losses.append(mean_loss)
        
        if mean_loss < best_val_loss:
            best_val_loss, best_model = mean_loss, model.state_dict()
            if save_path is not None:
                torch.save(best_model, save_path)

        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: train {train_losses[-1]:.4f}  val {val_losses[-1]:.4f}")
            
        # # early stopping (if validation loss does not improve for 5 epochs, stop training)
        # if len(val_losses) >= patience:
        #     recent = val_losses[-patience:]
        #     # all of the last `patience` losses exceeded the best ever
        #     if all(v > best_val_loss for v in recent):
        #         print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs)")
        #         break

    return train_losses, val_losses, best_model
    
def generate_patient_embeddings(model, 
                                gmp,
                                cd_binary,
                                best_model,
                                device='cpu', 
                                batch_size=64, 
                                latent_dim=128, 
                                patient_ids=None, 
                                save_path=None,
                                dataloader_path=None):
    
    if dataloader_path is not None:
        # Load the dataset from the provided path
        full_loader = torch.load(dataloader_path)
    else:
        full_dataset = MultimodalDataset(
            gmp=gmp, 
            cd_binary=cd_binary, 
            patient_ids=patient_ids  # Optional.
        )
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    if best_model:
        model.load_state_dict(best_model)
    model.eval()
    all_latent_embeddings = []
    
    with torch.no_grad():
        for batch in full_loader:
            x_gmp = batch['x_gmp'].to(device)
            x_cd_binary = batch['cd_binary'].to(device)     
            _, _, _, latent = model(x_gmp, x_cd_binary)
            all_latent_embeddings.append(latent.cpu().numpy())
            
    all_latent_embeddings = np.concatenate(all_latent_embeddings, axis=0)
    
    # save the latent embeddings to a CSV file
    latent_cols = [f'latent_{i}' for i in range(latent_dim)]
    latent_df = pd.DataFrame(all_latent_embeddings, columns=latent_cols)
    
    if patient_ids is not None:
        latent_df.insert(0, 'Patient', patient_ids)  # Insert patient IDs as the first column
    else:
        latent_df.insert(0, 'Patient', np.arange(len(all_latent_embeddings)))
        
    if save_path:
        latent_df.to_csv(save_path, index=False)
        
    return all_latent_embeddings, latent_df.drop(columns=['Patient'])


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

def construct_ds_loaders(X, y_os, y_status, batch_size=64, device='cpu', whole_dataset = True):
    X = X.values.astype(np.float32)
    y_os = y_os.values.astype(np.float32)
    y_status = y_status.values.astype(np.float32)
    
    X_train, X_temp, y_os_train, y_os_temp, y_status_train, y_status_temp = train_test_split(X, y_os, y_status, test_size=0.3, random_state=42)
    X_val, X_test, y_os_val, y_os_test, y_status_val, y_status_test = train_test_split(X_temp, y_os_temp, y_status_temp, test_size=0.5, random_state=42)
    
    X_train, y_os_train, y_status_train = map(torch.tensor, (X_train, y_os_train, y_status_train))
    X_val, y_os_val, y_status_val = map(torch.tensor, (X_val, y_os_val, y_status_val))
    X_test, y_os_test, y_status_test = map(torch.tensor, (X_test, y_os_test, y_status_test))
    
    X_train, X_val, X_test = X_train.float(), X_val.float(), X_test.float()
    y_os_train, y_os_val, y_os_test = y_os_train.float(), y_os_val.float(), y_os_test.float()
    y_status_train, y_status_val, y_status_test = y_status_train.float(), y_status_val.float(), y_status_test.float()
    
    X_train, y_os_train, y_status_train = X_train.to(device), y_os_train.to(device), y_status_train.to(device)
    X_val, y_os_val, y_status_val = X_val.to(device), y_os_val.to(device), y_status_val.to(device)
    X_test, y_os_test, y_status_test = X_test.to(device), y_os_test.to(device), y_status_test.to(device)

    # normalize the y_os_train, y_os_val, y_os_test using log1p
    y_os_train = torch.log1p(y_os_train)
    y_os_val = torch.log1p(y_os_val)
    y_os_test = torch.log1p(y_os_test)
    
    # Create dataset objects
    train_dataset = SurvivalDataset(X_train, y_os_train, y_status_train)
    val_dataset   = SurvivalDataset(X_val, y_os_val, y_status_val)
    test_dataset  = SurvivalDataset(X_test, y_os_test, y_status_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # For validation and test we use full batches (since they are smaller)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    if whole_dataset:
        return X_train, y_os_train, y_status_train, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader

# -------------- Downstream Performance Evaluation Functions -------------- #

def cox_partial_likelihood_loss(risk_scores, times, events):
    """
    Computes the Cox Partial Likelihood loss for survival analysis.

    Parameters:
    - risk_scores: Tensor of predicted risk scores (higher = higher risk).
    - times: Tensor of survival times.
    - events: Tensor indicating event occurrence (1 = event, 0 = censored).

    Returns:
    - Negative log Cox partial likelihood loss.
    """
    # Sort survival times in descending order
    sorted_indices = torch.argsort(times, descending=True)
    sorted_risk = risk_scores[sorted_indices]
    sorted_events = events[sorted_indices]

    # Compute log-cumulative sum of exp(risk) -> LogSumExp trick for numerical stability
    risk_cumsum = torch.logcumsumexp(sorted_risk, dim=0)

    # Select only events (uncensored cases)
    event_mask = sorted_events == 1
    loss = -torch.sum(sorted_risk[event_mask] - risk_cumsum[event_mask])

    return loss / (event_mask.sum() + 1e-8)  # Normalize by number of events

def deepsurv_loss(risk_scores, times, events, model, l2_reg=1e-4):
    loss = cox_partial_likelihood_loss(risk_scores, times, events)
    l2_penalty = sum(param.norm(2) for param in model.parameters()) * l2_reg
    return loss + l2_penalty


def smooth_concordance_loss(risk_scores, times, events, sigma=0.1, eps=1e-8):
    """
    Computes a smooth, differentiable surrogate for the concordance index.
    
    For each valid pair (i, j) (with times[i] < times[j] and event[i] == 1),
    the loss is defined as:
    
        L = mean( sigmoid((risk_scores[j] - risk_scores[i]) / sigma) )
    
    This encourages risk_scores[i] to be larger than risk_scores[j].
    
    Parameters:
      risk_scores: Tensor of shape [N] (predicted risk scores).
      times: Tensor of shape [N] (survival times).
      events: Tensor of shape [N] (event indicators, 1 if event, 0 if censored).
      sigma: Temperature parameter controlling the smoothness of the penalty.
      eps: Small constant to avoid division by zero.
      
    Returns:
      Mean smooth concordance loss (scalar).
    """
    risk_scores = risk_scores.squeeze()
    times = times.squeeze()
    events = events.squeeze()
    
    time_i = times.unsqueeze(0)  # [1, N]
    time_j = times.unsqueeze(1)  # [N, 1]
    event_i = events.unsqueeze(0)  # [1, N]
    
    # Valid pairs: time_i < time_j and event_i == 1
    valid_pairs = (time_i < time_j) & (event_i == 1)
    
    if valid_pairs.sum() == 0:
        return torch.tensor(0.0, device=risk_scores.device)
    
    # Compute pairwise difference: (risk_scores[j] - risk_scores[i])
    diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)
    loss = torch.sigmoid(diff / sigma)
    loss = loss[valid_pairs]
    return loss.mean()

def downstream_performance(X, 
                           y_os, 
                           y_status,
                           device='cpu',
                           ds_batch_size = 1024,
                           whole_dataset = True,
                           ds_model = 'MLP',
                           ds_epoch = 100,
                           input_dim = 256,
                           ds_lr = 0.001,
                           ds_l2_reg = 1e-4,
                           fig_save_path = None,
                           test_results_saving_path = None,
                           model_save_path = None,
                           verbose=True):
    """
    Evaluates the downstream performance of the model on the validation set.
    Returns the average loss and accuracy for the classification task.
    """
    if whole_dataset:
        X_train, y_os_train, y_status_train, val_loader, test_loader = construct_ds_loaders(X, y_os, y_status, batch_size=ds_batch_size, device=device, whole_dataset=True)
    else:
        train_loader, val_loader, test_loader = construct_ds_loaders(X, y_os, y_status, batch_size=ds_batch_size, whole_dataset=False)
    
    if verbose:
        if whole_dataset:
            print(f"Training on {len(X_train)} samples, validating on {len(val_loader.dataset)} samples, testing on {len(test_loader.dataset)} samples.")
        else:
            print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples, testing on {len(test_loader.dataset)} samples.")
            
    if ds_model == 'MLP':
        model = MLP(input_dim=input_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.3).to(device)
    elif ds_model == 'sa':
        model = SelfAttentionSurvival(input_dim=input_dim, hidden_dim=128, num_heads=4, num_layers=2).to(device)
    else:
        raise ValueError(f"Unknown model type: {ds_model}")
    
    print(f"Using model: {ds_model} with input dimension: {input_dim}")
    
    optimizer = optim.Adam(model.parameters(), lr=ds_lr)
    
    loss_fn = deepsurv_loss
    train_losses, val_losses, val_c_indices = [], [], []
    
    best_model = None
    best_val_loss = float('inf')
    best_c_index = 0.0
    
    if verbose:
        print(f"Starting training for {ds_epoch} epochs...")
        
    for epoch in range(ds_epoch):
        model.train()
        epoch_loss = 0.0
        model.train()

        if whole_dataset:
            # Convert tensors to DataLoader
            risk_scores = model(X_train)
            loss = loss_fn(risk_scores, y_os_train, y_status_train, model, l2_reg=ds_l2_reg)
            
            optimizer.zero_grad()
            loss.backward()  # Backpropagation
            optimizer.step()
            epoch_loss = loss.item()
            
        else:
            running_loss = 0.0
            for x_batch, y_os_batch, y_status_batch in train_loader:
                x_batch, y_os_batch, y_status_batch = x_batch.to(device), y_os_batch.to(device), y_status_batch.to(device)
                risk_scores = model(x_batch)
                loss = loss_fn(risk_scores, y_os_batch, y_status_batch, model, l2_reg=ds_l2_reg)
                
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
            for x_batch, y_os_batch, y_status_batch in val_loader:
                x_batch, y_os_batch, y_status_batch = x_batch.to(device), y_os_batch.to(device), y_status_batch.to(device)
                
                risk_scores = model(x_batch)
                loss = loss_fn(risk_scores, y_os_batch, y_status_batch, model, l2_reg=ds_l2_reg)
                val_loss = loss.item()
                val_losses.append(val_loss)
                
                val_risk_np = risk_scores.cpu().numpy()
                val_times_np = y_os_batch.cpu().numpy()
                val_events_np = y_status_batch.cpu().numpy()
                val_time_orig = np.expm1(val_times_np)

                c_index = concordance_index(val_time_orig, -val_risk_np, event_observed=val_events_np)
                val_c_indices.append(c_index)
                
        if (epoch + 1) % 10 == 0 and verbose:
            print(f"Epoch {epoch + 1}/{ds_epoch}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, C-Index: {c_index:.4f}")
            
        if val_loss < best_val_loss and c_index > best_c_index:
            best_val_loss = val_loss
            best_c_index = c_index
            best_model = model.state_dict()
            if verbose:
                print(f"New best model found at epoch {epoch + 1} with validation loss: {best_val_loss:.4f} and C-Index: {best_c_index:.4f}")
        
    print(f"Training completed. Best Validation Loss: {min(val_losses):.4f}, Best C-Index: {max(val_c_indices):.4f}")
    
    if verbose:
        epoch_arr = np.arange(1, ds_epoch + 1)
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epoch_arr, train_losses, label='Train Loss')
        plt.plot(epoch_arr, val_losses, label='Validation Loss')
        plt.title('Losses over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(epoch_arr, val_c_indices, label='Validation C-Index')
        plt.title('C-Index over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('C-Index')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if fig_save_path:
            plt.savefig(fig_save_path)
        plt.show()
    
    if best_model:
        model.load_state_dict(best_model)
        
    model.eval()
    
    all_risk = []
    all_time = []
    all_event = []
    
    with torch.no_grad():
        for test_X, test_time, test_event in test_loader:
            test_X, test_time, test_event = test_X.to(device), test_time.to(device), test_event.to(device)
            test_risk = model(test_X)  # Get risk scores for the test set
            
            # Compute loss on the test set
            whole_loss = loss_fn(test_risk, test_time, test_event, model, l2_reg=ds_l2_reg)
            
            # Convert log-transformed times back to original scale for c-index
            test_risk_np = test_risk.cpu().numpy().squeeze()    # risk scores: shape (batch,)
            test_time_np = test_time.cpu().numpy().squeeze()      # survival times (log-transformed)
            test_event_np = test_event.cpu().numpy().squeeze()    # event indicators
            test_time_orig = np.expm1(test_time_np)
            c_index_whole = concordance_index(test_time_orig, -test_risk_np, test_event_np)
            
            all_risk.append(test_risk_np)
            all_time.append(test_time_orig)
            all_event.append(test_event_np)
                       
    # Concatenate predictions from all batches.
    all_risk = np.concatenate(all_risk, axis=0)
    all_time = np.concatenate(all_time, axis=0)
    all_event = np.concatenate(all_event, axis=0)

    # --- Stratify patients into risk groups ---
    # Here we use the median risk score as the cutoff.
    median_risk = np.median(all_risk)
    group_labels = np.where(all_risk > median_risk, "High Risk", "Low Risk")

    # --- Fit Kaplan-Meier curves for each group ---
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    high_mask = group_labels == "High Risk"
    low_mask = group_labels == "Low Risk"

    kmf_high.fit(durations=all_time[high_mask],
                event_observed=all_event[high_mask],
                label="High Risk")
    kmf_low.fit(durations=all_time[low_mask],
                event_observed=all_event[low_mask],
                label="Low Risk")

    # --- Plot the Kaplan-Meier curves ---
    if verbose:
        plt.figure(figsize=(10, 6))
        ax = kmf_high.plot_survival_function(ci_show=True)
        kmf_low.plot_survival_function(ax=ax, ci_show=True)
        plt.xlabel("Time (Months)")
        plt.ylabel("Survival Probability")
        plt.title("Kaplan-Meier Curves: High Risk vs Low Risk")
        plt.show()

    # --- Optionally, print the log-rank test result to compare curves ---
    from lifelines.statistics import logrank_test
    results = logrank_test(all_time[high_mask], all_time[low_mask],
                        event_observed_A=all_event[high_mask],
                        event_observed_B=all_event[low_mask])
    
    print("Log-rank test p-value:", results.p_value)
    print(f"Test Loss: {whole_loss.item():.4f}, Test c-index: {c_index_whole:.4f}")
    
    if test_results_saving_path:
        with open(test_results_saving_path, 'w') as f:
            f.write(f"Test Loss: {whole_loss.item():.4f}\n")
            f.write(f"Test c-index: {c_index_whole:.4f}\n")
            f.write(f"Log-rank test p-value: {results.p_value}\n")
            
    # Save the model
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        
    return train_losses, val_losses, val_c_indices, c_index_whole, model, best_model

class IntegratedModel(nn.Module):
    def __init__(self, ae_model, ds_model, ae_type, gmp_index):
        super(IntegratedModel, self).__init__()
        self.ae_model = ae_model
        self.ds_model = ds_model
        self.ae_type = ae_type
        self.gmp_index = gmp_index
        
    def forward(self, x):
        if self.ae_type == "mm":
            latent = self.ae_model.encode(x[:, :self.gmp_index], x[:, self.gmp_index:])
        else:
            # For other AE types, use the encoder method.
            latent = self.ae_model.encode(x)
        risk_scores = self.ds_model(latent)
        return risk_scores

def interpret_model(ae_model, 
                    ds_model, 
                    ae_best_model, 
                    ds_best_model, 
                    ae_type,
                    gmp_index,
                    all_columns,
                    X_test,
                    verbose = True,
                    sample_size = None,
                    device='cpu'):
    
    # Assume ae_model, ds_model, best_model, ds_best_model, ae_type, and GMP are defined elsewhere.
    ae_model = ae_model.to(device)
    ds_model = ds_model.to(device)
    
    ae_model.load_state_dict(ae_best_model)
    ds_model.load_state_dict(ds_best_model)
    
    ae_model.eval()
    ds_model.eval()
    
    integrated_model = IntegratedModel(ae_model, ds_model, ae_type, gmp_index).to(device)
    integrated_model.eval()
    
    # 1. Sample Output
    sample_input = X_test[0:1]  # shape: [1, num_genes]
    sample_input = sample_input.to(device)
    baseline = torch.zeros_like(sample_input)

    # Instantiate the IntegratedGradients object with your model.
    ig = IntegratedGradients(integrated_model)

    # Compute attributions.
    # For regression outputs, 'target' can be set to 0 (since risk score is a single output).
    attributions, delta = ig.attribute(sample_input,
                                        baseline,
                                        target=0,
                                        return_convergence_delta=True)
        
    # Move attributions to CPU and convert to numpy.
    attr_np = attributions.cpu().detach().numpy()[0]
    # Compute absolute values and get indices of the top 20 features.
    abs_attr = np.abs(attr_np)
    top_indices = np.argsort(abs_attr)[::-1][:20]
    
    if verbose:
        # Print the top 20 features along with their attributions.
        print("Top 20 Features by Integrated Gradients:")
        for idx in top_indices:
            print(f"{all_columns[idx]}: Attribution = {attr_np[idx]:.4f}, Absolute = {abs_attr[idx]:.4f}")

    # Optionally, plot the attributions of the top features.
    top_feature_names = [all_columns[idx] for idx in top_indices]
    top_attr_values = attr_np[top_indices]

    plt.figure(figsize=(12, 6))
    plt.barh(top_feature_names[::-1], top_attr_values[::-1])
    plt.xlabel("Attribution Value")
    plt.title("Top 20 Feature Attributions In One Sample")
    plt.show()

    print("Convergence Delta:", delta.item())
    
    all_attrs = []

    num_samples = X_test.shape[0]
    if sample_size:
        num_samples = min(sample_size, X_test.shape[0])

    for i in range(num_samples):
        sample = X_test[i:i+1].to(device)
        attr, _ = ig.attribute(sample, baseline, target=0, return_convergence_delta=True)
        all_attrs.append(attr.cpu().detach().numpy()[0])
    mean_attrs = np.mean(np.abs(np.array(all_attrs)), axis=0)

    top_indices_global = np.argsort(mean_attrs)[::-1][:20]
    
    if verbose:
        print("\nTop 20 global features by average absolute attribution:")
        for idx in top_indices_global:
            print(f"{all_columns[idx]}: Mean Abs Attribution = {mean_attrs[idx]:.4f}")

    top_feature_gloabl_names = [all_columns[idx] for idx in top_indices_global]
    top_attr_global_values = mean_attrs[top_indices_global]

    plt.figure(figsize=(12, 6))
    plt.barh(top_feature_gloabl_names[::-1], top_attr_global_values[::-1])
    plt.xlabel("Attribution Value")
    plt.title("Top 20 Feature Attributions Computed Across All Samples")
    plt.show()

    print("Convergence Delta:", delta.item())