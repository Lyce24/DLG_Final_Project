import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import umap
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from captum.attr import IntegratedGradients

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1 = 128, hidden_dim2 = 64, dropout=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
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

def prepare_binary_data(gene_mutations, cd_features, data_labels, device='cpu', batch_size=3000, whole_dataset=False):
    if 'Patient' in gene_mutations.columns:
        gene_mutations = gene_mutations.drop(columns=['Patient'], axis=1)
    else:
        pass

    # Get binary mutation features; convert to float32 for PyTorch
    GMP = gene_mutations

    CD_BINARY = data_labels[cd_features]
    
    # Get the column names for each modality.
    gmp_columns = gene_mutations.columns.tolist()
    cd_columns = CD_BINARY.columns.tolist()
    all_columns = gmp_columns + cd_columns  # final feature order in X
    
    GMP = GMP.values.astype(np.float32)
    CD_BINARY = CD_BINARY.values.astype(np.float32)

    CD = CD_BINARY
    X = np.hstack([GMP, CD])  # Combine gene mutations and clinical data (GMP, CD_BINARY, CD_NUMERIC)

    y_os = data_labels["OS_MONTHS"].values  # Overall survival time
    y_status = data_labels["OS_STATUS"].values  # Overall survival status (0 = alive (censored), 1 = dead (event))

    # Train-Validation-Test Split
    X_train, X_temp, y_os_train, y_os_temp, y_status_train, y_status_temp = train_test_split(
        X, y_os, y_status, test_size=0.3
    )
    X_val, X_test, y_os_val, y_os_test, y_status_val, y_status_test = train_test_split(
        X_temp, y_os_temp, y_status_temp, test_size=0.5
    )

    # Convert to PyTorch tensors
    X_train, y_os_train, y_status_train = map(torch.tensor, (X_train, y_os_train, y_status_train))
    X_val, y_os_val, y_status_val = map(torch.tensor, (X_val, y_os_val, y_status_val))
    X_test, y_os_test, y_status_test = map(torch.tensor, (X_test, y_os_test, y_status_test))

    # Move to float tensors
    X_train, X_val, X_test = X_train.float(), X_val.float(), X_test.float()
    y_os_train, y_os_val, y_os_test = y_os_train.float(), y_os_val.float(), y_os_test.float()
    y_status_train, y_status_val, y_status_test = y_status_train.float(), y_status_val.float(), y_status_test.float()

    print(f"Train Size: {X_train.shape}, Validation Size: {X_val.shape}, Test Size: {X_test.shape}")

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
    if whole_dataset:
        # For whole dataset, we use a single batch
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # For validation and test we use full batches (since they are smaller)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train.shape, all_columns

def prepare_data(gene_mutations, data_labels, device='cpu', batch_size=3000, whole_dataset=False):
    if 'Patient' in gene_mutations.columns:
        gene_mutations = gene_mutations.drop(columns=['Patient'], axis=1)
    else:
        pass

    # Get binary mutation features; convert to float32 for PyTorch
    GMP = gene_mutations

    CD_BINARY = data_labels[['highest_stage_recorded', 'CNS_BRAIN', 'LIVER', 'LUNG', 'Regional', 'Distant', 'CANCER_TYPE_BREAST', 'CANCER_TYPE_COLON', 'CANCER_TYPE_LUNG', 'CANCER_TYPE_PANCREAS', 'CANCER_TYPE_PROSTATE']]
    CD_NUMERIC = data_labels[['CURRENT_AGE_DEID', 'TMB_NONSYNONYMOUS', 'FRACTION_GENOME_ALTERED']]
    
    # Get the column names for each modality.
    gmp_columns = gene_mutations.columns.tolist()
    cd_columns = CD_BINARY.columns.tolist() + CD_NUMERIC.columns.tolist()
    all_columns = gmp_columns + cd_columns  # final feature order in X
    
    GMP = GMP.values.astype(np.float32)
    CD_BINARY = CD_BINARY.values.astype(np.float32)
    CD_NUMERIC = CD_NUMERIC.values.astype(np.float32)

    # --- Process CURRENT_AGE_DEID (left-skewed) ---
    # Reflect the age values: higher ages become lower values.
    max_age = np.max(CD_NUMERIC[:, 0])
    age_reflected = max_age - CD_NUMERIC[:, 0]

    # Apply log transformation to the reflected age.
    age_log = np.log1p(age_reflected)  # log1p ensures numerical stability for zero values.

    # Standardize the transformed age.
    scaler_age = StandardScaler()
    age_normalized = scaler_age.fit_transform(age_log.reshape(-1, 1))

    # --- Process TMB_NONSYNONYMOUS and FRACTION_GENOME_ALTERED (right-skewed) ---
    # Apply log1p transformation to both features.
    tmb_log = np.log1p(CD_NUMERIC[:, 1])
    frac_log = np.log1p(CD_NUMERIC[:, 2])

    # Standardize the transformed features.
    scaler_tmb = StandardScaler()
    tmb_normalized = scaler_tmb.fit_transform(tmb_log.reshape(-1, 1))

    scaler_frac = StandardScaler()
    frac_normalized = scaler_frac.fit_transform(frac_log.reshape(-1, 1))

    # --- Combine normalized features ---
    # The resulting cd_numeric_normalized will have the same shape as the original.
    CD_NUMERIC = np.hstack([age_normalized, tmb_normalized, frac_normalized])
    CD = np.hstack([CD_BINARY, CD_NUMERIC])
    X = np.hstack([GMP, CD])  # Combine gene mutations and clinical data (GMP, CD_BINARY, CD_NUMERIC)

    y_os = data_labels["OS_MONTHS"].values  # Overall survival time
    y_status = data_labels["OS_STATUS"].values  # Overall survival status (0 = alive (censored), 1 = dead (event))

    # Train-Validation-Test Split
    X_train, X_temp, y_os_train, y_os_temp, y_status_train, y_status_temp = train_test_split(
        X, y_os, y_status, test_size=0.3
    )
    X_val, X_test, y_os_val, y_os_test, y_status_val, y_status_test = train_test_split(
        X_temp, y_os_temp, y_status_temp, test_size=0.5
    )

    # Convert to PyTorch tensors
    X_train, y_os_train, y_status_train = map(torch.tensor, (X_train, y_os_train, y_status_train))
    X_val, y_os_val, y_status_val = map(torch.tensor, (X_val, y_os_val, y_status_val))
    X_test, y_os_test, y_status_test = map(torch.tensor, (X_test, y_os_test, y_status_test))

    # Move to float tensors
    X_train, X_val, X_test = X_train.float(), X_val.float(), X_test.float()
    y_os_train, y_os_val, y_os_test = y_os_train.float(), y_os_val.float(), y_os_test.float()
    y_status_train, y_status_val, y_status_test = y_status_train.float(), y_status_val.float(), y_status_test.float()

    print(f"Train Size: {X_train.shape}, Validation Size: {X_val.shape}, Test Size: {X_test.shape}")

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
    if whole_dataset:
        # For whole dataset, we use a single batch
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # For validation and test we use full batches (since they are smaller)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train.shape, all_columns

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

def run_experiment(train_loader,
                    val_loader,
                    test_loader,
                    device='cpu',
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
    if ds_model == 'MLP':
        model = MLP(input_dim=input_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.3).to(device)
    else:
        raise ValueError(f"Unknown model type: {ds_model}")
    
    if verbose:
        print(f"Using model: {ds_model} with input dimension: {input_dim}")
    
    optimizer = optim.Adam(model.parameters(), lr=ds_lr)
    
    loss_fn = deepsurv_loss
    train_losses, val_losses, val_c_indices = [], [], []
    
    best_model = None
    best_val_loss = float('inf')
    best_c_index = 0.0
    
    print(f"Starting training for {ds_epoch} epochs...")
    for epoch in range(ds_epoch):
        model.train()
        epoch_loss = 0.0
        
        for x_batch, y_os_batch, y_status_batch in train_loader:
            x_batch, y_os_batch, y_status_batch = x_batch.to(device), y_os_batch.to(device), y_status_batch.to(device)
            
            risk_scores = model(x_batch)
            loss = loss_fn(risk_scores, y_os_batch, y_status_batch, model, l2_reg=ds_l2_reg)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
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