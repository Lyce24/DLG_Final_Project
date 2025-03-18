# %%
import pandas as pd
from baseline_models import MLP, CNNSurvival, LSTMSurvival, TransformerSurvival, GeneRiskNet
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from loss_functions import hybrid_survival_loss

# save the patients_df
df = pd.read_csv('../data/msk_2024_mutations_final.csv')

# only select the columns we need first 263 columns and the last 2 columns
df = df.iloc[:, :263].join(df.iloc[:, -2:])

# %%

# Extract features (mutation presence) and target (survival months & status)
X = df.drop(columns=["Patient", "OS_MONTHS", "OS_STATUS"])
y_os = df["OS_MONTHS"].values  # Overall survival time
y_status = df["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in x else 0).values  # Convert status to binary

# Train-Validation-Test Split
X_train, X_temp, y_os_train, y_os_temp, y_status_train, y_status_temp = train_test_split(
    X, y_os, y_status, test_size=0.3, random_state=42
)
X_val, X_test, y_os_val, y_os_test, y_status_val, y_status_test = train_test_split(
    X_temp, y_os_temp, y_status_temp, test_size=0.5, random_state=42
)

# Convert to PyTorch tensors
X_train, y_os_train, y_status_train = map(torch.tensor, (X_train.values, y_os_train, y_status_train))
X_val, y_os_val, y_status_val = map(torch.tensor, (X_val.values, y_os_val, y_status_val))
X_test, y_os_test, y_status_test = map(torch.tensor, (X_test.values, y_os_test, y_status_test))

# Move to float tensors
X_train, X_val, X_test = X_train.float(), X_val.float(), X_test.float()
y_os_train, y_os_val, y_os_test = y_os_train.float(), y_os_val.float(), y_os_test.float()
y_status_train, y_status_val, y_status_test = y_status_train.float(), y_status_val.float(), y_status_test.float()

# Assume X_train exists; if not, default to an example input dimension of 10.
input_dim = X_train.shape[1] if 'X_train' in globals() else 10

print(f"Train Size: {X_train.shape}, Validation Size: {X_val.shape}, Test Size: {X_test.shape}")


if __name__ == "__main__":
    import argparse
    from lifelines.utils import concordance_index
    from sklearn.metrics import roc_auc_score
    
    parser = argparse.ArgumentParser(description="Train a survival model on mutation data.")
    parser.add_argument("--model_type", type=str, default="MLP", help="Type of model to train")
    parser.add_argument("--mode", type=str, default="full", help="Mode: train or test")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    
    args = parser.parse_args()
    
    model_type = args.model_type
    mode = args.mode
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr

    if model_type == "MLP":
        model = MLP(input_dim)
    elif model_type == "CNN":
        model = CNNSurvival(input_dim)
    elif model_type == "LSTM":
        model = LSTMSurvival(input_dim, hidden_size=64, num_layers=1)
    elif model_type == "TRA":
        model = TransformerSurvival(input_dim, d_model=64, nhead=4, num_layers=2)
    elif model_type == "RES":
        model = GeneRiskNet(input_dim)
    else:
        raise ValueError("Invalid model type selected.")

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_train, y_os_train, y_status_train = X_train.to(device), y_os_train.to(device), y_status_train.to(device)
    X_val, y_os_val, y_status_val = X_val.to(device), y_os_val.to(device), y_status_val.to(device)
    X_test, y_os_test, y_status_test = X_test.to(device), y_os_test.to(device), y_status_test.to(device)

    # normalize the y_os_train, y_os_val, y_os_test using log1p
    y_os_train = torch.log1p(y_os_train)
    y_os_val = torch.log1p(y_os_val)
    y_os_test = torch.log1p(y_os_test)

    from torch.utils.data import Dataset, DataLoader

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

    # Create dataset objects
    train_dataset = SurvivalDataset(X_train, y_os_train, y_status_train)
    val_dataset   = SurvivalDataset(X_val, y_os_val, y_status_val)
    test_dataset  = SurvivalDataset(X_test, y_os_test, y_status_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # For validation and test we use full batches (since they are smaller)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


    train_losses, val_losses, val_c_indices, val_aurocs = [], [], [], []

    for epoch in range(num_epochs):
        # ---- Training Phase ----
        epoch_loss = 0.0
        if mode == "batch":
            model.train()
            for batch_X, batch_time, batch_event in train_loader:
                batch_X, batch_time, batch_event = batch_X.to(device), batch_time.to(device), batch_event.to(device)
                optimizer.zero_grad()
                risk = model(batch_X)
                loss = hybrid_survival_loss(risk, batch_time, batch_event, alpha=0.5, margin=0.0)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(train_dataset)
            
        elif mode == "full":
            model.train()
            optimizer.zero_grad()
            risk = model(X_train)
            loss = hybrid_survival_loss(risk, y_os_train, y_status_train, alpha=0.5, margin=0.0)
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
            
        else:
            raise ValueError("Invalid mode selected. Choose 'batch' or 'full'.")
        
        train_losses.append(epoch_loss)
        # ---- Validation Phase ----
        model.eval()
        with torch.no_grad():
            # We compute validation loss and predictions on the full validation set
            for val_X, val_time, val_event in val_loader:
                val_X, val_time, val_event = val_X.to(device), val_time.to(device), val_event.to(device)
                val_risk = model(val_X)
                val_loss = hybrid_survival_loss(val_risk, val_time, val_event, alpha=0.5, margin=0.0)
                # Convert log-transformed times back to original scale for c-index
                val_risk_np = val_risk.cpu().numpy()
                val_time_np = val_time.cpu().numpy()
                val_event_np = val_event.cpu().numpy()
                val_time_orig = np.expm1(val_time_np)
                c_index = concordance_index(val_time_orig, -val_risk_np, val_event_np)
                try:
                    auroc = roc_auc_score(val_event_np, val_risk_np)
                except Exception:
                    auroc = np.nan
        
        val_losses.append(val_loss.item())
        val_c_indices.append(c_index)
        val_aurocs.append(auroc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss {epoch_loss:.4f}, Val Loss {val_loss.item():.4f}, "
                f"c-index {c_index:.4f}, AUROC {auroc:.4f}")

    model.eval()
    with torch.no_grad():
        for test_X, test_time, test_event in test_loader:
            test_X, test_time, test_event = test_X.to(device), test_time.to(device), test_event.to(device)
            test_risk = model(test_X)
            test_loss = hybrid_survival_loss(test_risk, test_time, test_event, alpha=0.5, margin=0.0)
            test_risk_np = test_risk.cpu().numpy()
            test_time_np = test_time.cpu().numpy()
            test_event_np = test_event.cpu().numpy()
            test_time_orig = np.expm1(test_time_np)
            test_c_index = concordance_index(test_time_orig, -test_risk_np, test_event_np)
            try:
                test_auroc = roc_auc_score(test_event_np, test_risk_np)
            except Exception:
                test_auroc = np.nan

    print("\nFinal Test Metrics:")
    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"Test c-index: {test_c_index:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    
    with open(f'../results/{model_type}_{mode}_{batch_size}_{num_epochs}_{lr}.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss.item():.4f}\n")
        f.write(f"Test c-index: {test_c_index:.4f}\n")
        f.write(f"Test AUROC: {test_auroc:.4f}\n")
        
    # Save the model
    torch.save(model.state_dict(), f'../models/checkpoints/{model_type}_{mode}_{batch_size}_{num_epochs}_{lr}.pt')

    epochs_arr = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_arr, train_losses, label="Train Loss")
    plt.plot(epochs_arr, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_arr, val_c_indices, label="Val c-index", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("c-index")
    plt.title("Validation c-index")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_arr, val_aurocs, label="Val AUROC", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title("Validation AUROC")
    plt.legend()

    plt.tight_layout()
    # save the plot
    plt.savefig(f'../results/{model_type}_{mode}_{batch_size}_{num_epochs}_{lr}.png')