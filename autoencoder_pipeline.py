# %%
import numpy as np
import pandas as pd
import torch
from utils.ae_script import (prepare_data, 
                             create_dataset, 
                             training_loop, 
                             generate_patient_embeddings,
                             downstream_performance)

from utils.autoencoder_model import (MultimodalCrossAttentionAutoencoder,
                                     Autoencoder)
threshold = 10
df = pd.read_csv(f"./data/msk_2024_fe_{threshold}.csv")

# locate the columns index for OS_MONTHS
os_months_index = df.columns.get_loc("OS_MONTHS")

gene_mutations = df.iloc[:, :os_months_index]  # Exclude the first column (ID) and OS_MONTHS
data_labels = df.iloc[:, os_months_index:]  # Include only the OS_MONTHS column as labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
GMP, CD_BINARY, CD_NUMERIC, patient_ids = prepare_data(gene_mutations, data_labels)
    # %%
num_positives = np.sum(GMP, axis=0)  # Count the number of positive mutations for each gene
num_negatives = GMP.shape[0] - num_positives  # Count the number of negative mutations for each gene
pos_weight = num_negatives / (num_positives + 1e-6)  # Avoid division by zero
pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)  # Move to the same device as the model

def run_experiment(batch_size = 4096,
                     ae_type='msaae',
                     training_method='normal',
                     backbone='attn_v2',
                     latent_dim=256,
                     hidden_dim=256,
                     num_layers=2,
                     dropout=0.3,
                     num_heads=4,
                     num_epochs=70,
                     learning_rate=0.001,
                     l2_lambda=1e-4,
                     mask_ratio=0.3,
                     noise_std=0.1,
                     noise_rate=0.1,
                     ds_batch_size=3000,
                     whole_dataset=False,
                     ds_model='MLP',
                     ds_epoch=100,
                     ds_lr=0.001,
                     ds_l2_reg=0.0001,
                     ae_save_path=None,
                     patient_rep_save_path=None,
                     ae_losses_plot_path=None,
                     ds_fig_save_path=None,
                     ds_test_results=None,
                     ds_model_save_path=None,
                     fig_save_path_prefix=None):
    # parameters
    batch_size = batch_size

    # %%
    train_loader, val_loader = create_dataset(
        GMP, CD_BINARY, CD_NUMERIC, 
        batch_size=batch_size, 
        train_split=0.85
    )

    # %%
    ae_type = ae_type
    training_method = training_method  # Training method ('normal', 'masked', or 'denoising')
    backbone = backbone  # Backbone architecture ('mlp', 'residual', 'attn', 'attn_v2', or 'self_attn')

    latent_dim = latent_dim  # Latent dimension for the autoencoder
    hidden_dim = hidden_dim  # Hidden dimension for the autoencoder
    num_layers = num_layers  # Number of layers for the backbone
    dropout = dropout  # Dropout rate for the backbone
    num_heads = num_heads  # Number of attention heads for the backbone

    ae_save_path = ae_save_path
    patient_rep_save_path = patient_rep_save_path
    ae_losses_plot_path = ae_losses_plot_path
    ds_fig_save_path = ds_fig_save_path
    ds_test_results = ds_test_results
    ds_model_save_path = ds_model_save_path

    if ae_type == "msaae":
        input_dim = GMP.shape[1] + CD_BINARY.shape[1] + CD_NUMERIC.shape[1]
    elif ae_type == "gmp":
        input_dim = GMP.shape[1]

    if ae_type == "mcaae":
        model = MultimodalCrossAttentionAutoencoder(
            input_dim_gmp=GMP.shape[1], 
            input_dim_cd=CD_BINARY.shape[1] + CD_NUMERIC.shape[1],
            hidden_dim=hidden_dim, 
            latent_dim=latent_dim, 
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers
        ).to(device)
        
    elif ae_type == "msaae" or ae_type == "gmp":
        model = Autoencoder(
            input_dim=input_dim, 
            latent_dim=latent_dim, 
            backbone=backbone,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(device)
    else:
        raise ValueError("Invalid model type. Choose 'multimodal' or 'autoencoder'.")

    # %%
    num_epochs = num_epochs  # Number of epochs for training
    learning_rate = learning_rate  # Learning rate for training
    l2_lambda = l2_lambda  # L2 regularization lambda
    mask_ratio = mask_ratio  # Mask ratio for masked autoencoder
    noise_std = noise_std  # Standard deviation for Gaussian noise in denoising autoencoder
    noise_rate = noise_rate  # Noise rate for denoising autoencoder
    
    train_losses, val_losses, best_model = training_loop(
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            num_epochs = num_epochs,
            learning_rate = learning_rate,
            pos_weight= pos_weight,
            device= device,
            training_method = training_method,
            ae_type = ae_type,
            l2_lambda = l2_lambda,
            mask_ratio = mask_ratio,
            noise_std = noise_std,
            noise_rate = noise_rate,
            ae_save_path = ae_save_path,
            ae_losses_plot_path = ae_losses_plot_path,
            verbose=False
    )

    latent_rep, latent_df = generate_patient_embeddings(
        model = model,
        gmp = GMP,
        cd_binary = CD_BINARY,
        cd_numeric = CD_NUMERIC,
        best_model = best_model,
        ae_type = ae_type,
        device = device,
        latent_dim = latent_dim,
        patient_ids = patient_ids,
        save_path = patient_rep_save_path
    )

    # %%
    y_os = data_labels['OS_MONTHS']
    y_status = data_labels['OS_STATUS']
    ds_batch_size = ds_batch_size
    whole_dataset = whole_dataset
    input_dim = latent_df.shape[1]
    print(f"Input dimension for downstream task: {input_dim}")
    ds_model = ds_model  # Model for downstream task ('MLP' or 'CNN')
    ds_epoch = ds_epoch  # Epochs for downstream task
    ds_lr = ds_lr  # Learning rate for downstream task
    ds_l2_reg = ds_l2_reg  # L2 regularization for downstream task


    train_losses, val_losses, val_c_indices, c_index_whole = downstream_performance(latent_df,
                                                                                        y_os, 
                                                                                        y_status,
                                                                                        device,
                                                                                        ds_batch_size,
                                                                                        whole_dataset,
                                                                                        ds_model,
                                                                                        ds_epoch,
                                                                                        input_dim,
                                                                                        ds_lr,
                                                                                        ds_l2_reg,
                                                                                        ds_fig_save_path,
                                                                                        ds_test_results,
                                                                                        ds_model_save_path,
                                                                                        verbose=False)
    
    return c_index_whole

def test(times,
            batch_size = 4096,
            ae_type='msaae',
            training_method='normal',
            backbone='attn_v2',
            latent_dim=256,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            num_heads=4,
            num_epochs=70,
            learning_rate=0.001,
            l2_lambda=1e-4,
            mask_ratio=0.3,
            noise_std=0.1,
            noise_rate=0.1,
            ds_batch_size=3000,
            whole_dataset=False,
            ds_model='MLP',
            ds_epoch=100,
            ds_lr=0.001,
            ds_l2_reg=0.0001,
            ae_save_path=None,
            patient_rep_save_path=None,
            ae_losses_plot_path=None,
            ds_fig_save_path=None,
            ds_test_results=None,
            ds_model_save_path=None,
            fig_save_path_prefix=None):
    
    all_metrics = []
    for i in range(times):
        print(f"Running experiment {i+1}...")
        result = run_experiment(
                    batch_size=batch_size,
                    ae_type=ae_type,
                    training_method=training_method,
                    backbone=backbone,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    num_heads=num_heads,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    l2_lambda=l2_lambda,
                    mask_ratio=mask_ratio,
                    noise_std=noise_std,
                    noise_rate=noise_rate,
                    ds_batch_size=ds_batch_size,
                    whole_dataset=whole_dataset,
                    ds_model=ds_model,
                    ds_epoch=ds_epoch,
                    ds_lr=ds_lr,
                    ds_l2_reg=ds_l2_reg,
                    ae_save_path=ae_save_path,
                    patient_rep_save_path=patient_rep_save_path,
                    ae_losses_plot_path=ae_losses_plot_path,
                    ds_fig_save_path=ds_fig_save_path,
                    ds_test_results=ds_test_results,
                    ds_model_save_path=ds_model_save_path,
                    fig_save_path_prefix=fig_save_path_prefix)
        all_metrics.append(result)
    
    all_metrics = np.array(all_metrics)
    mean_c_index = np.mean(all_metrics)
    std_c_index = np.std(all_metrics)
    sem_c_index = std_c_index / np.sqrt(len(all_metrics))
    
    file_path = f"./results/tests/{ae_type}_{training_method}_{backbone}_results.csv"
    with open(file_path, 'w') as f:
        f.write(f"Mean C-Index: {mean_c_index}\n")
        f.write(f"Std C-Index: {std_c_index}\n")
        f.write(f"SEM C-Index: {sem_c_index}\n")
        
    print(f"Results saved to {file_path}")
    
    error_bar_path = f"./results/tests/{ae_type}_{training_method}_{backbone}_error_bar.png"
    import matplotlib.pyplot as plt
    # Now plot the mean c-index with error bars (using SEM as error).
    plt.figure(figsize=(6, 4))
    # For example, if you have one point:
    plt.errorbar(1, mean_c_index, yerr=sem_c_index, fmt='o', label='c-index')
    plt.xlim(0, 2)
    plt.xlabel("Experiment")
    plt.ylabel("c-index")
    plt.title("c-index with error bars (SEM) over 100 runs")
    plt.legend()
    if fig_save_path_prefix:
        plt.savefig(error_bar_path)

if __name__ == "__main__":
    for ae_type in ['msaae', 'gmp', 'mcaae']:
        for training_method in ['normal', 'denoising', 'masked']:
            test(
                times = 100,
                ae_type=ae_type,
                training_method=training_method,
                backbone='attn_v2'
            )
            
    for ae_type in ['msaae', 'gmp']:
        for training_method in ['normal', 'denoising', 'masked']:
            test(
                times = 100,
                ae_type=ae_type,
                training_method=training_method,
                backbone='mlp'
            )