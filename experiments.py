import numpy as np
import pandas as pd
import torch
from utils.ae_script import (prepare_data, 
                             create_dataset, 
                             training_loop, 
                             generate_patient_embeddings,
                             downstream_performance)

# from utils.baseline_script import (prepare_data,run_experiment)
import utils.baseline_script as bs

from utils.autoencoder_model import (MultimodalAutoencoder,
                                     Autoencoder)

threshold = 15
df = pd.read_csv(f"./data/msk_2024_fe_{threshold}.csv")

# locate the columns index for OS_MONTHS
os_months_index = df.columns.get_loc("OS_MONTHS")

gene_mutations = df.iloc[:, :os_months_index]  # Exclude the first column (ID) and OS_MONTHS
data_labels = df.iloc[:, os_months_index:]  # Include only the OS_MONTHS column as labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Test The Baseline Model
def test_baseline_model(iterations=100,
                        ds_model='MLP',
                        ds_epoch=100,
                        ds_lr=0.001,
                        ds_l2_reg=1e-4,
                        save_path ="./results/tests/bs_model_mlp.csv"):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, test_loader, input_dim, _ = bs.prepare_data(
        gene_mutations,
        data_labels,
        device = device,
        batch_size = 3000,
        whole_dataset = True
    )
    
    c_index_list = []
    for i in range(iterations):
        print(f"Running iteration {i+1}...")
        _, _, _, c_index_whole, _, _ = bs.run_experiment(
                                                        train_loader,
                                                        val_loader,
                                                        test_loader,
                                                        device = device,
                                                        ds_model = ds_model,
                                                        ds_epoch = ds_epoch,
                                                        input_dim = input_dim[1],
                                                        ds_lr = ds_lr,
                                                        ds_l2_reg = ds_l2_reg,
                                                        fig_save_path = None,
                                                        test_results_saving_path = None,
                                                        model_save_path = None,
                                                        verbose=False
                                                    )
        c_index_list.append(c_index_whole)
        
    c_index_whole = np.array(c_index_list)
    mean_c_index = np.mean(c_index_whole)
    std_c_index = np.std(c_index_whole)
    sem_c_index = std_c_index / np.sqrt(len(c_index_whole))
    print(f"Mean C-Index: {mean_c_index}")
    print(f"Std C-Index: {std_c_index}")
    print(f"SEM C-Index: {sem_c_index}")
    
    # Save the results
    if save_path:
        with open(save_path, 'w') as f:
            f.write(f"Mean C-Index: {mean_c_index}\n")
            f.write(f"Std C-Index: {std_c_index}\n")
            f.write(f"SEM C-Index: {sem_c_index}\n")

    print(f"Results saved to {save_path}")

### Test The Autoencoder Model
def test_ae_model(iterations=100,
                    batch_size = 4096,
                    ae_type='mm', # 'mm', 'combined', or 'gmp'
                    training_method='normal', # 'normal', 'masked', or 'denoising'
                    backbone='self_attn', # 'mlp', 'self_attn'
                    latent_dim=256,
                    hidden_dim=256,
                    num_layers=2, 
                    dropout=0.3,
                    num_heads=4, # Number of attention heads for the backbone
                    gmp_num_layers=2, # Number of layers for the GMP encoder
                    cd_num_layers=1, # Number of layers for the CD encoder
                    fusion_mode="concat",  # "cross_attention" or "concat" or "gated"
                    cross_attn_mode="shared", # "stacked" or "shared"
                    cross_attn_layers=2, # Number of cross-attention layers
                    num_epochs=70, # Number of epochs for training (for autoencoder)
                    learning_rate=5e-4,
                    l2_lambda=1e-4,
                    mask_ratio=0.3, # Mask ratio for masked autoencoder
                    noise_std=0.1, # Standard deviation for Gaussian noise in denoising autoencoder
                    noise_rate=0.1, # Noise rate for denoising autoencoder
                    ds_batch_size=3000, # Batch size for downstream task
                    whole_dataset=True, # True for whole dataset, False for train/val split
                    ds_model='MLP',
                    ds_epoch=100,
                    ds_lr=0.001,
                    ds_l2_reg=0.0001):
    
    cd_bin = ['highest_stage_recorded', 'CNS_BRAIN', 'LIVER', 'LUNG', 'Regional', 'Distant', 'CANCER_TYPE_BREAST', 'CANCER_TYPE_COLON', 'CANCER_TYPE_LUNG', 'CANCER_TYPE_PANCREAS', 'CANCER_TYPE_PROSTATE']
    cd_num = ['CURRENT_AGE_DEID', 'TMB_NONSYNONYMOUS', 'FRACTION_GENOME_ALTERED']
    
    GMP, CD_BINARY, CD_NUMERIC, patient_ids, _, _ = prepare_data(gene_mutations, data_labels, cd_bin=cd_bin, cd_num=cd_num, device=device)
    num_positives = np.sum(GMP, axis=0)  # Count the number of positive mutations for each gene
    num_negatives = GMP.shape[0] - num_positives  # Count the number of negative mutations for each gene
    pos_weight = num_negatives / (num_positives + 1e-6)  # Avoid division by zero
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)  # Move to the same device as the model
    
    train_loader, val_loader = create_dataset(
        GMP, CD_BINARY, CD_NUMERIC, 
        batch_size=batch_size, 
        train_split=0.85
    )

    if ae_type == "combined":
        input_dim = GMP.shape[1] + CD_BINARY.shape[1] + CD_NUMERIC.shape[1]
    elif ae_type == "gmp":
        input_dim = GMP.shape[1]
    print("Data loaded and prepared.")
    
    all_metrics = []
    for i in range(iterations):
        print(f"Running experiment {i+1}...")
        if ae_type == "mm":
            model = MultimodalAutoencoder(
                input_dim_gmp=GMP.shape[1], 
                input_dim_cd=CD_BINARY.shape[1] + CD_NUMERIC.shape[1],
                backbone=backbone,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                gmp_num_layers=gmp_num_layers, # Number of layers for the GMP encoder
                cd_num_layers=cd_num_layers, # Number of layers for the CD encoder
                dropout=dropout, # Dropout rate
                fusion_mode=fusion_mode, # "cross_attention" or "concat" or "gated"
                cross_attn_mode=cross_attn_mode, # "stacked" or "shared"
                cross_attn_layers=cross_attn_layers, # Number of cross-attention layers
                cd_encoder_mode=backbone
            ).to(device)
            
        elif ae_type == "combined" or ae_type == "gmp":
            model = Autoencoder(
                input_dim=input_dim, 
                latent_dim=latent_dim, 
                backbone=backbone,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_heads=num_heads,
                num_layers=num_layers,
                use_pos=True, # Whether to use positional encoding
                num_tokens=1  # Number of tokens (1 for single token - default)
            ).to(device)
        else:
            raise ValueError("Invalid model type. Choose 'mm', 'gmp', or 'combined'.")
        
        print(f"Testing model (ae_type = {ae_type}, training_method = {training_method}, backbone = {backbone}, fusion_mode = {fusion_mode}, cross_attn_mode = {cross_attn_mode}, cross_attn_layers = {cross_attn_layers}, cd_num_layers = {cd_num_layers})")
        
        _, _, best_model = training_loop(
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
                ae_save_path = None,
                ae_losses_plot_path = None,
                verbose=False
        )

        _, latent_df = generate_patient_embeddings(
            model = model,
            gmp = GMP,
            cd_binary = CD_BINARY,
            cd_numeric = CD_NUMERIC,
            best_model = best_model,
            ae_type = ae_type,
            device = device,
            latent_dim = latent_dim,
            patient_ids = patient_ids,
            save_path = None
        )

        y_os = data_labels['OS_MONTHS']
        y_status = data_labels['OS_STATUS']
        ds_input_dim = latent_df.shape[1]
        _, _, _, c_index_whole, _, _ = downstream_performance(latent_df,
                                                        y_os, 
                                                        y_status,
                                                        device,
                                                        ds_batch_size,
                                                        whole_dataset,
                                                        ds_model,
                                                        ds_epoch,
                                                        ds_input_dim,
                                                        ds_lr,
                                                        ds_l2_reg,
                                                        None,
                                                        None,
                                                        None,
                                                        verbose=False)
        all_metrics.append(c_index_whole)
    
    all_metrics = np.array(all_metrics)
    mean_c_index = np.mean(all_metrics)
    std_c_index = np.std(all_metrics)
    sem_c_index = std_c_index / np.sqrt(len(all_metrics))
    
    file_path = f"./results/tests/{ae_type}_{training_method}_{backbone}_{fusion_mode}_{cross_attn_mode}_{cross_attn_layers}_{cd_num_layers}_{ds_model}.csv"
    with open(file_path, 'w') as f:
        f.write(f"Mean C-Index: {mean_c_index}\n")
        f.write(f"Std C-Index: {std_c_index}\n")
        f.write(f"SEM C-Index: {sem_c_index}\n")
        
    print(f"Results saved to {file_path}")

if __name__ == "__main__":
    # Test the baseline model
    # test_baseline_model(
    #     iterations=100,
    #     ds_model='MLP',
    #     ds_epoch=100,
    #     ds_lr=0.001,
    #     ds_l2_reg=1e-4,
    #     save_path ="./results/tests/bs_model_mlp.csv"
    # )
    
    for backbone in ['mlp', 'self_attn']:
        for ae_type in ['combined', 'gmp']:
            test_ae_model(
                iterations=100,
                ae_type=ae_type,
                training_method='normal',
                backbone=backbone,
                hidden_dim=512,
                num_layers=2,
                dropout=0.3,
                num_epochs=70, # Number of epochs for training (for autoencoder)
                learning_rate=5e-4,
                l2_lambda=1e-4,
                whole_dataset=True, # True for whole dataset, False for train/val split
                ds_model='MLP',
                ds_epoch=100,
                ds_lr=0.001,
                ds_l2_reg=0.0001
            )
    
    for training_method in ['normal', 'denoising', 'masked']:
        for backbone in ['mlp', 'self_attn']:
            for fusion_mode in ['cross_attention', 'concat']:
                test_ae_model(
                    iterations=100,
                    ae_type='mm',
                    training_method=training_method,
                    backbone=backbone,
                    hidden_dim=512,
                    fusion_mode=fusion_mode
                )
                
    for training_method in ['normal', 'denoising', 'masked']:
        for cd_layers in [1, 2]:
            for cross_attn_mode in ['stacked', 'shared']:
                for cross_attn_layers in [1, 2]:
                    test_ae_model(
                        iterations=100,
                        ae_type='mm',
                        training_method=training_method,
                        backbone='self_attn',
                        hidden_dim=512,
                        fusion_mode='cross_attention',
                        cd_num_layers=cd_layers,
                        cross_attn_mode=cross_attn_mode,
                        cross_attn_layers=cross_attn_layers
                    )