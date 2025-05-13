import numpy as np
import pandas as pd
import torch
from utils.ae_script import (
    prepare_data,
    create_dataset,
    combined_ae_training_loop,
    multimodal_ae_training_loop,
    generate_patient_embeddings,
    downstream_performance,
)

# from utils.baseline_script import (prepare_data,run_experiment)
import utils.baseline_script as bs

from models.surv_mac import Surv_MAC, VanillaMaskedAutoencoder

threshold = 10
df = pd.read_csv(f"./data/msk_2024_fe_{threshold}.csv")

# locate the columns index for OS_MONTHS
os_months_index = df.columns.get_loc("OS_MONTHS")

gene_mutations = df.iloc[:, :os_months_index]  # Exclude the first column (ID) and OS_MONTHS
data_labels = df.iloc[:, os_months_index:]  # Include only the OS_MONTHS column as labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Test The Baseline Model
def test_baseline_model(iterations=50,
                        ds_model='MLP',
                        ds_epoch=100,
                        ds_lr=0.001,
                        ds_l2_reg=1e-4,
                        save_path = f"./results/tests/bs_model_mlp_{threshold}.csv"):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cd_features = ['highest_stage_recorded', 'CNS_BRAIN', 'LIVER', 'LUNG', 'Regional', 'Distant', 'CANCER_TYPE_BREAST', 'CANCER_TYPE_COLON', 'CANCER_TYPE_LUNG', 'CANCER_TYPE_PANCREAS', 'CANCER_TYPE_PROSTATE', "AGE_GE_65", "TMB_GE_10", "FRACTION_GENOME_ALTERED_GE_0.2"]
    train_loader, val_loader, test_loader, input_dim, _ = bs.prepare_binary_data(
        gene_mutations,
        cd_features,
        data_labels,
        device = device,
        batch_size = 3000,
        whole_dataset = True
    )
    
    c_index_list = []
    for i in range(iterations):
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
        print(f"Iteration {i+1} C-Index: {c_index_whole}")
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
def test_surv_mac(iterations=100,
                    latent_dim=256,
                    hidden_dim=512,
                    proj_dim=128,
                    intra_gmp=('res','res','se'),
                    intra_cd=('res', 'se'),
                    fusion_method='bi_film',
                    expansion_factor=2,
                    baseline=False,
                    dropout=0.2,
                    training_method='normal', # 'normal', 'denoising', 'masked'
                    mask_ratio=0.2,
                    beta=0.1,
                    ds_batch_size=3000, # Batch size for downstream task
                    whole_dataset=True, # True for whole dataset, False for train/val split
                    ds_model='MLP',
                    ds_epoch=100,
                    ds_lr=0.001,
                    ds_l2_reg=0.0001,
                    num_epochs = 80,
                    learning_rate = 1e-4,
                    l2_lambda = 1e-4,
                    batch_size = 1024):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 'highest_stage_recorded', 'CNS_BRAIN', 'LIVER', 'LUNG', 'Regional', 'Distant', 'CANCER_TYPE_BREAST', 'CANCER_TYPE_COLON', 'CANCER_TYPE_LUNG', 'CANCER_TYPE_PANCREAS', 'CANCER_TYPE_PROSTATE'
    cd_bin = ['highest_stage_recorded', 'CNS_BRAIN', 'LIVER', 'LUNG', 'Regional', 'Distant', 'CANCER_TYPE_BREAST', 'CANCER_TYPE_COLON', 'CANCER_TYPE_LUNG', 'CANCER_TYPE_PANCREAS', 'CANCER_TYPE_PROSTATE', "AGE_GE_65", "TMB_GE_10", "FRACTION_GENOME_ALTERED_GE_0.2"]
    GMP, CD_BINARY, _, _, _ = prepare_data(gene_mutations, data_labels, cd_bin=cd_bin, device=device)
    
    num_positives = np.sum(GMP, axis=0)  # Count the number of positive mutations for each gene
    num_negatives = GMP.shape[0] - num_positives  # Count the number of negative mutations for each gene
    pos_weight = num_negatives / (num_positives + 1e-6)  # Avoid division by zero
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    combined_data = np.hstack([GMP, CD_BINARY])
    num_positives_whole = np.sum(combined_data, axis=0)  # Count the number of positive mutations for each gene
    num_negatives_whole = combined_data.shape[0] - num_positives_whole  # Count the number of negative mutations for each gene
    pos_weight_whole = num_negatives_whole / (num_positives_whole + 1e-6)  # Avoid division by zero
    pos_weight_whole = torch.tensor(pos_weight_whole, dtype=torch.float32).to(device)
    
    # parameters
    batch_size = batch_size

    train_loader, val_loader = create_dataset(
        GMP, CD_BINARY, 
        batch_size=batch_size, 
        train_split=0.85
    )

    model = Surv_MAC(
        num_genes=GMP.shape[1],
        num_cd_fields=CD_BINARY.shape[1],
        intra_gmp=intra_gmp,
        intra_cd=intra_cd,
        fusion_method=fusion_method,
        expansion_factor=expansion_factor,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        proj_dim=proj_dim,
        dropout=dropout,
        baseline=baseline,
    ).to(device)
    
    _, _, best_model =  multimodal_ae_training_loop(model, 
                                                    train_loader, 
                                                    val_loader, 
                                                    num_epochs=num_epochs, 
                                                    learning_rate=learning_rate, 
                                                    pos_weight=pos_weight,
                                                    device=device, 
                                                    method=training_method, 
                                                    l2_lambda=l2_lambda, 
                                                    mask_ratio=mask_ratio,
                                                    alpha=1,
                                                    beta=0.5,
                                                    gamma=1,
                                                    verbose=False)
    print("Model training completed.")
    
    all_metrics = []
    for i in range(iterations):
        print(f"Running experiment {i+1}...")
        
        _, latent_df = generate_patient_embeddings(
            model = model,
            gmp = GMP,
            cd_binary = CD_BINARY,
            best_model = best_model,
            device = device,
            latent_dim = latent_dim,
            patient_ids = None,
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
    
    file_path = f"./results/surv_mac/intra_gmp_{intra_gmp}_intra_cd_{intra_cd}_latent_dim_{latent_dim}_hidden_dim_{hidden_dim}_proj_dim_{proj_dim}_fusion_method_{fusion_method}_expansion_factor_{expansion_factor}_baseline_{baseline}_{training_method}_{mask_ratio}_{beta}.txt"
    with open(file_path, 'w') as f:
        f.write(f"Mean C-Index: {mean_c_index}\n")
        f.write(f"Std C-Index: {std_c_index}\n")
        f.write(f"SEM C-Index: {sem_c_index}\n")
        
    print(f"Results saved to {file_path}")
        

if __name__ == "__main__":
    fusion_method = ['bi_film']
    intra_gmp = [('res','res','se')]
    intra_cd = [('res', 'se')]
    
    hidden_dim = 512
    latent_dim = 256
    proj_dim = 128
    
    for fusion in fusion_method:
        for intra_gmp_ in intra_gmp:
            for intra_cd_ in intra_cd:
                for baseline in [False, True]:
                    test_surv_mac(
                        iterations=100,
                        latent_dim=latent_dim,
                        hidden_dim= hidden_dim,
                        proj_dim=proj_dim,
                        intra_gmp=intra_gmp_,
                        intra_cd=intra_cd_,
                        fusion_method=fusion,
                        expansion_factor=1,
                        baseline=baseline,
                        dropout=0.2,
                        training_method='normal', # 'normal', 'contrastive', 'masked'
                        mask_ratio=0,
                        beta=0,
                        ds_batch_size=3000, # Batch size for downstream task
                        whole_dataset=True, # True for whole dataset, False for train/val split
                        ds_model='MLP',
                        ds_epoch=100,
                        ds_lr=0.001,
                        ds_l2_reg=0.0001,
                        num_epochs = 80,
                        learning_rate = 1e-4,
                        l2_lambda = 1e-4,
                    )
                    
                    for mask_ratio in [0.3, 0.5]:
                        test_surv_mac(
                            iterations=100,
                            latent_dim=latent_dim,
                            hidden_dim= hidden_dim,
                            proj_dim=proj_dim,
                            intra_gmp=intra_gmp_,
                            intra_cd=intra_cd_,
                            fusion_method=fusion,
                            expansion_factor=1,
                            baseline=baseline,
                            dropout=0.2,
                            training_method='masked', # 'normal', 'contrastive', 'masked'
                            mask_ratio=mask_ratio,
                            beta=0,
                            ds_batch_size=3000, # Batch size for downstream task
                            whole_dataset=True, # True for whole dataset, False for train/val split
                            ds_model='MLP',
                            ds_epoch=100,
                            ds_lr=0.001,
                            ds_l2_reg=0.0001,
                            num_epochs = 80,
                            learning_rate = 1e-4,
                            l2_lambda = 1e-4,
                        )
                        
                        for mask_ratio in [0.3, 0.5]:
                            for beta in [0.3, 0.5]:
                                test_surv_mac(
                                    iterations=100,
                                    latent_dim=latent_dim,
                                    hidden_dim= hidden_dim,
                                    proj_dim=proj_dim,
                                    intra_gmp=intra_gmp_,
                                    intra_cd=intra_cd_,
                                    fusion_method=fusion,
                                    expansion_factor=1,
                                    baseline=baseline,
                                    dropout=0.2,
                                    training_method='contrastive', # 'normal', 'contrastive', 'masked'
                                    mask_ratio=mask_ratio,
                                    beta=beta,
                                    ds_batch_size=3000, # Batch size for downstream task
                                    whole_dataset=True, # True for whole dataset, False for train/val split
                                    ds_model='MLP',
                                    ds_epoch=100,
                                    ds_lr=0.001,
                                    ds_l2_reg=0.0001,
                                    num_epochs = 80,
                                    learning_rate = 1e-4,
                                    l2_lambda = 1e-4,
                                )
