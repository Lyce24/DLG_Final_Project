#!/bin/bash

# Slurm commands
#SBATCH --partition=gpus             # Use the GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPUs
#SBATCH --nodelist=gpu1703           # Explicitly request GPUs 1701
#SBATCH --time=24:00:00              # Maximum runtime
#SBATCH --mem=24G                    # Memory allocation
#SBATCH -J test_train               # Job name
#SBATCH -o ./output/test_train-%j.out  # Standard output
#SBATCH -e ./output/test_train-%j.err  # Standard error

# parser.add_argument('--model_backbone', type=str, required= True, help='Model Backbone (inceptionv3, convlstm, vit, effnet)')
# parser.add_argument('--pretrained', type=str2bool, default=False, help='Use pre-trained weights')
# parser.add_argument('--log_min_max', type=str2bool, default=False, help='Use log min max normalization')
# parser.add_argument('--da', type=str2bool, default=True, help='Use data augmentation')
# parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
# parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
# parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
# parser.add_argument('--save_name', type=str, required=False, help='Name of the model checkpoint to save')
# parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')  
# parser.add_argument('--l2', type=float, default=0.0, help='L2 regularization')

# v5 - 16 batch size, 100 epochs, no l2, no log min max, data augmentation, no patience, no patience, hidden_dim_2 = 1024

# mass_pred => multihead_attention + extractors

# Run jobs in parallel, assigning specific GPUs using CUDA_VISIBLE_DEVICES

python patient.py