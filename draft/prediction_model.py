import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleLayerFFN(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=64, dropout=0.3):
        super(SingleLayerFFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)  # Ensure output is (batch,)

# 0. MLP-Based Survival Model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1 = 256, hidden_dim2 = 256, dropout=0.3):
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
        return x.squeeze(-1)  # Ensure output is (batch,)
    
class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1 = 64, hidden_dim2 = 32, dropout=0.3):
        super(SmallMLP, self).__init__()
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
        return x.squeeze(-1)  # Ensure output is (batch,)
    
# ---------------------------
# Define Transformer/Self-Attention Risk Predictor
# ---------------------------
class RiskPredictor(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=1):
        """
        A small transformer-based head for risk prediction.
        - Projects each latent feature (token) from scalar to d_model.
        - Adds a learned positional embedding.
        - Processes the sequence with a Transformer encoder.
        - Pools the sequence and outputs a risk score.
        """
        super(RiskPredictor, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project each latent scalar (token) to a d_model vector.
        self.token_embedding = nn.Linear(1, d_model)
        # Learnable positional embedding for each token.
        self.pos_embedding = nn.Parameter(torch.randn(input_dim, d_model))
        
        # Transformer Encoder: using one or more layers.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final linear layer that outputs a single risk score.
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        """
        x: tensor of shape (batch_size, input_dim)
        """
        batch_size, seq_len = x.shape
        # Reshape so that each latent dimension is treated as a token:
        # (batch_size, input_dim) -> (batch_size, input_dim, 1)
        x = x.unsqueeze(2)
        # Project each token to d_model:
        # shape becomes (batch_size, input_dim, d_model)
        x = self.token_embedding(x)
        # Add positional embeddings: (input_dim, d_model) is broadcasted over batch.
        x = x + self.pos_embedding.unsqueeze(0)
        # Transformer expects input of shape (sequence_length, batch_size, d_model)
        x = x.transpose(0, 1)  # now shape (input_dim, batch_size, d_model)
        # Pass through the Transformer encoder.
        x = self.transformer_encoder(x)
        # Pool over the sequence dimension (mean pooling)
        x = x.mean(dim=0)  # shape: (batch_size, d_model)
        # Output risk score:
        risk = self.fc(x)  # shape: (batch_size, 1)
        return risk.squeeze(1)  # shape: (batch_size)

# ---------------------------
# Self-Attention Risk Predictor
# ---------------------------
class SelfAttentionRiskPredictor(nn.Module):
    def __init__(self, input_dim=128, d_model=32):
        """
        Lightweight self-attention mechanism over the latent features.
        - Each of the 128 latent features (tokens) is first projected from 1 to d_model.
        - We then compute scaled dot-product attention over the tokens.
        - The attended representation is aggregated (mean-pooled) and fed into a linear layer to output a risk score.
        """
        super(SelfAttentionRiskPredictor, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project each latent scalar (token) to d_model dimensions.
        self.token_proj = nn.Linear(1, d_model)
        
        # Define layers to compute query, key, value for self-attention.
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Final layer for risk score.
        self.fc_out = nn.Linear(d_model, 1)
        self.scale = d_model ** 0.5

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 128)
        """
        batch_size = x.size(0)
        # Treat each of the 128 latent dimensions as a token.
        # Reshape to (batch_size, seq_len=128, token_dim=1)
        x = x.unsqueeze(-1)
        # Project tokens: (batch_size, 128, d_model)
        tokens = self.token_proj(x)
        # Compute Q, K, V
        Q = self.query(tokens)  # (batch_size, 128, d_model)
        K = self.key(tokens)    # (batch_size, 128, d_model)
        V = self.value(tokens)  # (batch_size, 128, d_model)
        # Compute scaled dot-product attention scores: (batch_size, 128, 128)
        att_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        att_weights = torch.softmax(att_scores, dim=-1)
        # Compute weighted sum: (batch_size, 128, d_model)
        att_output = torch.bmm(att_weights, V)
        # Aggregate the token representations (mean pooling)
        agg = att_output.mean(dim=1)  # (batch_size, d_model)
        # Compute risk score (scalar per patient)
        risk = self.fc_out(agg)  # (batch_size, 1)
        return risk.squeeze(1)   # (batch_size)