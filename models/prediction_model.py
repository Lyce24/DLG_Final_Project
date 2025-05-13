import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 0. MLP-Based Survival Model
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
    
# ---------------------------
# Define Transformer/Self-Attention Risk Predictor
# ---------------------------
# 2. LSTM-Based Survival Model
class LSTMSurvival(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1):
        super(LSTMSurvival, self).__init__()
        # Each feature is treated as a time step with dimension=1.
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, input_dim) -> reshape to (batch, seq_len, 1)
        x = x.unsqueeze(-1)  # (batch, input_dim, 1)
        output, (hn, cn) = self.lstm(x)
        risk = self.fc(hn[-1])
        return risk.squeeze(-1)

# 3. Transformer-Based Survival Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # For odd d_model, handle the extra dimension
            pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class TransformerSurvival(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerSurvival, self).__init__()
        # Treat each feature as a time step with embedding dimension=1, then project to d_model.
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: (batch, input_dim) -> reshape to (batch, seq_len, 1)
        x = x.unsqueeze(-1)  # (batch, input_dim, 1)
        x = self.embedding(x) # (batch, input_dim, d_model)
        x = self.pos_encoder(x)  # (batch, input_dim, d_model)
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch, d_model)
        x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        x = torch.mean(x, dim=0)  # Pool over sequence dimension -> (batch, d_model)
        risk = self.fc(x)         # (batch, 1)
        return risk.squeeze(-1)
    
# ---------------------------
# Model Architecture: GeneRiskNet
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = out + residual  # Residual connection
        return F.relu(out)

class GeneRiskNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_residual_blocks=3, dropout_rate=0.3):
        super(GeneRiskNet, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_residual_blocks)])
        self.fc_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.bn_in(self.fc_in(x)))
        x = self.res_blocks(x)
        risk = self.fc_out(x)
        return risk.squeeze(-1)  # Output shape: (batch,)
# ---------------------------
# Self-Attention Risk Predictor
# ---------------------------
class ImprovedAttentionBlock(nn.Module):
    """
    Transformer-like block with:
      1) Multi-head Self-Attention (with dropout)
      2) Residual Connection + LayerNorm
      3) Feed-forward Subnetwork (with GELU activation and dropout)
      4) Another Residual Connection + LayerNorm
    """
    def __init__(self, embed_dim, num_heads=1, dropout=0.1, ff_hidden_multiplier=4):
        super(ImprovedAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_multiplier * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_multiplier * embed_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Self-attention sublayer.
        attn_output, _ = self.mha(x, x, x)
        x = self.ln1(x + self.dropout(attn_output))
        # Feed-forward sublayer.
        ff_output = self.ff(x)
        out = self.ln2(x + self.dropout(ff_output))
        return out

class SelfAttentionSurvival(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_heads=4):
        super(SelfAttentionSurvival, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attention_blocks = nn.Sequential(*[ImprovedAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.attention_blocks(x)
        risk = self.fc(x).squeeze(-1)
        return risk    