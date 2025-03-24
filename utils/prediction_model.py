import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return x.squeeze(-1)  # Ensure output is (batch,)
    
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