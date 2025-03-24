import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionClassifier(nn.Module):
    """
    Classification head that uses self-attention.
    It projects the latent vector into a sequence of tokens, applies an attention block,
    then aggregates the tokens to produce classification logits.
    """
    def __init__(self, latent_dim, num_heads=1, dropout=0.3, num_classes=1, seq_len=4):
        super().__init__()
        self.seq_len = seq_len
        # Project latent vector to a sequence of tokens.
        self.token_proj = nn.Linear(latent_dim, latent_dim * seq_len)
        # Apply a self-attention block
        self.attn_block = AttentionBlock(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout)
        # MLP head for classification after pooling the tokens.
        self.mlp_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, z):
        """
        z shape: [batch_size, latent_dim]
        """
        B = z.size(0)
        # Project to tokens and reshape: [B, seq_len, latent_dim]
        tokens = self.token_proj(z).view(B, self.seq_len, -1)
        # Refine tokens via self-attention.
        tokens = self.attn_block(tokens)
        # Aggregate tokens (using mean pooling, can also use a CLS token)
        pooled = tokens.mean(dim=1)
        # Final classification logits.
        logits = self.mlp_head(pooled)
        return logits 
    
class MLPClassifier(nn.Module):
    """
    A simple MLP classifier that takes a latent vector and produces class logits.
    """
    def __init__(self, latent_dim, num_classes=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, z):
        """
        z shape: [batch_size, latent_dim]
        """
        logits = self.mlp(z)
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ImprovedAttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32, num_heads=4, dropout=0.3, num_layers=2):
        """
        Args:
          input_dim   : Dimensionality of the input features.
          hidden_dim  : Dimensionality of the hidden representations.
          latent_dim  : Bottleneck (latent) dimensionality.
          num_heads   : Number of attention heads.
          dropout     : Dropout rate.
          num_layers  : Number of stacked attention blocks in both encoder and decoder.
        """
        super(ImprovedAttentionAutoencoder, self).__init__()
        # -- ENCODER --
        # Project input to hidden_dim.
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        # Stack attention blocks.
        self.encoder_blocks = nn.Sequential(*[
            ImprovedAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Project hidden representation to latent space.
        self.encoder_fc2 = nn.Linear(hidden_dim, latent_dim)
        
        # -- DECODER --
        # Project latent vector to hidden representation.
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU()
        )
        # Stack attention blocks.
        self.decoder_blocks = nn.Sequential(*[
            ImprovedAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Project hidden representation back to input dimension.
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        """
        Encodes input x into a latent representation z.
        Args:
          x: Tensor of shape [batch_size, input_dim]
        Returns:
          latent: Tensor of shape [batch_size, latent_dim]
        """
        x = self.encoder_fc(x)  # [batch, hidden_dim]
        # We add a sequence dimension (here seq_len = 1).
        x = x.unsqueeze(1)      # [batch, 1, hidden_dim]
        x = self.encoder_blocks(x)  # [batch, 1, hidden_dim]
        x = x.squeeze(1)        # [batch, hidden_dim]
        latent = self.encoder_fc2(x)  # [batch, latent_dim]
        return latent
    
    def decode(self, z):
        """
        Decodes latent representation z back to input space.
        Args:
          z: Tensor of shape [batch, latent_dim]
        Returns:
          out: Tensor of shape [batch, input_dim]
        """
        x = self.decoder_fc(z)  # [batch, hidden_dim]
        x = x.unsqueeze(1)      # [batch, 1, hidden_dim]
        x = self.decoder_blocks(x)  # [batch, 1, hidden_dim]
        x = x.squeeze(1)        # [batch, hidden_dim]
        out = self.decoder_fc2(x)  # [batch, input_dim]
        return out
    
    def forward(self, x):
        """
        Returns:
          x_recon_logits: Reconstructed input.
          None: Placeholder for compatibility.
          latent: Latent representation.
        """
        latent = self.encode(x)
        x_recon_logits = self.decode(latent)
        return x_recon_logits, None, latent

# -------------------------------
# Cross-Attention Block for Multimodal Fusion
# -------------------------------
class CrossAttentionBlock(nn.Module):
    """
    A block that allows one modality (query) to attend to another (key/value).
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.3):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        # query: [batch, seq_len, embed_dim]
        # key_value: [batch, seq_len, embed_dim]
        attn_output, _ = self.cross_attn(query, key_value, key_value)
        out = self.ln(query + self.dropout(attn_output))
        return out


# -------------------------------
# Multimodal Cross-Attention Autoencoder
# -------------------------------
class MultimodalCrossAttentionAutoencoder(nn.Module):
    """
    An autoencoder that integrates gene mutation profile (GMP) and clinical data (CD)
    via separate encoders and cross-attention fusion.
    """
    def __init__(self, input_dim_gmp, input_dim_cd, hidden_dim=256, latent_dim=256,
                 num_heads=4, dropout=0.3, num_layers=2):
        super(MultimodalCrossAttentionAutoencoder, self).__init__()
        
        # --- GMP Encoder ---
        self.encoder_gmp_fc = nn.Sequential(
            nn.Linear(input_dim_gmp, hidden_dim),
            nn.GELU()
        )
        self.encoder_gmp_blocks = nn.Sequential(*[
            ImprovedAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # --- CD Encoder ---
        self.encoder_cd_fc = nn.Sequential(
            nn.Linear(input_dim_cd, hidden_dim),
            nn.GELU()
        )
        self.encoder_cd_blocks = nn.Sequential(*[
            ImprovedAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # --- Cross-Attention for Fusion ---
        # Let each modality attend to the other.
        self.cross_attn_gmp_to_cd = CrossAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_cd_to_gmp = CrossAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        
        # Fusion: Concatenate the two fused representations and project to latent space.
        self.fusion_fc = nn.Linear(hidden_dim * 2, latent_dim)
        
        # --- GMP Decoder ---
        self.decoder_gmp_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU()
        )
        self.decoder_gmp_blocks = nn.Sequential(*[
            ImprovedAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.decoder_gmp_fc2 = nn.Linear(hidden_dim, input_dim_gmp)
        
        # --- CD Decoder ---
        self.decoder_cd_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU()
        )
        self.decoder_cd_blocks = nn.Sequential(*[
            ImprovedAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.decoder_cd_fc2 = nn.Linear(hidden_dim, input_dim_cd)
        
        
    def encode(self, x_gmp, x_cd):
        """
        Encode each modality and fuse them via cross-attention.
        Args:
          x_gmp: [batch, input_dim_gmp]
          x_cd:  [batch, input_dim_cd]
        Returns:
          latent: [batch, latent_dim] fused latent representation.
        """
        # --- GMP Encoding ---
        gmp = self.encoder_gmp_fc(x_gmp)       # [batch, hidden_dim]
        gmp = gmp.unsqueeze(1)                 # [batch, 1, hidden_dim]
        gmp = self.encoder_gmp_blocks(gmp)       # [batch, 1, hidden_dim]
        
        # --- CD Encoding ---
        cd = self.encoder_cd_fc(x_cd)            # [batch, hidden_dim]
        cd = cd.unsqueeze(1)                    # [batch, 1, hidden_dim]
        cd = self.encoder_cd_blocks(cd)          # [batch, 1, hidden_dim]
        
        # --- Cross-Attention Fusion ---
        # Let GMP attend to CD and vice versa.
        gmp_fused = self.cross_attn_gmp_to_cd(gmp, cd)  # [batch, 1, hidden_dim]
        cd_fused = self.cross_attn_cd_to_gmp(cd, gmp)     # [batch, 1, hidden_dim]
        
        # Concatenate the fused representations along the feature dimension.
        fused = torch.cat([gmp_fused.squeeze(1), cd_fused.squeeze(1)], dim=1)  # [batch, 2*hidden_dim]
        latent = self.fusion_fc(fused)  # [batch, latent_dim]
        return latent
    
    def decode(self, latent):
        """
        Decode the latent representation back into both modalities.
        Args:
          latent: [batch, latent_dim]
        Returns:
          recon_gmp: Reconstructed gene mutation profile.
          recon_cd:  Reconstructed clinical data.
        """
        # --- GMP Decoder ---
        gmp = self.decoder_gmp_fc(latent)       # [batch, hidden_dim]
        gmp = gmp.unsqueeze(1)                  # [batch, 1, hidden_dim]
        gmp = self.decoder_gmp_blocks(gmp)      # [batch, 1, hidden_dim]
        gmp = gmp.squeeze(1)                    # [batch, hidden_dim]
        recon_gmp = self.decoder_gmp_fc2(gmp)     # [batch, input_dim_gmp]
        
        # --- CD Decoder ---
        cd = self.decoder_cd_fc(latent)         # [batch, hidden_dim]
        cd = cd.unsqueeze(1)                    # [batch, 1, hidden_dim]
        cd = self.decoder_cd_blocks(cd)         # [batch, 1, hidden_dim]
        cd = cd.squeeze(1)                      # [batch, hidden_dim]
        recon_cd = self.decoder_cd_fc2(cd)        # [batch, input_dim_cd]
        
        return recon_gmp, recon_cd
    
    def forward(self, x_gmp, x_cd):
        """
        Forward pass through the multimodal autoencoder.
        Returns:
          recon_gmp: Reconstruction of the gene mutation profile.
          recon_cd : Reconstruction of the clinical data.
          risk_score: Prediction output from the fused latent representation.
          latent   : The fused latent representation.
        """
        latent = self.encode(x_gmp, x_cd)
        recon_gmp, recon_cd = self.decode(latent)
        return recon_gmp, recon_cd, latent

# -------------------------------
# Self-Attention Autoencoder for a Single Concatenated Input
# -------------------------------
class SelfAttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32, seq_len=3, num_heads=4, dropout=0.3, num_layers=2):
        """
        Args:
          input_dim   : Dimensionality of the concatenated input features (e.g. GMP + CD_binary + CD_numeric).
          hidden_dim  : Dimensionality of the token embeddings.
          latent_dim  : Bottleneck (latent) dimensionality.
          seq_len     : Number of tokens to split the input into.
          num_heads   : Number of attention heads.
          dropout     : Dropout rate.
          num_layers  : Number of stacked attention blocks in both encoder and decoder.
        """
        super(SelfAttentionAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 1. Encoder: Project the input vector into a sequence of tokens.
        #    The projection maps input_dim -> (seq_len * hidden_dim)
        self.input_proj = nn.Linear(input_dim, seq_len * hidden_dim)
        # Learnable positional embeddings for the tokens.
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, hidden_dim))
        
        # 2. Encoder: Self-attention blocks applied on the token sequence.
        self.encoder_blocks = nn.Sequential(*[
            ImprovedAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        # 3. Aggregate the tokens (here, by mean pooling) and project to latent space.
        self.encoder_fc2 = nn.Linear(hidden_dim, latent_dim)
        
        # 4. Decoder: Map latent vector to an initial hidden representation.
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU()
        )
        # 5. Decoder: Project to a sequence of tokens.
        self.decoder_proj = nn.Linear(hidden_dim, seq_len * hidden_dim)
        # 6. Decoder: Self-attention blocks on the token sequence.
        self.decoder_blocks = nn.Sequential(*[
            ImprovedAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        # 7. Decoder: Reconstruct the original input from the flattened token representations.
        self.decoder_fc2 = nn.Linear(seq_len * hidden_dim, input_dim)
        
    def encode(self, x):
        """
        Encodes input x into a latent representation.
        Args:
          x: Tensor of shape [batch_size, input_dim]
        Returns:
          latent: Tensor of shape [batch_size, latent_dim]
        """
        batch_size = x.size(0)
        # Project input into a sequence of tokens.
        tokens = self.input_proj(x)  # [batch, seq_len * hidden_dim]
        tokens = tokens.view(batch_size, self.seq_len, self.hidden_dim)  # [batch, seq_len, hidden_dim]
        # Add positional embeddings.
        tokens = tokens + self.pos_embedding.unsqueeze(0)  # [batch, seq_len, hidden_dim]
        # Process tokens through self-attention blocks.
        tokens = self.encoder_blocks(tokens)  # [batch, seq_len, hidden_dim]
        # Aggregate tokens (e.g. by mean pooling) to form a single vector.
        tokens_agg = tokens.mean(dim=1)  # [batch, hidden_dim]
        # Project aggregated representation to latent space.
        latent = self.encoder_fc2(tokens_agg)  # [batch, latent_dim]
        return latent
    
    def decode(self, latent):
        """
        Decodes latent representation z back to input space.
        Args:
          latent: Tensor of shape [batch, latent_dim]
        Returns:
          out: Tensor of shape [batch, input_dim]
        """
        batch_size = latent.size(0)
        # Project latent vector to a hidden representation.
        hidden = self.decoder_fc(latent)  # [batch, hidden_dim]
        # Project to a sequence of tokens.
        tokens = self.decoder_proj(hidden)  # [batch, seq_len * hidden_dim]
        tokens = tokens.view(batch_size, self.seq_len, self.hidden_dim)  # [batch, seq_len, hidden_dim]
        # Process tokens with self-attention blocks.
        tokens = self.decoder_blocks(tokens)  # [batch, seq_len, hidden_dim]
        # Flatten tokens.
        tokens_flat = tokens.view(batch_size, -1)  # [batch, seq_len * hidden_dim]
        # Reconstruct the input.
        out = self.decoder_fc2(tokens_flat)  # [batch, input_dim]
        return out
    
    def forward(self, x):
        """
        Args:
          x: Tensor of shape [batch_size, input_dim]
        Returns:
          x_recon: Reconstructed input of shape [batch_size, input_dim]
          latent: Latent representation of shape [batch_size, latent_dim]
        """
        latent = self.encode(x)
        x_recon = self.decode(latent)
        return x_recon, None, latent

class AttentionBlock(nn.Module):
    """
    A simple Transformer-like attention block with:
    1) Multi-head Self-Attention
    2) Residual Connection + LayerNorm
    3) Feed-forward Subnetwork
    4) Another Residual Connection + LayerNorm
    """
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super().__init__()
        
        # Multi-Head Self-Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # so input is [batch, seq_len, embed_dim]
        )
        
        # Layer Norms
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward sublayer
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
    def forward(self, x):
        """
        x shape: [batch_size, seq_len, embed_dim]
        """
        # 1) Multi-head self-attention
        attn_output, _ = self.mha(x, x, x)  # queries=keys=values=x
        
        # 2) Residual + LayerNorm
        x = self.ln1(x + attn_output)
        
        # 3) Feed-forward
        ff_output = self.ff(x)
        
        # 4) Residual + LayerNorm
        out = self.ln2(x + ff_output)
        return out
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        shortcut = self.shortcut(x)
        return self.relu(out + shortcut)
    
# -----------------------------------------------------------------
# Basic Autoencoder Models
# -----------------------------------------------------------------
class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super(MLPAutoencoder, self).__init__()
        # ENCODER: input_dim -> 512 -> 256 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, latent_dim)
        )
        # DECODER: latent_dim -> 256 -> 512 -> input_dim
        # Note: We do NOT apply a final Sigmoid here because BCEWithLogitsLoss expects logits.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, input_dim)
            # We do not use Sigmoid here as BCEWithLogitsLoss applies sigmoid internally.
        )
        
        
    def encode(self, x):
        """
        Encodes input x into a latent representation z.
        x shape: [batch_size, input_dim]
        """
        x = self.encoder(x)
        return x
    
    def decode(self, z):
        """
        Decodes latent representation z back to reconstruction.
        z shape: [batch_size, latent_dim]
        """
        x_recon_logits = self.decoder(z)
        return x_recon_logits


    def forward(self, x):
        z = self.encode(x)
        x_recon_logits = self.decode(z)
        
        return x_recon_logits, None, z
        
class ResidualAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ResidualAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(input_dim, 256),
            ResidualBlock(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            ResidualBlock(latent_dim, 256),
            ResidualBlock(256, input_dim)
        )
    
    def encode(self, x):
        """Encodes input x into latent representation."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decodes latent representation z back into the reconstruction."""
        return self.decoder(z)    
    
    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, None, latent  # No need for mu in this case
    
class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32, num_heads=4, dropout=0.3):
        """
        Args:
            input_dim   : Dimensionality of the input features.
            hidden_dim  : Dimensionality inside the encoder/decoder prior to the latent space.
            latent_dim  : Bottleneck (latent) dimensionality.
            num_heads   : Number of attention heads in the self-attention block.
            dropout     : Dropout rate for the attention mechanism.
        """
        super().__init__()
        
        # -- ENCODER --
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_attn = AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.encoder_fc2 = nn.Linear(hidden_dim, latent_dim)
        
        # -- DECODER --
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_attn = AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        """
        Encodes input x into a latent representation z.
        x shape: [batch_size, input_dim]
        """
        x = self.encoder_fc1(x)  # [batch_size, hidden_dim]
        x = F.relu(x)
        
        # For multi-head attention, we typically have a sequence dimension
        # but here we have only 1 "token" per sample. We'll treat each sample
        # as [batch_size, seq_len=1, hidden_dim].
        x = x.unsqueeze(1)      # [batch_size, 1, hidden_dim]
        x = self.encoder_attn(x)   # [batch_size, 1, hidden_dim]
        x = x.squeeze(1)       # [batch_size, hidden_dim]
        
        # Project to latent dim
        z = self.encoder_fc2(x) # [batch_size, latent_dim]
        return z

    def decode(self, z):
        """
        Decodes latent representation z back to reconstruction.
        z shape: [batch_size, latent_dim]
        """
        x = self.decoder_fc1(z)  # [batch_size, hidden_dim]
        x = F.relu(x)
        
        x = x.unsqueeze(1)    # [batch_size, 1, hidden_dim]
        x = self.decoder_attn(x)
        x = x.squeeze(1)      # [batch_size, hidden_dim]
        
        # Final reconstruction
        out = self.decoder_fc2(x)  # [batch_size, input_dim]
        return out

    def forward(self, x):
        """
        Returns both reconstructed x_hat and latent embedding z.
        """
        z = self.encode(x)
        x_recon_logits = self.decode(z)
        
        return x_recon_logits, None, z  # No need for mu in this case
    
class Autoencoder(nn.Module):
    def __init__(self, input_dim, 
                        latent_dim=128,
                        backbone = 'mlp', 
                        hidden_dim=256,
                        dropout=0.3,
                        num_heads=4,
                        num_layers=2):
        """
        Args:
          input_dim: Number of features in the input.
          latent_dim: Dimensionality of the latent space.
          backbone: Backbone architecture for the autoencoder ('mlp' or 'residual').
        """
        super(Autoencoder, self).__init__()
        
        if backbone == 'mlp':
            self.autoencoder = MLPAutoencoder(input_dim, latent_dim)
        elif backbone == 'residual':
            self.autoencoder = ResidualAutoencoder(input_dim, latent_dim)
        elif backbone == 'attn':
            self.autoencoder = AttentionAutoencoder(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_heads=num_heads, dropout=dropout)
        elif backbone == 'attn_v2':
            self.autoencoder = ImprovedAttentionAutoencoder(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_heads=num_heads, dropout=dropout, num_layers=num_layers)
        elif backbone == 'self_attn':
            self.autoencoder = SelfAttentionAutoencoder(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, seq_len=3, num_heads=num_heads, dropout=dropout)
        else:
            raise ValueError("Invalid backbone architecture. Choose 'mlp', 'residual', 'rnn', or 'attention'.")
        
    def encode(self, x):
        """
        Encodes input x into a latent representation z.
        x shape: [batch_size, input_dim]
        """
        z = self.autoencoder.encode(x)
        return z
    
    def decode(self, z):
        """
        Decodes latent representation z back to reconstruction.
        z shape: [batch_size, latent_dim]
        """
        x_recon_logits = self.autoencoder.decode(z)
        return x_recon_logits
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, None, z  # No need for mu in this case

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, 
                        latent_dim=128, 
                        num_heads=4,
                        hidden_dim=256,
                        dropout=0.3,
                        backbone='mlp'):
        """
        Args:
          input_dim: Dimensionality of the input features.
          latent_dim: Dimensionality of the latent representation.
          num_classes: Number of classes for the classification head.
        """
        super(ContrastiveModel, self).__init__()
        # Encoder: input -> hidden -> latent representation
        if backbone == 'mlp':
            self.autoencoder = MLPAutoencoder(input_dim, latent_dim)
        elif backbone == 'residual':
            self.autoencoder = ResidualAutoencoder(input_dim, latent_dim)
        elif backbone == 'attention':
            self.autoencoder = AttentionAutoencoder(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_heads=num_heads, dropout=dropout)
        else:
            raise ValueError("Invalid backbone architecture. Choose 'mlp', 'residual', or 'attention'.")
        
        # Projection head: maps latent representation to a space for contrastive learning.
        # self.projection_head = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.BatchNorm1d(latent_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(latent_dim, latent_dim)
        # )
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            AttentionBlock(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout),
            nn.Linear(latent_dim, latent_dim)
        )

    def encode(self, x):
        """
        Encodes input x into a latent representation z.
        x shape: [batch_size, input_dim]
        """
        z = self.autoencoder.encode(x)
        return z
        
    def forward(self, x):
        latent = self.encode(x)
        projection = self.projection_head(latent)
        return projection, latent

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=128, num_classes=6):
        """
        Args:
          input_dim: Number of features in the input.
          latent_dim: Dimensionality of the latent space.
          num_classes: Number of output classes for classification.
        """
        super(VAE, self).__init__()
        # Encoder: maps input to hidden features then to latent parameters.
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder: reconstructs input from latent representation.
        self.fc2 = nn.Linear(latent_dim, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, input_dim)
        
        # Classification head: predicts labels from latent representation.
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def encode_process(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        mu, logvar = self.encode_process(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z):
        h3 = F.relu(self.bn2(self.fc2(z)))
        # For reconstruction, we return logits (e.g., for BCEWithLogitsLoss)
        return self.fc3(h3)
    
    def forward(self, x):
        mu, logvar = self.encode_process(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        class_logits = self.classifier(z)
        return reconstruction, mu, logvar, class_logits, z