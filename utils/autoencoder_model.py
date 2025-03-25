import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- CLASSIFIER MODELS --------------------------
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

# --------------------------- BLOCKS ---------------------------
class AttentionBlock(nn.Module):
    """
    Transformer-like block with:
      1) Multi-head Self-Attention (with dropout)
      2) Residual Connection + LayerNorm
      3) Feed-forward Subnetwork (with GELU activation and dropout)
      4) Another Residual Connection + LayerNorm
    """
    def __init__(self, embed_dim, num_heads=1, dropout=0.1, ff_hidden_multiplier=4):
        super(AttentionBlock, self).__init__()
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
    
class Identity(nn.Module):
    def forward(self, x):
        return x
    
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

# --------------------------- AUTOENCODER MODELS ---------------------------
class SelfAttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32, num_heads=4, dropout=0.3, num_layers=2, use_pos=True, num_tokens=1):
        """
        Args:
          input_dim   : Dimensionality of the input features.
          hidden_dim  : Dimensionality of the hidden representations.
          latent_dim  : Bottleneck (latent) dimensionality.
          num_heads   : Number of attention heads.
          dropout     : Dropout rate.
          num_layers  : Number of stacked attention blocks in both encoder and decoder.
        """
        super(SelfAttentionAutoencoder, self).__init__()
        self.use_pos = use_pos
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        assert input_dim % num_tokens == 0, "Input dimension must be divisible by the number of tokens."

        # -- ENCODER --
        if num_tokens == 1:
            # Original: project input to hidden_dim
            self.encoder_fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU()
            )
        else:
            # Project input to (num_tokens * hidden_dim) then reshape to sequence.
            self.encoder_fc = nn.Sequential(
                nn.Linear(input_dim, num_tokens * hidden_dim),
                nn.GELU()
            )
            if use_pos:
                self.pos_embedding_enc = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
        
        # Stack attention blocks for encoder.
        self.encoder_blocks = nn.Sequential(*[
            AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Final projection from aggregated hidden representation to latent space.
        self.encoder_fc2 = nn.Linear(hidden_dim, latent_dim)
        
        # -- DECODER --
        if num_tokens == 1:
            # Project latent to hidden_dim.
            self.decoder_fc = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU()
            )
        else:
            # Project latent to (num_tokens * hidden_dim) then reshape to sequence.
            self.decoder_fc = nn.Sequential(
                nn.Linear(latent_dim, num_tokens * hidden_dim),
                nn.GELU()
            )
            if use_pos:
                self.pos_embedding_dec = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
                
        # Stack attention blocks for decoder.
        self.decoder_blocks = nn.Sequential(*[
            AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Final projection from aggregated hidden representation back to input space.
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        """
        Encodes input x into a latent representation z.
        Args:
          x: Tensor of shape [batch_size, input_dim]
        Returns:
          latent: Tensor of shape [batch_size, latent_dim]
        """
        batch = x.size(0)
        if self.num_tokens == 1:
            x_proj = self.encoder_fc(x)           # [batch, hidden_dim]
            x_seq = x_proj.unsqueeze(1)             # [batch, 1, hidden_dim]
        else:
            x_proj = self.encoder_fc(x)           # [batch, num_tokens * hidden_dim]
            x_seq = x_proj.view(batch, self.num_tokens, self.hidden_dim)  # [batch, num_tokens, hidden_dim]
            if self.use_pos:
                x_seq = x_seq + self.pos_embedding_enc  # add positional embeddings
        
        # Process sequence with attention blocks.
        x_seq = self.encoder_blocks(x_seq)        # [batch, num_tokens, hidden_dim]
        # Aggregate tokens: here we use mean pooling over the token dimension.
        x_agg = x_seq.mean(dim=1)                  # [batch, hidden_dim]
        latent = self.encoder_fc2(x_agg)           # [batch, latent_dim]
        return latent
    
    def decode(self, z):
        """
        Decodes latent representation z back to input space.
        Args:
          z: Tensor of shape [batch, latent_dim]
        Returns:
          out: Tensor of shape [batch, input_dim]
        """
        batch = z.size(0)
        if self.num_tokens == 1:
            x_proj = self.decoder_fc(z)            # [batch, hidden_dim]
            x_seq = x_proj.unsqueeze(1)             # [batch, 1, hidden_dim]
        else:
            x_proj = self.decoder_fc(z)            # [batch, num_tokens * hidden_dim]
            x_seq = x_proj.view(batch, self.num_tokens, self.hidden_dim)  # [batch, num_tokens, hidden_dim]
            if self.use_pos:
                x_seq = x_seq + self.pos_embedding_dec  # add positional embeddings
        
        # Process sequence with decoder attention blocks.
        x_seq = self.decoder_blocks(x_seq)         # [batch, num_tokens, hidden_dim]
        # Aggregate tokens via mean pooling.
        x_agg = x_seq.mean(dim=1)                   # [batch, hidden_dim]
        out = self.decoder_fc2(x_agg)               # [batch, input_dim]
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

class MultimodalCrossAttentionAutoencoder(nn.Module):
    """
    An autoencoder that integrates gene mutation profile (GMP) and clinical data (CD)
    via separate encoders and cross-attention fusion.
    """
    def __init__(self, 
                 input_dim_gmp, 
                 input_dim_cd, 
                 hidden_dim=256, 
                 latent_dim=256,
                 num_heads=4, 
                 dropout=0.3, 
                 gmp_num_layers=2,
                 cd_num_layers=2,
                 cross_attn_mode="stacked",    # "stacked" or "shared"
                 cross_attn_layers=2,
                 cd_encoder_mode="attention",   # "mlp" or "attention"
                 gmp_use_pos=True,
                 gmp_tokens=1):                # number of tokens for GMP branch (default=1)
        
        super(MultimodalCrossAttentionAutoencoder, self).__init__()
        
        self.input_dim_gmp = input_dim_gmp
        self.input_dim_cd = input_dim_cd
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.gmp_num_layers = gmp_num_layers
        self.cd_num_layers = cd_num_layers
        self.cross_attn_mode = cross_attn_mode
        self.gmp_use_pos = gmp_use_pos
        self.gmp_tokens = gmp_tokens
        
        # --- GMP Encoder ---
        # If multiple tokens are used, adjust the output dimension accordingly.
        assert input_dim_gmp % gmp_tokens == 0, "Input dimension must be divisible by the number of tokens."
        
        if gmp_tokens == 1:
            self.encoder_gmp_fc = nn.Sequential(
                nn.Linear(input_dim_gmp, hidden_dim),
                nn.GELU()
            )
        else:
            self.encoder_gmp_fc = nn.Sequential(
                nn.Linear(input_dim_gmp, gmp_tokens * hidden_dim),
                nn.GELU()
            )
            if gmp_use_pos:
                self.gmp_pos_embedding = nn.Parameter(torch.randn(1, gmp_tokens, hidden_dim))
        
        # Use self-attention blocks for GMP branch.
        self.encoder_gmp_blocks = nn.Sequential(*[
            AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(gmp_num_layers)
        ])
        
        # --- CD Encoder ---
        self.encoder_cd_fc = nn.Sequential(
            nn.Linear(input_dim_cd, hidden_dim),
            nn.GELU()
        )
        if cd_encoder_mode == "attention":
            self.encoder_cd_blocks = nn.Sequential(*[
                AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(cd_num_layers)
            ])
        else:  # "mlp" mode: use identity (no attention)
            self.encoder_cd_blocks = Identity()
        
        # --- Cross-Attention for Fusion ---
        # Use ModuleList for stacked mode, or a shared module.
        if cross_attn_mode == "stacked":
            self.cross_attn_gmp_to_cd = nn.ModuleList([
                CrossAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(cross_attn_layers)
            ])
            self.cross_attn_cd_to_gmp = nn.ModuleList([
                CrossAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(cross_attn_layers)
            ])
        elif cross_attn_mode == "shared":
            self.shared_cross_attn = CrossAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        else:
            raise ValueError("Invalid cross_attn_mode. Choose 'stacked' or 'shared'.")
        
        # Fusion: Concatenate the two fused representations and project to latent space.
        self.fusion_fc = nn.Linear(hidden_dim * 2, latent_dim)
        
        # --- GMP Decoder ---
        if gmp_tokens == 1:
            self.decoder_gmp_fc = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU()
            )
        else:
            self.decoder_gmp_fc = nn.Sequential(
                nn.Linear(latent_dim, gmp_tokens * hidden_dim),
                nn.GELU()
            )
            if gmp_use_pos:
                self.gmp_pos_embedding_dec = nn.Parameter(torch.randn(1, gmp_tokens, hidden_dim))
                
        self.decoder_gmp_blocks = nn.Sequential(*[
            AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(gmp_num_layers)
        ])
        self.decoder_gmp_fc2 = nn.Linear(hidden_dim, input_dim_gmp)
        
        # --- CD Decoder ---
        self.decoder_cd_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU()
        )
        if cd_encoder_mode == "attention":
            self.decoder_cd_blocks = nn.Sequential(*[
                AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(cd_num_layers)
            ])
        else:
            self.decoder_cd_blocks = Identity()
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
        batch = x_gmp.size(0)
        # --- GMP Encoding ---
        gmp = self.encoder_gmp_fc(x_gmp)  # either [batch, hidden_dim] or [batch, gmp_tokens*hidden_dim]
        if self.gmp_tokens == 1:
            gmp = gmp.unsqueeze(1)  # shape: [batch, 1, hidden_dim]
        else:
            gmp = gmp.view(batch, self.gmp_tokens, self.hidden_dim)  # [batch, gmp_tokens, hidden_dim]
            if self.gmp_use_pos:
                gmp = gmp + self.gmp_pos_embedding  # add positional embeddings if desired
        
        gmp = self.encoder_gmp_blocks(gmp)  # [batch, tokens, hidden_dim]
        
        # --- CD Encoding ---
        cd = self.encoder_cd_fc(x_cd)  # [batch, hidden_dim]
        cd = cd.unsqueeze(1)          # [batch, 1, hidden_dim]
        cd = self.encoder_cd_blocks(cd)  # either processed with attention blocks or identity
        
        # --- Cross-Attention Fusion ---
        # Expected shapes: gmp: [batch, gmp_tokens, hidden_dim] (or [batch, 1, hidden_dim] if gmp_tokens==1)
        #                  cd:  [batch, 1, hidden_dim]
        if self.cross_attn_mode == "stacked":
            # For GMP -> CD: iterate over the ModuleList manually.
            query_gmp = gmp
            key_cd = cd
            for block in self.cross_attn_gmp_to_cd:
                query_gmp = block(query_gmp, key_cd)
            gmp_fused = query_gmp # [batch, seq_len, hidden_dim]
            
            # For CD -> GMP:
            query_cd = cd
            key_gmp = gmp
            for block in self.cross_attn_cd_to_gmp:
                query_cd = block(query_cd, key_gmp)
            cd_fused = query_cd
        else:  # shared mode
            query_gmp = self.shared_cross_attn(gmp, cd) # batch first in the cross-attention block
            query_cd = self.shared_cross_attn(cd, gmp)
            gmp_fused = query_gmp
            cd_fused = query_cd
        
        # For fusion, we squeeze the sequence dimension.
        # For GMP, if multiple tokens exist, you may choose to pool (e.g., mean) over tokens.
        if self.gmp_tokens == 1:
            gmp_rep = gmp_fused.squeeze(1)  # [batch, hidden_dim]
        else:
            gmp_rep = gmp_fused.mean(dim=1)   # [batch, hidden_dim]
        cd_rep = cd_fused.squeeze(1)         # [batch, hidden_dim]
        
        # Concatenate the representations and project to latent space.
        fused = torch.cat([gmp_rep, cd_rep], dim=1)  # [batch, 2*hidden_dim]
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
        batch = latent.size(0)
        # --- GMP Decoder ---
        gmp_dec = self.decoder_gmp_fc(latent)  # [batch, hidden_dim] or [batch, gmp_tokens*hidden_dim]
        if self.gmp_tokens == 1:
            gmp_dec = gmp_dec.unsqueeze(1)  # [batch, 1, hidden_dim]
        else:
            gmp_dec = gmp_dec.view(batch, self.gmp_tokens, self.hidden_dim)  # [batch, gmp_tokens, hidden_dim]
            if self.gmp_use_pos:
                gmp_dec = gmp_dec + self.gmp_pos_embedding_dec
        gmp_dec = self.decoder_gmp_blocks(gmp_dec)  # [batch, tokens, hidden_dim]
        # For reconstruction, if multiple tokens, you might pool or flatten.
        
        if self.gmp_tokens == 1:
            gmp_dec = gmp_dec.squeeze(1)  # [batch, hidden_dim]
        else:
            gmp_dec = gmp_dec.mean(dim=1)  # [batch, hidden_dim]
        recon_gmp = self.decoder_gmp_fc2(gmp_dec)  # [batch, input_dim_gmp]
        
        # --- CD Decoder ---
        cd_dec = self.decoder_cd_fc(latent)  # [batch, hidden_dim]
        cd_dec = cd_dec.unsqueeze(1)          # [batch, 1, hidden_dim]
        cd_dec = self.decoder_cd_blocks(cd_dec) # process with attention or identity
        cd_dec = cd_dec.squeeze(1)            # [batch, hidden_dim]
        recon_cd = self.decoder_cd_fc2(cd_dec)  # [batch, input_dim_cd]
        
        return recon_gmp, recon_cd
    
    def forward(self, x_gmp, x_cd):
        """
        Forward pass through the multimodal autoencoder.
        Returns:
          recon_gmp: Reconstruction of the gene mutation profile.
          recon_cd : Reconstruction of the clinical data.
          latent   : The fused latent representation.
        """
        latent = self.encode(x_gmp, x_cd)
        recon_gmp, recon_cd = self.decode(latent)
        return recon_gmp, recon_cd, latent

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
    
class Autoencoder(nn.Module):
    def __init__(self, input_dim, 
                        latent_dim=128,
                        backbone = 'mlp', 
                        hidden_dim=256,
                        dropout=0.3,
                        num_heads=4,
                        num_layers=2,
                        use_pos=True,
                        num_tokens=1):
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
        elif backbone == 'self_attn':
            self.autoencoder = SelfAttentionAutoencoder(input_dim, 
                                                        hidden_dim=hidden_dim, 
                                                        latent_dim=latent_dim, 
                                                        num_heads=num_heads, 
                                                        dropout=dropout, 
                                                        num_layers=num_layers,
                                                        use_pos=use_pos,
                                                        num_tokens=num_tokens)
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
    
class HybridTransformerMLPAutoencoder(nn.Module):
    def __init__(self, input_dim, tokens_num=30, token_dim=128, latent_dim=128,
                 nhead=8, num_layers=3, dropout=0.1):
        """
        Hybrid autoencoder that uses an initial MLP to split the input into patches (tokens)
        and transformer encoder/decoder layers to capture inter-token dependencies.
        
        Args:
            input_dim (int): Total number of input features.
            tokens_num (int): Number of tokens (patches). Must evenly divide input_dim.
            token_dim (int): Dimension for each token embedding.
            latent_dim (int): Dimension of the latent representation.
            nhead (int): Number of attention heads for transformer layers.
            num_layers (int): Number of transformer layers (for both encoder and decoder).
            dropout (float): Dropout probability.
        """
        super(HybridTransformerMLPAutoencoder, self).__init__()
        assert input_dim % tokens_num == 0, "input_dim must be divisible by tokens_num"
        
        self.input_dim = input_dim
        self.tokens_num = tokens_num
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.patch_size = input_dim // tokens_num  # Number of features per token

        # Encoder MLP: maps input vector to token embeddings.
        # Output shape: (batch, tokens_num * token_dim)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim, tokens_num * token_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Learnable positional embedding for tokens in the encoder.
        self.pos_embedding_enc = nn.Parameter(torch.randn(1, tokens_num, token_dim))
        
        # Transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Aggregate token representations (e.g. via mean pooling) and project to latent space.
        self.fc_enc = nn.Linear(token_dim, latent_dim)

        # Decoder: map latent vector back to token sequence.
        self.fc_dec = nn.Linear(latent_dim, tokens_num * token_dim)
        
        # Learnable positional embedding for tokens in the decoder.
        self.pos_embedding_dec = nn.Parameter(torch.randn(1, tokens_num, token_dim))
        
        # Transformer decoder layers (using an encoder-style stack for decoding).
        decoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Final projection: for each token, reconstruct its patch (of size patch_size).
        self.patch_reconstruction = nn.Linear(token_dim, self.patch_size)
        
    def encode(self, x):
        """
        Encodes the input vector into a latent representation.
        
        Args:
            x: Tensor of shape (batch, input_dim)
        Returns:
            latent: Tensor of shape (batch, latent_dim)
        """
        batch_size = x.size(0)
        # Map input to tokens: (batch, tokens_num * token_dim)
        tokens = self.encoder_mlp(x)
        # Reshape into sequence: (batch, tokens_num, token_dim)
        tokens = tokens.view(batch_size, self.tokens_num, self.token_dim)
        # Add positional encoding.
        tokens = tokens + self.pos_embedding_enc
        # Transformer expects (sequence_length, batch, d_model).
        tokens = tokens.transpose(0, 1)
        encoded_tokens = self.transformer_encoder(tokens)
        # Back to shape: (batch, tokens_num, token_dim)
        encoded_tokens = encoded_tokens.transpose(0, 1)
        # Aggregate tokens (mean pooling).
        token_agg = encoded_tokens.mean(dim=1)
        # Project aggregated token to latent space.
        latent = self.fc_enc(token_agg)
        return latent

    def decode(self, latent):
        """
        Decodes the latent representation back to a reconstruction.
        
        Args:
            latent: Tensor of shape (batch, latent_dim)
        Returns:
            x_recon: Reconstructed input of shape (batch, input_dim)
        """
        batch_size = latent.size(0)
        # Expand latent into a token sequence.
        dec_tokens = self.fc_dec(latent)
        dec_tokens = dec_tokens.view(batch_size, self.tokens_num, self.token_dim)
        # Add decoder positional encoding.
        dec_tokens = dec_tokens + self.pos_embedding_dec
        # Prepare for transformer decoder.
        dec_tokens = dec_tokens.transpose(0, 1)
        decoded_tokens = self.transformer_decoder(dec_tokens)
        decoded_tokens = decoded_tokens.transpose(0, 1)
        # For each token, reconstruct the patch.
        patches = self.patch_reconstruction(decoded_tokens)  # (batch, tokens_num, patch_size)
        # Flatten tokens to obtain the final reconstruction.
        x_recon = patches.view(batch_size, self.input_dim)
        return x_recon

    def forward(self, x):
        latent = self.encode(x)
        x_recon = self.decode(latent)
        return x_recon, latent
    
if __name__ == '__main__':
    # Test the autoencoder models
    gmp_dim = 1000
    cd_dim = 100
    
    # Test the Multi-Modal Autoencoder
    model = MultimodalCrossAttentionAutoencoder(input_dim_gmp=gmp_dim, input_dim_cd=cd_dim, gmp_tokens=10)
    x_gmp = torch.randn(32, gmp_dim)
    x_cd = torch.randn(32, cd_dim)
    recon_gmp, recon_cd, latent = model(x_gmp, x_cd)
    print("Multi-Modal Autoencoder:")
    print("Reconstruction GMP:", recon_gmp.size())
    print("Reconstruction CD:", recon_cd.size())
    print("Latent:", latent.size())
    
    
    # Test the Self-Attention Autoencoder
    model = SelfAttentionAutoencoder(input_dim=gmp_dim, hidden_dim=256, latent_dim=32, num_heads=4, dropout=0.3, num_tokens=11)
    x = torch.randn(32, gmp_dim)
    recon, _, latent = model(x)
    print("\nSelf-Attention Autoencoder:")
    print("Reconstruction:", recon.size())
    print("Latent:", latent.size())