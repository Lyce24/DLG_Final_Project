import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- Prediction Model ---------------------------
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

class Autoencoder(nn.Module):
    def __init__(self, input_dim, 
                        latent_dim=128,
                        backbone = 'mlp', 
                        hidden_dim=256,
                        dropout=0.3,
                        num_heads=4,
                        num_layers=2,
                        use_pos=True,
                        num_tokens=1,
                        prediction=False,
                        prediction_model='mlp'):
        """
        Args:
          input_dim: Number of features in the input.
          latent_dim: Dimensionality of the latent space.
          backbone: Backbone architecture for the autoencoder ('mlp' or 'residual').
        """
        super(Autoencoder, self).__init__()
        
        if backbone == 'mlp':
            self.autoencoder = MLPAutoencoder(input_dim, latent_dim)
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
            raise ValueError("Invalid backbone architecture. Choose 'mlp', 'self-attn'.")
        
        self.prediction = prediction
        if prediction:
            if prediction_model == 'mlp':
                self.prediction_model = MLP(latent_dim)
            else:
                raise ValueError("Invalid prediction model. Choose 'mlp'.")
        
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
        if self.prediction:
            # If prediction is enabled, pass the latent representation through the prediction model.
            pred = self.prediction_model(z)
            return x_recon, None, z, pred
        else:
            return x_recon, None, z, None # No need for mu in this case

class MultimodalAutoencoder(nn.Module):
    """
    An autoencoder that integrates gene mutation profile (GMP) and clinical data (CD)
    via separate encoders and cross-attention fusion.
    """
    def __init__(self, 
                 input_dim_gmp, 
                 input_dim_cd, 
                 hidden_dim=512, 
                 latent_dim=256,
                 backbone='mlp', # 'mlp' or 'self_attn'
                 num_heads=4, 
                 dropout=0.3, 
                 gmp_num_layers=2,
                 cd_num_layers=2,
                 fusion_mode='cross_attention',  # "cross_attention" or "concat" or "gated"
                 cross_attn_mode="stacked",    # "stacked" or "shared"
                 cross_attn_layers=2,
                 cd_encoder_mode="attention",   # "mlp" or "attention" or "plain"
                 prediction=False,
                 prediction_model='mlp',
                ):                # number of tokens for GMP branch (default=1)
        
        super(MultimodalAutoencoder, self).__init__()
        
        self.input_dim_gmp = input_dim_gmp
        self.input_dim_cd = input_dim_cd
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.gmp_num_layers = gmp_num_layers
        self.cd_num_layers = cd_num_layers
        
        self.fusion_mode = fusion_mode
        self.cross_attn_mode = cross_attn_mode
        
        self.prediction = prediction
        

        # --- GMP Encoder ---
        self.encoder_gmp_fc = nn.Sequential(
            nn.Linear(input_dim_gmp, hidden_dim),
            nn.GELU()
        )
        
        if backbone == 'self_attn':
            # Use self-attention blocks for GMP branch.
            self.encoder_gmp_blocks = nn.Sequential(*[
                AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(gmp_num_layers)
            ])
        elif backbone == 'mlp':
            # Use MLP for GMP branch.
            self.encoder_gmp_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ) for _ in range(gmp_num_layers)
            ])
        else:
            raise ValueError("Invalid backbone architecture. Choose 'mlp' or 'self_attn'.")
        
        # --- CD Encoder ---
        self.encoder_cd_fc = nn.Sequential(
            nn.Linear(input_dim_cd, hidden_dim),
            nn.GELU()
        )
        if cd_encoder_mode == "self_attn":
            self.encoder_cd_blocks = nn.Sequential(*[
                AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(cd_num_layers)
            ])
        elif cd_encoder_mode == 'mlp':
            self.encoder_cd_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ) for _ in range(cd_num_layers)
            ])
        elif cd_encoder_mode == "plain":
            self.encoder_cd_blocks = Identity()
        else:
            raise ValueError("Invalid CD encoder mode. Choose 'mlp', 'self_attn', or 'plain'.")
        
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
        
        # --- Gated Fusion ---
        if fusion_mode == 'gated':
            self.gate_fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.Sigmoid()
            )
        
        # Fusion: Concatenate the two fused representations and project to latent space.
        self.fusion_fc = nn.Linear(hidden_dim * 2, latent_dim)
        
        # --- GMP Decoder ---
        self.decoder_gmp_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU()
        )
        if backbone == 'self_attn':
            self.decoder_gmp_blocks = nn.Sequential(*[
                AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(gmp_num_layers)
            ])
        elif backbone == 'mlp':
            self.decoder_gmp_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ) for _ in range(gmp_num_layers)
            ])
        else:
            raise ValueError("Invalid backbone architecture. Choose 'mlp' or 'self_attn'.")
        self.decoder_gmp_fc2 = nn.Linear(hidden_dim, input_dim_gmp)
        
        # --- CD Decoder ---
        self.decoder_cd_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU()
        )
        if cd_encoder_mode == "self_attn":
            self.decoder_cd_blocks = nn.Sequential(*[
                AttentionBlock(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(cd_num_layers)
            ])
        elif cd_encoder_mode == 'mlp':
            self.decoder_cd_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ) for _ in range(cd_num_layers)
            ])
        elif cd_encoder_mode == "plain":
            self.decoder_cd_blocks = Identity()
        else:
            raise ValueError("Invalid CD decoder mode. Choose 'mlp', 'self_attn', or 'plain'.")
        self.decoder_cd_fc2 = nn.Linear(hidden_dim, input_dim_cd)
        
        if prediction:
            if prediction_model == 'mlp':
                self.prediction_model = MLP(latent_dim)
            else:
                raise ValueError("Invalid prediction model. Choose 'mlp'.")
        else:
            self.prediction_model = None
        
    def encode(self, x_gmp, x_cd):
        """
        Encode each modality and fuse them via cross-attention.
        Args:
          x_gmp: [batch, input_dim_gmp]
          x_cd:  [batch, input_dim_cd]
        Returns:
          latent: [batch, latent_dim] fused latent representation.
        """
        # Encode GMP
        gmp = self.encoder_gmp_fc(x_gmp).unsqueeze(1)
        gmp = self.encoder_gmp_blocks(gmp)

        # Encode CD
        cd = self.encoder_cd_fc(x_cd).unsqueeze(1)
        cd = self.encoder_cd_blocks(cd)
        
        # Fusion
        if self.fusion_mode == 'cross_attention':
            # GMP -> CD
            query_gmp, query_cd = gmp, cd
           
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
                
            gmp_rep = gmp_fused.squeeze(1)
            cd_rep = cd_fused.squeeze(1)
        else:
            gmp_rep = gmp.squeeze(1)
            cd_rep = cd.squeeze(1)
        
        # Concatenate or gated
        rep = torch.cat([gmp_rep, cd_rep], dim=-1)
        if self.fusion_mode == 'gated':
            rep = self.gate_fc(rep) * rep

        # Project to latent
        return self.fusion_fc(rep)
    
    def decode(self, latent):
        """
        Decode the latent representation back into both modalities.
        Args:
          latent: [batch, latent_dim]
        Returns:
          recon_gmp: Reconstructed gene mutation profile.
          recon_cd:  Reconstructed clinical data.
        """
        # Decode GMP
        gmp_dec = self.decoder_gmp_fc(latent).unsqueeze(1)
        gmp_dec = self.decoder_gmp_blocks(gmp_dec).squeeze(1)
        recon_gmp = self.decoder_gmp_fc2(gmp_dec)
        
        # Decode CD
        cd_dec = self.decoder_cd_fc(latent).unsqueeze(1)
        cd_dec = self.decoder_cd_blocks(cd_dec).squeeze(1)
        recon_cd = self.decoder_cd_fc2(cd_dec)

        return recon_gmp, recon_cd
    
    def forward(self, x_gmp, x_cd):
        latent = self.encode(x_gmp, x_cd)
        recon_gmp, recon_cd = self.decode(latent)
        if self.prediction:
            # If prediction is enabled, pass the latent representation through the prediction model.
            pred = self.prediction_model(latent)
            return recon_gmp, recon_cd, latent, pred
        else:
            return recon_gmp, recon_cd, latent, None

    
if __name__ == '__main__':
    # Test the autoencoder models
    gmp_dim = 1000
    cd_dim = 100
    
    # Test the Multi-Modal Autoencoder
    model = MultimodalAutoencoder(input_dim_gmp=gmp_dim, input_dim_cd=cd_dim,
                                  backbone='mlp',
                                  fusion_mode='gated',
                                  cross_attn_mode='stacked',
                                  cd_encoder_mode='self_attn')
    x_gmp = torch.randn(32, gmp_dim)
    x_cd = torch.randn(32, cd_dim)
    recon_gmp, recon_cd, latent, _ = model(x_gmp, x_cd)
    print("Multi-Modal Autoencoder:")
    print("Reconstruction GMP:", recon_gmp.size())
    print("Reconstruction CD:", recon_cd.size())
    print("Latent:", latent.size())