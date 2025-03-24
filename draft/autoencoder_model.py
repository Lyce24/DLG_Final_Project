import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class AttentionClassifier(nn.Module):
    """
    Classification head that uses self-attention.
    It projects the latent vector into a sequence of tokens, applies an attention block,
    then aggregates the tokens to produce classification logits.
    """
    def __init__(self, latent_dim, num_heads=1, dropout=0.1, num_classes=1, seq_len=4):
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
    
# -----------------------------------------------------------------
# Autoencoder Models
# -----------------------------------------------------------------
class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, num_heads=1, dropout=0.1, task = 'recon', num_task = 1, classification_head = 'mlp'):
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
        
        # -- Classifier -
        if task == 'recon':
            self.classifier = None
        elif task == 'joint':
            assert classification_head in ['mlp', 'attention'], "Invalid classification head. Choose 'mlp' or 'attention'."
            if classification_head == 'attention':
                self.classifier = AttentionClassifier(latent_dim, num_heads=num_heads, dropout=dropout, num_classes=num_task)
            else:  # 'mlp'
                self.classifier = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    
                    nn.Linear(128, num_task),  # Binary: 1 logit | Multi-class: num_classes
                )
        else:
            raise ValueError("Invalid task. Choose 'recon' or 'joint'.")
        
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
        
        # Classify
        if self.classifier is None:
            return x_recon_logits, None, z
        else:
            # If classifier exists, get class logits
            assert self.classifier is not None, "Classifier should be defined for classification task."
            # Pass the latent representation through the classifier
            class_logits = self.classifier(z)
            return x_recon_logits, class_logits, z
        
class MutationAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, task = 'recon', num_task = 1, classification_head = 'mlp'):
        super(MutationAutoencoder, self).__init__()
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
        
        if task == 'recon':
            self.classifier = None
        elif task == 'joint':
            assert classification_head in ['mlp', 'attention'], "Invalid classification head. Choose 'mlp' or 'attention'."
            if classification_head == 'attention':
                self.classifier = AttentionClassifier(latent_dim, num_heads=1, dropout=0.1, num_classes=num_task)
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    
                    nn.Linear(64, num_task)  # Binary: 1 logit | Multi-class: num_classes
                )
        else:
            raise ValueError("Invalid task. Choose 'recon' or 'joint'.")
        
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
        
        if self.classifier is None:
            return x_recon_logits, None, z
        else:
            # Pass the latent representation through the classifier
            assert self.classifier is not None, "Classifier should be defined for classification task."
            class_logits = self.classifier(z)
            return x_recon_logits, class_logits, z