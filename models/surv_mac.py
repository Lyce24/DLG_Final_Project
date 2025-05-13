import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List, Sequence, Union
import math
import torch, torch.nn as nn
from torch.nn.utils.rnn import pad_sequence          # helper for ragged → padded

# ------------------------------------------------------------
# Intra‑modality blocks
# ------------------------------------------------------------
class ResidualMLPBlock(nn.Module):
    """
    Residual MLP Block: two-layer MLP with skip connection
    """
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0, expansion_factor: int = 4):
        super().__init__()
        hidden_dim = hidden_dim or dim * expansion_factor
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim) if dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.layerscale = nn.Parameter(torch.ones(dim) * 1e-6)  # LayerScale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim]
        residual = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = residual + self.layerscale * out  # residual connection with layer scale
        return self.norm(out)

class GatedMLPBlock(nn.Module):
    """
    gMLP-style block that **preserves** input dimension (residual).
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1, expansion_factor: int = 2):
        super().__init__()
        hidden_dim = hidden_dim or dim * expansion_factor
        self.norm = nn.LayerNorm(dim)
        self.fc_in = nn.Linear(dim, expansion_factor * hidden_dim, bias=False)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.layerscale = nn.Parameter(torch.ones(dim) * 1e-6)  # LayerScale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        content, gate = self.fc_in(y).chunk(2, dim=-1)
        y = self.act(content) * torch.sigmoid(gate)  # [B, hidden]
        y = self.fc_out(y)  # back to dim
        y = self.drop(y)
        return x + self.layerscale * y  # residual connection with layer scale

class SEBlock(nn.Module):
    """Channel re-weighting for flat vectors (no 1-D pooling needed)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C]
        w = self.fc(x)          # squeeze step is identity for flat vector
        return x * w

class GLUBlock(nn.Module):
    """GLU: (W1x) ⊙ σ(W2x). Maintains output dim == input dim."""

    def __init__(self, dim, hidden = None, dropout = 0.0):
        super().__init__()
        hidden = hidden or dim * 2
        self.fc = nn.Linear(dim, 2 * dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        c, g = self.fc(y).chunk(2, dim=-1)
        y = c * torch.sigmoid(g)
        y = self.drop(y)
        return x + y
    
class SelfGateBlock(nn.Module):
    """
    Parametric element-wise gating on a single vector x ∈ ℝ^{B×H}.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.3):
        super().__init__()
        # one learnable scale per channel, init to 1.0
        self.scale   = nn.Parameter(torch.ones(embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.ln      = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H]
        gated = x * self.scale            # element-wise modulation
        out   = x + self.dropout(gated)   # residual + dropout
        return self.ln(out)               # LayerNorm for stability

# ------------------------------------------------------------
# Cross‑modality blocks
# ------------------------------------------------------------
class ConcatFusion(nn.Module):
    """Pre‑norm MLP on [h_g‖h_c] with double-dropout and LayerNorm."""
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(2 * d)
        self.fc1   = nn.Linear(2 * d, 2 * d, bias=False)
        self.act   = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(2 * d, 2 * d, bias=False)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(2 * d)

    def forward(self, h_g, h_c):
        x = torch.cat([h_g, h_c], dim=-1)
        y = self.norm1(x)
        y = self.act(self.fc1(y))
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return self.norm2(y)
    
class FiLMFusion(nn.Module):
    """FiLM with pre‑norm, dropout, and post‑fusion LayerNorm."""
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.norm  = nn.LayerNorm(d)
        self.gamma_beta = nn.Linear(d, 2 * d, bias=False)
        self.proj       = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, 2 * d, bias=False)
        )
        self.norm2 = nn.LayerNorm(2 * d)

    def forward(self, h_g, h_c):
        hc     = self.norm(h_c)
        gamma, beta = self.gamma_beta(hc).chunk(2, dim=-1)
        hg     = self.norm(h_g)
        h_mod  = gamma * hg + beta
        y      = self.proj(torch.cat([hc, h_mod], dim=-1))
        return self.norm2(y)
    
class BidirectionalFiLMFusion(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.norm_g = nn.LayerNorm(d)
        self.norm_c = nn.LayerNorm(d)
        # Modulate h_g using h_c
        self.gamma_beta_g = nn.Linear(d, 2 * d, bias=False)
        # Modulate h_c using h_g
        self.gamma_beta_c = nn.Linear(d, 2 * d, bias=False)
        self.proj       = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, 2 * d, bias=False)
        )
        self.norm2 = nn.LayerNorm(2 * d)

    def forward(self, h_g, h_c):
        hg_norm, hc_norm = self.norm_g(h_g), self.norm_c(h_c)
        # Cross-modulation
        gamma_g, beta_g = self.gamma_beta_g(hc_norm).chunk(2, dim=-1)
        h_mod_g = gamma_g * hg_norm + beta_g
        gamma_c, beta_c = self.gamma_beta_c(hg_norm).chunk(2, dim=-1)
        h_mod_c = gamma_c * hc_norm + beta_c
        # Combine and project
        y = self.proj(torch.cat([h_mod_g, h_mod_c], dim=-1))
        return self.norm2(y)

class LowRankBilinearFusion(nn.Module):
    def __init__(self, d: int, rank: int = None, dropout: float = 0.1):
        super().__init__()
        r = rank or d // 4
        self.U = nn.Linear(d, r, bias=False)
        self.V = nn.Linear(d, r, bias=False)
        self.out = nn.Linear(r, 2 * d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(2 * d)

    def forward(self, h_g, h_c):
        u = self.U(h_g)
        v = self.V(h_c)
        y = self.out(u * v)
        y = self.drop(y)
        return self.norm(y)

class GatedFuse(nn.Module):
    """Single‑fc GLU fusion with LayerNorm and dropout."""
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(2 * d)
        self.fc    = nn.Linear(2 * d, 4 * d, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(2 * d)

    def forward(self, h_g, h_c):
        x = torch.cat([h_g, h_c], dim=-1)
        x = self.norm1(x)
        c, g = self.fc(x).chunk(2, dim=-1)
        y = c * torch.sigmoid(g)
        y = self.drop(y)
        return self.norm2(y)
    
class MoEFusion(nn.Module):
    """MoE fusion with LayerNorm and dropout in experts; outputs 2d."""
    def __init__(self, d: int, k: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Linear(d, k, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 2 * d, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d, 2 * d, bias=False)
            ) for _ in range(k)
        ])
        self.norm2 = nn.LayerNorm(2 * d)

    def forward(self, h_g, h_c):
        hc = self.norm(h_c)
        weights = torch.softmax(self.gate(hc), dim=-1)  # [B, k]
        hg = self.norm(h_g)
        expert_outs = torch.stack([e(hg) for e in self.experts], dim=-1)  # [B, 2d, k]
        y = (expert_outs * weights.unsqueeze(1)).sum(dim=-1)               # [B, 2d]
        return self.norm2(y)                                             # [B, 2d]

class CrossAttnFusion(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, num_heads=1, dropout=dropout, batch_first=True)
        
    def forward(self, h_g, h_c):
        # h_c as query, h_g as key/value
        q = self.norm_q(h_c).unsqueeze(1)  # [B,1,d]
        kv = self.norm_kv(h_g).unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)
        return torch.cat([h_c, attn_out.squeeze(1)], dim=-1) # [B,2d]

# ------------------ Autoencoders ------------------------------------------
class Surv_MAC(nn.Module):
    def __init__(self, 
                 num_genes: int, 
                 num_cd_fields: int,
                 intra_gmp: Union[str, Sequence[str]] = ('res', 'res'),
                 intra_cd: Union[str, Sequence[str]] = ('res',),
                 fusion_method = 'film', # 'film', 'bilinear', 'gated', 'moe', 'concat', 'crossattn'
                 expansion_factor: int = 2,
                 hidden_dim: int = 256, 
                 latent_dim: int = 128, 
                 proj_dim: int = 64, 
                 dropout: float = 0.2,
                 baseline = False):
        
        super().__init__()
        self.num_genes = num_genes
        self.num_cd_fields = num_cd_fields
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.mask_token_gmp = nn.Parameter(torch.randn(num_genes) * 1e-2)
        self.mask_token_cd  = nn.Parameter(torch.randn(num_cd_fields) * 1e-2)
        
        self.intra_gmp_blocks = list(intra_gmp) if isinstance(intra_gmp, (list, tuple)) else [intra_gmp]
        self.intra_cd_blocks  = list(intra_cd)  if isinstance(intra_cd,  (list, tuple)) else [intra_cd]

        # encoder: input → hidden → hidden
        self.encoder_gmp = self._build_intra(
            input_dim=num_genes,
            blocks=self.intra_gmp_blocks,
            expansion_factor=expansion_factor,
            hidden_dim=hidden_dim,
            dropout=dropout * 2
        )
        
        self.encoder_cd = self._build_intra(
            input_dim=num_cd_fields,
            blocks=self.intra_cd_blocks,
            expansion_factor=expansion_factor,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        if fusion_method == 'concat':
            # concat → 2d MLP → GLU gating
            self.fusion = nn.Sequential(
                ConcatFusion(d=hidden_dim, dropout=dropout),  # [B,2d] out
                GLUBlock(dim=2*hidden_dim, dropout=dropout)   # still [B,2d] out
            )
        elif fusion_method == 'film':
            # FiLM now returns [B,2d]
            self.fusion = FiLMFusion(d=hidden_dim, dropout=dropout)
        elif fusion_method == 'bi_film':
            # Bidirectional FiLM now returns [B,2d]
            self.fusion = BidirectionalFiLMFusion(d=hidden_dim, dropout=dropout)
        elif fusion_method == 'bilinear':
            # low-rank bilinear returns [B,2d]
            self.fusion = LowRankBilinearFusion(d=hidden_dim, rank=hidden_dim//2, dropout=dropout)
        elif fusion_method == 'gated':
            # a single GLU fuse on concat = [B,2d]
            self.fusion = GatedFuse(d=hidden_dim, dropout=dropout)
        elif fusion_method == 'moe':
            # MoE now yields [B,2d]
            self.fusion = MoEFusion(d=hidden_dim, k=4, dropout=dropout)
        elif fusion_method == 'crossattn':
            # cross-attention fusion
            self.fusion = CrossAttnFusion(d=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method!r}")

        # Projection to latent
        self.latent_proj = nn.Linear(hidden_dim * 2, latent_dim)
        
        # prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        # --- Decoders ---
        self.decoder_gmp =nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_genes)
        )
        self.decoder_cd  = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cd_fields)
        )
        self.baseline = baseline
    
    def _apply_mask(self, x, mask, mask_token):
        # mask: [B, D] (float or bool)
        if mask is None:
            return x
        mask_bool = mask.bool()
        mask_tok  = mask_token.unsqueeze(0).expand_as(x)
        return torch.where(mask_bool, mask_tok, x)
    
    def _build_intra(
        self,
        input_dim: int,
        blocks: Sequence[str],
        hidden_dim: int,
        dropout: float,
        expansion_factor: int = 1
    ) -> nn.Sequential:
        """
        Build a stack of intra-modality blocks:
          - Initial: Linear(input_dim→hidden_dim) + Gelu/Dropout/LayerNorm
          - For each block name in `blocks`, append the corresponding module + Dropout
        Supported block names: 'residual', 'se', 'glu', 'gatedmlp'
        """
        modules: List[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        ]
        # Map names to factories
        factory = {
            'res': lambda: ResidualMLPBlock(dim=hidden_dim, dropout=dropout, expansion_factor=expansion_factor),
            'se':       lambda: SEBlock(hidden_dim),
            'glu':      lambda: GLUBlock(hidden_dim, dropout=dropout),
            'gated': lambda: GatedMLPBlock(hidden_dim, dropout=dropout, expansion_factor=expansion_factor),
        }
        for blk in blocks:
            if blk not in factory:
                raise ValueError(f"Invalid intra block: {blk}")
            modules.append(factory[blk]())
            modules.append(nn.Dropout(dropout))
        return nn.Sequential(*modules)
    
    def encode(self, x_gmp, x_cd, m_gmp=None, m_cd=None):
        if m_gmp is None: m_gmp = torch.zeros_like(x_gmp)
        if m_cd  is None: m_cd  = torch.zeros_like(x_cd)
        
        # 1) inject mask tokens
        x_gmp_masked = self._apply_mask(x_gmp, m_gmp, self.mask_token_gmp)
        x_cd_masked  = self._apply_mask(x_cd, m_cd, self.mask_token_cd)
        
        h_gmp = self.encoder_gmp(x_gmp_masked)  # [B, hidden_dim] 
        h_cd  = self.encoder_cd(x_cd_masked)   # [B, hidden_dim]
       
        # cross-attn fusion
        if not self.baseline:
            rep = self.fusion(h_gmp, h_cd)  # [B, hidden_dim * 2]
        else:
            # no cross-attn fusion
            rep = torch.cat([h_gmp, h_cd], dim=-1)  # [B, hidden_dim * 2]
                        
        return self.latent_proj(rep)  # [B, latent_dim]
    
    def decode(self, z):
        recon_gmp = self.decoder_gmp(z)
        recon_cd  = self.decoder_cd(z)
        return recon_gmp, recon_cd
        
    def forward(self, x_gmp, x_cd, m_gmp=None, m_cd=None):
        z            = self.encode(x_gmp, x_cd, m_gmp, m_cd)
        h            = F.normalize(self.prediction_head(z), dim=1)
        recon_gmp, recon_cd = self.decode(z)
        return recon_gmp, recon_cd, h, z

class VanillaMaskedAutoencoder(nn.Module):
    def __init__(self, num_genes: int, 
                        num_cd_fields: int, 
                        hidden_dim: int,
                        latent_dim, 
                        proj_dim):
        super().__init__()
        # a learnable mask token (same size as each feature vector)
        self.mask_token = nn.Parameter(torch.randn(num_genes + num_cd_fields) * 1e-2)

        # encoder: input → hidden → latent
        self.encoder = nn.Sequential(
            nn.Linear(num_genes + num_cd_fields, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # decoder: latent → hidden → reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_genes + num_cd_fields),
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def encode(self, x_gmp, x_cd, m_gmp=None, m_cd=None):
        """
        x:    [B, input_dim]
        mask: [B, input_dim] float or bool, 1 = masked, 0 = keep
        """
        if m_gmp is None: m_gmp = torch.zeros_like(x_gmp)
        if m_cd  is None: m_cd  = torch.zeros_like(x_cd)
        
        # 1) concatenate GMP and CD features
        x = torch.cat([x_gmp, x_cd], dim=1)  # [B, input_dim]
        m = torch.cat([m_gmp, m_cd], dim=1)  # [B, input_dim]
        
        # 2) inject mask tokens
        m = m.bool()
        # expand mask_token → [B, D]
        mask_tok = self.mask_token.unsqueeze(0).expand_as(x)
        # replace masked positions
        x_masked = torch.where(m, mask_tok, x)  # [B, input_dim]
        
        # 2) encode → decode
        z     = self.encoder(x_masked)   # [B, latent_dim]
        return z

    def forward(self, x_gmp, x_cd, m_gmp=None, m_cd=None):
        z = self.encode(x_gmp, x_cd, m_gmp, m_cd)
        recon = self.decoder(z)          # [B, input_dim]
        h = self.prediction_head(z)  # [B, proj_dim]
        h = F.normalize(h, dim=1)  # [B, proj_dim]
        return recon, None, h, z


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Surv_MAC(num_genes=1000, 
                        num_cd_fields=11, 
                        intra_gmp=('res', 'res', 'se'),
                        intra_cd=('glu'),
                        fusion_method='film', # 'film', 'bilinear', 'gated', 'moe', 'concat'
                        hidden_dim=256, 
                        latent_dim=128, 
                        proj_dim=64, 
                        dropout=0.3,
                        baseline=False).to(device)
    
    gmp_example = torch.randn(1024, 1000).to(device)  # Example batch of gene mutation data
    cd_example = torch.randn(1024, 11).to(device)  # Example batch of clinical data
    masked_gmp = torch.ones_like(gmp_example).to(device)  # Example mask for gene mutation data
    masked_cd = torch.ones_like(cd_example).to(device)  # Example mask for clinical data

    # Forward pass
    recon_gmp, recon_cd, h, z = model(gmp_example, cd_example, masked_gmp, masked_cd)
    print(f"Reconstructed_1: {recon_gmp.shape}")
    print(f"Reconstructed_2: {recon_cd.shape}") if recon_cd is not None else None
    print(f"Latent representation: {z.shape}")