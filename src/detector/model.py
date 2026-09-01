import torch
import torch.nn as nn


class ForensicResidualBlock(nn.Module):
    # Standard residual block because vanishing gradients give me trust issues
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection magic: keep the good gradients flowing
        return x + self.drop2(self.act2(self.ln2(self.fc2(self.drop1(self.act1(self.ln1(self.fc1(x))))))))


class ForensicAcousticDeepfakeNet(nn.Module):
    # Neural net trained to sniff out vocoder artifacts and robot vibes

    def __init__(self, input_dim: int = 60, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        # Project our 60 forensic acoustic features into latent space
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # Stack residual blocks for deep representation learning
        self.res1 = ForensicResidualBlock(hidden_dim, dropout=dropout)
        self.res2 = ForensicResidualBlock(hidden_dim, dropout=dropout)
        self.res3 = ForensicResidualBlock(hidden_dim, dropout=dropout)
        
        # Binary head: 0 = human, 1 = deepfake bot
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features and return unscaled logits (useful for training & Platt calibration)."""
        feat = self.input_layer(x)
        feat = self.res3(self.res2(self.res1(feat)))
        return self.head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw sigmoid probability."""
        return torch.sigmoid(self.forward_logits(x))

