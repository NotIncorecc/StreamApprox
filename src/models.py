import torch
import torch.nn as nn

class MLPApproximator(nn.Module):
    """
    Baseline 1 & Method A: Concatenates visual and audio features and projects
    them to the teacher embedding space using a Multi-Layer Perceptron (MLP).
    """
    def __init__(self, input_dim=640, hidden_dims=[512, 1024], output_dim=1024, dropout=0.1):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.BatchNorm1d(h_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = h_d
        layers.append(nn.Linear(in_d, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, img_emb, aud_emb):
        # Concatenate along the feature dimension
        x = torch.cat([img_emb, aud_emb], dim=-1)
        return self.network(x)


class TransformerFusionApproximator(nn.Module):
    """
    Method B: Uses a Transformer Encoder to fuse image and audio features.
    Treats the features as tokens, adds a [CLS] token, and projects the
    final [CLS] representation to the teacher embedding space.
    """
    def __init__(self, img_dim=512, aud_dim=128, embed_dim=256, num_heads=4, 
                 num_layers=2, output_dim=1024, dropout=0.1):
        super().__init__()
        # Token projection layers
        self.img_proj = nn.Linear(img_dim, embed_dim)
        self.aud_proj = nn.Linear(aud_dim, embed_dim)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learned modality type embeddings to distinguish image and audio
        self.modality_embeddings = nn.Parameter(torch.zeros(1, 3, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final output projection head
        self.fc_out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.modality_embeddings, std=0.02)
        
    def forward(self, img_emb, aud_emb):
        batch_size = img_emb.size(0)
        
        # Project inputs to shared embed_dim
        # img_emb shape: [B, img_dim] -> [B, 1, embed_dim]
        # aud_emb shape: [B, aud_dim] -> [B, 1, embed_dim]
        z_img = self.img_proj(img_emb).unsqueeze(1)
        z_aud = self.aud_proj(aud_emb).unsqueeze(1)
        
        # Tile CLS token for the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate tokens: [CLS, IMG, AUD] (shape: [B, 3, embed_dim])
        tokens = torch.cat([cls_tokens, z_img, z_aud], dim=1)
        
        # Add modality/position embeddings
        tokens = tokens + self.modality_embeddings
        
        # Process with Transformer Encoder
        # Output shape: [B, 3, embed_dim]
        features = self.transformer(tokens)
        
        # Extract the representation of the [CLS] token (index 0)
        # Shape: [B, embed_dim]
        cls_rep = features[:, 0]
        
        # Project to target teacher embedding space
        return self.fc_out(cls_rep)
