import torch
import torch.nn as nn
import torchvision.models

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # Input: [batch_size, seq_len, embed_dim]
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class MultiBranchModel(nn.Module):
    def __init__(self, num_triplets, num_instruments, num_verbs, num_targets, seq_len=5):
        super(MultiBranchModel, self).__init__()
        self.seq_len = seq_len

        # Pretrained ResNet50 for feature extraction
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity() # Remove ResNet's final classification layer

        # Masked Multi-Head Attention Layers
        embed_dim = 2048
        num_heads = 8
        self.mh_attention1 = MaskedMultiHeadAttention(embed_dim, num_heads)
        self.mh_attention2 = MaskedMultiHeadAttention(embed_dim, num_heads)

        # Branches for individual predictions
        self.instrument_branch = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_instruments),
            nn.Sigmoid()
        )
        self.verb_branch = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_verbs),
            nn.Sigmoid()
        )
        self.target_branch = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_targets),
            nn.Sigmoid()
        )

        # Triplet prediction branch
        self.triplet_branch = nn.Sequential(
            nn.Linear(embed_dim + num_instruments + num_verbs + num_targets, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_triplets),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features from ResNet
        batch_size, seq_len, c, h, w = x.size()  # Input shape: [batch, seq_len, C, H, W]
        x = x.view(batch_size * seq_len, c, h, w)  # Flatten sequence for ResNet
        features = self.feature_extractor(x)  # [batch_size * seq_len, embed_dim]
        if hasattr(features, "logits"):
            features = features.logits  
        features = features.view(batch_size, seq_len, -1)  # Reshape to [batch, seq_len, embed_dim]

        # Apply masked multi-head attention to sequence features
        features = self.mh_attention1(features)
        features = self.mh_attention2(features)

        # Pass through individual branches for sequence-level prediction
        instrument_logits = self.instrument_branch(features)  # [batch, seq_len, num_instruments]
        verb_logits = self.verb_branch(features)  # [batch, seq_len, num_verbs]
        target_logits = self.target_branch(features)  # [batch, seq_len, num_targets]

        # Concatenate sequence features and logits for triplet prediction
        combined_features = torch.cat(
            (features, instrument_logits, verb_logits, target_logits), dim=2
        )
        triplet_logits = self.triplet_branch(combined_features)  # [batch, seq_len, num_triplets]

        return instrument_logits, verb_logits, target_logits, triplet_logits


