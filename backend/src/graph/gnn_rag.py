import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, GATv2Conv

class FinancialReasoningGNN(nn.Module):
    def __init__(self, in_channels=3, edge_channels=1, hidden=64, out=32):
        super().__init__()
        
        # SYSTEM 1: Structural Track (Math/Fundamentals)
        # We use an MLP to help NNConv interpret raw metrics like Revenue & Market Cap
        sys1_mlp = nn.Sequential(
            nn.Linear(edge_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels * hidden),
            nn.LayerNorm(in_channels * hidden), # Prevents the "blowing up"
            nn.Tanh() # Squashes the output to a safe [-1, 1] range
        )
        self.conv_sys1 = NNConv(in_channels, hidden, sys1_mlp)
        
        # SYSTEM 2: Narrative Track (News/Sentiment)
        # We use GATv2 with 4 heads to catch different 'themes' in news sentiment
        self.conv_sys2 = GATv2Conv(in_channels, hidden, heads=4, edge_dim=edge_channels)
        
        # THE FUSION LAYER: Merges both streams
        # 4 heads from Sys2 + 1 stream from Sys1 = hidden * 5
        self.fusion = nn.Sequential(
            nn.Linear(hidden + (hidden * 4), hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, out)
        )

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        # 1. Parallel Execution: Both 'brains' process the graph simultaneously
        # System 1: Extracts structural patterns (Sector/Peer weight)
        emb_s1 = self.conv_sys1(x, edge_index, edge_attr).relu()
        
        # System 2: Extracts narrative importance (News/Hype weight)
        emb_s2, (edge_index_attn, alpha) = self.conv_sys2(
            x, edge_index, edge_attr, return_attention_weights=True
        )
        emb_s2 = emb_s2.relu()
        
        # 2. Fusion: Concatenate the outputs
        combined = torch.cat([emb_s1, emb_s2], dim=-1)
        
        # 3. Final Projection: Generate the latent stock representation
        z = self.fusion(combined)
        
        if return_attention:
            return z, (edge_index_attn, alpha)
        return z