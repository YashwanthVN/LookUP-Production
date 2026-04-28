from torch_geometric_temporal.nn.recurrent import TGCN
import torch.nn as nn

class TemporalFinancialGNN(nn.Module):
    """TGN for dynamic competitor relationships"""
    
    def __init__(self, node_features=32, hidden=64):
        super().__init__()
        self.tgcn = TGCN(node_features, hidden)
        self.linear = nn.Linear(hidden, 16)
    
    def forward(self, snapshots):
        h = None
        for x, edge_index in snapshots:
            h = self.tgcn(x, edge_index, h)
        return self.linear(h)