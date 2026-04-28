import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import networkx as nx

# ---------- FinancialKG ----------
class FinancialKG:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        
    def build_mini_kg(self):
        companies = ["AAPL", "MSFT", "GOOGL"]
        metrics = ["pe_ratio", "revenue"]
        quarters = ["2024Q1", "2024Q2"]
        
        for c in companies:
            self.graph.add_node(f"company_{c}", type="company", feat=[1.0, 0.0, 0.0])
        for m in metrics:
            self.graph.add_node(f"metric_{m}", type="metric", feat=[0.0, 1.0, 0.0])
        for q in quarters:
            self.graph.add_node(f"time_{q}", type="time", feat=[0.0, 0.0, 1.0])
            
        self.graph.add_edge("company_AAPL", "metric_pe_ratio", relation="HAS_METRIC")
        self.graph.add_edge("metric_pe_ratio", "time_2024Q2", relation="MEASURED_AT", value=29.5)
        
    def to_pyg_data(self):
        node_list = list(self.graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        x = torch.tensor([self.graph.nodes[n]['feat'] for n in node_list], dtype=torch.float)

        edge_index = []
        for u, v in self.graph.edges(keys=False):   # ← FIX HERE
            edge_index.append([node_to_idx[u], node_to_idx[v]])

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

# ---------- FinancialGNNRAG ----------
class FinancialGNNRAG(torch.nn.Module):
    def __init__(self, in_channels=3, hidden=64, out=32):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=4)
        self.conv2 = GATConv(hidden*4, hidden, heads=4)
        self.conv3 = GATConv(hidden*4, out, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# ---------- Test ----------
kg = FinancialKG()
kg.build_mini_kg()
data = kg.to_pyg_data()
print("Number of nodes:", data.x.shape[0])
print("Number of edges:", data.edge_index.shape[1])

model = FinancialGNNRAG()
out = model(data.x, data.edge_index)
print("Output shape:", out.shape)   # Expected: [num_nodes, 32]