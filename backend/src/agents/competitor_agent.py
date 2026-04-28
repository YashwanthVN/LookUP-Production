import torch
import torch.nn.functional as F
from .state import LOOKUPState
from src.kg_holder import get_kg

def competitor_agent(state: LOOKUPState) -> dict:
    kg = get_kg()
    if kg is None: return {"competitor_candidates": []}

    primary_ticker = state["tickers"][0]
    
    # 1. Sync Data and Node List
    data = kg.to_pyg_data()
    node_list = list(kg.graph.nodes)
    
    # 2. Forward Pass for Embeddings
    kg.gnn_model.eval()
    with torch.no_grad():
        # FIXED UNPACKING: 
        # The model returns: embeddings, (edge_index, alpha)
        out = kg.gnn_model(data.x, data.edge_index, data.edge_attr, return_attention=True)
        embeddings = out[0] # This isolates 'z' (the embeddings)
    
    # 3. Robust Node Lookup
    target_idx = kg.get_node_index(primary_ticker)
    
    if target_idx is None:
        return {"competitor_candidates": []}

    # Ensure we have a 2D tensor for cosine_similarity [1, feature_dim]
    primary_emb = embeddings[target_idx].view(1, -1)
    
    # 4. Find ALL company nodes and compute similarity
    similarities = []
    for i, node_name in enumerate(node_list):
        if "company_" in node_name and i != target_idx:
            other_emb = embeddings[i].view(1, -1)
            
            # Compute Cosine Similarity in the latent space
            sim = F.cosine_similarity(primary_emb, other_emb).item()
            ticker_label = node_name.replace("company_", "")
            similarities.append((ticker_label, sim))
    
    # Sort by mathematical proximity in the GNN's feature space
    similarities.sort(key=lambda x: x[1], reverse=True)
    candidates = [f"{t} (Sim: {s:.4f})" for t, s in similarities[:5]]
    
    return {"competitor_candidates": candidates}