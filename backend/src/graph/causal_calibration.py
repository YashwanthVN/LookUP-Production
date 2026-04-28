import torch
import torch.nn as nn

def calibrate_attention(kg_obj, training_scenarios, pyg_data):
    optimizer = torch.optim.Adam(kg_obj.gnn_model.parameters(), lr=1e-3)
    kg_obj.gnn_model.train()
    
    pre_std = list(kg_obj.gnn_model.parameters())[0].std().item()
    if pre_std == 0:
        print("⚠️ Warning: Model weights are ZERO before training starts. Forcing re-init.")
        for layer in kg_obj.gnn_model.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)): # or GNN equivalent
                nn.init.xavier_uniform_(layer.weight)
                
    print(f"🚀 Starting Stabilized Calibration on {len(training_scenarios)} scenarios...")

    for epoch in range(50): # Increased epochs since LR is lower
        optimizer.zero_grad()
        
        # 2. Forward Pass
        z, (edge_index_attn, alpha) = kg_obj.gnn_model(
            pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr, return_attention=True
        )
        
        if torch.isnan(alpha).any():
            print(f"❌ CRITICAL: NaN detected in Alpha at Epoch {epoch}. Stopping.")
            break
        
        total_loss = torch.tensor(0.0, device=pyg_data.x.device)
        edges_found = 0
        
        for scenario in training_scenarios:
            true_edge_idx = kg_obj.find_edge_index(
                scenario['trigger_headline'], 
                scenario['target_ticker'], 
                pyg_data
            )
            
            if true_edge_idx is not None:
                edges_found += 1
                weight = scenario.get('weight', 1.0)
                # alpha is [num_edges, heads]. Average across heads.
                current_attention = alpha[true_edge_idx].mean()
                
                # Use Mean Squared Error style loss for smoother gradients
                total_loss = total_loss + torch.pow(1.0 - current_attention, 2) * scenario['weight']

        if edges_found > 0:
            # Normalize loss by number of edges to prevent huge gradients
            avg_loss = total_loss / edges_found
            avg_loss.backward()
            
            # 3. Gradient Clipping (Prevents 'nan' by capping updates)
            torch.nn.utils.clip_grad_norm_(kg_obj.gnn_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Edges: {edges_found} | Avg Loss: {avg_loss.item():.6f}")

    print("✅ Stabilized Calibration Complete.")