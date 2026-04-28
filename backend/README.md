# LOOKUP: Graph-Augmented Agentic RAG for Real-Time Competitor Intelligence

**Status**: Initial setup – building GNN‑RAG fusion with temporal graphs and domain‑adapted reranking.

## Quick Start
1. Clone this repo
2. `python -m venv .venv && .venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your API keys
5. Run `python src/main.py`

## Windows Setup (Verified)

1. **Install Python 3.11.9** (critical – PyG wheels require Python ≤3.11)
   ```powershell
   winget install python.python.3.11
   ```

2. **Create virtual environment**
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install PyTorch 2.6.0 (CPU)**
   ```powershell
   pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install PyG binaries** (pre‑compiled, no compilation needed)
   ```powershell
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
   ```

5. **Install remaining dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

6. **Verify Setup**
    ```powershell
    python -c "from src.graph import FinancialGNNRAG; print('GNN module OK')"
    ```

## Architecture
[Diagram & description – coming soon]

## Publication Target
ACL 2026 / ICLR 2026