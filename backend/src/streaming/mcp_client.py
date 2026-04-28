import subprocess
import json
from typing import Dict, Any

class YFinanceMCPClient:
    """Bridge to Yahoo Finance MCP server"""
    
    def __init__(self, server_cmd="yfnhanced-mcp"):
        self.cmd = server_cmd
        self.proc = None
        
    def start(self):
        self.proc = subprocess.Popen(
            self.cmd.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Real‑time quote with quality scoring"""
        req = json.dumps({"tool": "get_quote", "params": {"symbol": symbol}})
        self.proc.stdin.write(req + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())