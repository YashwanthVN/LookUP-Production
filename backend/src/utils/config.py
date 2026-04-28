import os
from dotenv import load_dotenv

load_dotenv()

def load_config():
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "mcp_command": os.getenv("MCP_SERVER_COMMAND", "yfnhanced-mcp"),
    }