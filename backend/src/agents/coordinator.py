"""Graph assembly + ticker/peer resolution.

Stateless: no global knowledge graph, no checkpointer.
coordinator -> move -> news -> retriever -> write
News is gathered whatever the price check says. With no unusual move it is shown
as context only; the writer never presents it as an explanation.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langgraph.graph import END, StateGraph

from . import profile as profile_mod
from .move_agent import move_agent
from .news_agent import news_agent
from .retriever_agent import retriever_agent
from .state import LOOKUPState
from .writer import writer_agent

_BACKEND = Path(__file__).resolve().parents[2]
_UTILS = _BACKEND / "src" / "utils"
MAX_PEERS = 5


def normalize(ticker: str) -> str:
    """Strip exchange suffixes for matching."""
    return ticker.upper().replace(".NS", "").replace(".BO", "")


def _is_indian(t: str) -> bool:
    return t.endswith(".NS") or t.endswith(".BO")


def _load_universe(is_indian: bool) -> Dict[str, List[str]]:
    path = _UTILS / ("in_tickers.json" if is_indian else "us_tickers.json")
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_peers(input_ticker: str) -> Tuple[str, List[str], Optional[str]]:
    """Return (qualified_primary, peers, sector). Falls back to the bare ticker."""
    t = input_ticker.upper()
    for indian in (_is_indian(t), not _is_indian(t)):
        universe = _load_universe(indian)
        target = normalize(t)
        for sector, members in universe.items():
            for m in members:
                if normalize(m) == target:
                    peers = [p for p in members if normalize(p) != target][:MAX_PEERS]
                    return m, peers, sector
    return t, [], None


def coordinator_agent(state: LOOKUPState) -> dict:
    primary, peers, _sector = resolve_peers(state["tickers"][0])
    now = pd.Timestamp.now(tz="UTC").isoformat()
    return {
        "tickers": [primary] + peers,
        "as_of": now,
        "generated_at": now,
        "profile": profile_mod.get_profile(primary),
    }


def create_graph():
    wf = StateGraph(LOOKUPState)
    wf.add_node("coordinator", coordinator_agent)
    wf.add_node("move", move_agent)
    wf.add_node("news", news_agent)
    wf.add_node("retriever", retriever_agent)
    wf.add_node("write", writer_agent)
    wf.set_entry_point("coordinator")
    wf.add_edge("coordinator", "move")
    wf.add_edge("move", "news")
    wf.add_edge("news", "retriever")
    wf.add_edge("retriever", "write")
    wf.add_edge("write", END)
    return wf.compile()