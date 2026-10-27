"""Check the latest complete session's market-adjusted move.

event      -> the most recent unusual session of the last 5 (preferring the
              direction the user asked about); mode 'explained'; the evidence
              window is moved to end at that session's open (nothing published
              after the open is used)
no_event   -> mode 'none'; recent coverage is still gathered, as context only
unavailable-> mode 'unchecked'; proceed, but the report says the move is unverified
Set LOOKUP_SKIP_MOVE_CHECK=1 to bypass (used by the labelling export).
"""
import os

from src.pricing import move_check
from src.pricing.move_check import query_direction

from .state import LOOKUPState


def move_agent(state: LOOKUPState) -> dict:
    if os.getenv("LOOKUP_SKIP_MOVE_CHECK") == "1":
        move = {"status": "unavailable", "reason": "skipped"}
    else:
        move = move_check.check_move(state["tickers"][0],
                                     direction=query_direction(state.get("query", "")))
    if move["status"] == "event":
        peers = state["tickers"][1:]
        ctx = move_check.peer_context(peers, move["date"]) if peers else []
        return {"move": move, "mode": "explained", "as_of": move["session_open"],
                "peer_context": ctx}
    if move["status"] == "no_event":
        return {"move": move, "mode": "none"}
    return {"move": move, "mode": "unchecked"}