"""Check the price move, either for a day the user named or for the recent sessions.

No date given (default):
  event      -> the most recent unusual session (|z| >= 2.5) of the last 5 (preferring the
                direction the user asked about); mode 'explained'; the evidence
                window ends at that session's open
  notable    -> like event, but below the z threshold: a raw move >= 3% or a large
                intraday swing. Reported as a weaker signal.
  no_event   -> mode 'none'; recent coverage is still gathered, as context only
A date given (state['target_date'] or a date written in the question):
  exactly that session is tested, once. event -> 'explained', no_event -> 'none',
  and in both cases the evidence window ends at that session's open.
A no_event session with a large intraday swing is treated the same way: its window
ends at its open and post-open coverage is gathered (a crash that recovered by the
close still has a cause).
  If it was not a complete trading session, the request abstains.
unavailable -> mode 'unchecked'; the report says the move is unverified
Set LOOKUP_SKIP_MOVE_CHECK=1 to bypass (used by the labelling export).
"""
import datetime as dt
import os

import pandas as pd

from src.pricing import move_check
from src.pricing.move_check import query_direction
from src.pricing.query_dates import find_date

from .state import LOOKUPState


def _target(state: LOOKUPState):
    raw = state.get("target_date")
    if raw:
        try:
            return dt.date.fromisoformat(str(raw)[:10])
        except ValueError:
            return None
    d, _ = find_date(state.get("query", ""), pd.Timestamp.now(tz="UTC").date())
    return d


def move_agent(state: LOOKUPState) -> dict:
    target = _target(state)
    if os.getenv("LOOKUP_SKIP_MOVE_CHECK") == "1":
        move = {"status": "unavailable", "reason": "skipped"}
    else:
        kw = {"direction": query_direction(state.get("query", ""))}
        if target is not None:
            kw["target_date"] = target
        move = move_check.check_move(state["tickers"][0], **kw)
        if target is not None and "requested_date" not in move:
            move["requested_date"] = target.isoformat()   # keep the ask visible whatever came back
    if move["status"] in ("event", "notable"):
        peers = state["tickers"][1:]
        ctx = move_check.peer_context(peers, move["date"]) if peers else []
        return {"move": move, "mode": "explained", "as_of": move["session_open"],
                "peer_context": ctx}
    if move["status"] == "no_event":
        out = {"move": move, "mode": "none"}
        if move.get("requested_date") or (move.get("facts") or {}).get("intraday_flag"):
            out["as_of"] = move["session_open"]     # explain that session, not "now"
        return out
    return {"move": move, "mode": "unchecked"}