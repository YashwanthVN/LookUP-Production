"""Honest report writer.

States what was done and what was not. It makes no causal claim: the ranking
is textual relevance of headlines to the query, not validated attribution.
"""
from .state import LOOKUPState


def _line(i: int, d: dict) -> str:
    who = "" if d["is_primary"] else f" [{d['ticker']}]"
    return (f"{i}. {d['headline']}{who} - {d.get('source') or 'unknown'}, "
            f"{d['published_at'][:16]}Z")


def writer_agent(state: LOOKUPState) -> dict:
    primary = state["tickers"][0]
    ranked = state.get("ranked") or []
    pool = state.get("news_pool") or []
    window = f"72h before {state['as_of'][:16]}Z"

    if not pool:
        return {
            "abstained": True,
            "abstain_reason": "no_timestamped_news_in_window",
            "final_report": (
                f"No timestamped news for {primary} or its peers was found in the "
                "last 72 hours, so no explanation is offered."
            ),
        }

    own = sum(1 for d in pool if d["is_primary"])
    if own == 0:
        lines = [
            f"No headlines about {primary} itself were found in the {window}, "
            "so no explanation of its move is offered.",
            "",
            "Sector context from peers (not about this company):",
        ] + [_line(i, d) for i, d in enumerate(ranked, 1)]
        return {"abstained": True, "abstain_reason": "no_news_for_primary_ticker",
                "final_report": "\n".join(lines)}

    lines = [f"### Most relevant recent headlines for {primary}",
             f"*Ranked by cross-encoder relevance from {len(pool)} headlines "
             f"({own} about {primary}) in the {window}. This is a relevance "
             "ranking, not a verified explanation of the price move.*", ""]
    lines += [_line(i, d) for i, d in enumerate(ranked, 1)]
    return {"abstained": False, "abstain_reason": "", "final_report": "\n".join(lines)}