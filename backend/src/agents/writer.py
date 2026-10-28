"""Honest report writer.

States what was done and what was not. It makes no causal claim: the ranking
is textual relevance of headlines to the company, not validated attribution.
Headlines published after the session opened are labelled as such: they may
report the move rather than cause it.
"""
import math

from src.pricing.move_check import query_direction

from .state import LOOKUPState

SAME_DIR_Z = 1.5


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _line(i: int, d: dict) -> str:
    tag = {"company": "", "market": " [market news]"}.get(d.get("scope"), f" [{d['ticker']}]")
    chk = d.get("claim_check")
    price = f" *(price check: {chk['detail']})*" if chk else ""
    return (f"{i}. {d['headline']}{tag} - {d.get('source') or 'unknown'}, "
            f"{d['published_at'][:16]}Z{price}")


def facts_line(facts) -> str:
    """Open/low/high/close of the session versus the previous close (price data only)."""
    if not facts:
        return ""
    return (f"Session {facts['date']} versus the previous close of {facts['prev_close']:,.2f}: "
            f"open {_pct(facts['gap'])}, low {_pct(facts['low_ret'])}, "
            f"high {_pct(facts['high_ret'])}, close {_pct(facts['close_ret'])}.")


def _numbered(items, start=1):
    return [_line(i, d) for i, d in enumerate(items, start)]


def _move_text(move: dict, primary: str) -> str:
    verb = "fell" if move["ret"] < 0 else "rose"
    return (f"{primary} {verb} {abs(move['ret']) * 100:.1f}% on {move['date']} "
            f"(market-adjusted {_pct(move['abnormal'])} vs {move['benchmark']} "
            f"{_pct(move['bench_ret'])}, z = {move['z']:+.1f}).")


def _premise_note(state: LOOKUPState, move: dict) -> str:
    asked = query_direction(state.get("query", ""))
    if asked and move.get("direction") and asked != move["direction"]:
        actual = "fell" if move["direction"] == "down" else "rose"
        return f"Note: your question assumes the stock {'fell' if asked == 'down' else 'rose'}, but it {actual}."
    return ""


def peer_line(move: dict, peers: list) -> str:
    """Price-only sector context: did peers move the same way that session?"""
    if not peers:
        return ""
    sign = 1 if move["abnormal"] > 0 else -1
    same = [p for p in peers if p["abnormal"] * sign > 0 and abs(p["z"]) >= SAME_DIR_Z]
    detail = ", ".join(f"{p['ticker'].split('.')[0]} {_pct(p['abnormal'])}" for p in peers)
    if len(same) >= math.ceil(len(peers) / 2):
        read = "which suggests a sector-wide move"
    elif not same:
        read = "which suggests a move specific to this stock"
    else:
        read = "a mixed picture"
    return (f"Peers the same day (market-adjusted): {detail}. {len(same)} of {len(peers)} moved "
            f"the same way with |z| >= {SAME_DIR_Z}, {read}. (Price data only: this says "
            "how widespread the move was, not why it happened.)")


def _searched(diag: dict) -> str:
    rows = []
    for t in (diag or {}).get("tiers", []):
        label = t["tier"].replace("google_news", "Google News")
        rows.append(f"{label} ({'error ' + t['error'] if t.get('error') else str(t['company_added']) + ' naming the company'})")
    return "Searched: " + "; ".join(rows) + "." if rows else ""


def writer_agent(state: LOOKUPState) -> dict:
    primary = state["tickers"][0]
    move = state.get("move") or {}
    mode = state.get("mode", "unchecked")
    diag = state.get("diagnostics") or {}
    pool = state.get("news_pool") or []
    ranked = state.get("ranked") or []
    pre = [d for d in ranked if d.get("phase", "pre") == "pre"]
    post = [d for d in ranked if d.get("phase") == "post"]
    own_pre = [d for d in pre if d.get("scope") == "company"]
    own_post = [d for d in post if d.get("scope") == "company"]
    searched = _searched(diag)

    # ── asked-for date was not a complete session ────────────────────────────
    if move.get("requested_date") and move.get("status") == "unavailable":
        return {"abstained": True, "abstain_reason": "date_not_available",
                "final_report": f"### No price check for {primary} on {move['requested_date']}\n"
                                f"There is no complete trading session for that date in the price "
                                f"history ({move.get('reason', 'unknown')}). It may be a weekend or "
                                "holiday, too recent, or too old. No explanation is offered."}

    # ── no unusual move: no explanation, coverage as context only ────────────
    if move.get("status") == "no_event":
        recent = move.get("recent") or []
        n = move.get("lookback", len(recent) or 1)
        facts = move.get("facts")
        swing = bool(facts and facts.get("intraday_flag"))
        if move.get("requested_date"):
            text = (f"### No unusual move for {primary} on {move['date']}\n"
                    f"{_pct(move['ret'])}, market-adjusted {_pct(move['abnormal'])} vs "
                    f"{move['benchmark']} {_pct(move['bench_ret'])}, z = {move['z']:+.1f}, below the "
                    f"|z| >= {move['z_threshold']} threshold used to flag events.")
        else:
            text = (f"### No unusual move for {primary}\n"
                    f"None of the last {n} complete sessions had a market-adjusted move of "
                    f"|z| >= {move['z_threshold']} (the threshold used to flag events). "
                    f"Latest ({move['date']}): {_pct(move['ret'])}, market-adjusted "
                    f"{_pct(move['abnormal'])} vs {move['benchmark']} {_pct(move['bench_ret'])}, "
                    f"z = {move['z']:+.1f}.")
            if recent:
                big = max(recent, key=lambda r: abs(r["z"]))
                text += (f" Largest in the window: {big['date']}, {_pct(big['ret'])} raw, "
                         f"z = {big['z']:+.1f}.")
        fl = facts_line(facts)
        if fl:
            text += f"\n\n{fl}"
        if swing:
            text += (" The session had a large intraday swing that did not hold into the close, "
                     "which a close-to-close check does not capture.")
        else:
            text += " There is no flagged price move to explain."
        note = _premise_note(state, move)
        if note:
            text += f"\n\n{note}"
        if own_pre or own_post:
            text += ("\n\nCoverage of the company is listed below as context only, ranked by "
                     "relevance. It is not a verified explanation of a price move.\n")
            idx = 1
            if own_pre:
                text += "\n**Published before the session opened**\n\n" + "\n".join(_numbered(own_pre, idx))
                idx += len(own_pre)
            if own_post:
                text += ("\n\n**Published after the session opened** - these may report the move "
                         "rather than cause it.\n\n" + "\n".join(_numbered(own_post, idx)))
        return {"abstained": True, "abstain_reason": "intraday_only" if swing else "no_unusual_move",
                "final_report": text}

    # ── header ───────────────────────────────────────────────────────────────
    if mode == "explained":
        head = f"### {_move_text(move, primary)}"
        notes = [n for n in (facts_line(move.get("facts")), _premise_note(state, move),
                             f"Most recent unusual move among the last {move['lookback']} complete sessions."
                             if move.get("lookback", 1) > 1 else "",
                             peer_line(move, state.get("peer_context") or [])) if n]
    else:
        head = f"### Recent headlines for {primary}"
        notes = [f"Price move not verified ({move.get('reason', 'unknown')}); "
                 "these headlines are context only."]
    out = [head, ""]
    for n in notes:
        out += [f"*{n}*", ""]

    # ── nothing at all ───────────────────────────────────────────────────────
    if not pool:
        out += ["No timestamped news for this company or its peers was found, so no explanation "
                f"is offered. {searched}"]
        return {"abstained": True, "abstain_reason": "no_timestamped_news_in_window",
                "final_report": "\n".join(out).strip()}

    # ── no headline names the company (before or after the open) ────────────
    if not own_pre and not own_post:
        out += [f"No headlines naming {primary} were found, so no explanation of its move is "
                f"offered. {searched}", "",
                "Context from peers and general market news (not about this company):"]
        out += _numbered(pre)
        return {"abstained": True, "abstain_reason": "no_news_for_primary_ticker",
                "final_report": "\n".join(out)}

    # ── explanation candidates ──────────────────────────────────────────────
    n_own = diag.get("own_count", len(own_pre))
    out += [f"*Headlines are ranked by cross-encoder relevance from {len(pool)} found "
            f"({n_own} naming {primary} before the open"
            + (f", {diag.get('post_count', len(own_post))} after" if post else "")
            + "). This is a relevance ranking, not a verified explanation of the price move.*", ""]
    idx = 1
    if pre:
        out += ["**Published before the session opened**" if mode == "explained" else "**Headlines**", ""]
        out += _numbered(pre, idx)
        idx += len(pre)
    elif mode == "explained":
        out += [f"No headlines naming {primary} were published before the session opened. "
                f"{searched}"]
    if post:
        out += ["", "**Published after the session opened** - these may report the move rather "
                "than cause it, and are outside the validated pre-open evidence window.", ""]
        out += _numbered(post, idx)
    return {"abstained": False, "abstain_reason": "", "final_report": "\n".join(out)}