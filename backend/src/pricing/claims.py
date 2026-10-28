"""Check a headline's price claim ("Infosys shares crash 4%") against the session's prices.

Headlines are not facts. A figure can be an intraday snapshot, a different
security, a different date, or simply wrong. This compares the claim with the
numbers of the session being explained (open gap, intraday low/high, close, all
versus the previous close) and says which, if any, it matches.

Only claims that attach directly to the company name are read ("<name> [shares]
<verb> N%"). "net leaps 225%" (profit) and "Dr Reddy's rallies 7%" (another
company) are ignored. Tolerance is the headline's own rounding: half a unit of its last decimal place
("4%" -> 0.55 points, "2.7%" -> 0.10) plus 0.05.
"""
import re
from typing import Any, Dict, Optional

DOWN = ("fall falls fell falling drop drops dropped dip dips dipped slump slumps slumped "
        "tumble tumbles tumbled crash crashes crashed plunge plunges plunged sink sinks sank "
        "decline declines declined slide slides slid lose loses lost slip slips slipped down").split()
UP = ("rise rises rose rising jump jumps jumped surge surges surged soar soars soared gain gains "
      "gained climb climbs climbed rally rallies rallied up advance advances advanced spike "
      "spikes spiked leap leaps leapt").split()

_VERB = "|".join(sorted(DOWN + UP, key=len, reverse=True))
_FILL = r"(?:by |as much as |nearly |over |about |around |almost |more than |up to |to )?"
_SUBJ = r"(?:\s+(?:ltd\.?|limited|shares?|stock|stocks|share price|stock price|scrip))*"


def extract_claim(headline: str, name: str) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    rx = re.compile(rf"{re.escape(name)}{_SUBJ}\W*\b({_VERB})\b\s+{_FILL}(\d+(?:\.\d+)?)\s?%", re.I)
    m = rx.search(headline)
    if not m:
        return None
    direction = "down" if m[1].lower() in DOWN else "up"
    dec = len(m[2].split(".")[1]) if "." in m[2] else 0
    return {"direction": direction, "pct": float(m[2]), "tol": 0.5 * 10 ** -dec + 0.05}


def _pp(x: float) -> str:
    return f"{x:+.1f}%"


def verify(claim: Dict[str, Any], facts: Dict[str, Any]) -> Dict[str, Any]:
    sign = -1.0 if claim["direction"] == "down" else 1.0
    c = sign * claim["pct"]
    tol = claim.get("tol", 0.55)
    close, gap = facts["close_ret"] * 100, facts["gap"] * 100
    extreme = (facts["low_ret"] if sign < 0 else facts["high_ret"]) * 100
    session = (f"open {_pp(gap)}, {'low' if sign < 0 else 'high'} {_pp(extreme)}, "
               f"close {_pp(close)}")
    base = {"claim": _pp(c)}
    for label, val in (("close", close), ("open gap", gap),
                       ("intraday low" if sign < 0 else "intraday high", extreme)):
        if val * sign > 0 and abs(val - c) <= tol:
            return {**base, "verdict": "matches", "detail":
                    f"headline figure {_pp(c)} matches the session's {label} ({session})"}
    if extreme * sign > 0 and abs(c) <= abs(extreme) + tol:
        return {**base, "verdict": "within_range", "detail":
                f"headline figure {_pp(c)} lies inside the session's range, so it is likely an "
                f"intraday snapshot ({session})"}
    if extreme * sign > 0 and abs(c) <= abs(extreme) + 0.5:
        return {**base, "verdict": "slightly_beyond", "detail":
                f"headline figure {_pp(c)} is slightly beyond the recorded session range; data "
                f"sources differ by small amounts ({session})"}
    return {**base, "verdict": "contradicted", "detail":
            f"headline figure {_pp(c)} is NOT supported by the session's prices ({session})"}


def check(headline: str, name: str, facts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not facts:
        return None
    claim = extract_claim(headline, name)
    return verify(claim, facts) if claim else None