"""Player-name normalization.

Name joining is the tax on this whole project. nflverse keys on `gsis_id`, ESPN uses
its own IDs and display names, and the two disagree constantly: "D.J. Moore" against
"DJ Moore", "Marquise Brown" against "Hollywood Brown", suffixes present on one side
and absent on the other. A silent join failure is the worst outcome available — the
board still builds, it just quietly omits the one player that mattered.

So this module exists to make matching deterministic, and every caller that uses it is
expected to report what failed to match rather than dropping it.

From Phase 2 onward the primary join is on IDs via the ffverse crosswalk
(`patron.data.nflverse.load_id_crosswalk`), and this is the fallback. It is the
primary path only for the manual override file, where entries are written by hand.
"""

from __future__ import annotations

import re
import unicodedata

#: Generational suffixes, which appear inconsistently across sources.
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

#: Punctuation that varies by source: "A.J." vs "AJ", "Ja'Marr" vs "JaMarr",
#: "Amon-Ra" vs "Amon Ra".
_PUNCTUATION = re.compile(r"[.'’`\-]")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9 ]")
_WHITESPACE = re.compile(r"\s+")


def normalize(name: str) -> str:
    """Reduce a display name to a comparable key.

    Lowercases, strips accents and punctuation, and drops generational suffixes.

    >>> normalize("D.J. Moore")
    'dj moore'
    >>> normalize("Amon-Ra St. Brown")
    'amonra st brown'
    >>> normalize("Harold Fannin Jr.")
    'harold fannin'
    """
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")

    # Punctuation is removed rather than replaced with a space, so "A.J." collapses to
    # "aj" the way the sources that omit the periods already write it.
    collapsed = _PUNCTUATION.sub("", ascii_only.lower())
    cleaned = _NON_ALPHANUMERIC.sub(" ", collapsed)

    parts = [part for part in _WHITESPACE.split(cleaned) if part]
    while len(parts) > 1 and parts[-1] in _SUFFIXES:
        parts.pop()

    return " ".join(parts)


def match_key(name: str, position: str) -> tuple[str, str]:
    """A normalized name paired with position.

    Position disambiguates the genuine name collisions in the league — there are two
    Michael Carters and two Josh Allens — cheaply and without a fuzzy match.
    """
    return normalize(name), position.upper()
