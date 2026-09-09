"""
team_codes.py

Three NFL franchises relocated during our 2015-2025 window and nflverse
correctly uses the city-accurate code for each era:
  - St. Louis Rams (STL, through 2015)      -> Los Angeles Rams (LA, 2016+)
  - San Diego Chargers (SD, through 2016)   -> LA Chargers (LAC, 2017+)
  - Oakland Raiders (OAK, through 2019)     -> Las Vegas Raiders (LV, 2020+)

That's the historically correct label, but it's the wrong join key for a
power-rating model: it's the same roster/front office/coaching continuity,
and treating "LA 2016" as a brand-new team with no history would silently
reset a real team's rating to a default 1500 and throw away legitimate
history for no football reason.

This module provides ONE normalization applied consistently everywhere a
team code is used as a modeling key (Elo, team_game_features, etc.), so a
franchise's history stays continuous across a relocation. The original,
era-accurate city code is never deleted from the raw data - only the
*modeling* key is normalized.
"""

RELOCATION_MAP = {
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
}


def normalize_team(code):
    if code is None:
        return code
    return RELOCATION_MAP.get(code, code)
