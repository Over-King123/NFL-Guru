"""
fetch_odds.py

Pulls current NFL point-spread lines from The Odds API
(https://the-odds-api.com) - this is the LIVE market data source, separate
from nflverse's historical closing lines (which only exist after a game is
over). Needs an API key: set the ODDS_API_KEY environment variable.

NOTE - written before a real API key was available to test against, so the
JSON parsing follows the documented response shape but has not been
exercised against a live response yet. If field names or nesting have
changed, this will need a quick fix on the first real run - check for that
before trusting its output blindly.

Sign convention: The Odds API returns "point" in the standard sportsbook
display convention - NEGATIVE means that side is favored (e.g. -3.5 for a
3.5-point favorite). Our own model (elo_implied_spread, and nflverse's
historical spread_line) use the OPPOSITE convention: POSITIVE means the
home team is favored (verified empirically in elo_baseline.py). This
script converts on the way in, so everything downstream stays in the
"positive = home favored" convention consistently. Do not skip this
conversion or every comparison will be silently backwards.

Cost note: one call to /odds for NFL with regions=us, markets=spreads costs
a handful of credits (markets x regions), not per-game - so calling this
once a day stays comfortably inside the 500-credits/month free tier.

Output: data/processed/current_odds.csv
"""

import os
import sys
import json
import subprocess
from pathlib import Path

import pandas as pd

SPORT = "americanfootball_nfl"
URL_TMPL = (
    "https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    "?apiKey={key}&regions=us&markets=spreads&oddsFormat=american&dateFormat=iso"
)

TEAM_NAME_TO_CODE = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def main(out_dir: str = "data/processed"):
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ODDS_API_KEY is not set - skipping live odds fetch. "
              "current_week_predictions will keep market_spread blank.")
        sys.exit(0)  # not a hard failure - daily_update.py should still proceed with Elo-only output

    url = URL_TMPL.format(sport=SPORT, key=key)
    result = subprocess.run(
        ["curl", "-sS", "--max-time", "30", url], capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        print(f"FAILED to reach The Odds API: {result.stderr[:300]}")
        sys.exit(1)

    try:
        events = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Response was not valid JSON (likely an API error body): {result.stdout[:500]}")
        sys.exit(1)

    if isinstance(events, dict) and events.get("message"):
        # The Odds API returns {"message": "..."} on auth/quota errors
        print(f"The Odds API returned an error: {events['message']}")
        sys.exit(1)

    rows = []
    for event in events:
        home_full = event.get("home_team")
        away_full = event.get("away_team")
        home_code = TEAM_NAME_TO_CODE.get(home_full)
        away_code = TEAM_NAME_TO_CODE.get(away_full)
        if home_code is None or away_code is None:
            print(f"  (skip) unrecognized team name(s): {home_full!r} / {away_full!r} - "
                  f"TEAM_NAME_TO_CODE may need updating")
            continue

        home_points = []
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "spreads":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_full and outcome.get("point") is not None:
                        home_points.append(float(outcome["point"]))

        if not home_points:
            continue

        avg_book_point = sum(home_points) / len(home_points)
        # CONVERT sign convention: The Odds API is negative=favored; we use
        # positive=home-favored (see module docstring) - so negate here.
        market_spread_home = -avg_book_point

        rows.append({
            "home_team": home_code, "away_team": away_code,
            "home_team_full": home_full, "away_team_full": away_full,
            "commence_time": event.get("commence_time"),
            "market_spread_home": round(market_spread_home, 2),
            "n_bookmakers": len(home_points),
        })

    out = pd.DataFrame(rows)
    out_path = Path(out_dir) / "current_odds.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"current_odds.csv: {len(out)} games, avg {out['n_bookmakers'].mean() if len(out) else 0:.1f} books/game")


if __name__ == "__main__":
    main()
