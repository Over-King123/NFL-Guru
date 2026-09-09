"""
schedule_transform.py

Shared logic for turning nflverse's raw schedules/games.csv into our three
processed tables (games, lines, situational). Extracted out of
build_features.py so the lightweight daily pipeline (daily_update.py) can
reuse it without pulling in the full play-by-play pipeline.
"""

from pathlib import Path

import pandas as pd


def build_games_and_lines(raw_games_csv: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_games_csv, low_memory=False)

    games = df[[
        "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
        "away_team", "home_team", "away_score", "home_score", "result", "total",
        "overtime", "location", "away_rest", "home_rest", "div_game",
        "roof", "surface", "temp", "wind", "away_qb_name", "home_qb_name",
        "away_coach", "home_coach", "referee", "stadium",
    ]].copy()
    games.rename(columns={"result": "home_margin", "total": "combined_score"}, inplace=True)
    games["home_margin_check"] = games["home_score"] - games["away_score"]

    lines = df[[
        "game_id", "season", "week",
        "spread_line", "home_spread_odds", "away_spread_odds",
        "total_line", "over_odds", "under_odds",
        "home_moneyline", "away_moneyline",
    ]].copy()
    lines.insert(3, "line_snapshot", "closing")
    lines.insert(4, "opening_spread", pd.NA)
    lines.insert(5, "opening_total", pd.NA)

    situational = games[[
        "game_id", "season", "week", "home_team", "away_team",
        "home_rest", "away_rest", "div_game", "roof", "surface", "temp", "wind",
    ]].copy()

    games.to_csv(out_dir / "games.csv", index=False)
    lines.to_csv(out_dir / "lines.csv", index=False)
    situational.to_csv(out_dir / "situational.csv", index=False)
    print(f"games.csv: {len(games):,} rows | lines.csv: {len(lines):,} rows | situational.csv: {len(situational):,} rows")

    return games, lines, situational
