"""
predict_current_week.py

Uses the exact same Elo machinery as elo_baseline.py (imported, not
duplicated) to answer one forward-looking question: for the NEXT unplayed
week on the schedule, what spread does Elo imply for each game, using every
completed game up through today?

This does NOT produce a market comparison by itself - The Odds API isn't
wired up yet (as of this run). The market_spread / gap columns are left
blank on purpose rather than filled with a guess, so nothing here is
mistaken for a live line. Once live odds are connected, this script's
predictions slot straight into that comparison.

Output: data/processed/current_week_predictions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from elo_baseline import run_elo, fit_walk_forward_conversion, normalize_team, START_RATING, SEASON_REVERSION

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def main():
    games = pd.read_csv(PROCESSED / "games.csv")
    completed = games[games["home_score"].notna() & games["away_score"].notna()].copy()
    upcoming_all = games[games["home_score"].isna()].copy()

    if upcoming_all.empty:
        print("No upcoming (unplayed) games found in games.csv - nothing to predict.")
        return

    # "current week" = the earliest season/week that still has an unplayed game
    upcoming_all = upcoming_all.sort_values(["season", "week", "gameday"])
    current_season = upcoming_all.iloc[0]["season"]
    current_week = upcoming_all.iloc[0]["week"]
    upcoming = upcoming_all[
        (upcoming_all["season"] == current_season) & (upcoming_all["week"] == current_week)
    ].copy()
    print(f"Current week on the schedule: season {current_season}, week {current_week} "
          f"({len(upcoming)} unplayed games)")

    print(f"Running Elo over {len(completed):,} completed games (through {completed['gameday'].max()})...")
    elo_df, ratings, last_season_seen = run_elo(completed)

    # fit the spread conversion using ALL completed history (this IS the
    # walk-forward rule applied to "predict a season we have zero games of
    # yet" - by definition every completed season is strictly in the past)
    x = elo_df["elo_diff_raw"].to_numpy()
    y = elo_df["home_margin"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    print(f"Conversion fit on {len(elo_df):,} completed games: home_margin = {a:.3f} + {b:.5f} * elo_diff")

    rows = []
    for _, g in upcoming.iterrows():
        home_raw, away_raw = g["home_team"], g["away_team"]
        home, away = normalize_team(home_raw), normalize_team(away_raw)

        for team in (home, away):
            if team not in ratings:
                ratings[team] = START_RATING  # a team we've never seen play a completed game
            elif last_season_seen.get(team) != current_season:
                # this team's rating is still "as of last season" - apply the
                # same season-transition reversion used mid-run, so a
                # not-yet-played-this-season team isn't given a stale edge
                ratings[team] = START_RATING + (1 - SEASON_REVERSION) * (ratings[team] - START_RATING)
                last_season_seen[team] = current_season

        elo_home = ratings[home]
        elo_away = ratings[away]
        elo_diff_raw = elo_home - elo_away
        elo_diff_for_prob = elo_home + 48.0 - elo_away  # HOME_ADV, matching elo_baseline.py
        implied_home_win_prob = 1.0 / (1.0 + 10 ** (-elo_diff_for_prob / 400.0))
        implied_home_margin = a + b * elo_diff_raw

        rows.append({
            "game_id": g["game_id"], "season": current_season, "week": current_week,
            "gameday": g["gameday"], "home_team_raw": home_raw, "away_team_raw": away_raw,
            "home_team": home, "away_team": away,
            "elo_home_rating": round(elo_home, 1), "elo_away_rating": round(elo_away, 1),
            "elo_implied_home_win_prob": round(float(implied_home_win_prob), 4),
            "elo_implied_spread": round(float(implied_home_margin), 2),
            "market_spread": np.nan,   # left blank - no live odds source connected yet
            "spread_gap": np.nan,
            "odds_source": "not connected",
        })

    out = pd.DataFrame(rows)

    odds_path = PROCESSED / "current_odds.csv"
    if odds_path.exists():
        odds = pd.read_csv(odds_path)
        if len(odds) > 0:
            out = out.drop(columns=["market_spread", "spread_gap", "odds_source"])
            out = out.merge(
                odds[["home_team", "away_team", "market_spread_home", "n_bookmakers"]],
                on=["home_team", "away_team"], how="left",
            )
            out = out.rename(columns={"market_spread_home": "market_spread"})
            out["spread_gap"] = out["elo_implied_spread"] - out["market_spread"]
            out["odds_source"] = np.where(
                out["market_spread"].notna(),
                "The Odds API (" + out["n_bookmakers"].astype("Int64").astype(str) + " books avg)",
                "not matched - see fetch_odds.py TEAM_NAME_TO_CODE",
            )
            out = out.drop(columns=["n_bookmakers"])
            print(f"Merged live odds: {out['market_spread'].notna().sum()}/{len(out)} games matched")
        else:
            print("current_odds.csv was empty (ODDS_API_KEY likely unset) - market_spread stays blank")

    out.to_csv(PROCESSED / "current_week_predictions.csv", index=False)
    print(f"\ncurrent_week_predictions.csv: {len(out)} games")
    print(out[["game_id", "home_team", "away_team", "elo_implied_spread", "market_spread", "spread_gap"]].to_string(index=False))


if __name__ == "__main__":
    main()
