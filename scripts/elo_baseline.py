"""
elo_baseline.py

Baseline #1 from the roadmap: an Elo-style power rating, chosen specifically
because it's the simplest thing that could work, fully transparent, and easy
to audit by hand - which is the point of doing this before anything fancier.

Method (documented in full so every number is traceable, not a black box):

  Rating update, applied game-by-game in true chronological order:
    expected_home_win_prob = 1 / (1 + 10^(-(elo_home + HOME_ADV - elo_away) / 400))
    actual_result           = 1 (home win), 0 (home loss), 0.5 (tie)
    mov_multiplier           = ln(|margin| + 1) * 2.2 / (0.001 * elo_diff_winner + 2.2)
                               (margin-of-victory multiplier, the standard
                               538 NFL Elo adjustment - a 1-point win updates
                               ratings less than a 30-point win, but with
                               diminishing returns so blowouts don't dominate)
    elo_home_new             = elo_home + K * mov_multiplier * (actual_result - expected_home_win_prob)
    elo_away_new             = elo_away - (elo_home_new - elo_home)   [zero-sum]

  K = 20, HOME_ADV = 48 Elo points - both fixed, literature-typical defaults
  for NFL Elo (not fit on our data). They are named constants below;
  revisit them once we have a real basis (e.g. a grid search on
  out-of-sample calibration) for choosing something else - "start simple"
  does not mean "assume the first number is right forever."

  Between seasons, each team's rating is pulled 1/3 of the way back toward
  1500 (roster turnover, coaching changes, draft - a team's true strength
  does not simply carry over 100%). This is the same idea 538 uses.

  Franchise relocations (STL->LA, SD->LAC, OAK->LV) are treated as one
  continuous team - see team_codes.py - so a relocation doesn't reset a
  real team's history to a default rating for no football reason.

  Ratings start in 1999 (as far back as nflverse's schedule file goes) even
  though our EPA-based features only cover 2015+. That gives 16 seasons of
  "burn-in" before our evaluation window starts, which matters: a freshly
  initialized Elo model needs time to converge, and we don't want to fool
  ourselves by evaluating a baseline while it's still warming up.

  Elo-diff -> point-spread conversion: fit by a SEPARATE walk-forward linear
  regression of actual home_margin on pre-game elo_diff (no HOME_ADV baked
  into this one - the regression's own intercept estimates home-field
  advantage in POINTS empirically, which we also cross-check against the
  fixed 48-Elo-point assumption above). The regression for a given season
  is fit using ONLY seasons strictly before it (expanding window) - so, for
  example, 2015's conversion coefficients come from 1999-2014 only.

  Sign convention (verified empirically, not assumed - see
  evaluate_elo.py's sanity check): nflverse's spread_line is POSITIVE when
  the home team is favored (e.g. +27 for a 27-point home favorite) - the
  opposite of typical sportsbook display ("-3.5"). elo_implied_spread is
  built to match that: it IS the predicted home margin, unnegated.

Output: data/processed/elo_ratings.csv (one row per scored game, with
pre-game ratings, implied win probability, implied spread, and the actual
result/market lines alongside for easy comparison) and
data/processed/elo_conversion_by_season.csv (the fitted regression
coefficients per season, for transparency/audit).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from team_codes import normalize_team

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

K = 20.0
HOME_ADV = 48.0
START_RATING = 1500.0
SEASON_REVERSION = 1.0 / 3.0  # fraction pulled back toward 1500 between seasons


def mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    if margin == 0:
        return 1.0
    return np.log(abs(margin) + 1.0) * (2.2 / (0.001 * elo_diff_winner + 2.2))


def run_elo(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    ratings = {}          # normalized_team -> current elo
    last_season_seen = {} # normalized_team -> last season we touched them

    rows = []
    for _, g in games.iterrows():
        home_raw, away_raw = g["home_team"], g["away_team"]
        home, away = normalize_team(home_raw), normalize_team(away_raw)
        season = g["season"]

        for team in (home, away):
            if team not in ratings:
                ratings[team] = START_RATING
                last_season_seen[team] = season
            elif last_season_seen[team] != season:
                # first game of a new season for this team: revert toward mean
                ratings[team] = START_RATING + (1 - SEASON_REVERSION) * (ratings[team] - START_RATING)
                last_season_seen[team] = season

        elo_home_pre = ratings[home]
        elo_away_pre = ratings[away]
        elo_diff_raw = elo_home_pre - elo_away_pre  # no home_adv - used for the spread regression
        elo_diff_for_update = elo_home_pre + HOME_ADV - elo_away_pre

        expected_home = 1.0 / (1.0 + 10 ** (-elo_diff_for_update / 400.0))

        home_score, away_score = g["home_score"], g["away_score"]
        margin = home_score - away_score
        if margin > 0:
            actual_home = 1.0
            elo_diff_winner = elo_diff_for_update
        elif margin < 0:
            actual_home = 0.0
            elo_diff_winner = -elo_diff_for_update
        else:
            actual_home = 0.5
            elo_diff_winner = 0.0

        mult = mov_multiplier(margin, elo_diff_winner)
        delta = K * mult * (actual_home - expected_home)

        rows.append({
            "game_id": g["game_id"], "season": season, "week": g["week"], "gameday": g["gameday"],
            "home_team_raw": home_raw, "away_team_raw": away_raw,
            "home_team": home, "away_team": away,
            "elo_home_pre": elo_home_pre, "elo_away_pre": elo_away_pre,
            "elo_diff_raw": elo_diff_raw,
            "elo_implied_home_win_prob": expected_home,
            "home_score": home_score, "away_score": away_score, "home_margin": margin,
        })

        ratings[home] = elo_home_pre + delta
        ratings[away] = elo_away_pre - delta

    return pd.DataFrame(rows), ratings, last_season_seen


def fit_walk_forward_conversion(elo_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each season, fit home_margin ~ a + b*elo_diff_raw using only
    strictly earlier seasons (expanding window). Adds elo_implied_spread and
    elo_implied_home_margin columns; also returns the per-season coefficients
    for transparency."""
    seasons = sorted(elo_df["season"].unique())
    coef_rows = []
    elo_df = elo_df.copy()
    elo_df["elo_implied_home_margin"] = np.nan
    elo_df["elo_implied_spread"] = np.nan

    for season in seasons:
        train = elo_df[elo_df["season"] < season]
        test_idx = elo_df["season"] == season
        if len(train) < 100:  # not enough history yet to fit anything meaningful
            continue
        x = train["elo_diff_raw"].to_numpy()
        y = train["home_margin"].to_numpy()
        b, a = np.polyfit(x, y, 1)  # y = a + b*x  (polyfit returns [b, a] for deg=1)

        pred_margin = a + b * elo_df.loc[test_idx, "elo_diff_raw"]
        elo_df.loc[test_idx, "elo_implied_home_margin"] = pred_margin
        # IMPORTANT, verified empirically against nflverse's games.csv (see
        # evaluate_elo.py's sign-convention check): in THIS dataset,
        # spread_line is POSITIVE when the home team is favored (e.g.
        # spread_line=+27 for a 27-point home favorite) - the opposite of
        # the usual sportsbook-display convention ("-3.5" for a favorite).
        # elo_implied_spread must match that convention to be comparable to
        # spread_line at all, so it is the predicted home margin AS-IS, not
        # negated.
        elo_df.loc[test_idx, "elo_implied_spread"] = pred_margin

        home_adv_points = a
        home_adv_in_elo_points = a / b if b != 0 else np.nan
        coef_rows.append({
            "season": season, "n_train_games": len(train),
            "intercept_a_points": a, "slope_b_points_per_elo": b,
            "home_adv_points_empirical": home_adv_points,
            "home_adv_elo_points_equiv": home_adv_in_elo_points,
        })

    return elo_df, pd.DataFrame(coef_rows)


def main():
    games = pd.read_csv(PROCESSED / "games.csv")
    scored = games[games["home_score"].notna() & games["away_score"].notna()].copy()
    print(f"Running Elo over {len(scored):,} scored games, seasons {scored['season'].min()}-{scored['season'].max()}...")

    elo_df, final_ratings, final_last_season_seen = run_elo(scored)
    elo_df, coef_df = fit_walk_forward_conversion(elo_df)

    # attach market lines for easy downstream comparison
    lines = pd.read_csv(PROCESSED / "lines.csv")[[
        "game_id", "spread_line", "home_spread_odds", "away_spread_odds",
        "total_line", "over_odds", "under_odds", "home_moneyline", "away_moneyline",
    ]]
    elo_df = elo_df.merge(lines, on="game_id", how="left")

    elo_df.to_csv(PROCESSED / "elo_ratings.csv", index=False)
    coef_df.to_csv(PROCESSED / "elo_conversion_by_season.csv", index=False)

    print(f"elo_ratings.csv: {len(elo_df):,} rows")
    print(f"elo_conversion_by_season.csv: {len(coef_df)} seasons with a fitted conversion")
    print("\nFitted conversion coefficients by season (last 12):")
    print(coef_df.tail(12).to_string(index=False))

    # quick final-rating leaderboard as a sanity check a human can eyeball
    final_ratings = (
        elo_df.sort_values(["home_team", "gameday"]).groupby("home_team").tail(1)
    )
    print("\n(sanity check only, not a formal output) most recent home Elo snapshot per team, sample:")
    print(final_ratings[["home_team", "season", "week", "elo_home_pre"]].sort_values("elo_home_pre", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
