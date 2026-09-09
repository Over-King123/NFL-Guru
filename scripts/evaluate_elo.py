"""
evaluate_elo.py

Honest evaluation of the Elo baseline - NOT a search for an edge. The only
questions this script answers:
  1. Does Elo beat trivial baselines (coin flip / home team always wins)?
  2. How does Elo's calibration compare to the market's own (vig-removed)
     implied probability? (Expectation: market wins. If Elo wins, that is
     a reason to be suspicious of a bug before it's a reason to celebrate.)
  3. Does Elo's implied spread actually track the market's closing spread,
     directionally and in magnitude? (A sign/convention bug here would
     silently invalidate everything downstream, so this is checked first
     and explicitly, not assumed.)

Evaluation window: seasons 2015-2025 (matches the EPA feature pipeline's
coverage), even though Elo itself was run from 1999 for a longer burn-in.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
EVAL_SEASONS = range(2015, 2026)


def american_to_implied_prob(ml: pd.Series) -> pd.Series:
    return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))


def brier(prob, actual):
    return np.mean((prob - actual) ** 2)


def log_loss(prob, actual, eps=1e-6):
    p = np.clip(prob, eps, 1 - eps)
    return -np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p))


def main():
    df = pd.read_csv(PROCESSED / "elo_ratings.csv")
    df = df[df["season"].isin(EVAL_SEASONS)].copy()
    df = df[df["elo_implied_spread"].notna()].copy()  # need a fitted conversion available
    print(f"Evaluating {len(df):,} games, seasons {df['season'].min()}-{df['season'].max()}\n")

    # ---- 0. sign-convention sanity check, done first and explicitly -----
    # Verified by hand against extreme games (e.g. 2013 DEN (home) vs JAX:
    # spread_line=+27, DEN won by 16 - a big positive spread_line here means
    # the HOME team is the big favorite). So in this dataset, positive
    # spread_line = home favored - the opposite of "-3.5" sportsbook
    # display convention. elo_implied_spread is built to match this.
    corr_check = df[["spread_line", "home_margin"]].corr().iloc[0, 1]
    print(f"[sanity check] corr(market spread_line, actual home_margin) = {corr_check:.3f} "
          f"(expected clearly POSITIVE in this dataset: positive spread_line = home favored = bigger home_margin)")
    if corr_check < 0:
        print("  !! UNEXPECTED SIGN - the spread_line convention assumption is likely wrong. Stop and check before trusting anything below.")
    print()

    # ---- 1. spread comparison: does Elo's cheap spread track the market? -
    spread_corr = df[["elo_implied_spread", "spread_line"]].corr().iloc[0, 1]
    spread_mae = (df["elo_implied_spread"] - df["spread_line"]).abs().mean()
    print(f"Elo implied spread vs market closing spread_line:")
    print(f"  correlation: {spread_corr:.3f}  (both use the same +=home-favored convention here, so positive & high = Elo is 'in the neighborhood' of the market, as expected for a simple model)")
    print(f"  mean absolute difference: {spread_mae:.2f} points\n")

    # ---- 2. straight-up accuracy: Elo vs naive baselines vs market favorite
    su = df[df["home_margin"] != 0].copy()  # drop ties for SU accuracy
    su["home_won"] = (su["home_margin"] > 0).astype(int)
    su["elo_pick_home"] = (su["elo_implied_home_win_prob"] > 0.5).astype(int)

    home_ml_prob = american_to_implied_prob(su["home_moneyline"])
    away_ml_prob = american_to_implied_prob(su["away_moneyline"])
    su["market_favors_home"] = (home_ml_prob > away_ml_prob).astype(int)
    su["market_prob_home_devigged"] = home_ml_prob / (home_ml_prob + away_ml_prob)

    elo_acc = (su["elo_pick_home"] == su["home_won"]).mean()
    home_always_acc = su["home_won"].mean()  # accuracy of "always pick home team"
    market_fav_acc = (su["market_favors_home"] == su["home_won"]).mean()

    print(f"Straight-up accuracy ({len(su):,} decided games, ties excluded):")
    print(f"  coin flip (reference)              : 0.500")
    print(f"  always pick home team               : {home_always_acc:.3f}")
    print(f"  Elo                                 : {elo_acc:.3f}")
    print(f"  market moneyline favorite           : {market_fav_acc:.3f}")
    print()

    # ---- 3. calibration: Brier score & log loss, Elo vs market -----------
    elo_brier = brier(su["elo_implied_home_win_prob"], su["home_won"])
    mkt_brier = brier(su["market_prob_home_devigged"], su["home_won"])
    elo_ll = log_loss(su["elo_implied_home_win_prob"], su["home_won"])
    mkt_ll = log_loss(su["market_prob_home_devigged"], su["home_won"])
    print("Calibration (lower is better for both metrics):")
    print(f"  Brier score   - Elo: {elo_brier:.4f}   Market (vig-removed): {mkt_brier:.4f}")
    print(f"  Log loss      - Elo: {elo_ll:.4f}   Market (vig-removed): {mkt_ll:.4f}")
    if elo_brier < mkt_brier or elo_ll < mkt_ll:
        print("  !! Elo beat the market on a calibration metric. For a same-day, no-injury-data, "
              "no-weather Elo number, that is a reason to look for a bug (e.g. leakage) before "
              "it's a reason to celebrate.")
    print()

    # ---- 4. per-season breakdown ------------------------------------------
    print("Per-season SU accuracy (Elo vs home-always vs market favorite) and Brier scores:")
    season_rows = []
    for season, g in su.groupby("season"):
        season_rows.append({
            "season": season,
            "n_games": len(g),
            "elo_su_acc": (g["elo_pick_home"] == g["home_won"]).mean(),
            "home_always_acc": g["home_won"].mean(),
            "market_fav_acc": (g["market_favors_home"] == g["home_won"]).mean(),
            "elo_brier": brier(g["elo_implied_home_win_prob"], g["home_won"]),
            "market_brier": brier(g["market_prob_home_devigged"], g["home_won"]),
            "spread_corr": g[["elo_implied_spread", "spread_line"]].corr().iloc[0, 1],
            "spread_mae": (g["elo_implied_spread"] - g["spread_line"]).abs().mean(),
        })
    season_df = pd.DataFrame(season_rows)
    pd.set_option("display.width", 140)
    print(season_df.round(3).to_string(index=False))

    season_df.to_csv(PROCESSED / "elo_evaluation_by_season.csv", index=False)

    # ---- 5. exploratory only: what if we "bet" every time Elo disagrees --
    # with the market by some margin? Reported for curiosity, NOT as a
    # validated edge - no bet sizing, no CLV, no threshold-tuning, and no
    # holdout discipline has been applied here. That is the actual next
    # phase of the roadmap (Betting Logic + Backtest Engine), not this step.
    disagreement = df.copy()
    disagreement["disagreement_points"] = disagreement["elo_implied_spread"] - disagreement["spread_line"]
    big_disagree = disagreement[disagreement["disagreement_points"].abs() >= 3]
    print(f"\n(exploratory, NOT a validated signal) games where Elo's spread differs from the "
          f"market's by >= 3 points: {len(big_disagree):,} of {len(disagreement):,} "
          f"({len(big_disagree)/len(disagreement):.1%}). This is expected model noise, not "
          f"evidence of an edge - real edge evaluation happens in the backtest engine, with a "
          f"pre-committed threshold and full CLV tracking, per the roadmap.")


if __name__ == "__main__":
    main()
