"""
build_features.py

Builds the point-in-time, leak-safe feature pipeline described in the
architecture: raw play-by-play -> per-team-per-game stat lines -> rolling
"as-of-kickoff" features that only ever look backward in time.

Output tables (written to data/processed/):
  - games.csv             one row per game: ids, season/week, teams, scores,
                           margin, rest days, divisional flag, weather/roof
  - lines.csv             one row per game: opening is NOT available from this
                           free source (nflverse's games.csv carries the
                           closing line only) - flagged clearly so it's never
                           confused with a true opening line. Includes
                           per-side spread/total odds (for real vig removal,
                           not just an assumed -110/-110).
  - team_game_raw.csv     one row per team per game: that game's own EPA/
                           success-rate/efficiency numbers (NOT point-in-time
                           - this is what happened IN that game)
  - team_game_features.csv   the point-in-time version of the above: for each
                           team-game, rolling stats computed ONLY from that
                           team's games strictly BEFORE this one. This is the
                           table a model is allowed to train/predict on.
  - situational.csv       rest days, home/away, divisional, travel note

Point-in-time method (two parallel windows, both shifted by 1 game so the
current game's own outcome is never included):
  - season_to_date: expanding mean within the current season only
  - last8: rolling mean over the trailing 8 games, crossing season
    boundaries (continuity for early-season weeks)
A team's very first game in the dataset (2015 week 1, no prior season on
file) has no history and is left as NaN on purpose - that's a real "we don't
know yet" state, not something to silently paper over.
"""

import gzip
import glob
import sqlite3
from pathlib import Path

import pandas as pd

from team_codes import normalize_team
from schedule_transform import build_games_and_lines as _build_games_and_lines

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

PBP_COLS = [
    "game_id", "season", "week", "game_date", "posteam", "defteam",
    "home_team", "away_team", "play_type", "pass", "rush", "epa", "success",
    "yards_gained", "down", "ydstogo", "yardline_100", "qb_dropback",
    "sack", "interception", "fumble_lost", "touchdown", "qb_hit",
    "posteam_score", "defteam_score",
]


# ---------------------------------------------------------------------------
# 1. games + lines tables (from schedules/games.csv)
# ---------------------------------------------------------------------------

def build_games_and_lines():
    games, lines, situational = _build_games_and_lines(RAW / "schedules" / "games.csv", OUT)
    return games


# ---------------------------------------------------------------------------
# 2. team_game_raw (per-team-per-game stat lines from play-by-play)
# ---------------------------------------------------------------------------

def load_pbp_all_seasons():
    frames = []
    for f in sorted(glob.glob(str(RAW / "pbp" / "play_by_play_*.csv.gz"))):
        with gzip.open(f, "rt", newline="") as fh:
            df = pd.read_csv(fh, usecols=lambda c: c in PBP_COLS, low_memory=False)
        frames.append(df)
        print(f"  loaded {Path(f).name}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)


def build_team_game_raw(pbp: pd.DataFrame) -> pd.DataFrame:
    # scrimmage plays only (pass or run), with a valid EPA value
    scrim = pbp[(pbp["epa"].notna()) & ((pbp["pass"] == 1) | (pbp["rush"] == 1))].copy()

    scrim["is_explosive"] = (
        ((scrim["pass"] == 1) & (scrim["yards_gained"] >= 20))
        | ((scrim["rush"] == 1) & (scrim["yards_gained"] >= 10))
    ).astype(int)
    scrim["is_third_down_conv"] = (
        (scrim["down"] == 3) & (scrim["yards_gained"] >= scrim["ydstogo"])
    ).astype(int)
    scrim["is_third_down"] = (scrim["down"] == 3).astype(int)
    scrim["is_red_zone_play"] = (scrim["yardline_100"] <= 20).astype(int)
    scrim["is_red_zone_td"] = (scrim["is_red_zone_play"] == 1) & (scrim["touchdown"] == 1)
    scrim["is_turnover"] = (
        (scrim["interception"] == 1) | (scrim["fumble_lost"] == 1)
    ).astype(int)
    # sacks/qb hits are only meaningful relative to dropbacks
    scrim["is_dropback"] = (scrim["qb_dropback"] == 1).astype(int)

    def side_agg(side_col: str, prefix: str) -> pd.DataFrame:
        g = scrim.groupby(["game_id", "season", "week", side_col], dropna=True)
        out = g.agg(
            n_plays=("epa", "size"),
            epa_per_play=("epa", "mean"),
            success_rate=("success", "mean"),
            yards_per_play=("yards_gained", "mean"),
            explosive_rate=("is_explosive", "mean"),
            turnover_rate=("is_turnover", "mean"),
            red_zone_td_rate=("is_red_zone_td", "mean"),
        ).reset_index()

        # third-down conversion needs its own denominator (plays on 3rd down)
        td = g.apply(
            lambda x: x.loc[x["is_third_down"] == 1, "is_third_down_conv"].mean()
        ).rename("third_down_conv_rate").reset_index()
        out = out.merge(td, on=["game_id", "season", "week", side_col], how="left")

        # pass/rush split EPA
        pass_epa = scrim[scrim["pass"] == 1].groupby(["game_id", side_col])["epa"].mean().rename("pass_epa_per_play")
        rush_epa = scrim[scrim["rush"] == 1].groupby(["game_id", side_col])["epa"].mean().rename("rush_epa_per_play")
        out = out.merge(pass_epa.reset_index(), on=["game_id", side_col], how="left")
        out = out.merge(rush_epa.reset_index(), on=["game_id", side_col], how="left")

        # sack rate & pressure (qb_hit) rate per dropback
        db = scrim[scrim["is_dropback"] == 1]
        sack_rate = db.groupby(["game_id", side_col])["sack"].mean().rename("sack_rate_per_dropback")
        qb_hit_rate = db.groupby(["game_id", side_col])["qb_hit"].mean().rename("qb_hit_rate_per_dropback")
        out = out.merge(sack_rate.reset_index(), on=["game_id", side_col], how="left")
        out = out.merge(qb_hit_rate.reset_index(), on=["game_id", side_col], how="left")

        out.rename(columns={side_col: "team"}, inplace=True)
        # Normalize relocated-franchise codes (STL->LA, SD->LAC, OAK->LV) so
        # a team's rolling history stays continuous across a relocation
        # instead of silently resetting - see team_codes.py.
        out["team"] = out["team"].map(normalize_team)
        out.columns = [c if c in ("game_id", "season", "week", "team") else f"{prefix}_{c}" for c in out.columns]
        return out

    off = side_agg("posteam", "off")
    deff = side_agg("defteam", "def")

    merged = off.merge(deff, on=["game_id", "season", "week", "team"], how="outer")
    return merged


# ---------------------------------------------------------------------------
# 3. point-in-time rolling features
# ---------------------------------------------------------------------------

STAT_COLS_TEMPLATE = [
    "n_plays", "epa_per_play", "success_rate", "yards_per_play",
    "explosive_rate", "turnover_rate", "red_zone_td_rate",
    "third_down_conv_rate", "pass_epa_per_play", "rush_epa_per_play",
    "sack_rate_per_dropback", "qb_hit_rate_per_dropback",
]


def build_point_in_time(team_game_raw: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    # need a real chronological order per team, including across seasons
    date_map = games.set_index("game_id")["gameday"].to_dict()
    tg = team_game_raw.copy()
    tg["gameday"] = tg["game_id"].map(date_map)
    tg = tg.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    stat_cols = [f"off_{c}" for c in STAT_COLS_TEMPLATE] + [f"def_{c}" for c in STAT_COLS_TEMPLATE]
    stat_cols = [c for c in stat_cols if c in tg.columns]

    out_frames = [tg[["game_id", "season", "week", "team", "gameday"]].copy()]

    for col in stat_cols:
        # shift(1) is what makes this point-in-time: the CURRENT game's own
        # stat line is excluded, only strictly earlier games contribute.
        #
        # season_to_date: shift WITHIN (team, season) so week 1 of a new
        # season has no prior-season game leaking in as a one-game sample -
        # it is genuinely NaN ("no data yet this season"), which is honest,
        # not a gap to paper over.
        within_season_shift = tg.groupby(["team", "season"], sort=False)[col].shift(1)
        season_to_date = within_season_shift.groupby(
            [tg["team"], tg["season"]], sort=False
        ).apply(lambda s: s.expanding(min_periods=1).mean())
        season_to_date = season_to_date.reset_index(level=[0, 1], drop=True).reindex(tg.index)

        # last8: shift across the team's FULL history (crosses season
        # boundaries on purpose) then roll over the trailing 8 games - this
        # is what carries a real, multi-game prior-season signal into early
        # weeks of a new season, instead of a single noisy game.
        all_history_shift = tg.groupby("team", sort=False)[col].shift(1)
        last8 = all_history_shift.groupby(tg["team"], sort=False).apply(
            lambda s: s.rolling(window=8, min_periods=1).mean()
        )
        last8 = last8.reset_index(level=0, drop=True).reindex(tg.index)

        out_frames.append(pd.Series(season_to_date.values, name=f"{col}_std"))
        out_frames.append(pd.Series(last8.values, name=f"{col}_last8"))

    features = pd.concat(out_frames, axis=1)
    return features


def main():
    print("Building games/lines/situational tables...")
    games = build_games_and_lines()

    print("\nLoading play-by-play (all seasons)...")
    pbp = load_pbp_all_seasons()
    print(f"Total pbp rows: {len(pbp):,}")

    print("\nAggregating per-team-per-game raw stats...")
    team_game_raw = build_team_game_raw(pbp)
    team_game_raw.to_csv(OUT / "team_game_raw.csv", index=False)
    print(f"team_game_raw.csv: {len(team_game_raw):,} rows, {len(team_game_raw.columns)} columns")

    print("\nBuilding point-in-time rolling features (this is the leak-safe table)...")
    features = build_point_in_time(team_game_raw, games)
    features.to_csv(OUT / "team_game_features.csv", index=False)
    print(f"team_game_features.csv: {len(features):,} rows, {len(features.columns)} columns")

    print("\nDone.")


if __name__ == "__main__":
    main()
