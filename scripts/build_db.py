"""
build_db.py

Loads the processed CSVs into a single SQLite file so the data structure is
queryable and auditable in one place (per the roadmap's "design the
database" step), without requiring any extra dependencies - sqlite3 is
Python's standard library.

Tables created:
  games, lines, situational, team_game_raw, team_game_features, injuries,
  snap_counts (season-level raw injury/snap tables are loaded as-is, one
  table each per source year concatenated together)
"""

import glob
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DB_PATH = ROOT / "db" / "nfl_model.db"


def load_csv_table(conn, csv_path: Path, table_name: str, **read_kwargs):
    df = pd.read_csv(csv_path, **read_kwargs)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"  {table_name}: {len(df):,} rows, {len(df.columns)} columns")


def load_multi_season_csv(conn, pattern: str, table_name: str):
    frames = []
    for f in sorted(glob.glob(pattern)):
        frames.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"  {table_name}: {len(df):,} rows, {len(df.columns)} columns (from {len(frames)} season files)")


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)

    print("Loading processed tables...")
    load_csv_table(conn, PROCESSED / "games.csv", "games")
    load_csv_table(conn, PROCESSED / "lines.csv", "lines")
    load_csv_table(conn, PROCESSED / "situational.csv", "situational")
    load_csv_table(conn, PROCESSED / "team_game_raw.csv", "team_game_raw")
    load_csv_table(conn, PROCESSED / "team_game_features.csv", "team_game_features")
    if (PROCESSED / "elo_ratings.csv").exists():
        load_csv_table(conn, PROCESSED / "elo_ratings.csv", "elo_ratings")
        load_csv_table(conn, PROCESSED / "elo_conversion_by_season.csv", "elo_conversion_by_season")
    if (PROCESSED / "elo_evaluation_by_season.csv").exists():
        load_csv_table(conn, PROCESSED / "elo_evaluation_by_season.csv", "elo_evaluation_by_season")

    print("Loading raw injuries/snap_counts (per-season files, concatenated)...")
    load_multi_season_csv(conn, str(RAW / "injuries" / "injuries_*.csv"), "injuries")
    load_multi_season_csv(conn, str(RAW / "snap_counts" / "snap_counts_*.csv"), "snap_counts")

    print("Indexing on game_id / team / season+week for query speed...")
    idx_stmts = [
        "CREATE INDEX IF NOT EXISTS idx_games_id ON games(game_id)",
        "CREATE INDEX IF NOT EXISTS idx_lines_id ON lines(game_id)",
        "CREATE INDEX IF NOT EXISTS idx_tgf_game ON team_game_features(game_id)",
        "CREATE INDEX IF NOT EXISTS idx_tgf_team_season_week ON team_game_features(team, season, week)",
        "CREATE INDEX IF NOT EXISTS idx_tgr_game ON team_game_raw(game_id)",
    ]
    for stmt in idx_stmts:
        conn.execute(stmt)
    conn.commit()
    conn.close()
    print(f"\nSQLite DB written to {DB_PATH} ({DB_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
