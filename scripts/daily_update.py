"""
daily_update.py

The single entry point the daily scheduled job runs. Deliberately lightweight:
none of this touches the 200MB+ historical play-by-play archive - Elo only
needs game scores (games.csv, ~2MB), so the whole thing runs in well under a
minute even from a completely empty environment.

Steps (each is its own script, run as a subprocess so one failing doesn't
take the whole process down with it - see the try/except around the
non-critical odds fetch):
  1. fetch_schedule.py     - fresh games.csv from nflverse (scores, closing
                              lines - closing lines are irrelevant for
                              *future* games but harmless to have)
  2. schedule_transform.py - (invoked via build_features.py's helper import,
                              done inline here) -> processed games/lines/
                              situational tables
  3. elo_baseline.py       - full walk-forward Elo history, 1999-present
  4. fetch_odds.py         - live current-week spreads from The Odds API
                              (needs ODDS_API_KEY set - skips gracefully if
                              not, current_week stays Elo-only that day)
  5. predict_current_week.py - Elo implied spread for the next unplayed
                              week, merged with live odds if step 4 produced any
  6. build_workbook.py     - regenerates nfl_model_v1.xlsx from scratch

Usage: python3 daily_update.py
Exits non-zero only if a step whose output later steps truly depend on
fails (schedule fetch, Elo, or the workbook build itself).
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def run(script_name: str, critical: bool = True) -> bool:
    print(f"\n{'='*60}\n{script_name}\n{'='*60}")
    result = subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)])
    ok = result.returncode == 0
    if not ok:
        level = "FATAL" if critical else "non-critical, continuing"
        print(f"[{level}] {script_name} exited with code {result.returncode}")
        if critical:
            sys.exit(1)
    return ok


def main():
    run("fetch_schedule.py")

    # lightweight games/lines/situational transform (no pbp needed)
    sys.path.insert(0, str(SCRIPTS_DIR))
    from schedule_transform import build_games_and_lines
    root = SCRIPTS_DIR.parent
    build_games_and_lines(root / "data" / "raw" / "schedules" / "games.csv", root / "data" / "processed")

    run("elo_baseline.py")
    run("fetch_odds.py", critical=False)  # missing/failed key -> Elo-only output, not a hard stop
    run("predict_current_week.py")
    run("build_workbook.py")

    print("\nDaily update complete. nfl_model_v1.xlsx is in /mnt/user-data/outputs/ - "
          "deliver it with SendUserFile in this session.")


if __name__ == "__main__":
    main()
