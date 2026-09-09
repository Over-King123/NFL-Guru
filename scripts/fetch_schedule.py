"""
fetch_schedule.py

Lightweight fetch of ONLY the schedules/games file (~2MB) - unlike
fetch_data.py's full historical pull, this is meant to run fast, every day,
as part of daily_update.py. It's the same nflverse source, just without
downloading 11 seasons of play-by-play every time.
"""

import subprocess
import sys
from pathlib import Path

URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


def main(dest_dir: str = "data/raw/schedules"):
    dest = Path(dest_dir) / "games.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-sS", "-L", "--max-time", "60", "-w", "%{http_code}", "-o", str(dest), URL],
        capture_output=True, text=True,
    )
    code = result.stdout.strip()
    if code != "200" or not dest.exists() or dest.stat().st_size == 0:
        print(f"FAILED to fetch games.csv (http {code}): {result.stderr[:300]}")
        sys.exit(1)
    print(f"games.csv: {dest.stat().st_size:,} bytes -> {dest}")


if __name__ == "__main__":
    main()
