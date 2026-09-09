"""
fetch_data.py

Downloads historical NFL data from the nflverse-data GitHub release assets
(https://github.com/nflverse/nflverse-data). This is the free, open-source
data source recommended in the architecture discussion: play-by-play (the
basis for EPA/success-rate/efficiency features), schedules+results (which
also carry historical closing betting lines), player stats, injuries, and
snap counts.

Notes on this environment:
  - pyarrow/fastparquet are NOT installed and PyPI is blocked for this
    session, so we cannot read .parquet files. nflverse publishes plain
    .csv / .csv.gz assets for every release too, so we use those instead.
    (If you later run this on your own machine with pyarrow available,
    switch to the .parquet URLs for smaller/faster downloads.)
  - Downloads are done with curl (subprocess) rather than requests/urllib,
    since curl reliably picks up this environment's HTTPS_PROXY + CA bundle.

Usage:
    python3 fetch_data.py
"""

import subprocess
import sys
from pathlib import Path

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Seasons to pull. 2015-2025 gives us 11 seasons (~3000 games) of point-in-time
# feature history, comfortably inside the "5-10 seasons" target from the plan.
SEASONS = list(range(2015, 2026))


def download(url: str, dest: Path, label: str) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {label} already present ({dest.stat().st_size:,} bytes)")
        return True
    print(f"  [fetch] {label} ...", end=" ", flush=True)
    result = subprocess.run(
        ["curl", "-sS", "-L", "--max-time", "120", "-w", "%{http_code}",
         "-o", str(dest), url],
        capture_output=True, text=True,
    )
    code = result.stdout.strip()
    if code != "200" or not dest.exists() or dest.stat().st_size == 0:
        print(f"FAILED (http {code}) - {result.stderr.strip()[:200]}")
        if dest.exists():
            dest.unlink()
        return False
    print(f"ok ({dest.stat().st_size:,} bytes)")
    return True


def main():
    failures = []

    print(f"Downloading {len(SEASONS)} seasons of play-by-play ({SEASONS[0]}-{SEASONS[-1]})...")
    for season in SEASONS:
        url = f"{BASE}/pbp/play_by_play_{season}.csv.gz"
        dest = RAW_DIR / "pbp" / f"play_by_play_{season}.csv.gz"
        if not download(url, dest, f"pbp {season}"):
            failures.append(url)

    print("\nDownloading schedules/games (includes historical closing lines & scores)...")
    url = f"{BASE}/schedules/games.csv"
    dest = RAW_DIR / "schedules" / "games.csv"
    if not download(url, dest, "schedules/games.csv"):
        failures.append(url)

    print("\nDownloading combined player_stats...")
    url = f"{BASE}/player_stats/player_stats.csv.gz"
    dest = RAW_DIR / "player_stats" / "player_stats.csv.gz"
    if not download(url, dest, "player_stats.csv.gz"):
        failures.append(url)

    print(f"\nDownloading injuries ({SEASONS[0]}-{SEASONS[-1]}, per-season files)...")
    for season in SEASONS:
        url = f"{BASE}/injuries/injuries_{season}.csv"
        dest = RAW_DIR / "injuries" / f"injuries_{season}.csv"
        if not download(url, dest, f"injuries {season}"):
            failures.append(url)

    print(f"\nDownloading snap_counts ({SEASONS[0]}-{SEASONS[-1]}, per-season files)...")
    for season in SEASONS:
        url = f"{BASE}/snap_counts/snap_counts_{season}.csv"
        dest = RAW_DIR / "snap_counts" / f"snap_counts_{season}.csv"
        if not download(url, dest, f"snap_counts {season}"):
            failures.append(url)

    print("\n" + "=" * 60)
    if failures:
        print(f"DONE with {len(failures)} failures:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All downloads completed successfully.")


if __name__ == "__main__":
    main()
