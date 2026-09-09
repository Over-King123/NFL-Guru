# NFL Guru — Data-Driven NFL Point-Spread Model

A serious, leak-safe NFL point-spread model, built up in stages: data pipeline → Elo baseline →
(next) logistic regression → backtest engine → betting logic. See `nfl_model_v1.xlsx` (generated
by this repo, delivered separately) for the human-facing audit trail — README tab there explains
the sheets, the two real bugs that were found and fixed along the way, and exactly what this
project does and does not claim yet (no ROI, no bet sizing, no CLV, no betting threshold — that's
all still ahead).

## What's here

- `team_codes.py` — franchise-relocation continuity (STL→LA, SD→LAC, OAK→LV), used everywhere a
  team code is a modeling key.
- `schedule_transform.py` / `fetch_schedule.py` — turns nflverse's raw schedule file into our
  `games` / `lines` / `situational` tables. Lightweight (~2MB), safe to re-run daily.
- `fetch_data.py` — the FULL historical pull (11 seasons of play-by-play, ~200MB). One-time /
  occasional use, not part of the daily job.
- `build_features.py` — the point-in-time, leak-safe EPA/success-rate feature pipeline from
  play-by-play. Not needed for Elo; feeds the next model (logistic regression).
- `build_db.py` — loads everything into a single SQLite file for ad-hoc querying.
- `validate_leakage.py` — independent recomputation that audits the feature pipeline for
  look-ahead leakage. Run this after touching `build_features.py`.
- `elo_baseline.py` — the Elo model itself: MOV-adjusted updates, home-field edge, season
  reversion, walk-forward margin→spread conversion. Fully documented in its own docstring,
  including the spread_line sign-convention gotcha (nflverse: positive = home favored).
- `evaluate_elo.py` — honest evaluation against naive baselines and the market (not a search for
  an edge — see its docstring).
- `predict_current_week.py` — Elo's implied spread for the next unplayed week on the schedule.
- `fetch_odds.py` — live current-week spreads from The Odds API (needs `ODDS_API_KEY`; degrades
  gracefully to Elo-only output if unset or the call fails).
- `build_workbook.py` — generates `nfl_model_v1.xlsx` (README / Season Summary / Games audit
  trail / Current Week / Team Elo Trajectory tabs) to `/mnt/user-data/outputs/`.
- `daily_update.py` — the single entry point for the daily job: schedule → Elo → odds → current
  week → workbook. Does NOT touch the play-by-play archive, so it runs in well under a minute even
  starting from nothing.

## Running the daily update

```bash
export ODDS_API_KEY=...   # optional - omit to get Elo-only current-week predictions
python3 scripts/daily_update.py
```

Then run the xlsx skill's `recalc.py` on the output workbook (formulas are written but not
pre-calculated by openpyxl) before delivering it — see the calling session's instructions.

## Requirements

`pandas` and `openpyxl` (both preinstalled in the Cowork/Claude Code sandbox this is designed to
run in — see `requirements.txt` if running elsewhere).
