"""
build_workbook.py

Builds nfl_model_v1.xlsx - the audit-friendly Excel workbook, per the
original request to keep an Excel-based tracking/model workbook so the
model's math can be checked by hand before anything gets automated further.

Sheets:
  README            - what everything means, and the two gotchas that were
                       actually found and fixed while building this (the
                       franchise-relocation join bug, and the spread_line
                       sign convention)
  Season Summary    - season-by-season accuracy/calibration, computed with
                       LIVE FORMULAS against the Games sheet (not pasted-in
                       numbers) + 2 charts
  Games             - one row per game, 2015-2025: raw data (scores, market
                       lines, Elo output) alongside LIVE FORMULAS for vig
                       removal, pick-correctness, and Brier terms - this is
                       the actual audit trail
  Team Elo Trajectory - illustrative line chart of a few teams' rating
                       history (presented as data, since it's visualizing
                       already-computed model output, not a new calculation)
"""

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT_PATH = Path("/mnt/user-data/outputs/nfl_model_v1.xlsx")

FONT = "Arial"
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(name=FONT, bold=True, color="FFFFFF")
TITLE_FONT = Font(name=FONT, bold=True, size=14)
BODY_FONT = Font(name=FONT, size=10)
NOTE_FONT = Font(name=FONT, size=10, italic=True, color="666666")


def style_header_row(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center")


def autosize(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def build_readme(wb, min_season, max_season, n_games):
    ws = wb.active
    ws.title = "README"
    ws["A1"] = "NFL Betting Model - Step 1 & 2 Workbook"
    ws["A1"].font = TITLE_FONT
    lines = [
        "",
        "This workbook is the audit companion to the Python pipeline (fetch_data.py, build_features.py,",
        "elo_baseline.py, evaluate_elo.py) - everything here should be traceable back to those scripts.",
        "",
        "SHEETS",
        "Season Summary  - season-by-season accuracy & calibration. The numbers here are LIVE FORMULAS",
        "                  (AVERAGEIFS / COUNTIF against the Games sheet), not pasted-in results - change",
        "                  or filter the Games sheet and these recompute.",
        f"Games           - one row per game, {min_season}-{max_season} ({n_games:,} games). Raw scores/market lines/Elo",
        "                  output are data; the vig-removal, pick-correctness, and Brier-score columns are",
        "                  live formulas so you can see exactly how each number is derived.",
        "Team Elo Trajectory - illustrative rating history for a handful of teams (chart only, for a feel",
        "                  of how Elo moves game to game - not a new calculation).",
        "",
        "TWO THINGS THAT WERE WRONG AND GOT FIXED BEFORE THIS WAS BUILT",
        "1. Franchise relocations (STL->LA Rams, SD->LAC Chargers, OAK->LV Raiders) were initially treated",
        "   as brand-new teams with zero history the moment they moved, which would have silently reset a",
        "   real team's rating/features to a default. Fixed in team_codes.py - these are now one continuous",
        "   franchise for modeling purposes; the original city-accurate labels are preserved in games.csv.",
        "2. nflverse's spread_line uses the OPPOSITE sign convention from a sportsbook display: POSITIVE",
        "   means the HOME team is favored (verified against real blowouts, e.g. Denver was a +27 home",
        "   favorite over Jacksonville in 2013 and won by 16) - not '-3.5' like a betting app shows. Every",
        "   spread comparison in this workbook uses that convention. Getting this backwards would have",
        "   silently inverted every result.",
        "",
        "WHAT THIS DOES **NOT** SHOW YOU YET (on purpose)",
        "There is no ROI, no bet sizing, no CLV, and no betting threshold anywhere in this workbook. Elo is",
        "baseline #1 - it exists to prove a simple model beats a coin flip, which it does (see Season",
        "Summary), and to be a floor the next model (logistic regression) has to actually beat. The market's",
        "own closing line still out-predicts Elo on every metric here, which is the expected, honest result -",
        "not a setback. Edge evaluation is a later, separate step (Betting Logic + Backtest Engine) that",
        "needs a pre-committed threshold and a full bet ledger before any number from it should be trusted.",
    ]
    for i, line in enumerate(lines, start=2):
        ws.cell(row=i, column=1, value=line).font = NOTE_FONT if line.strip() and not line.isupper() else BODY_FONT
    ws["A6"].font = Font(name=FONT, bold=True, size=11)
    ws["A16"].font = Font(name=FONT, bold=True, size=11)
    ws["A23"].font = Font(name=FONT, bold=True, size=11)
    autosize(ws, [125])
    return ws


def build_games_sheet(wb, elo_df):
    ws = wb.create_sheet("Games")
    headers = [
        "game_id", "season", "week", "home_team", "away_team", "home_score", "away_score",
        "home_margin", "spread_line", "elo_implied_spread", "spread_diff",
        "elo_implied_home_win_prob", "home_moneyline", "away_moneyline",
        "implied_prob_home", "implied_prob_away", "market_prob_home_devigged",
        "elo_pick_home", "market_pick_home", "home_won", "elo_correct", "market_correct",
        "elo_brier_term", "market_brier_term", "spread_abs_diff",
    ]
    ws.append(headers)
    style_header_row(ws, 1, len(headers))

    df = elo_df.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    for i, row in df.iterrows():
        r = i + 2
        ws.cell(r, 1, row["game_id"])
        ws.cell(r, 2, int(row["season"]))
        ws.cell(r, 3, int(row["week"]) if pd.notna(row["week"]) else None)
        ws.cell(r, 4, row["home_team"])
        ws.cell(r, 5, row["away_team"])
        ws.cell(r, 6, row["home_score"])
        ws.cell(r, 7, row["away_score"])
        ws.cell(r, 8, f"=F{r}-G{r}")
        ws.cell(r, 9, row["spread_line"] if pd.notna(row["spread_line"]) else None)
        ws.cell(r, 10, round(float(row["elo_implied_spread"]), 3) if pd.notna(row["elo_implied_spread"]) else None)
        ws.cell(r, 11, f"=J{r}-I{r}")
        ws.cell(r, 12, round(float(row["elo_implied_home_win_prob"]), 4))
        ws.cell(r, 13, row["home_moneyline"] if pd.notna(row["home_moneyline"]) else None)
        ws.cell(r, 14, row["away_moneyline"] if pd.notna(row["away_moneyline"]) else None)
        ws.cell(r, 15, f'=IFERROR(IF(M{r}<0,-M{r}/(-M{r}+100),100/(M{r}+100)),"")')
        ws.cell(r, 16, f'=IFERROR(IF(N{r}<0,-N{r}/(-N{r}+100),100/(N{r}+100)),"")')
        ws.cell(r, 17, f'=IFERROR(O{r}/(O{r}+P{r}),"")')
        ws.cell(r, 18, f"=IF(L{r}>0.5,1,0)")
        ws.cell(r, 19, f'=IFERROR(IF(O{r}>P{r},1,0),"")')
        ws.cell(r, 20, f'=IF(F{r}>G{r},1,IF(F{r}<G{r},0,""))')
        ws.cell(r, 21, f'=IF(T{r}="","",IF(R{r}=T{r},1,0))')
        ws.cell(r, 22, f'=IF(OR(T{r}="",S{r}=""),"",IF(S{r}=T{r},1,0))')
        ws.cell(r, 23, f'=IF(T{r}="","",(L{r}-T{r})^2)')
        ws.cell(r, 24, f'=IF(OR(T{r}="",Q{r}=""),"",(Q{r}-T{r})^2)')
        ws.cell(r, 25, f"=ABS(K{r})")

    for c in range(1, len(headers) + 1):
        for r in range(2, len(df) + 2):
            ws.cell(r, c).font = BODY_FONT
    autosize(ws, [16, 8, 6, 11, 11, 11, 11, 12, 12, 17, 12, 22, 14, 14, 15, 15, 22, 14, 15, 10, 12, 15, 13, 15, 15])
    ws.freeze_panes = "A2"
    return ws, len(df)


def build_season_summary(wb, seasons):
    ws = wb.create_sheet("Season Summary", 1)  # position right after README
    headers = ["season", "n_games", "elo_su_accuracy", "home_always_accuracy", "market_su_accuracy",
               "elo_brier", "market_brier", "elo_spread_mae"]
    ws.append(headers)
    style_header_row(ws, 1, len(headers))

    for i, season in enumerate(seasons):
        r = i + 2
        ws.cell(r, 1, int(season))
        ws.cell(r, 2, f"=COUNTIF(Games!$B:$B,A{r})")
        ws.cell(r, 3, f"=AVERAGEIFS(Games!$U:$U,Games!$B:$B,A{r})")
        ws.cell(r, 4, f"=AVERAGEIFS(Games!$T:$T,Games!$B:$B,A{r})")
        ws.cell(r, 5, f"=AVERAGEIFS(Games!$V:$V,Games!$B:$B,A{r})")
        ws.cell(r, 6, f"=AVERAGEIFS(Games!$W:$W,Games!$B:$B,A{r})")
        ws.cell(r, 7, f"=AVERAGEIFS(Games!$X:$X,Games!$B:$B,A{r})")
        ws.cell(r, 8, f"=AVERAGEIFS(Games!$Y:$Y,Games!$B:$B,A{r})")

    total_row = len(seasons) + 2
    lo, hi = seasons[0], seasons[-1]
    ws.cell(total_row, 1, f"{lo}-{hi}")
    ws.cell(total_row, 2, f"=SUM(B2:B{total_row - 1})")
    ws.cell(total_row, 3, f'=AVERAGEIFS(Games!$U:$U,Games!$B:$B,">={lo}",Games!$B:$B,"<={hi}")')
    ws.cell(total_row, 4, f'=AVERAGEIFS(Games!$T:$T,Games!$B:$B,">={lo}",Games!$B:$B,"<={hi}")')
    ws.cell(total_row, 5, f'=AVERAGEIFS(Games!$V:$V,Games!$B:$B,">={lo}",Games!$B:$B,"<={hi}")')
    ws.cell(total_row, 6, f'=AVERAGEIFS(Games!$W:$W,Games!$B:$B,">={lo}",Games!$B:$B,"<={hi}")')
    ws.cell(total_row, 7, f'=AVERAGEIFS(Games!$X:$X,Games!$B:$B,">={lo}",Games!$B:$B,"<={hi}")')
    ws.cell(total_row, 8, f'=AVERAGEIFS(Games!$Y:$Y,Games!$B:$B,">={lo}",Games!$B:$B,"<={hi}")')
    for c in range(1, 9):
        ws.cell(total_row, c).font = Font(name=FONT, bold=True)

    for c in [3, 4, 5, 6, 7]:
        for r in range(2, total_row + 1):
            ws.cell(r, c).number_format = "0.000"
    for r in range(2, total_row + 1):
        ws.cell(r, 8).number_format = "0.00"

    for c in range(1, 9):
        for r in range(2, total_row + 1):
            if ws.cell(r, c).font is None or ws.cell(r, c).font.name != FONT:
                ws.cell(r, c).font = BODY_FONT
    autosize(ws, [10, 10, 15, 19, 17, 11, 13, 14])
    ws.freeze_panes = "A2"

    # --- charts -----------------------------------------------------------
    n = len(seasons)
    acc_chart = BarChart()
    acc_chart.title = "Straight-Up Accuracy by Season"
    acc_chart.y_axis.title = "Accuracy"
    acc_chart.x_axis.title = "Season"
    acc_chart.height, acc_chart.width = 9, 20
    cats = Reference(ws, min_col=1, min_row=2, max_row=n + 1)
    for col, name in [(3, "Elo"), (4, "Home team always"), (5, "Market favorite")]:
        data = Reference(ws, min_col=col, min_row=1, max_row=n + 1)
        acc_chart.add_data(data, titles_from_data=True)
    acc_chart.set_categories(cats)
    ws.add_chart(acc_chart, "J2")

    brier_chart = BarChart()
    brier_chart.title = "Brier Score by Season (lower = better calibrated)"
    brier_chart.y_axis.title = "Brier score"
    brier_chart.x_axis.title = "Season"
    brier_chart.height, brier_chart.width = 9, 20
    for col, name in [(6, "Elo"), (7, "Market (vig-removed)")]:
        data = Reference(ws, min_col=col, min_row=1, max_row=n + 1)
        brier_chart.add_data(data, titles_from_data=True)
    brier_chart.set_categories(cats)
    ws.add_chart(brier_chart, "J20")

    return ws


def build_current_week_sheet(wb, current_week_df):
    ws = wb.create_sheet("Current Week", 2)  # after README, Season Summary
    season = int(current_week_df["season"].iloc[0])
    week = int(current_week_df["week"].iloc[0])

    ws["A1"] = f"Current Week: Elo vs Market - {season} Week {week}"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = ("Side-by-side only - no bet flag, no unit sizing, no recommendation. Elo has NOT beaten "
                "the market in backtesting so far (see Season Summary) - this is what a simple model "
                "thinks, next to what the market thinks, nothing more.")
    ws["A2"].font = NOTE_FONT
    has_market = "market_spread" in current_week_df.columns and current_week_df["market_spread"].notna().any()
    if has_market:
        ws["A3"] = ("market_spread comes from The Odds API (averaged across books - see odds_source per "
                    "row for how many). spread_gap = elo_implied_spread - market_spread, both in the "
                    "'positive = home favored' convention used throughout this workbook.")
    else:
        ws["A3"] = ("market_spread is blank: no live odds source is connected yet (The Odds API pending "
                    "setup). Once connected, this sheet fills in and the gap column becomes meaningful.")
    ws["A3"].font = NOTE_FONT

    headers = ["game_id", "home_team", "away_team", "gameday", "elo_home_rating", "elo_away_rating",
               "elo_implied_home_win_prob", "elo_implied_spread", "market_spread", "spread_gap", "odds_source"]
    start_row = 5
    for j, h in enumerate(headers, start=1):
        ws.cell(start_row, j, h)
    style_header_row(ws, start_row, len(headers))

    df = current_week_df.sort_values("gameday")
    for i, row in df.iterrows():
        r = start_row + 1 + list(df.index).index(i)
        ws.cell(r, 1, row["game_id"])
        ws.cell(r, 2, row["home_team"])
        ws.cell(r, 3, row["away_team"])
        ws.cell(r, 4, str(row["gameday"]))
        ws.cell(r, 5, row["elo_home_rating"])
        ws.cell(r, 6, row["elo_away_rating"])
        ws.cell(r, 7, row["elo_implied_home_win_prob"])
        ws.cell(r, 8, row["elo_implied_spread"])
        ws.cell(r, 9, row.get("market_spread") if pd.notna(row.get("market_spread")) else None)
        ws.cell(r, 10, row.get("spread_gap") if pd.notna(row.get("spread_gap")) else None)
        ws.cell(r, 11, row.get("odds_source", "not connected"))
        for c in range(1, len(headers) + 1):
            ws.cell(r, c).font = BODY_FONT

    autosize(ws, [16, 11, 11, 12, 15, 15, 22, 16, 14, 12, 15])
    ws.freeze_panes = f"A{start_row + 1}"
    return ws


def build_trajectory_sheet(wb, elo_df, teams):
    ws = wb.create_sheet("Team Elo Trajectory")
    lo, hi = int(elo_df["season"].min()), int(elo_df["season"].max())
    ws["A1"] = f"Illustrative Elo rating trajectory (pre-game rating), {lo}-{hi}, selected teams"
    ws["A1"].font = Font(name=FONT, bold=True, size=12)
    ws["A2"] = "X-axis is the Nth game since 2015 for that team (not calendar-aligned across teams)."
    ws["A2"].font = NOTE_FONT

    long_rows = []
    for _, row in elo_df.sort_values(["gameday", "game_id"]).iterrows():
        long_rows.append({"team": row["home_team"], "gameday": row["gameday"], "elo_pre": row["elo_home_pre"]})
        long_rows.append({"team": row["away_team"], "gameday": row["gameday"], "elo_pre": row["elo_away_pre"]})
    long_df = pd.DataFrame(long_rows)
    long_df = long_df[long_df["team"].isin(teams)].sort_values(["team", "gameday"])
    long_df["game_num"] = long_df.groupby("team").cumcount() + 1

    pivot = long_df.pivot_table(index="game_num", columns="team", values="elo_pre")
    pivot = pivot.reset_index()

    headers = ["game_num"] + list(pivot.columns[1:])
    start_row = 4
    ws.cell(start_row, 1, "game_num")
    for j, team in enumerate(pivot.columns[1:], start=2):
        ws.cell(start_row, j, team)
    style_header_row(ws, start_row, len(headers))

    for i, row in pivot.iterrows():
        r = start_row + 1 + i
        ws.cell(r, 1, int(row["game_num"]))
        for j, team in enumerate(pivot.columns[1:], start=2):
            val = row[team]
            ws.cell(r, j, round(float(val), 1) if pd.notna(val) else None)

    n_rows = len(pivot)
    chart = LineChart()
    chart.title = "Elo rating over time (selected teams)"
    chart.y_axis.title = "Elo rating"
    chart.x_axis.title = "Game # since 2015"
    chart.height, chart.width = 12, 26
    data = Reference(ws, min_col=2, max_col=len(headers), min_row=start_row, max_row=start_row + n_rows)
    cats = Reference(ws, min_col=1, min_row=start_row + 1, max_row=start_row + n_rows)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    ws.add_chart(chart, f"A{start_row + n_rows + 3}")

    autosize(ws, [10] + [10] * (len(headers) - 1))
    return ws


def main():
    elo_df = pd.read_csv(PROCESSED / "elo_ratings.csv")
    elo_df = elo_df[elo_df["elo_implied_spread"].notna()].copy()
    # Lower bound fixed at 2015 (matches the EPA feature pipeline's coverage,
    # for when that gets layered in later). Upper bound is NOT hardcoded -
    # it's whatever the most recent season with a completed, scored game is,
    # so 2026 starts appearing here on its own as soon as games are played,
    # instead of silently staying frozen at 2025.
    max_season = int(elo_df["season"].max())
    elo_df = elo_df[elo_df["season"].between(2015, max_season)].copy()

    seasons = sorted(int(s) for s in elo_df["season"].unique())
    n_games = len(elo_df)

    wb = Workbook()
    build_readme(wb, seasons[0], seasons[-1], n_games)  # uses wb.active - must be created first, stays tab 1
    games_ws, n_games_check = build_games_sheet(wb, elo_df)
    build_season_summary(wb, seasons)

    cw_path = PROCESSED / "current_week_predictions.csv"
    if cw_path.exists():
        current_week_df = pd.read_csv(cw_path)
        build_current_week_sheet(wb, current_week_df)

    build_trajectory_sheet(wb, elo_df, teams=["KC", "SF", "BUF", "NE", "LA", "DAL"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_PATH)
    print(f"Saved {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes), {n_games} games, {len(seasons)} seasons")


if __name__ == "__main__":
    main()
