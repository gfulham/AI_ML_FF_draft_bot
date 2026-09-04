"""Recommend a pick from a live Google Sheet or a downloaded sheet CSV.

The sheet must contain player projections plus these boolean columns:
- Pick: TRUE once any manager drafts the player.
- My Team: TRUE for players drafted by this draft slot.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

from draft_sim_class import DraftSim
from RL_draft_env import LEAGUE_CONFIGS

BASE_DIR = Path(__file__).resolve().parent
GOOGLE_KEY_FILE = BASE_DIR / "config" / "FantasyDraftBotKey.json"
DEFAULT_SPREADSHEET = "Fantasy Data 2026"
WORKSHEETS = {
    "ppr": "All Data PPR 2WR",
    "keeper": "All Data PPR 3WR",
    "ppr_fd": "All Data PPR plus FD",
}
REQUIRED_COLUMNS = {"Player", "Position", "ESPN ADP", "ESPN ECR", "VOR", "Pick", "My Team"}


def nested_defaultdict():
    """Compatibility factory for Q-tables serialized by RL_draft_env.py."""
    return defaultdict(float)


def parse_sheet_bool(values: pd.Series, column: str) -> pd.Series:
    """Convert Google Sheets booleans while rejecting ambiguous values."""
    normalized = values.fillna(False).astype(str).str.strip().str.lower()
    valid_values = {"true", "false", "", "nan"}
    invalid = normalized[~normalized.isin(valid_values)]
    if not invalid.empty:
        raise ValueError(f"{column} must contain only TRUE or FALSE. Invalid values: {invalid.unique().tolist()}")
    return normalized.eq("true")


def prepare_draft_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize live-sheet data without changing draft flags."""
    data = data.copy()
    data.columns = data.columns.str.strip()
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Sheet is missing required columns: {sorted(missing)}")

    for column in ("ESPN ADP", "ESPN ECR", "VOR"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Player", "Position", "ESPN ADP", "ESPN ECR", "VOR"])
    data["Player"] = data["Player"].astype(str).str.strip()
    if data["Player"].duplicated().any():
        raise ValueError("Player names must be unique in the sheet.")

    data["Pick"] = parse_sheet_bool(data["Pick"], "Pick")
    data["My Team"] = parse_sheet_bool(data["My Team"], "My Team")
    if (data["My Team"] & ~data["Pick"]).any():
        raise ValueError("Every player marked My Team must also be marked Pick.")
    return data


def load_google_sheet(spreadsheet: str, worksheet: str) -> pd.DataFrame:
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
    except ImportError as error:
        raise RuntimeError(
            "Google Sheets support requires gspread and oauth2client. "
            "Install them with: pip install gspread oauth2client"
        ) from error
    if not GOOGLE_KEY_FILE.exists():
        raise FileNotFoundError(f"Google service-account key not found: {GOOGLE_KEY_FILE}")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(str(GOOGLE_KEY_FILE), scope)
    client = gspread.authorize(credentials)
    return pd.DataFrame(client.open(spreadsheet).worksheet(worksheet).get_all_records())


def build_live_simulator(data: pd.DataFrame, config: dict, draft_slot: int) -> DraftSim:
    """Build model state from picked flags and the user's roster only."""
    num_teams = 12
    if not 1 <= draft_slot <= num_teams:
        raise ValueError(f"draft slot must be between 1 and {num_teams}")

    simulator = DraftSim(
        data,
        num_rounds=config.get("num_rounds", 12),
        num_teams=num_teams,
        my_pick=draft_slot - 1,
        starter_limits=config["starter_limits"],
        position_limits=config["position_limits"],
    )
    picked = set(data.loc[data["Pick"], "Player"])
    if len(picked) > simulator.total_picks:
        raise ValueError(f"The sheet has {len(picked)} picks, exceeding the {simulator.total_picks}-pick draft.")

    # The Q-table state needs availability, global pick count, and our roster only.
    # Sheet row order is rankings, not chronological draft order, so do not replay it.
    simulator.available_set.difference_update(picked)
    simulator.rosters[simulator.my_pick] = [
        simulator._player_dict[player]
        for player in data.loc[data["My Team"], "Player"]
    ]
    simulator.current_pick = len(picked)
    simulator.done = simulator.current_pick >= simulator.total_picks
    return simulator


def get_ranked_recommendations(simulator: DraftSim, q_table: dict, count: int) -> list[tuple[str, float]]:
    state = simulator.get_state()
    candidates = simulator.get_next_n_per_position()
    my_team = simulator.rosters[simulator.my_pick]
    legal_candidates = [
        player
        for player in candidates
        if simulator.can_add_player(my_team, simulator._player_dict[player], simulator.my_pick)
    ]
    ranked = [(player, q_table[state][player]) for player in legal_candidates if player in q_table[state]]
    return sorted(ranked, key=lambda item: item[1], reverse=True)[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Get Q-table recommendations from the current draft sheet.")
    parser.add_argument("--draft-slot", type=int, required=True, help="Your one-based draft position (1-12).")
    parser.add_argument("--league", choices=LEAGUE_CONFIGS, default="ppr_fd")
    parser.add_argument("--csv", type=Path, help="Use a downloaded sheet CSV instead of Google Sheets.")
    parser.add_argument("--spreadsheet", default=DEFAULT_SPREADSHEET)
    parser.add_argument("--worksheet", help="Override the league's default worksheet name.")
    parser.add_argument("--top", type=int, default=10, help="Number of recommendations to display.")
    args = parser.parse_args()

    config = LEAGUE_CONFIGS[args.league]
    raw_data = pd.read_csv(args.csv) if args.csv else load_google_sheet(
        args.spreadsheet, args.worksheet or WORKSHEETS[args.league]
    )
    simulator = build_live_simulator(prepare_draft_frame(raw_data), config, args.draft_slot)
    with Path(config["q_file"]).open("rb") as file_handle:
        q_table = pickle.load(file_handle)

    recommendations = get_ranked_recommendations(simulator, q_table, args.top)
    print(f"Picks recorded: {simulator.current_pick}")
    print(f"Your draft slot: {args.draft_slot}")
    print("Your roster:", ", ".join(player["Player"] for player in simulator.rosters[simulator.my_pick]) or "(empty)")
    if recommendations:
        print("Recommendations:")
        for rank, (player, value) in enumerate(recommendations, start=1):
            print(f"{rank}. {player}: {value:.3f}")
    else:
        print("No learned Q-values exist for this state and available candidate set.")


if __name__ == "__main__":
    main()
