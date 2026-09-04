"""Reinforcement-learning draft training helpers.

This module should be imported for its functions, not for side effects.
Run it directly to train the Q-table.
"""

from __future__ import annotations

import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import season_simulation
from draft_sim_class import DraftSim
from sample_player_outcomes import (
    all_samples,
    load_latest_player_data,
    load_samples_from_csv,
    load_weekly_matchups_sos_data,
    sample_player_points,
    standardize_name,
)
from season_simulation import simulate_league

# Each league gets its own player-data subfolder, season-sample CSV, Q-table/reward file,
# and starting-lineup requirements (the keeper league starts 3 WR instead of 2).
BASE_DIR = Path(__file__).resolve().parent

LEAGUE_CONFIGS = {
    "ppr": {
        "sample_csv": BASE_DIR / "data" / "ppr" / "player_samples.csv",
        "q_file": BASE_DIR / "q_table_ppr_2026.pkl",
        "reward_history_file": BASE_DIR / "reward_history_ppr_2026.csv",
        "evaluation_history_file": BASE_DIR / "evaluation_history_ppr_2026.csv",
        "starter_limits": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        "position_limits": {"QB": 2, "RB": 8, "WR": 8, "TE": 3},
        "waiver_weekly_projections": {"QB": 16.0, "RB": 6.5, "WR": 7.5, "TE": 7.5},
    },
    "keeper": {
        "sample_csv": BASE_DIR / "data" / "keeper" / "player_samples.csv",
        "keeper_file": BASE_DIR / "data" / "keepers_2026.csv",
        "num_rounds": 14,
        "q_file": BASE_DIR / "q_table_keeper_2026.pkl",
        "reward_history_file": BASE_DIR / "reward_history_keeper_2026.csv",
        "evaluation_history_file": BASE_DIR / "evaluation_history_keeper_2026.csv",
        "starter_limits": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1},
        "position_limits": {"QB": 2, "RB": 7, "WR": 7, "TE": 3},
        "waiver_weekly_projections": {"QB": 16.0, "RB": 6.5, "WR": 7.5, "TE": 7.5},
    },
    "ppr_fd": {
        "sample_csv": BASE_DIR / "data" / "ppr_fd" / "player_samples.csv",
        "q_file": BASE_DIR / "q_table_ppr_fd_2026.pkl",
        "reward_history_file": BASE_DIR / "reward_history_ppr_fd_2026.csv",
        "evaluation_history_file": BASE_DIR / "evaluation_history_ppr_fd_2026.csv",
        "starter_limits": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2},
        "position_limits": {"QB": 2, "RB": 8, "WR": 8, "TE": 3},
        "waiver_weekly_projections": {"QB": 19.0, "RB": 7.5, "WR": 9.0, "TE": 9.0},
    },
}


def load_keepers(keeper_file: Path, my_keepers: tuple[str, str], my_team: int = 1) -> list[dict[str, int | str]]:
    """Load fixed league keepers and the selected two-player keeper scenario for my team."""
    keeper_frame = pd.read_csv(keeper_file)
    keeper_frame.columns = keeper_frame.columns.str.strip()
    required_columns = {"Team", "Keeper_1", "Keeper_1_cost"}
    if missing_columns := required_columns - set(keeper_frame.columns):
        raise ValueError(f"Keeper file is missing columns: {sorted(missing_columns)}")

    keeper_frame = keeper_frame.dropna(subset=["Keeper_1", "Keeper_1_cost"]).copy()
    keeper_frame["Team"] = pd.to_numeric(keeper_frame["Team"], errors="raise").astype(int)
    keeper_frame["Keeper_1_cost"] = pd.to_numeric(keeper_frame["Keeper_1_cost"], errors="raise").astype(int)
    keeper_frame["Keeper_1"] = keeper_frame["Keeper_1"].astype(str).str.strip()

    available_my_keepers = set(keeper_frame.loc[keeper_frame["Team"] == my_team, "Keeper_1"])
    selected_my_keepers = set(my_keepers)
    if len(my_keepers) != 2 or len(selected_my_keepers) != 2:
        raise ValueError("Choose exactly two different keepers for Team 1.")
    unknown_keepers = selected_my_keepers - available_my_keepers
    if unknown_keepers:
        raise ValueError(f"Team {my_team} keeper choice is not in {keeper_file.name}: {sorted(unknown_keepers)}")

    fixed_keepers = keeper_frame[keeper_frame["Team"] != my_team]
    selected_keepers = keeper_frame[
        (keeper_frame["Team"] == my_team) & keeper_frame["Keeper_1"].isin(selected_my_keepers)
    ]
    keeper_frame = pd.concat([fixed_keepers, selected_keepers], ignore_index=True)
    return [
        {"team": row.Team, "player": row.Keeper_1, "cost": row.Keeper_1_cost}
        for row in keeper_frame.itertuples(index=False)
    ]


def keeper_scenario_file(path: Path, my_keepers: tuple[str, str]) -> Path:
    """Keep Q-tables and history distinct for each Team 1 keeper pairing."""
    suffix = "_".join(sorted(my_keepers))
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def nested_defaultdict():
    return defaultdict(float)


def build_q_table(q_file: str = "q_table_2026.pkl"):
    """Initializes the first q table, or loads it if exists"""
    if os.path.exists(q_file):
        with open(q_file, "rb") as handle:
            q_table = pickle.load(handle)
        print("Loaded existing Q-table.")
    else:
        q_table = defaultdict(nested_defaultdict)
        print("Initialized new Q-table.")
    return q_table


#def load_latest_player_data(folder: str = "data", prefix: str = "player_data_") -> pd.DataFrame:
#    """Load the latest player data CSV matching the naming convention.
#    Needed to know the latest player to draft."""
#    files = sorted(Path(folder).glob(f"{prefix}*.csv"))
#    if not files:
#        raise FileNotFoundError(f"No files found with prefix '{prefix}' in {folder}")
#
#    latest = files[-1]
#    print(f"Loading latest player data from: {latest}")
#    return pd.read_csv(latest)


def draft_using_q_agent(
    Q,
    df,
    epsilon: float,
    print_draft: bool = True,
    starter_limits: dict | None = None,
    position_limits: dict | None = None,
    my_pick: int | None = None,
    keepers: list[dict[str, int | str]] | None = None,
    num_rounds: int = 12,
    sos_df: pd.DataFrame | None = None,
    weekly_volatility: float | None = None,
    waiver_weekly_projections: dict[str, float] | None = None,
    q_agent_teams: set[int] | None = None,
):
    """Run one draft and return the draft environment, log, and state-action history.
    DraftSim is a file / class created to help simulate a real draft"""
    env = DraftSim(
        df,
        num_rounds=num_rounds,
        my_pick=my_pick,
        starter_limits=starter_limits,
        position_limits=position_limits,
        keepers=keepers,
        sos_df=sos_df,
        weekly_volatility=weekly_volatility,
        waiver_weekly_projections=waiver_weekly_projections,
    )
    player_dict = env._player_dict

    # Teams controlled by the Q-agent; every other team follows the ADP-offset board.
    if q_agent_teams is None:
        q_agent_teams = {env.my_pick}
    q_agent_teams = set(q_agent_teams)
    if not q_agent_teams:
        raise ValueError("q_agent_teams must contain at least one team.")
    invalid_teams = [team for team in q_agent_teams if not 0 <= team < env.num_teams]
    if invalid_teams:
        raise ValueError(f"q_agent_teams out of range for {env.num_teams} teams: {invalid_teams}")
    env.q_agent_teams = q_agent_teams

    picks_log = []
    history = []
    policy_audit = []

    while not env.done:
        # Get the next candidates for the current pick and filter them based on legality
        candidate_names = env.get_next_n_per_position()
        team_idx = env.snake_order[env.current_pick]
        is_q_agent = team_idx in q_agent_teams
        if is_q_agent:
            # State and draftable-position rules are relative to the picking team.
            env.my_pick = team_idx
        draftable_positions = env.get_draftable_positions(team_idx)

        # Filter candidates to only those that are legal for the current team and pick
        legal_candidates = [
            player_name
            for player_name in candidate_names
            if player_name in player_dict
            and player_dict[player_name]["Position"] in draftable_positions
        ]

        if not legal_candidates:
            raise RuntimeError(f"No legal draft candidates available for team {team_idx} at pick {env.current_pick}")

        rolling_adp = None
        adp_offset = None
        if is_q_agent:
            state = env.get_state()
            if random.random() < epsilon:
                action = random.choice(legal_candidates)
                pick_type = "Random"
            else:
                q_vals = {candidate: Q[state][candidate] for candidate in legal_candidates}
                action = max(q_vals, key=q_vals.get)
                pick_type = "Q-Max"
                policy_audit.append(
                    {
                        "state": state,
                        "selected_player": action,
                        "ranked_q_values": sorted(q_vals.items(), key=lambda item: item[1], reverse=True),
                    }
                )
        else:
            adp_candidates = sorted(
                (
                    player_name
                    for player_name in env.available_set
                    if player_dict[player_name]["Position"] in draftable_positions
                ),
                key=lambda player_name: (
                    0.5 * player_dict[player_name]["ESPN ADP"]
                    + 0.5 * player_dict[player_name]["ESPN ECR"],
                    player_dict[player_name]["ESPN ADP"],
                ),
            )
            if not adp_candidates:
                raise RuntimeError(f"No legal ADP candidates available for team {team_idx} at pick {env.current_pick}")
            board_start = env.current_pick + 1
            adp_offset = DraftSim.get_offset_for_adp(board_start, board_start)
            board_index = min(adp_offset, len(adp_candidates) - 1)
            action = adp_candidates[board_index]
            rolling_adp = board_start + board_index
            pick_type = "ADP-Offset"

        if print_draft:
            sampled_points = sample_player_points(pd.Series(player_dict[action]), n=1)[0]
            picks_log.append(
                {
                    "Pick #": env.current_pick,
                    "ADP": player_dict[action]["ESPN ADP"],
                    "Rolling ADP": rolling_adp,
                    "ADP Offset": adp_offset,
                    "Team": team_idx,
                    "Player": action,
                    "Position": player_dict[action]["Position"],
                    "FP": player_dict[action]["points"],
                    "Sampled FP": sampled_points,
                    "Pick Type": pick_type,
                }
            )

        env.step(action)
        if is_q_agent:
            history.append((team_idx, state, action))

    draft_df = pd.DataFrame(picks_log) if print_draft else pd.DataFrame()

    if print_draft:
        print(f"Q-agent teams: {sorted(q_agent_teams)}")
        for team_id in range(env.num_teams):
            print(f"\nTeam {team_id} Draft:")
            print(draft_df[draft_df["Team"] == team_id][["Player", "Position", "FP", "Sampled FP", "Pick Type"]].reset_index(drop=True))

    env.policy_audit = policy_audit
    return env, draft_df, history


def print_training_inspection(env, draft_log, season_result, block_number, reward):
    """Print one complete draft and season outcome for a training block."""
    print(f"\n{'=' * 24} Block {block_number} Inspection {'=' * 24}")
    q_agent_teams = sorted(getattr(env, "q_agent_teams", {env.my_pick}))
    print(f"Q-agent teams: {q_agent_teams} | Reward: ${reward:.2f}")
    print("\nDraft rosters:")
    # just return the roster which the q-agent team drafted
    #for team_id in range(env.num_teams):
        #roster = env.rosters[team_id]
        #players = ", ".join(f"{p.get('Player', p.get('Name', ''))} ({p.get('Position', '')})" for p in roster)
        #print(f"Team {team_id}: {players}")
    for team_id in q_agent_teams:
        roster = env.rosters[team_id]
        players = ", ".join(f"{p.get('Player', p.get('Name', ''))} ({p.get('Position', '')})" for p in roster)
        print(f"Team {team_id}: {players}")

    #print("\nRegular season:")
    #for weekly_result in season_result["regular_season"]:
    #    matchups = " | ".join(
    #        f"T{matchup['teams'][0]} {matchup['scores'][0]:.1f} - "
    #        f"T{matchup['teams'][1]} {matchup['scores'][1]:.1f} (W: T{matchup['winner']})"
    #        for matchup in weekly_result["matchups"]
    #    )
    #    print(f"Week {weekly_result['week']}: {matchups}")

    print("\nRegular-season standings:")
    for seed, team_id in enumerate(season_result["seeds"], start=1):
        print(
            f"{seed}. Team {team_id}: "
            f"{season_result['regular_season_points_for'][team_id]:.1f} points"
        )

    print("\nPlayoffs:")
    playoffs = season_result["playoffs"]
    print(f"Week 15 byes: {', '.join(f'Team {team}' for team in playoffs['bye_teams'])}")
    for week_name in ("week_15", "week_16", "week_17"):
        scores = playoffs[week_name]
        print(week_name.replace("_", " ").title() + ": " + ", ".join(
            f"Team {team} {score:.1f}" for team, score in scores.items()
        ))
    print("\nFinal placements:")
    for place, team_id in enumerate(season_result["final_placements"], start=1):
        print(f"{place}. Team {team_id}: Reward ${season_result['team_rewards'][team_id]:.1f}")

    # Retired H2H reporting printed matchup winners and a championship game.
    # championship_scores = season_result["playoffs"]["championship_scores"]
    # champion = season_result["playoffs"]["champion"]
    # runner_up = season_result["playoffs"]["runner_up"]


def evaluate_q_agent(
    Q,
    df,
    episodes: int = 25,
    seed: int = 20260828,
    starter_limits: dict | None = None,
    position_limits: dict | None = None,
    my_pick: int | None = None,
    keepers: list[dict[str, int | str]] | None = None,
    num_rounds: int = 12,
    sos_df: pd.DataFrame | None = None,
    weekly_volatility: float | None = None,
    waiver_weekly_projections: dict[str, float] | None = None,
):
    """Evaluate one greedy draft over multiple independently sampled seasons."""
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        env, _, _ = draft_using_q_agent(
            Q,
            df,
            epsilon=0.0,
            print_draft=False,
            starter_limits=starter_limits,
            position_limits=position_limits,
            my_pick=my_pick,
            keepers=keepers,
            num_rounds=num_rounds,
            sos_df=sos_df,
            weekly_volatility=weekly_volatility,
            waiver_weekly_projections=waiver_weekly_projections,
        )
        rewards = []
        placements = []
        first_place_finishes = 0
        top_three_finishes = 0
        playoff_appearances = 0

        for _ in range(episodes):
            season_result = env.get_season_result()
            team_id = env.my_pick
            reward = season_result["team_rewards"][team_id]
            placement = season_result["placement_by_team"][team_id]
            rewards.append(reward)
            placements.append(placement)
            first_place_finishes += placement == 1
            top_three_finishes += placement <= 3
            playoff_appearances += team_id in season_result["seeds"][:6]

        # Retired H2H evaluation used $7 champion and $5 runner-up payouts.
        # payout = 7.0 if season_result["playoffs"]["champion"] == team_id else 5.0 if season_result["playoffs"]["runner_up"] == team_id else 0.0
        return {
            "evaluation_seasons": episodes,
            "average_reward": float(np.mean(rewards)),
            "average_placement": float(np.mean(placements)),
            "first_place_rate": first_place_finishes / episodes,
            "top_three_rate": top_three_finishes / episodes,
            "playoff_rate": playoff_appearances / episodes,
        }
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def evaluate_draft_slots(
    Q,
    df,
    drafts_per_slot: int = 100,
    seasons_per_draft: int = 25,
    seed: int = 20260903,
    starter_limits: dict | None = None,
    position_limits: dict | None = None,
    keepers: list[dict[str, int | str]] | None = None,
    num_rounds: int = 12,
    sos_df: pd.DataFrame | None = None,
    weekly_volatility: float | None = None,
    waiver_weekly_projections: dict[str, float] | None = None,
):
    """Evaluate a greedy Q-table at every draft slot using paired season simulations."""
    if drafts_per_slot < 1:
        raise ValueError("drafts_per_slot must be at least 1.")
    if seasons_per_draft < 1:
        raise ValueError("seasons_per_draft must be at least 1.")

    original_season_rng = season_simulation.rng
    records = []
    try:
        for draft_slot in range(12):
            for draft_number in range(1, drafts_per_slot + 1):
                env, _, _ = draft_using_q_agent(
                    Q,
                    df,
                    epsilon=0.0,
                    print_draft=False,
                    starter_limits=starter_limits,
                    position_limits=position_limits,
                    my_pick=draft_slot,
                    keepers=keepers,
                    num_rounds=num_rounds,
                    sos_df=sos_df,
                    weekly_volatility=weekly_volatility,
                    waiver_weekly_projections=waiver_weekly_projections,
                )
                rewards = []
                placements = []
                playoff_appearances = 0
                for season_number in range(seasons_per_draft):
                    # Reuse the same season seed for every slot/draft pair to reduce comparison noise.
                    season_simulation.rng = np.random.default_rng(
                        seed + draft_number * seasons_per_draft + season_number
                    )
                    season_result = env.get_season_result()
                    team_id = env.my_pick
                    rewards.append(season_result["team_rewards"][team_id])
                    placements.append(season_result["placement_by_team"][team_id])
                    playoff_appearances += team_id in season_result["seeds"][:6]

                records.append(
                    {
                        "Draft Slot": draft_slot + 1,
                        "Draft Simulation": draft_number,
                        "Seasons Simulated": seasons_per_draft,
                        "Average Reward": float(np.mean(rewards)),
                        "Average Placement": float(np.mean(placements)),
                        "First Place Rate": float(np.mean(np.array(placements) == 1)),
                        "Top Three Rate": float(np.mean(np.array(placements) <= 3)),
                        "Playoff Rate": playoff_appearances / seasons_per_draft,
                    }
                )
    finally:
        season_simulation.rng = original_season_rng

    detail_df = pd.DataFrame(records)
    summary_df = detail_df.groupby("Draft Slot", as_index=False).agg(
        Drafts_Evaluated=("Draft Simulation", "count"),
        Seasons_Per_Draft=("Seasons Simulated", "first"),
        Mean_Reward=("Average Reward", "mean"),
        Reward_Std_Dev=("Average Reward", "std"),
        Mean_Placement=("Average Placement", "mean"),
        First_Place_Rate=("First Place Rate", "mean"),
        Top_Three_Rate=("Top Three Rate", "mean"),
        Playoff_Rate=("Playoff Rate", "mean"),
    )
    summary_df["Reward_95pct_CI"] = (
        1.96 * summary_df["Reward_Std_Dev"].fillna(0.0) / np.sqrt(summary_df["Drafts_Evaluated"])
    )
    summary_df = summary_df.sort_values("Mean_Reward", ascending=False, ignore_index=True)
    return detail_df, summary_df


def run_draft_slot_evaluation(
    league: str,
    my_keepers: tuple[str, str] | None = None,
    drafts_per_slot: int = 100,
    seasons_per_draft: int = 25,
):
    """Load a trained Q-table and export a greedy evaluation for every draft slot."""
    config = LEAGUE_CONFIGS[league]
    sample_path = Path(config["sample_csv"])
    load_samples_from_csv(str(sample_path))
    dataframe = load_latest_player_data(league_config=league)
    q_file = Path(config["q_file"])
    keepers = None
    if league == "keeper":
        if my_keepers is None:
            raise ValueError("Keeper slot evaluation requires exactly two keeper names.")
        keepers = load_keepers(Path(config["keeper_file"]), my_keepers, my_team=1)
        q_file = keeper_scenario_file(q_file, my_keepers)

    q_table = build_q_table(q_file=q_file)
    detail_df, summary_df = evaluate_draft_slots(
        q_table,
        dataframe,
        drafts_per_slot=drafts_per_slot,
        seasons_per_draft=seasons_per_draft,
        starter_limits=config["starter_limits"],
        position_limits=config["position_limits"],
        keepers=keepers,
        num_rounds=config.get("num_rounds", 12),
        sos_df=load_weekly_matchups_sos_data(),
        waiver_weekly_projections=config["waiver_weekly_projections"],
    )
    output_dir = sample_path.parent
    detail_path = output_dir / "draft_slot_evaluation.csv"
    summary_path = output_dir / "draft_slot_evaluation_summary.csv"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print("\nDraft slot ranking by mean reward:")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Draft-slot details: {detail_path}")
    print(f"Draft-slot summary: {summary_path}")
    return detail_df, summary_df


def _sample_q_agent_teams(num_teams: int, num_q_agents: int, my_pick: int | None) -> set[int]:
    """Choose which draft slots the Q-agent controls for one training episode."""
    if not 1 <= num_q_agents <= num_teams:
        raise ValueError(f"num_q_agents must be between 1 and {num_teams}: {num_q_agents}")
    if my_pick is not None:
        # A fixed draft slot (e.g. the keeper league's Team 1) is always Q-controlled.
        other_teams = [team for team in range(num_teams) if team != my_pick]
        return {my_pick, *random.sample(other_teams, num_q_agents - 1)}
    return set(random.sample(range(num_teams), num_q_agents))


def train_q_agent(
    Q, # Q-table for the agent
    df, # DataFrame containing player information for the draft
    alpha: float = 0.1, # Learning rate for the Q-agent
    gamma: float = 1.0, # Discount factor for future rewards
    epsilon: float = 0.3, # Exploration rate for the Q-agent
    max_blocks: int | None = None, # Maximum number of training blocks (None = run indefinitely until Ctrl+C)
    episodes_per_block: int = 500, # Number of episodes (draft simulations) per training block
    q_file: str = "q_table.pkl", # File path to save the Q-table
    reward_history_file: str = "reward_history.csv",
    evaluation_history_file: str = "evaluation_history.csv",
    starter_limits: dict | None = None,
    position_limits: dict | None = None,
    inspect_every_blocks: int | None = 10,
    evaluate_every_blocks: int | None = 10,
    evaluation_episodes: int = 25,
    seasons_per_draft: int = 5,
    my_pick: int | None = None,
    keepers: list[dict[str, int | str]] | None = None,
    num_rounds: int = 12,
    sos_df: pd.DataFrame | None = None,
    weekly_volatility: float | None = None,
    waiver_weekly_projections: dict[str, float] | None = None,
    num_q_agents: int = 1, # How many drafters the Q-agent controls; the rest use ADP-offset
    num_teams: int = 12,
):
    """Train the Q-table using repeated draft simulations."""
    df = pd.DataFrame(df)

    if not 1 <= num_q_agents <= num_teams:
        raise ValueError(f"num_q_agents must be between 1 and {num_teams}: {num_q_agents}")

    if all_samples:
        df = df[df["Player"].map(standardize_name).isin(all_samples)]

    if df.empty:
        raise ValueError("No player data available for training after filtering.")
    if seasons_per_draft < 1:
        raise ValueError("seasons_per_draft must be at least 1.")

    reward_history = []
    last_avg_reward = -float("inf")
    existing_records: list[dict] = []

    if os.path.exists(reward_history_file) and os.path.getsize(reward_history_file) > 0:
        try:
            existing_df = pd.read_csv(reward_history_file)
            existing_records = existing_df.to_dict("records")
            if "average_reward" in existing_df.columns and len(existing_df) > 0:
                last_avg_reward = float(existing_df["average_reward"].iloc[-1])
            elif len(existing_df) > 0:
                last_avg_reward = float(existing_df.iloc[-1, 0])
        except pd.errors.EmptyDataError:
            existing_records = []

    initial_episodes = len(existing_records) * episodes_per_block
    evaluation_history = []
    if os.path.exists(evaluation_history_file) and os.path.getsize(evaluation_history_file) > 0:
        try:
            evaluation_history = pd.read_csv(evaluation_history_file).to_dict("records")
        except pd.errors.EmptyDataError:
            evaluation_history = []

    block = 0
    
    print(f"Starting training with initial episodes: {initial_episodes}")
    print(f"Max blocks: {max_blocks if max_blocks is not None else '∞'}")
    print(f"Episodes per block: {episodes_per_block}")
    print(f"Initial epsilon: {epsilon}")
    print(f"Alpha: {alpha}")
    print(f"Starter limits: {starter_limits}")
    print(f"Position limits: {position_limits}")
    print(f"My pick: {my_pick}")
    print(f"Keepers: {keepers}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Seasons per draft: {seasons_per_draft}")
    print(f"Q-agent drafters per draft: {num_q_agents} of {num_teams}")
    try:
        while max_blocks is None or block < max_blocks:
            block_start = time.perf_counter()
            block_rewards = []

            for _ in range(episodes_per_block):
                q_agent_teams = _sample_q_agent_teams(num_teams, num_q_agents, my_pick)
                env, draft_log, history = draft_using_q_agent(
                    Q, df, epsilon=epsilon, print_draft=False, starter_limits=starter_limits, position_limits=position_limits
                    , my_pick=my_pick, keepers=keepers, num_rounds=num_rounds,
                    sos_df=sos_df, weekly_volatility=weekly_volatility,
                    waiver_weekly_projections=waiver_weekly_projections,
                    q_agent_teams=q_agent_teams,
                )
                season_results = [env.get_season_result() for _ in range(seasons_per_draft)]
                # Each Q-controlled team learns from its own season outcome.
                team_rewards = {
                    team: float(np.mean([
                        season_result["team_rewards"][team]
                        for season_result in season_results
                    ]))
                    for team in q_agent_teams
                }
                reward = float(np.mean(list(team_rewards.values())))
                block_rewards.append(reward)

                for team, state, action in history:
                    Q[state][action] += alpha * (team_rewards[team] - Q[state][action])

            block_time = time.perf_counter() - block_start
            epsilon = max(0.10, epsilon * 0.95)
            avg_reward = sum(block_rewards) / len(block_rewards)
            unique_states = len(Q)
            total_episodes = initial_episodes + (block + 1) * episodes_per_block
            delta = avg_reward - last_avg_reward if last_avg_reward != -float("inf") else 0.0

            record = {
                "average_reward": round(avg_reward, 4),
                "unique_states": unique_states,
                "block_time_sec": round(block_time, 2),
                "total_episodes": total_episodes,
            }
            reward_history.append(record)

            max_str = f"{max_blocks:2d}" if max_blocks is not None else "∞"
            print(
                f"Block {block + 1:2d}/{max_str} | "
                f"Time: {block_time:5.2f}s | "
                f"Total Ep: {total_episodes:6d} | "
                f"Unique States: {unique_states:6d} | "
                f"Avg Reward: {avg_reward:5.2f} | "
                f"Delta: {delta:+5.2f}"
            )

            if inspect_every_blocks and (block + 1) % inspect_every_blocks == 0:
                print_training_inspection(env, draft_log, season_results[-1], block + 1, reward)
            if evaluate_every_blocks and (block + 1) % evaluate_every_blocks == 0:
                evaluation = evaluate_q_agent(
                    Q,
                    df,
                    episodes=evaluation_episodes,
                    starter_limits=starter_limits,
                    position_limits=position_limits,
                    my_pick=my_pick,
                    keepers=keepers,
                    num_rounds=num_rounds,
                    sos_df=sos_df,
                    weekly_volatility=weekly_volatility,
                    waiver_weekly_projections=waiver_weekly_projections,
                )
                evaluation["block"] = block + 1
                evaluation_history.append(evaluation)
                print(
                    f"Evaluation: reward ${evaluation['average_reward']:.2f} | "
                    f"average place {evaluation['average_placement']:.2f} | "
                    f"first {evaluation['first_place_rate']:.1%} | "
                    f"top three {evaluation['top_three_rate']:.1%} | "
                    f"playoffs {evaluation['playoff_rate']:.1%}"
                )

            with open(q_file, "wb") as handle:
                pickle.dump(Q, handle)

            full_history = existing_records + reward_history
            pd.DataFrame(full_history).to_csv(reward_history_file, index=False)
            pd.DataFrame(evaluation_history).to_csv(evaluation_history_file, index=False)
            last_avg_reward = avg_reward
            block += 1

    except KeyboardInterrupt:
        print("\n[Stopped] Training interrupted by user. Saved Q-table and CSV logs successfully.")

    return Q, reward_history


def export_season_audit(env, draft_log, season_result, output_dir):
    """Write one draft and a 17-week player-level audit for every team."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    draft_path = output_dir / "season_audit_draft.csv"
    weekly_path = output_dir / "season_audit_weekly_lineups.csv"
    draft_log.to_csv(draft_path, index=False)

    playoff_scores = season_result["playoffs"]

    rows = []
    for team_id, roster in enumerate(env.rosters):
        projections = season_result["weekly_player_projections"][team_id, :len(roster), :]
        outcomes = season_result["weekly_player_outcomes"][team_id, :len(roster), :]
        lineup_slots = season_result["weekly_lineup_slots"][team_id]
        weekly_waivers = season_result["weekly_waivers"][team_id]
        regular_weekly_scores = {
            week["week"]: week["team_scores"][team_id]
            for week in season_result["regular_season"]
        }

        for week in range(1, 18):
            team_score = (
                regular_weekly_scores[week]
                if week <= 14
                else playoff_scores.get(f"week_{week}", {}).get(team_id)
            )
            for player_index, player in enumerate(roster):
                rows.append(
                    {
                        "Team": team_id + 1,
                        "Week": week,
                        "Player": player["Player"],
                        "Position": player["Position"],
                        "Season Projection": player.get("points"),
                        "Weekly Projection": projections[player_index, week - 1],
                        "Lineup Slot": lineup_slots[player_index, week - 1],
                        "Weekly Outcome": outcomes[player_index, week - 1],
                        "Team Score": team_score,
                        "Regular Season Seed": season_result["seeds"].index(team_id) + 1,
                        "Final Placement": season_result["placement_by_team"][team_id],
                    }
                )
            waiver = weekly_waivers[week - 1]
            if waiver:
                rows.append(
                    {
                        "Team": team_id + 1,
                        "Week": week,
                        "Player": f"Waiver {waiver['position']}",
                        "Position": waiver["position"],
                        "Season Projection": None,
                        "Weekly Projection": waiver["points"],
                        "Lineup Slot": waiver["position"],
                        "Weekly Outcome": waiver["points"],
                        "Team Score": team_score,
                        "Regular Season Seed": season_result["seeds"].index(team_id) + 1,
                        "Final Placement": season_result["placement_by_team"][team_id],
                    }
                )
    # Retired H2H audit columns: Opponent Team, My Team Score, Opponent Score, Matchup Winner.
    pd.DataFrame(rows).to_csv(weekly_path, index=False)
    return draft_path, weekly_path


def export_policy_audit(env, output_dir, top_n=10):
    """Print and export the state and highest-value legal actions for each Q-max pick."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "season_audit_q_policy.csv"
    rows = []
    for decision_number, decision in enumerate(env.policy_audit, start=1):
        state_details = DraftSim.describe_state(decision["state"])
        print(f"\nQ-max decision {decision_number} | Pick {decision['state'][0] + 1}")
        print(state_details)
        for rank, (player, q_value) in enumerate(decision["ranked_q_values"][:top_n], start=1):
            selected = player == decision["selected_player"]
            print(f"  {rank:2d}. {player}: {q_value:.4f}{' [selected]' if selected else ''}")
            rows.append(
                {
                    "Decision": decision_number,
                    "State": repr(decision["state"]),
                    **state_details,
                    "Q Rank": rank,
                    "Player": player,
                    "Q Value": q_value,
                    "Selected": selected,
                }
            )
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    return audit_path


def run_season_audit(league, my_keepers=None, policy_top_n=10):
    """Run one greedy draft and save draft and weekly lineup audit CSVs."""
    config = LEAGUE_CONFIGS[league]
    sample_path = Path(config["sample_csv"])
    load_samples_from_csv(str(sample_path))
    dataframe = load_latest_player_data(league_config=league)
    q_table = build_q_table(q_file=config["q_file"])
    my_pick = 0 if league == "keeper" else None
    keepers = load_keepers(Path(config["keeper_file"]), my_keepers, my_team=1) if league == "keeper" else None
    environment, draft_log, _ = draft_using_q_agent(
        q_table,
        dataframe,
        epsilon=0.0,
        print_draft=True,
        starter_limits=config["starter_limits"],
        position_limits=config["position_limits"],
        my_pick=my_pick,
        keepers=keepers,
        num_rounds=config.get("num_rounds", 12),
        sos_df=load_weekly_matchups_sos_data(),
        waiver_weekly_projections=config["waiver_weekly_projections"],
    )
    season_result = environment.get_season_result()
    output_dir = Path(config["sample_csv"]).parent
    draft_path, weekly_path = export_season_audit(environment, draft_log, season_result, output_dir)
    policy_path = export_policy_audit(environment, output_dir, top_n=policy_top_n)
    print(f"Season audit draft: {draft_path}")
    print(f"Season audit weekly lineups: {weekly_path}")
    print(f"Season audit Q-policy decisions: {policy_path}")


def example_roster_handoff(
    Q,
    df,
    epsilon: float = 0.2,
    weeks: int = 17,
    starter_limits: dict | None = None,
    position_limits: dict | None = None,
):
    """Example flow: draft rosters in RL, then simulate a season from those rosters."""
    env, draft_log, history = draft_using_q_agent(
        Q, df, epsilon=epsilon, print_draft=False, starter_limits=starter_limits, position_limits=position_limits
    )
    season_result = simulate_league(
        team_rosters=env.rosters,
        weeks=weeks,
        my_team_index=env.my_pick,
        starter_limits=starter_limits,
    )

    return {
        "env": env,
        "draft_log": draft_log,
        "history": history,
        "season_result": season_result,
        "reward": season_result["team_rewards"][env.my_pick],
    }


def main(league: str = "ppr_fd", my_keepers: tuple[str, str] | None = None, num_q_agents: int = 1):
    if league not in LEAGUE_CONFIGS:
        raise ValueError(f"Unknown league '{league}'. Choose one of: {list(LEAGUE_CONFIGS)}")
    config = LEAGUE_CONFIGS[league]

    keepers = None
    my_pick = None
    q_file = Path(config["q_file"])
    reward_history_file = Path(config["reward_history_file"])
    evaluation_history_file = Path(config["evaluation_history_file"])
    if league == "keeper":
        if my_keepers is None:
            raise ValueError("Keeper training requires exactly two names: --my-keepers PLAYER_1 PLAYER_2")
        my_pick = 0  # Team 1 in the CSV, which is the first position in the fixed draft order.
        keepers = load_keepers(Path(config["keeper_file"]), my_keepers, my_team=my_pick + 1)
        q_file = keeper_scenario_file(q_file, my_keepers)
        reward_history_file = keeper_scenario_file(reward_history_file, my_keepers)
        evaluation_history_file = keeper_scenario_file(evaluation_history_file, my_keepers)

    sample_path = Path(config["sample_csv"])
    if sample_path.exists():
        load_samples_from_csv(str(sample_path))
    else:
        print(f"Sample file not found: {sample_path}")

    df = load_latest_player_data(league_config = league)
    q_table = build_q_table(q_file=q_file)
    train_q_agent(
        q_table,
        df,
        q_file=q_file,
        reward_history_file=reward_history_file,
        evaluation_history_file=evaluation_history_file,
        starter_limits=config.get("starter_limits"),
        position_limits=config.get("position_limits"),
        my_pick=my_pick,
        keepers=keepers,
        num_rounds=config.get("num_rounds", 12),
        sos_df=load_weekly_matchups_sos_data(),
        waiver_weekly_projections=config["waiver_weekly_projections"],
        num_q_agents=num_q_agents,
    ) 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a fantasy football draft Q-table.")
    parser.add_argument("league", choices=LEAGUE_CONFIGS)
    parser.add_argument("--my-keepers", nargs=2, metavar=("PLAYER_1", "PLAYER_2"))
    parser.add_argument("--audit-season", action="store_true", help="Run one greedy draft and export a 17-week roster audit.")
    parser.add_argument("--audit-top", type=int, default=10, help="Number of ranked Q-value options per audit decision.")
    parser.add_argument("--evaluate-slots", action="store_true", help="Evaluate the trained Q-table at every draft slot.")
    parser.add_argument("--drafts-per-slot", type=int, default=100, help="Greedy drafts to evaluate for each draft slot.")
    parser.add_argument("--seasons-per-draft", type=int, default=25, help="Independent seasons to average for each evaluated draft.")
    parser.add_argument(
        "--num-q-agents",
        type=int,
        default=1,
        help="Number of drafters controlled by the Q-agent each training draft; the rest use ADP-offset. "
             "Slots are re-sampled every episode (a fixed keeper slot is always included).",
    )
    args = parser.parse_args()

    keeper_selection = tuple(args.my_keepers) if args.my_keepers else None
    if args.evaluate_slots:
        if args.league == "keeper" and keeper_selection is None:
            parser.error("--my-keepers PLAYER_1 PLAYER_2 is required for keeper slot evaluations.")
        if args.drafts_per_slot < 1 or args.seasons_per_draft < 1:
            parser.error("--drafts-per-slot and --seasons-per-draft must both be at least 1.")
        run_draft_slot_evaluation(
            args.league,
            keeper_selection,
            args.drafts_per_slot,
            args.seasons_per_draft,
        )
    elif args.audit_season:
        if args.league == "keeper" and keeper_selection is None:
            parser.error("--my-keepers PLAYER_1 PLAYER_2 is required for keeper audits.")
        if args.audit_top < 1:
            parser.error("--audit-top must be at least 1.")
        run_season_audit(args.league, keeper_selection, args.audit_top)
    else:
        if not 1 <= args.num_q_agents <= 12:
            parser.error("--num-q-agents must be between 1 and 12.")
        print(f"Starting training for league: {args.league}...")
        main(args.league, keeper_selection, num_q_agents=args.num_q_agents)
