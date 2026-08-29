"""Reinforcement-learning draft training helpers.

This module should be imported for its functions, not for side effects.
Run it directly to train the Q-table.
"""

from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from draft_sim_class import DraftSim
from sample_player_outcomes import all_samples, load_samples_from_csv, sample_player_points, load_latest_player_data
from season_simulation import simulate_league

# Each league gets its own player-data subfolder, season-sample CSV, Q-table/reward file,
# and starting-lineup requirements (the keeper league starts 3 WR instead of 2).
BASE_DIR = Path(__file__).resolve().parent

LEAGUE_CONFIGS = {
    "ppr": {
        "sample_csv": BASE_DIR / "data" / "ppr" / "player_samples.csv",
        "q_file": BASE_DIR / "q_table_ppr_2026.pkl",
        "reward_history_file": BASE_DIR / "reward_history_ppr_2026.csv",
        "starter_limits": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        "position_limits": {"QB": 3, "RB": 8, "WR": 8, "TE": 3},
    },
    "keeper": {
        "sample_csv": BASE_DIR / "data" / "keeper" / "player_samples.csv",
        "q_file": BASE_DIR / "q_table_keeper_2026.pkl",
        "reward_history_file": BASE_DIR / "reward_history_keeper_2026.csv",
        "starter_limits": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1},
        "position_limits": {"QB": 3, "RB": 7, "WR": 7, "TE": 3},
    },
    "ppr_fd": {
        "sample_csv": BASE_DIR / "data" / "ppr_fd" / "player_samples.csv",
        "q_file": BASE_DIR / "q_table_ppr_fd_2026.pkl",
        "reward_history_file": BASE_DIR / "reward_history_ppr_fd_2026.csv",
        "starter_limits": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        "position_limits": {"QB": 3, "RB": 8, "WR": 8, "TE": 3},
    },
}


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
):
    """Run one draft and return the draft environment, log, and state-action history.
    DraftSim is a file / class created to help simulate a real draft"""
    env = DraftSim(df, num_rounds=12, starter_limits=starter_limits, position_limits=position_limits)
    env.reset()
    state = env.get_state()
    # set index to "Player" for quick lookups
    player_lookup = df.set_index("Player")

    rolling_adp_lookup = env.available_players.set_index("Player")
    picks_log = []
    history = []

    while not env.done:
        # Get the next candidates for the current pick and filter them based on legality
        candidate_names = env.get_next_n_per_position()

        rolling_adp_lookup = env.available_players.set_index("Player")
        
        
        team_idx = env.snake_order[env.current_pick]

        # Filter candidates to only those that are legal for the current team and pick
        legal_candidates = [
            player_name
            for player_name in candidate_names
            if player_name in player_lookup.index
            and env.can_add_player(env.rosters[team_idx], player_lookup.loc[player_name].to_dict())
        ]

        if not legal_candidates:
            raise RuntimeError(f"No legal draft candidates available for team {team_idx} at pick {env.current_pick}")

        if team_idx == env.my_pick:
            if random.random() < epsilon:
                action = random.choice(legal_candidates)
                pick_type = "Random"
            else:
                q_vals = {candidate: Q[state][candidate] for candidate in legal_candidates}
                action = max(q_vals, key=q_vals.get)
                pick_type = "Q-Max"
        else:
            sorted_candidates = sorted(
                legal_candidates,
                key=lambda player_name: rolling_adp_lookup.loc[player_name, "Rolling ADP"],
            )
            top_player = sorted_candidates[0]
            adp = rolling_adp_lookup.loc[top_player, "Rolling ADP"]
            ecr = player_lookup.loc[top_player]["ESPN ECR"]
            offset = DraftSim.get_offset_for_adp(adp, ecr)
            pick_idx = min(max(int(round(offset)), 0), len(sorted_candidates) - 1)
            action = sorted_candidates[pick_idx]
            pick_type = "ADP-Offset"

        selected_rolling_adp = rolling_adp_lookup.loc[action, "Rolling ADP"]
        env.step(action)
        sampled_points = sample_player_points(player_lookup.loc[action], n=1)[0]

        picks_log.append(
            {
                "Pick #": env.current_pick,
                "ADP": player_lookup.loc[action]["ESPN ADP"],
                "Rolling ADP": selected_rolling_adp,
                "Team": team_idx,
                "Player": action,
                "Position": player_lookup.loc[action]["Position"],
                "FP": player_lookup.loc[action]["points"],
                "Sampled FP": sampled_points,
                "Pick Type": pick_type,
            }
        )
        history.append((state, action))
        state = env.get_state()

    if print_draft:
        draft_log = pd.DataFrame(picks_log)
        print(f"My Pick: {env.my_pick}")
        for team_id in range(env.num_teams):
            print(f"\nTeam {team_id} Draft:")
            print(draft_log[draft_log["Team"] == team_id][["Player", "Position", "FP", "Sampled FP", "Pick Type"]].reset_index(drop=True))

    return env, pd.DataFrame(picks_log), history


def train_q_agent(
    Q, # Q-table for the agent
    df, # DataFrame containing player information for the draft
    alpha: float = 0.1, # Learning rate for the Q-agent meaning how much new information overrides old information
    gamma: float = 1.0, # Discount factor for future rewards
    epsilon: float = 0.3, # Exploration rate for the Q-agent
    max_blocks: int = 50, # Maximum number of training blocks meaning how many times the agent will go through the entire training process
    episodes_per_block: int = 300, # Number of episodes (draft simulations) per training block
    q_file: str = "q_table.pkl", # File path to save the Q-table
    reward_history_file: str = "reward_history.csv",
    starter_limits: dict | None = None,
    position_limits: dict | None = None,
):
    """Train the Q-table using repeated draft simulations."""
    df = pd.DataFrame(df)

    if all_samples:
        df = df[df["Player"].isin(all_samples.keys())]

    if df.empty:
        raise ValueError("No player data available for training after filtering.")

    reward_history = []
    last_avg_reward = -float("inf")

    player_lookup = df.set_index("Player")

    for block in range(max_blocks):
        block_rewards = []

        for _ in range(episodes_per_block):
            env, draft_log, history = draft_using_q_agent(
                Q, df, epsilon=epsilon, print_draft=False, starter_limits=starter_limits, position_limits=position_limits
            )
            reward = env.get_reward(player_lookup)
            block_rewards.append(reward)

            for state, action in history:
                Q[state][action] += alpha * (reward - Q[state][action])

        epsilon = max(0.02, epsilon * 0.95)
        avg_reward = sum(block_rewards) / len(block_rewards)
        reward_history.append(avg_reward)
        print(f"Block {block + 1}: Avg Reward = {avg_reward:.2f} | Δ = {avg_reward - last_avg_reward:.2f}")

        with open(q_file, "wb") as handle:
            pickle.dump(Q, handle)

        existing: list[float] = []
        if os.path.exists(reward_history_file):
            existing = pd.read_csv(reward_history_file, header=None)[0].tolist()

        full_history = existing + reward_history
        pd.Series(full_history).to_csv(reward_history_file, index=False)
        last_avg_reward = avg_reward

    return Q, reward_history


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


def main(league: str = "ppr_fd"):
    if league not in LEAGUE_CONFIGS:
        raise ValueError(f"Unknown league '{league}'. Choose one of: {list(LEAGUE_CONFIGS)}")
    config = LEAGUE_CONFIGS[league]

    sample_path = Path(config["sample_csv"])
    if sample_path.exists():
        load_samples_from_csv(str(sample_path))
    else:
        print(f"Sample file not found: {sample_path}")

    df = load_latest_player_data(league_config = league)
    q_table = build_q_table(q_file=config["q_file"])
    train_q_agent(
        q_table,
        df,
        q_file=config["q_file"],
        reward_history_file=config["reward_history_file"],
        starter_limits=config.get("starter_limits"),
        position_limits=config.get("position_limits"),
    )


if __name__ == "__main__":
    # import sys to read command line arguments for league selection
    # Can run cmd line like: python RL_draft_env.py ppr
    import sys

    # Use the league argurment
    if len(sys.argv) > 1:
        league_arg = sys.argv[1]
    else:
        raise ValueError("No league specified")

    print(f"Starting training for league: {league_arg}...")

    main(league_arg)
