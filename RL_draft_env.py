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
        "evaluation_history_file": BASE_DIR / "evaluation_history_ppr_2026.csv",
        "starter_limits": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        "position_limits": {"QB": 3, "RB": 8, "WR": 8, "TE": 3},
    },
    "keeper": {
        "sample_csv": BASE_DIR / "data" / "keeper" / "player_samples.csv",
        "q_file": BASE_DIR / "q_table_keeper_2026.pkl",
        "reward_history_file": BASE_DIR / "reward_history_keeper_2026.csv",
        "evaluation_history_file": BASE_DIR / "evaluation_history_keeper_2026.csv",
        "starter_limits": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1},
        "position_limits": {"QB": 3, "RB": 7, "WR": 7, "TE": 3},
    },
    "ppr_fd": {
        "sample_csv": BASE_DIR / "data" / "ppr_fd" / "player_samples.csv",
        "q_file": BASE_DIR / "q_table_ppr_fd_2026.pkl",
        "reward_history_file": BASE_DIR / "reward_history_ppr_fd_2026.csv",
        "evaluation_history_file": BASE_DIR / "evaluation_history_ppr_fd_2026.csv",
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
    state = env.get_state()
    player_dict = env._player_dict

    picks_log = []
    history = []

    while not env.done:
        # Get the next candidates for the current pick and filter them based on legality
        candidate_names = env.get_next_n_per_position()
        team_idx = env.snake_order[env.current_pick]

        # Filter candidates to only those that are legal for the current team and pick
        legal_candidates = [
            player_name
            for player_name in candidate_names
            if player_name in player_dict
            and env.can_add_player(env.rosters[team_idx], player_dict[player_name])
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
                key=lambda player_name: player_dict[player_name]["ESPN ADP"],
            )
            top_player = sorted_candidates[0]
            adp = env.current_pick
            ecr = player_dict[top_player]["ESPN ECR"]
            offset = DraftSim.get_offset_for_adp(adp, ecr)
            pick_idx = min(max(int(round(offset)), 0), len(sorted_candidates) - 1)
            action = sorted_candidates[pick_idx]
            pick_type = "ADP-Offset"

        if print_draft:
            sampled_points = sample_player_points(pd.Series(player_dict[action]), n=1)[0]
            picks_log.append(
                {
                    "Pick #": env.current_pick,
                    "ADP": player_dict[action]["ESPN ADP"],
                    "Rolling ADP": env.current_pick,
                    "Team": team_idx,
                    "Player": action,
                    "Position": player_dict[action]["Position"],
                    "FP": player_dict[action]["points"],
                    "Sampled FP": sampled_points,
                    "Pick Type": pick_type,
                }
            )

        env.step(action)
        if team_idx == env.my_pick:
            history.append((state, action))
        state = env.get_state()

    draft_df = pd.DataFrame(picks_log) if print_draft else pd.DataFrame()

    if print_draft:
        print(f"My Pick: {env.my_pick}")
        for team_id in range(env.num_teams):
            print(f"\nTeam {team_id} Draft:")
            print(draft_df[draft_df["Team"] == team_id][["Player", "Position", "FP", "Sampled FP", "Pick Type"]].reset_index(drop=True))

    return env, draft_df, history


def print_training_inspection(env, draft_log, season_result, block_number, reward):
    """Print one complete draft and season outcome for a training block."""
    print(f"\n{'=' * 24} Block {block_number} Inspection {'=' * 24}")
    print(f"Q-agent team: {env.my_pick} | Reward: ${reward:.2f}")
    print("\nDraft rosters:")
    for team_id in range(env.num_teams):
        roster = env.rosters[team_id]
        players = ", ".join(f"{p.get('Player', p.get('Name', ''))} ({p.get('Position', '')})" for p in roster)
        print(f"Team {team_id}: {players}")

    print("\nRegular season:")
    for weekly_result in season_result["regular_season"]:
        matchups = " | ".join(
            f"T{matchup['teams'][0]} {matchup['scores'][0]:.1f} - "
            f"T{matchup['teams'][1]} {matchup['scores'][1]:.1f} (W: T{matchup['winner']})"
            for matchup in weekly_result["matchups"]
        )
        print(f"Week {weekly_result['week']}: {matchups}")

    print("\nPlayoff seeds:")
    for seed, team_id in enumerate(season_result["seeds"], start=1):
        print(
            f"{seed}. Team {team_id}: {season_result['standings'][team_id]} wins, "
            f"{season_result['regular_season_points_for'][team_id]:.1f} points"
        )

    print("\nPlayoffs:")
    for week_name in ("week_15", "week_16"):
        print(week_name.replace("_", " ").title() + ":")
        for matchup in season_result["playoffs"][week_name]:
            teams = matchup["teams"]
            scores = matchup["scores"]
            print(f"T{teams[0]} {scores[0]:.1f} - T{teams[1]} {scores[1]:.1f} (W: T{matchup['winner']})")
    championship_scores = season_result["playoffs"]["championship_scores"]
    champion = season_result["playoffs"]["champion"]
    runner_up = season_result["playoffs"]["runner_up"]
    championship_teams = season_result["playoffs"]["championship_teams"]
    print(
        f"Week 17 Championship: T{championship_teams[0]} {championship_scores[0]:.1f} - "
        f"T{championship_teams[1]} {championship_scores[1]:.1f} "
        f"(Champion: T{champion}; Runner-up: T{runner_up})"
    )


def evaluate_q_agent(
    Q,
    df,
    episodes: int = 50,
    seed: int = 20260828,
    starter_limits: dict | None = None,
    position_limits: dict | None = None,
):
    """Evaluate the greedy draft policy without updating its Q-values."""
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        rewards = []
        payouts = []
        champions = 0
        runner_ups = 0
        playoff_appearances = 0

        for _ in range(episodes):
            env, _, _ = draft_using_q_agent(
                Q,
                df,
                epsilon=0.0,
                print_draft=False,
                starter_limits=starter_limits,
                position_limits=position_limits,
            )
            season_result = env.get_season_result()
            team_id = env.my_pick
            reward = season_result["team_rewards"][team_id]
            payout = 7.0 if season_result["playoffs"]["champion"] == team_id else 5.0 if season_result["playoffs"]["runner_up"] == team_id else 0.0
            rewards.append(reward)
            payouts.append(payout)
            champions += payout == 7.0
            runner_ups += payout == 5.0
            playoff_appearances += team_id in season_result["seeds"][:6]

        return {
            "evaluation_episodes": episodes,
            "average_reward": float(np.mean(rewards)),
            "average_payout": float(np.mean(payouts)),
            "champion_rate": champions / episodes,
            "runner_up_rate": runner_ups / episodes,
            "playoff_rate": playoff_appearances / episodes,
        }
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def train_q_agent(
    Q, # Q-table for the agent
    df, # DataFrame containing player information for the draft
    alpha: float = 0.1, # Learning rate for the Q-agent
    gamma: float = 1.0, # Discount factor for future rewards
    epsilon: float = 0.3, # Exploration rate for the Q-agent
    max_blocks: int | None = None, # Maximum number of training blocks (None = run indefinitely until Ctrl+C)
    episodes_per_block: int = 1000, # Number of episodes (draft simulations) per training block
    q_file: str = "q_table.pkl", # File path to save the Q-table
    reward_history_file: str = "reward_history.csv",
    evaluation_history_file: str = "evaluation_history.csv",
    starter_limits: dict | None = None,
    position_limits: dict | None = None,
    inspect_every_blocks: int | None = 10,
    evaluate_every_blocks: int | None = 10,
    evaluation_episodes: int = 50,
):
    """Train the Q-table using repeated draft simulations."""
    df = pd.DataFrame(df)

    if all_samples:
        df = df[df["Player"].isin(all_samples.keys())]

    if df.empty:
        raise ValueError("No player data available for training after filtering.")

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
    try:
        while max_blocks is None or block < max_blocks:
            block_start = time.perf_counter()
            block_rewards = []

            for _ in range(episodes_per_block):
                env, draft_log, history = draft_using_q_agent(
                    Q, df, epsilon=epsilon, print_draft=False, starter_limits=starter_limits, position_limits=position_limits
                )
                season_result = env.get_season_result()
                reward = season_result["team_rewards"][env.my_pick]
                block_rewards.append(reward)

                for state, action in history:
                    Q[state][action] += alpha * (reward - Q[state][action])

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
                print_training_inspection(env, draft_log, season_result, block + 1, reward)
            if evaluate_every_blocks and (block + 1) % evaluate_every_blocks == 0:
                evaluation = evaluate_q_agent(
                    Q,
                    df,
                    episodes=evaluation_episodes,
                    starter_limits=starter_limits,
                    position_limits=position_limits,
                )
                evaluation["block"] = block + 1
                evaluation_history.append(evaluation)
                print(
                    f"Evaluation: payout ${evaluation['average_payout']:.2f} | "
                    f"champion {evaluation['champion_rate']:.1%} | "
                    f"runner-up {evaluation['runner_up_rate']:.1%} | "
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
        evaluation_history_file=config["evaluation_history_file"],
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
