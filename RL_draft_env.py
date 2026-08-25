# Training Script
# Reinforcement Learning Model using Q-Learning
# This notebook will simulate many drafts in order to find the perfect draft 

# Next Additions
# * Clean up your draft sim to log how often each player is picked and what pick number?
# * Output a CSV showing the model’s favorite players (Sim-ADP)?

import pandas as pd
import random
import pickle
import os
from pathlib import Path
from collections import defaultdict
from draft_sim_class import DraftSim
from sample_player_outcomes import pull_player_sample #Pulls a single sample from 500 samples for a player
from season_simulation import simulate_league

def nested_defaultdict():
    return defaultdict(float)

# 1. Import the Q-table from a pickle file or initialize it if it doesn't exist
# 2. Load the latest player data from a CSV file so we know who is available to draft, but wont this come from another notebook?
# 3. Draft using the Q-table, but some this is training, and will include exploration (random picks) to learn about the environment. 
# 4. Sim the season for the drafted team, in another notebook, return the reward for the team, and update the q-table
def build_q_table():
    q_file = "q_table_2026.pkl"
    if os.path.exists(q_file):
        with open(q_file, "rb") as f:
            Q = pickle.load(f)
        print("Loaded existing Q-table.")
    else:
        Q = defaultdict(nested_defaultdict)
        print("Initialized new Q-table.")
    return Q

def load_latest_player_data(folder="data", prefix="player_data_"):
    """New Player data files are updated often due to player news updating thier pre season projections.
    This function loads the latest player data CSV file based on the naming convention."""
    files = sorted(Path(folder).glob(f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found with prefix '{prefix}' in {folder}")
    latest = files[-1]
    print(f"Loading latest player data from: {latest}")
    return pd.read_csv(latest)

df = load_latest_player_data()

# this is a dictionary where keys are player names and values are lists of sampled points for that player
load_samples_from_csv(r"C:\Users\onlyu\OneDrive\Fantasy Football\data\data_clean\player_weekly_projections_ppr\projections_2025_wk0.csv") 


# Q is a dictionary where keys are states and values are dictionaries of actions with their Q-values
# df is a DataFrame containing player data with columns like "Player", "Position", "FP", and "Rolling ADP"
# DraftSim is a class that simulates a draft environment
def draft_using_q_agent(Q, df, epsilon, print_draft=True):
    """Draft using a Q-learning agent.

    Args:
        Q (dict): Q-table with states as keys and action-value dictionaries as values for bot to pull from
        df (pd.DataFrame): DataFrame containing player data.
        epsilon (float): Probability of choosing a random action (exploration).
        print_draft (bool): Whether to print the draft picks.

    Returns:
        list: Log of picks made during the draft.
    """
    # The draft_sim_class_greg_wrote module 
    # Randomly select a pick position between 0 and 11
    env = DraftSim(df, num_rounds= 12)
    # the env changes state after each pick, so we need to reset it
    env.reset()
    # Capture the initial state of the draft where there are no picks and no players drafted
    state = env.get_state()
    # Create a lookup for player data for quick access which is faster than searching through the DataFrame repeatedly
    player_lookup = df.set_index("Player")
    
    #Picks made during the draft
    picks_log = []
    history = []  # To store the history of (state, action) pairs for Q-learning updates

    # As long as the current pick is less than the total number of picks, continue drafting
    while not env.done:
        # Reruns a list o available players. Naming convetion is to match reinforcement learning
        #valid_actions = env.get_valid_actions()
        #print(f"Valid actions for pick {env.current_pick}: {valid_actions}")
        
        # Potential Players to pick
        candidate_names = env.get_next_n_per_position()
        #candidate_names = candidates_df["Player"].tolist()
        
        # Debugging output to check candidate names
        ##########################################
        #print("Candidate names:", candidate_names)
        ###########################################
        team_idx = env.snake_order[env.current_pick]
        legal_candidates = [
            p for p in candidate_names
            if env.can_add_player(env.rosters[team_idx], player_lookup.loc[p].to_dict())
        ]
        # Debugging output to check legal candidates
        ##############################################################
        #print(f"Team {team_idx} legal candidates:", legal_candidates)
        ###############################################################
        
        if team_idx == env.my_pick:
        # q_vals are essentially the average end results based on picking a player in this state of the draft 
            if random.random() < epsilon:
                action = random.choice(legal_candidates)
                pick_type = "Random"
            else:
                q_vals = {a: Q[state][a] for a in legal_candidates}
                action = max(q_vals, key=q_vals.get)
                pick_type = "Q-Max"
            env.step(action)
        else:
            # Sort legal_candidates by ADP
            sorted_candidates = sorted(
                legal_candidates,
                key=lambda p: player_lookup.loc[p]["Rolling ADP"]
            )
            # OLD CODE THAT OFFSET ONLY BY PICK BUT NOT POSITION
            ###########################################################
            #adp = player_lookup.loc[sorted_candidates[0]]["Rolling ADP"]
            #offset = env.get_offset_for_adp(adp)
            ##################################################################
            
            # Get ADP + Position for the top candidate
            top_player = sorted_candidates[0]
            adp = player_lookup.loc[top_player]["Rolling ADP"]
            #pos = player_lookup.loc[top_player]["Position"]
            ecr = player_lookup.loc[top_player]["ESPN ECR"]
            # Offset is now position-aware
            offset = env.get_offset_for_adp(adp, ecr)
            
            pick_idx = min(max(int(round(offset)), 0), len(sorted_candidates) - 1)
            #pick_idx = min(adp + offset, len(sorted_candidates) - 1)
            action = sorted_candidates[int(pick_idx)]
            pick_type = "ADP-Offset"
            env.step(action)
        
        # Print the pick details
        #print(f"Pick {env.current_pick}: Team {team_idx} selects {action} (FP: {player_lookup.loc[action]['FP']}, ADP: {player_lookup.loc[action]['Rolling ADP']})")
        sampled_points = sample_player_points(player_lookup.loc[action], n=1)[0]
        picks_log.append({
            "Pick #": env.current_pick,
            "ADP": player_lookup.loc[action]["ESPN ADP"],
            "Roling ADP": player_lookup.loc[action]["Rolling ADP"],
            "Team": team_idx,
            "Player": action,
            "Position": player_lookup.loc[action]["Position"],
            "FP": player_lookup.loc[action]["FP"],
            "Sampled FP": sampled_points,
            "Pick Type": pick_type  # <-- Add this line
            
        })
        #print(f"Team {team_idx} picks {action} with FP: {player_lookup.loc[action]['FP"]}")
        history.append((state, action))  # Store the (state, action) pair for Q-learning updates
        
        state = env.get_state()

    # Optionally print the full draft board
    if print_draft:
        df_log = pd.DataFrame(picks_log)
        print(f"My Pick: {env.my_pick}")
        for team_id in range(env.num_teams):
            print(f"\nTeam {team_id} Draft:")
            print(df_log[df_log["Team"] == team_id][["Player", "Position", "FP", "Sampled FP", "Pick Type"]].reset_index(drop=True))

    return env, pd.DataFrame(picks_log), history

#alpha = 0.1 # learning rate 10% new and 90  old knowledge
#gamma = 1.0 # 
#epsilon = 0.3 # exploration rate, How often the model picks randomly
#top_k_actions = 12 # Number of players to consider drafting at each position accroding to the Rolling ADP 

def train_q_agent(Q, df, alpha=0.1, gamma=1.0, epsilon=0.3, top_k_actions=12, max_blocks=50, episodes_per_block=300):

    df = pd.DataFrame(df)  # Ensure df is a DataFrame
    # Drop any player who isn't in the all_samples dictionary
    df = df[df["Player"].isin(all_samples.keys())]


    # Load or Initialize Q-table
    q_file = "q_table.pkl"
    if os.path.exists(q_file):
        with open(q_file, "rb") as f:
            Q = pickle.load(f)
        print("Loaded existing Q-table.")
    else:
        Q = defaultdict(nested_defaultdict)
        print("Initialized new Q-table.")

def evaluate_league_reward(team_rosters, player_points_map, matchup_df):
    from season_simulation import simulate_single_season, reward_for_team

    season_result = simulate_single_season(
        team_rosters=team_rosters,
        player_total_points=player_points_map,
        matchup_df=matchup_df,
        weeks=17,
    )

    return reward_for_team(season_result, team_index=my_team_index, top_prizes={1: 1.0, 2: 0.5})

def main():
    df = load_latest_player_data()
    q_table = build_q_table()
    train_q_agent(q_table, df)

if __name__ == "__main__":
    print("Starting training...")
    main()

# Training settings
max_blocks = 50       # max 100 batches
episodes_per_block = 300
#min_improvement = 1.0      # minimum increase in avg reward to continue
reward_history = []
last_avg_reward = -float("inf")

player_lookup = df.set_index("Player")

for block in range(max_blocks):
    block_rewards = []

    for episode in range(episodes_per_block):
        env, draft_log, history = draft_using_q_agent(Q, df, print_draft=False, epsilon=epsilon)
        reward = env.get_reward(player_lookup)
        # ONLY NEEDED FOR DEBUGGING
        ##############################################################
        # Print the selected starters for this episode
        #starters = env.get_starting_lineup(env.rosters[env.my_pick], player_lookup)
        #print("Selected starters for this draft:")
        #for p in starters:
        #    print(f"{p["Player"]} ({p["Position"]})")
        ###############################################################
        block_rewards.append(reward)
        # Update Q-table using the history of (state, action) pairs
        for s, a in history:
            Q[s][a] += alpha * (reward - Q[s][a])
        
        #draft_log["Episode"] = episode  # Optionally track which episode
        #draft_log.to_csv("all_draft_picks.csv", mode="a", header=not os.path.exists("all_draft_picks.csv"), index=False)
    # Modify Epsilon
    epsilon = max(0.02, epsilon * 0.95)  # decay epsilon after each block
    # Evaluate improvement
    avg_reward = sum(block_rewards) / len(block_rewards)
    reward_history.append(avg_reward)
    print(f"Block {block + 1}: Avg Reward = {avg_reward:.2f} | Δ = {avg_reward - last_avg_reward:.2f}")

    # Save progress 
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q, f)
        existing = []
        
    if os.path.exists("reward_history.csv"):
        existing = pd.read_csv("reward_history.csv", header=None)[0].tolist()

    full_history = existing + reward_history
    pd.Series(full_history).to_csv("reward_history.csv", index=False)

    """ Stopping condition
    if avg_reward - last_avg_reward < min_improvement:
        print("Training stopped — no significant improvement.")
        break"""
        

    last_avg_reward = avg_reward


def main():
    df = load_latest_player_data()
    q_table = build_q_table()
    train_q_agent(q_table, df)

#def main():
#    df = load_data()
#    q_table = build_q_table()
#    train(q_table, df)


if __name__ == "__main__":
    main()
"""
env, draft_log, history = draft_using_q_agent(Q, df, print_draft=True, epsilon=0.1)

# Print the starters for your team
starters = env.get_starting_lineup(env.rosters[env.my_pick], player_lookup)
print("\nSelected starters for this draft:")
for p in starters:
    print(f"{p['Player']} ({p['Position']}) ({p['FP']}) ([Sampled: {p['Sampled FP']}) ")    


# Print the starters for your team
print("\nSelected starters for this draft:")
starters = env.get_starting_lineup(env.rosters[env.my_pick], player_lookup)
# Create a lookup from draft_log for sampled points
sample_lookup = draft_log.set_index("Player")["Sampled FP"].to_dict()

for p in starters:
    player_name = p["Player"]
    sampled_fp = sample_lookup.get(player_name, "N/A")
    print(f"{player_name} ({p['Position']}) ({p['FP']}) [Sampled: {sampled_fp}]")
"""



