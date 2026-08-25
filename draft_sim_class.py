# Engine where sim logic is used

# Python script for simulating a fantasy football draft with positional limits and rolling ADP calculations
# Used for reinforcement learning, so there are no standard strategies to follow like VBD VOR ect..
# Reinforcement learning will also not be focing the simulation of players, so we won't need that either
# RL will simulate a whole draft

# How reinforcement learning works;
# Initialize the draft simulation with parameters like number of teams, rounds, and your pick position.
# Step function that is considered a single move / action / decision
# Track the step so we know when it is done
# Reset each step to do it again

import numpy as np
import random
# Sample player outcomes are used to simulate the variability in player performance, which is crucial for realistic draft simulations.
# Load samples from CSV provides the initial player data, while sample_player_points generates random outcomes based on Min, max, and stdevs
from sample_player_outcomes import load_samples_from_csv
from sample_player_outcomes import sample_player_points

load_samples_from_csv(r"C:\Users\onlyu\OneDrive\Fantasy Football\data\data_clean\player_weekly_projections_ppr\projections_2025_wk0.csv") 

class DraftSim:
    """This class simulates a fantasy football draft environment for reinforcement learning.
    It manages the draft state, including available players, team rosters, and draft order. 
    The simulation enforces positional limits and allows for dynamic updates to player availability and projected points based on the current pick number.
    """
    def __init__(self, player_pool, num_rounds, my_pick = None, num_teams=12, ):
        # instance variables
        self.num_teams = num_teams
        # can choose a pick number for when I know my pick, or random one for drafts we don't know
        self.my_pick = my_pick if my_pick is not None else random.randint(0, num_teams - 1) 
        self.current_pick_number = 0
        self.total_picks = num_teams * num_rounds
        self.rosters = [[] for _ in range(num_teams)]
        self.player_pool = player_pool.copy()
        self.available_players = player_pool.copy()
        self.draft_order = list(range(num_teams))
        """Having both self.player_pool and self.available_players allows the environment to always have access to the original, 
        complete list of players (self.player_pool), while separately managing the dynamic list of players still available for selection 
        (self.available_players). This separation is important for resetting the draft or for referencing the full player list without 
        interference from the ongoing draft state."""
        self.done = False

        self.STARTER_LIMITS = {
            "QB": 1,
            "RB": 2,
            "WR": 3,
            "TE": 1,
            "FLEX": 1,     # can be RB, WR, or TE
        }
        
        self.POSITION_LIMITS = {
        "QB": 2,
        "RB": 7,
        "WR": 7,
        "TE": 2,
        }
        # List of positions that qualify as FLEX
        self.FLEX_ELIGIBLE = ["RB", "WR", "TE"]
        
        self.snake_order = []
        
        for rnd in range(num_rounds):
            if rnd % 2 == 0:
                self.snake_order.extend(self.draft_order)
            else:
                self.snake_order.extend(self.draft_order[::-1])

    def get_next_n_per_position(self, n=6):
        result = set()
        for pos in ["QB", "RB", "WR", "TE"]:
            # Top n by Rolling ADP
            adp_top = self.available_players[self.available_players["Position"] == pos] \
                .sort_values("Rolling ADP").head(n)["Player"].tolist()
            # Top n by FP (VBD/projection)
            vbd_top = self.available_players[self.available_players["Position"] == pos] \
                .sort_values("FP", ascending=False).head(n)["Player"].tolist()
            # Combine and deduplicate
            result.update(adp_top)
            result.update(vbd_top)
        return list(result)
    
    # No self called, so static method
    # Static method means that the method does not depend on the instance of the class (self) 
    # and can be called on the class itself.
    @staticmethod
    
    def get_offset_for_adp(adp, ecr, weight=0.5):
        offset_ecr = 0.0968 * ecr + 3.63
        offset_adp = 0.165 * adp + 0.835

        blended_offset = weight * offset_adp + (1 - weight) * offset_ecr
        return max(round(np.random.normal(loc=0, scale=blended_offset)), 0)
    

    def update_rolling_adp(self):
        """Recalculates the "Rolling ADP" for all available players
        to help bot determine the top remaining players for each position, to help the bot choose best available player """
        sorted_df = self.available_players.sort_values("ESPN ADP").reset_index(drop=True)
        sorted_df["Rolling ADP"] = range(self.current_pick_number, self.current_pick_number + len(sorted_df))
        self.available_players["Rolling ADP"] = self.available_players["Player"].map(dict(zip(sorted_df["Player"], sorted_df["Rolling ADP"])))
        
    
    def can_add_player(self, roster, player):
        """Determines if a player can be added to a roster based on positional limits and starter requirements."""
        pos = player["Position"]
        position_counts = {p: 0 for p in self.POSITION_LIMITS}
        starter_counts = {p: 0 for p in self.STARTER_LIMITS}

        # Count players currently in roster
        for p in roster:
            p_pos = p["Position"]
            position_counts[p_pos] += 1
            if p_pos in self.STARTER_LIMITS and starter_counts[p_pos] < self.STARTER_LIMITS[p_pos]:
                starter_counts[p_pos] += 1
            elif p_pos in self.FLEX_ELIGIBLE and starter_counts["FLEX"] < self.STARTER_LIMITS["FLEX"]:
                starter_counts["FLEX"] += 1

        if position_counts[pos] >= self.POSITION_LIMITS[pos]:
            return False
        if starter_counts.get(pos, 0) < self.STARTER_LIMITS.get(pos, 0):
            return True
        if pos in self.FLEX_ELIGIBLE and starter_counts["FLEX"] < self.STARTER_LIMITS["FLEX"]:
            return True
        starters_filled = all(
            starter_counts.get(p, 0) >= self.STARTER_LIMITS[p]
            for p in self.STARTER_LIMITS if p != "FLEX"
        )
        return starters_filled
    
    
    # Perform a draft step by selecting a player for the current pick#
    # This method updates the draft state, including the rosters and available players.
    # It also checks if the draft is complete after the pick.
    def step(self, player_name):
        team_idx = self.snake_order[self.current_pick]
        player_row = self.available_players[self.available_players["Player"] == player_name]
        if player_row.empty:
            raise ValueError(f"Player {player_name} not available")
        player = player_row.iloc[0].to_dict()

        if not self.can_add_player(self.rosters[team_idx], player):
            raise ValueError(f"Cannot add player {player_name} to team {team_idx}")

        self.available_players = self.available_players[self.available_players["Player"] != player_name]
        self.rosters[team_idx].append(player)
        self.current_pick += 1
        self.update_rolling_adp()
        self.done = self.current_pick >= self.total_picks
        
    def reset(self):
        # Reset the draft simulation to its initial state
        self.current_pick = 0
        self.done = False
        self.rosters = [[] for _ in range(self.num_teams)]
        self.available_players = self.player_pool.copy()
        self.update_rolling_adp()

    def get_valid_actions(self):
        # Get a list of players that can be drafted based on the current state
        return self.available_players["Player"].tolist()

    def get_reward(self, player_lookup):
        # The calcualtion of the reward helps the bot determine if the action it took was good or bad
        # Should only calcualte at the end of the draft
        my_team = self.rosters[self.my_pick]
        starters = self.get_starting_lineup(my_team, player_lookup)
        return sum(
            #sample_player_points(player_lookup.loc[p["Player"]], n=1)[0]
            np.mean(sample_player_points(player_lookup.loc[p["Player"]], n=10))
            for p in starters
        )

    def get_state(self):
        """State is very important becuase it determines what the the bot sees and how it makes decisions
        The more possible states there are, the more scenarios the bot will have to learn from,
        Downside is its easy to explode the state space so there are too many states to learn from.
        Therefore minimizing the state space is important, with things like player tiers by position
        """
        # 0. What round are we in
        
        my_team = self.rosters[self.my_pick]  # // self.num_teams
        
        # 1. Count the number of players in each position for my team
        positions = [p['Position'] for p in my_team]
        pos_counts = {pos: positions.count(pos) for pos in ["QB", "RB", "WR", "TE"]}
        
        # 2. Roster Quality by position
        pos_tiers = {pos: [] for pos in ["QB", "RB", "WR", "TE"]}
        for p in my_team:
            pos_tiers[p['Position']].append(p['Tier'])
            
        # 3. Injury risk low med high by position
        # Removeing for now since it may not be necessary for the state representation, but can be added back in if needed
        injury_risk_counts = {pos: {"Low": 0, "Medium": 0, "High": 0} for pos in ["QB", "RB", "WR", "TE"]}
        for p in my_team:
            injury_risk_counts[p['Position']][p['Injury Risk']] += 1 
            
        # 4. Count of players remaining by position but only at the max tier
        # Going to remove as well for now since it may not be necessary for the state representation, but can be added back in if needed
        remaining_pos_tiers = {pos: [] for pos in ["QB", "RB", "WR", "TE"]}
        for p in self.available_players.to_dict('records'):
            if p['Tier'] == max([pl['Tier'] for pl in self.available_players.to_dict('records') if pl['Position'] == p['Position']]):
                remaining_pos_tiers[p['Position']].append(p['Tier'])
            
        
        
        # 5. Count the number of players in each position for all other teams
        #other_teams = self.rosters[:self.my_pick] + self.rosters[self.my_pick + 1:]
        #positions = [p['Position'] for team in other_teams for p in team]
        # Using to count positions across all other teams, but this may not be necessary for the state representation
        #other_team_counts = {pos: positions.count(pos) for pos in ["QB", "RB", "WR", "TE"]}
        """other_team_counts = [
            sum(1 for p in team if p['Position'] == pos)
            for team in self.rosters if team != my_team
            for pos in ["QB", "RB", "WR", "TE"]
        ]"""
        return (
            self.current_pick,  
            pos_counts.get("QB", 0),
            pos_counts.get("RB", 0),
            pos_counts.get("WR", 0),
            pos_counts.get("TE", 0),
            tuple(pos_tiers.get("QB", [])),
            tuple(pos_tiers.get("RB", [])),
            tuple(pos_tiers.get("WR", [])),
            tuple(pos_tiers.get("TE", []))
            #tuple(injury_risk_counts.get("QB", {"Low": 0, "Medium": 0, "High": 0}).values()),
            #tuple(injury_risk_counts.get("RB", {"Low": 0, "Medium": 0, "High": 0}).values()),
            #tuple(injury_risk_counts.get("WR", {"Low": 0, "Medium": 0, "High": 0}).values()),
            #tuple(injury_risk_counts.get("TE", {"Low": 0, "Medium": 0, "High": 0}).values()),
            #tuple(remaining_pos_tiers.get("QB", [])),
            #tuple(remaining_pos_tiers.get("RB", [])),
            #tuple(remaining_pos_tiers.get("WR", [])),
            #tuple(remaining_pos_tiers.get("TE", [])),
            #other_team_counts.get("QB", 0),
            #other_team_counts.get("RB", 0), 
            #other_team_counts.get("WR", 0),
            #other_team_counts.get("TE", 0),
        )
    def get_starting_lineup(self, team, player_lookup):
        starter_limits = self.STARTER_LIMITS
        starters = []
        used_indices = set()
        
        # 1. Fill each position except FLEX
        for pos, limit in starter_limits.items():
            if pos == "FLEX":
                continue
            # Get all players of this position not already used
            pos_players = [(i, p) for i, p in enumerate(team) if p["Position"] == pos and i not in used_indices]
            # Sample and select top players based on mean of 5 samples
            if pos_players:
                pos_samples = [
                    (i, float(np.mean(sample_player_points(player_lookup.loc[p["Player"]], n=10))))
                    for i, p in pos_players
                ]
                # Sort by projected points descending
                pos_samples.sort(key=lambda x: x[1], reverse=True)
                # Select top 'limit' players
                for i, _ in pos_samples[:limit]:
                    starters.append(team[i])
                    used_indices.add(i)
        # 2. Fill FLEX
        flex_limit = starter_limits.get("FLEX", 0)
        if flex_limit > 0:
            flex_eligible = ["RB", "WR", "TE"]
            # Get all FLEX-eligible players not already used
            flex_candidates = [
                (i, p) for i, p in enumerate(team)
                if p["Position"] in flex_eligible and i not in used_indices
            ]
            # Sample and select top players based on mean of 5 samples
            if flex_candidates:
                flex_samples = [
                    (i, float(np.mean(sample_player_points(player_lookup.loc[p["Player"]], n=10))))
                    for i, p in flex_candidates
                ]
                flex_samples.sort(key=lambda x: x[1], reverse=True)
                for i, _ in flex_samples[:flex_limit]:
                    starters.append(team[i])
                    used_indices.add(i)
                    
        return starters
    
    #def get_starting_lineup(self, team, player_lookup):
    #    starter_limits = self.STARTER_LIMITS
    #    starters = []
    #    # Track used player indices to avoid duplicates
    #    used_indices = set()
    #    
    #    # 1. Fill each position except FLEX
    #    for pos, limit in starter_limits.items():
    #        if pos == "FLEX":
    #            continue
    #        # Get all players of this position not already used
    #        pos_players = [(i, p) for i, p in enumerate(team) if p["Position"] == pos and i not in used_indices]
    #        # Sample and select top players based on projected points
    #        if pos_players:
    #            pos_samples = [
    #                (i, sample_player_points(player_lookup.loc[p["Player"]], n=1)[0])
    #                for i, p in pos_players
    #            ]
    #            # Sort by projected points descending
    #            pos_samples.sort(key=lambda x: x[1], reverse=True)
    #            #  Select top 'limit' players
    #            for i, _ in pos_samples[:limit]:
    #                starters.append(team[i])
    #                used_indices.add(i)
    #    # 2. Fill FLEX
    #    flex_limit = starter_limits.get("FLEX", 0)
    #    if flex_limit > 0:
    #        flex_eligible = ["RB", "WR", "TE"]
    #        # Get all FLEX-eligible players not already used
    #        flex_candidates = [
    #            (i, p) for i, p in enumerate(team)
    #            if p["Position"] in flex_eligible and i not in used_indices
    #        ]
    #        # Sample and select top players based on projected points
    #        if flex_candidates:
    #            flex_samples = [
    #                (i, sample_player_points(player_lookup.loc[p["Player"]], n=1)[0])
    #                for i, p in flex_candidates
    #            ]
    #            # 
    #            flex_samples.sort(key=lambda x: x[1], reverse=True)
    #            for i, _ in flex_samples[:flex_limit]:
    #                starters.append(team[i])
    #                used_indices.add(i)
    #                
    #    return starters