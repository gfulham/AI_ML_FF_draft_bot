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
# Caller is responsible for calling load_samples_from_csv(...) for the correct league before using this class.
from sample_player_outcomes import sample_player_points
from season_simulation import simulate_league


class DraftSim:
    """This class simulates a fantasy football draft environment for reinforcement learning.
    It manages the draft state, including available players, team rosters, and draft order. 
    The simulation enforces positional limits and allows for dynamic updates to player availability and projected points based on the current pick number.
    """
    def __init__(self, player_pool, num_rounds, my_pick = None, num_teams=12, starter_limits=None, position_limits=None):
        # instance variables
        self.num_teams = num_teams
        self.num_rounds = num_rounds
        # can choose a pick number for when I know my pick, or random one for drafts we don't know
        self.my_pick = my_pick if my_pick is not None else random.randint(0, num_teams - 1) 
        self.current_pick_number = 0
        self.total_picks = num_teams * num_rounds
        self.player_pool = player_pool.copy()
        
        # Fast dictionary and pre-sorted positional indexing for zero-overhead simulation
        records = self.player_pool.to_dict("records")
        self._player_dict = {p["Player"]: p for p in records}
        self.pos_players = {}
        for pos in ["QB", "RB", "WR", "TE"]:
            players_in_pos = [p for p in records if p["Position"] == pos]
            self.pos_players[pos] = {
                "adp": sorted(players_in_pos, key=lambda p: p["ESPN ADP"]),
                "points": sorted(players_in_pos, key=lambda p: p["points"], reverse=True),
                "ecr": sorted(players_in_pos, key=lambda p: p["ESPN ECR"]),
                "vor": sorted(players_in_pos, key=lambda p: p["VOR"], reverse=True),
            }

        self.draft_order = list(range(num_teams))
        self.done = False

        # Defaults match a standard 2-WR-starter league; pass starter_limits (e.g. from
        # RL_draft_env.LEAGUE_CONFIGS) to override for leagues like the 3-WR keeper league.
        self.STARTER_LIMITS = starter_limits
        self.POSITION_LIMITS = position_limits
        self.FLEX_ELIGIBLE = ["RB", "WR", "TE"]
        
        self.snake_order = []
        for rnd in range(num_rounds):
            if rnd % 2 == 0:
                self.snake_order.extend(self.draft_order)
            else:
                self.snake_order.extend(self.draft_order[::-1])
        
        self.reset()

    @property
    def available_players(self):
        """Return DataFrame of currently available players for backward compatibility."""
        return self.player_pool[self.player_pool["Player"].isin(self.available_set)]

    def get_next_n_per_position(self, n=2):
        result = set()
        for pos in ["QB", "RB", "WR", "TE"]:
            pos_data = self.pos_players[pos]
            for metric in ("adp", "points", "ecr"):
                count = 0
                for p in pos_data[metric]:
                    if p["Player"] in self.available_set:
                        result.add(p["Player"])
                        count += 1
                        if count >= n:
                            break
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
        """No-op kept for backwards compatibility."""
        pass
        
    
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
        if player_name not in self.available_set:
            raise ValueError(f"Player {player_name} not available")
        player = self._player_dict[player_name]

        if not self.can_add_player(self.rosters[team_idx], player):
            raise ValueError(f"Cannot add player {player_name} to team {team_idx}")

        self.available_set.remove(player_name)
        self.rosters[team_idx].append(player)
        # Round is 0-indexed by picks-per-team so far this pick
        current_round = self.current_pick // self.num_teams
        self.draft_history[current_round].append(player)
        self.current_pick += 1
        self.done = self.current_pick >= self.total_picks
        
    def reset(self):
        # Reset the draft simulation to its initial state
        self.current_pick = 0
        self.done = False
        self.rosters = [[] for _ in range(self.num_teams)]
        self.draft_history = [[] for _ in range(self.num_rounds)]
        self.available_set = set(self._player_dict.keys())

    def get_valid_actions(self):
        # Get a list of players that can be drafted based on the current state
        return [p for p in self.player_pool["Player"] if p in self.available_set]

    def get_season_result(self):
        """Simulate and return this draft's full season result."""
        return simulate_league(
            team_rosters=self.rosters,
            my_team_index=self.my_pick,
            starter_limits=self.STARTER_LIMITS,
        )

    def get_reward(self, player_lookup):
        """Return this draft slot's numeric reward from a simulated season."""
        season_result = self.get_season_result()
        return season_result["team_rewards"][self.my_pick]

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
        
        # 2. Track my own first 6 picks by position
        # self.rosters[self.my_pick] is already in draft order,
        # so the first 6 players are my first 6 selections.
        first_6_picks = [
            p["Position"]
            for p in my_team[:6]
        ]

        # Pad with "None" so the state always has exactly 6 values.
        # This keeps the state shape consistent during early rounds.
        first_6_picks += ["None"] * (6 - len(first_6_picks))

        # 3. Rank positions by the best available VOR
        position_vor = {}

        for pos in ["QB", "RB", "WR", "TE"]:
            found = False
            for p in self.pos_players[pos]["vor"]:
                if p["Player"] in self.available_set:
                    position_vor[pos] = p["VOR"]
                    found = True
                    break
            if not found:
                position_vor[pos] = -999

        # Sort positions from highest VOR to lowest VOR
        vor_ranking = sorted(
            position_vor,
            key=position_vor.get,
            reverse=True
        )

        # Convert position names to numbers
        position_code = {
            "QB": 0,
            "RB": 1,
            "WR": 2,
            "TE": 3
        }

        vor_rank = tuple(
            position_code[pos]
            for pos in vor_ranking
        )

        return (
            self.current_pick,

            # Roster composition
            pos_counts["QB"],
            pos_counts["RB"],
            pos_counts["WR"],
            pos_counts["TE"],

            # My first 6 picks
            *first_6_picks,

            # VOR ranking
            *vor_rank,
        )
