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

from itertools import count
import random
import re
from collections import defaultdict

import numpy as np
# Sample player outcomes are used to simulate the variability in player performance, which is crucial for realistic draft simulations.
# Caller is responsible for calling load_samples_from_csv(...) for the correct league before using this class.
from sample_player_outcomes import sample_player_points
from season_simulation import simulate_league


class DraftSim:
    """This class simulates a fantasy football draft environment for reinforcement learning.
    It manages the draft state, including available players, team rosters, and draft order. 
    The simulation enforces positional limits and allows for dynamic updates to player availability and projected points based on the current pick number.
    """
    STARTER_REACH_WINDOW = 24
    STATE_POSITIONS = ("QB", "RB", "WR", "TE")

    def __init__(
        self,
        player_pool,
        num_rounds,
        my_pick=None,
        num_teams=12,
        starter_limits=None,
        position_limits=None,
        keepers=None,
        sos_df=None,
        weekly_volatility=None,
        waiver_weekly_projections=None,
    ):
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
        self._normalized_player_names = {
            self._normalize_player_name(name): name
            for name in self._player_dict
        }
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
        self.sos_df = sos_df
        self.weekly_volatility = weekly_volatility
        self.waiver_weekly_projections = waiver_weekly_projections
        
        self.snake_order = []
        for rnd in range(num_rounds):
            if rnd % 2 == 0:
                self.snake_order.extend(self.draft_order)
            else:
                self.snake_order.extend(self.draft_order[::-1])

        self.keepers = keepers or []
        self.keeper_slots = self._build_keeper_slots()
        self.reset()

    @staticmethod
    def _normalize_player_name(name):
        """Match player names across CSVs regardless of spaces or punctuation."""
        return re.sub(r"[^a-z0-9]", "", str(name).lower())

    def _build_keeper_slots(self):
        """Map each in-draft keeper cost to the pick that is forfeited."""
        keeper_slots = defaultdict(list)
        for keeper in self.keepers:
            team = keeper["team"]
            cost = keeper["cost"]
            if not 1 <= team <= self.num_teams:
                raise ValueError(f"Keeper team must be between 1 and {self.num_teams}: {team}")
            if cost < 1:
                raise ValueError(f"Keeper cost must be a positive round number: {cost}")
            if cost <= self.num_rounds:
                round_start = (cost - 1) * self.num_teams
                pick_index = self.snake_order.index(team - 1, round_start, round_start + self.num_teams)
                keeper_slots[pick_index].append(keeper)
        return keeper_slots

    def _apply_keepers(self):
        seen_players = set()
        for keeper in self.keepers:
            keeper_name = keeper["player"]
            player_name = self._normalized_player_names.get(self._normalize_player_name(keeper_name))
            team_index = keeper["team"] - 1
            if player_name in seen_players:
                raise ValueError(f"Player appears more than once in the keeper file: {keeper_name}")
            if player_name is None:
                raise ValueError(f"Keeper is missing from the player pool: {keeper_name}")
            player = self._player_dict[player_name]
            if not self.can_add_player(self.rosters[team_index], player, team_index):
                raise ValueError(f"Keeper violates roster limits for team {keeper['team']}: {player_name}")
            self.rosters[team_index].append(player)
            self.available_set.remove(player_name)
            seen_players.add(player_name)

    def _skip_keeper_slots(self):
        while self.current_pick in self.keeper_slots:
            self.current_pick += 1
        self.done = self.current_pick >= self.total_picks

    @property
    def available_players(self):
        """Return DataFrame of currently available players for backward compatibility."""
        return self.player_pool[self.player_pool["Player"].isin(self.available_set)]

    def get_next_n_per_position(self, n=2):
        # Get the next n available players for each position based on different metrics
        # Top N players for each position and metric for a potential total of 3 * 4 * n players.
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
        #E.g player with ADP 10 and ECR 8 might get an offset of 1 or 2 based on the blended metric.
        # A player with ADP of 100 and ECR of 95 might get an offset of 5 or 6 based on the blended metric.
        return max(round(np.random.normal(loc=0, scale=blended_offset)), 0)
    

    def update_rolling_adp(self):
        """No-op kept for backwards compatibility."""
        pass
        
    
    def can_add_player(self, roster, player, team_idx=None):
        """Return whether a player fits positional limits and opponent starter needs.

        Opponents prioritize missing starters only when a player who can fill that
        slot is within the configured ADP or ECR reach window. The Q-agent's team
        only observes positional limits so it can learn any draft construction strategy.
        """
        pos = player["Position"]
        return pos in self.get_draftable_positions(team_idx, roster)

    def get_draftable_positions(self, team_idx, roster=None):
        """Return positions the active team may draft at the current pick."""
        if roster is None:
            roster = self.rosters[team_idx]
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

        positions_under_limit = {
            pos
            for pos, limit in self.POSITION_LIMITS.items()
            if position_counts[pos] < limit
        }
        if team_idx == self.my_pick:
            return positions_under_limit
        starters_filled = all(
            starter_counts.get(p, 0) >= self.STARTER_LIMITS[p]
            for p in self.STARTER_LIMITS if p != "FLEX"
        )
        flex_needed = starter_counts["FLEX"] < self.STARTER_LIMITS["FLEX"]
        if starters_filled and not flex_needed:
            return positions_under_limit

        missing_starter_positions = {
            position
            for position, limit in self.STARTER_LIMITS.items()
            if position != "FLEX" and starter_counts[position] < limit
        }
        current_draft_position = self.current_pick + 1
        nearby_missing_starter = any(
            candidate["Position"] in missing_starter_positions
            or (flex_needed and candidate["Position"] in self.FLEX_ELIGIBLE)
            for candidate_name in self.available_set
            if position_counts[(candidate := self._player_dict[candidate_name])["Position"]]
            < self.POSITION_LIMITS[candidate["Position"]]
            and min(
                abs(candidate["ESPN ADP"] - current_draft_position),
                abs(candidate["ESPN ECR"] - current_draft_position),
            ) <= self.STARTER_REACH_WINDOW
        )
        if not nearby_missing_starter:
            return positions_under_limit
        needed_positions = missing_starter_positions | (
            set(self.FLEX_ELIGIBLE) if flex_needed else set()
        )
        return positions_under_limit & needed_positions
    
    
    # Perform a draft step by selecting a player for the current pick#
    # This method updates the draft state, including the rosters and available players.
    # It also checks if the draft is complete after the pick.
    def step(self, player_name):
        team_idx = self.snake_order[self.current_pick]
        if player_name not in self.available_set:
            raise ValueError(f"Player {player_name} not available")
        player = self._player_dict[player_name]

        if not self.can_add_player(self.rosters[team_idx], player, team_idx):
            raise ValueError(f"Cannot add player {player_name} to team {team_idx}")

        self.available_set.remove(player_name)
        self.rosters[team_idx].append(player)
        # Round is 0-indexed by picks-per-team so far this pick
        current_round = self.current_pick // self.num_teams
        self.draft_history[current_round].append(player)
        self.current_pick += 1
        self._skip_keeper_slots()
        
    def reset(self):
        # Reset the draft simulation to its initial state
        self.current_pick = 0
        self.done = False
        self.rosters = [[] for _ in range(self.num_teams)]
        self.draft_history = [[] for _ in range(self.num_rounds)]
        self.available_set = set(self._player_dict.keys())
        self._apply_keepers()
        self._skip_keeper_slots()

    def get_valid_actions(self):
        # Get a list of players that can be drafted based on the current state
        return [p for p in self.player_pool["Player"] if p in self.available_set]

    def _next_my_pick(self):
        """Return the next non-forfeited pick belonging to the Q-agent."""
        for pick_index in range(self.current_pick + 1, self.total_picks):
            if self.snake_order[pick_index] == self.my_pick and pick_index not in self.keeper_slots:
                return pick_index
        return self.total_picks

    @classmethod
    def describe_state(cls, state):
        """Return named, human-readable values for a state tuple."""
        return {
            "State Pick Index": state[0],
            "Roster QB": state[1],
            "Roster RB": state[2],
            "Roster WR": state[3],
            "Roster TE": state[4],
            "Roster VOR Rank (Weak to Strong)": " > ".join(cls.STATE_POSITIONS[index] for index in state[5:9]),
            "Available VOR Rank (High to Low)": " > ".join(cls.STATE_POSITIONS[index] for index in state[9:13]),
            "QB Next-Pick Scarcity": state[13],
            "RB Next-Pick Scarcity": state[14],
            "WR Next-Pick Scarcity": state[15],
            "TE Next-Pick Scarcity": state[16],
        }

    def get_season_result(self):
        """Simulate and return this draft's full season result."""
        return simulate_league(
            team_rosters=self.rosters,
            my_team_index=self.my_pick,
            sos_df=self.sos_df,
            starter_limits=self.STARTER_LIMITS,
            weekly_volatility=self.weekly_volatility,
            waiver_weekly_projections=self.waiver_weekly_projections,
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
        
        # 2. Rank the team's positional total VOR from weakest to strongest.
        roster_vor = {
            pos: sum(player["VOR"] for player in my_team if player["Position"] == pos)
            for pos in self.STATE_POSITIONS
        }
        roster_vor_ranking = sorted(self.STATE_POSITIONS, key=roster_vor.get)

        # 3. Rank positions by the best available VOR from highest to lowest.
        position_vor = {}

        for pos in self.STATE_POSITIONS:
            for p in self.pos_players[pos]["vor"]:
                if p["Player"] in self.available_set:
                    position_vor[pos] = p["VOR"]
                    break
            else:
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

        roster_vor_rank = tuple(
            position_code[pos]
            for pos in roster_vor_ranking
        )

        # Bucket players likely to be drafted before the agent's actual next snake pick.
        # Buckets avoid raw projection values producing a unique Q-table state per draft.
        next_my_pick = self._next_my_pick()
        next_pick_scarcity = []
        for pos in self.STATE_POSITIONS:
            scarcity = 0
            for player in self.pos_players[pos]["adp"]:
                if player["Player"] in self.available_set and player["ESPN ADP"] <= next_my_pick + 1:
                    scarcity += 1
                    if scarcity == 2:
                        break
            next_pick_scarcity.append(scarcity)

        return (
            self.current_pick,

            # Roster composition
            pos_counts["QB"],
            pos_counts["RB"],
            pos_counts["WR"],
            pos_counts["TE"],

            # Total roster VOR rank, best-available VOR rank, and next-pick scarcity.
            #*roster_vor_rank,
            #*vor_rank,
            *next_pick_scarcity,
        )
