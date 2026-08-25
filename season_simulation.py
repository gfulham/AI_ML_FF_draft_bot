"""Season simulation utilities for the fantasy draft project.

This module is intentionally separate from the RL training notebook so the
training loop can stay focused on Q-learning and draft decisions, while this
script handles:
- weekly matchup weighting
- season scoring simulation
- final standings / finish rankings
- prize payout logic for top finishers

Typical usage:
    from season_simulation import simulate_league
    results = simulate_league(team_rosters, player_points, matchup_data)
"""

from __future__ import annotations

from typing import Dict, List, Iterable, Tuple, Any
import numpy as np
import pandas as pd


def matchup_multiplier(star_rating: int | None, bye_week: bool = False) -> float:
    """Convert a 1-5 matchup grade into a scoring multiplier.

    5 = best matchup, 1 = worst matchup, BYE = 0.
    """
    if bye_week:
        return 0.0

    if star_rating is None:
        star_rating = 3

    star_rating = max(1, min(5, int(star_rating)))
    return max(0.35, 1.0 + 0.15 * (star_rating - 3))


def distribute_season_points_by_matchups(
    total_points: float,
    weekly_matchups: List[Dict[str, Any]],
) -> Dict[int, float]:
    """Allocate a season total across a player's weekly schedule.

    The total still sums to the full-season projection, but good matchups get
    a larger share of the total and bad matchups get a smaller share.
    """
    multipliers = []
    for week in weekly_matchups:
        if week.get("bye", False):
            multipliers.append(0.0)
        else:
            multipliers.append(matchup_multiplier(week.get("star_rating")))

    total_mult = sum(multipliers)
    if total_mult == 0:
        return {int(w["week"]): 0.0 for w in weekly_matchups}

    weekly_expected: Dict[int, float] = {}
    for week in weekly_matchups:
        week_num = int(week["week"])
        if week.get("bye", False):
            weekly_expected[week_num] = 0.0
        else:
            weight = matchup_multiplier(week.get("star_rating"))
            weekly_expected[week_num] = total_points * (weight / total_mult)

    return weekly_expected


def sample_weekly_points_for_player(
    total_points: float,
    weekly_matchups: List[Dict[str, Any]],
    rng: np.random.Generator | None = None,
) -> Dict[int, float]:
    """Return one weekly scoring profile for a player.

    This draws a realistic week-by-week distribution while keeping the sum of the
    weekly scores close to the player's total-season projection.
    """
    if rng is None:
        rng = np.random.default_rng()

    weekly_expected = distribute_season_points_by_matchups(total_points, weekly_matchups)

    weekly_draws: Dict[int, float] = {}
    for week in weekly_matchups:
        week_num = int(week["week"])
        mu = weekly_expected.get(week_num, 0.0)

        if week.get("bye", False):
            weekly_draws[week_num] = 0.0
        else:
            sigma = max(1.0, mu * 0.35)
            weekly_draws[week_num] = max(0.0, float(rng.normal(mu, sigma)))

    return weekly_draws


def build_player_weekly_schedule(
    player_name: str,
    matchup_df: pd.DataFrame,
    weeks: int = 17,
) -> List[Dict[str, Any]]:
    """Create a weekly schedule entry for a single player from a cleaned matchup CSV.

    The matchup CSV is expected to contain columns like:
    - player_name
    - week
    - opponent
    - star_matchup_rating
    - opponent_rank_against_position
    """
    df = matchup_df[matchup_df["player_name"].astype(str).str.lower() == player_name.lower()].copy()
    if df.empty:
        # fallback schedule: neutral matchups if no data exists
        return [
            {"week": w, "star_rating": 3, "bye": False}
            for w in range(1, weeks + 1)
        ]

    schedule = []
    for _, row in df.sort_values("week").iterrows():
        week_num = int(row["week"])
        if week_num > weeks:
            continue

        bye_flag = str(row.get("opponent", "")).strip().upper() == "BYE"
        schedule.append(
            {
                "week": week_num,
                "star_rating": int(row.get("star_matchup_rating", 3) or 3),
                "opponent": row.get("opponent", ""),
                "bye": bye_flag,
            }
        )

    if not schedule:
        return [
            {"week": w, "star_rating": 3, "bye": False}
            for w in range(1, weeks + 1)
        ]

    filled = []
    existing_weeks = {int(item["week"]) for item in schedule}
    for w in range(1, weeks + 1):
        if w in existing_weeks:
            filled.append(next(item for item in schedule if int(item["week"]) == w))
        else:
            filled.append({"week": w, "star_rating": 3, "bye": False})

    return filled


def simulate_single_season(
    team_rosters: List[List[str]],
    player_total_points: Dict[str, float],
    matchup_df: pd.DataFrame,
    weeks: int = 17,
    rng: np.random.Generator | None = None,
) -> Dict[str, Any]:
    """Simulate one fantasy season for a league of teams.

    Args:
        team_rosters: list of team rosters, each roster = list of player names
        player_total_points: mapping of player name -> season projection
        matchup_df: dataframe with weekly matchup grades
        weeks: number of weeks in the season

    Returns:
        dictionary with weekly team scores, final season totals, and standings
    """
    if rng is None:
        rng = np.random.default_rng()

    weekly_team_scores = {team_idx: {week: 0.0 for week in range(1, weeks + 1)} for team_idx in range(len(team_rosters))}
    season_totals = {team_idx: 0.0 for team_idx in range(len(team_rosters))}

    for team_idx, roster in enumerate(team_rosters):
        for week in range(1, weeks + 1):
            team_week_score = 0.0
            for player_name in roster:
                total_points = float(player_total_points.get(player_name, 0.0))
                schedule = build_player_weekly_schedule(player_name, matchup_df, weeks=weeks)
                player_weekly_scores = sample_weekly_points_for_player(
                    total_points=total_points,
                    weekly_matchups=[entry for entry in schedule if int(entry["week"]) == week],
                    rng=rng,
                )
                team_week_score += player_weekly_scores.get(week, 0.0)

            weekly_team_scores[team_idx][week] = team_week_score
            season_totals[team_idx] += team_week_score

    standings = sorted(
        season_totals.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    finish_order = [team_idx for team_idx, _ in standings]
    return {
        "weekly_team_scores": weekly_team_scores,
        "season_totals": season_totals,
        "standings": finish_order,
        "ranking": standings,
    }


def prize_payouts(num_teams: int, top_prizes: Dict[int, float] | None = None) -> Dict[int, float]:
    """Return payout amounts by finish position.

    Example: {1: 1.0, 2: 0.5}
    """
    if top_prizes is None:
        top_prizes = {1: 1.0, 2: 0.5}

    payouts = {i: 0.0 for i in range(1, num_teams + 1)}
    for finish_pos, payout in top_prizes.items():
        if finish_pos <= num_teams:
            payouts[finish_pos] = payout

    return payouts


def reward_for_team(
    season_result: Dict[str, Any],
    team_index: int,
    top_prizes: Dict[int, float] | None = None,
) -> float:
    """Return reward for a specific team based on league finish.

    This is useful as the RL reward signal: first place = highest reward,
    second place = smaller positive reward, and lower placements = 0.
    """
    payouts = prize_payouts(len(season_result["season_totals"]), top_prizes)
    standings = season_result["standings"]
    rank_pos = standings.index(team_index) + 1
    return payouts.get(rank_pos, 0.0)


def simulate_league(
    team_rosters: List[List[str]],
    player_total_points: Dict[str, float],
    matchup_df: pd.DataFrame,
    weeks: int = 17,
    num_simulations: int = 100,
    top_prizes: Dict[int, float] | None = None,
) -> List[Dict[str, Any]]:
    """Run one or many season simulations and return league results.

    This is the high-level function the RL notebook can import and call.
    """
    if top_prizes is None:
        top_prizes = {1: 1.0, 2: 0.5}

    results = []
    for sim in range(num_simulations):
        season_result = simulate_single_season(
            team_rosters=team_rosters,
            player_total_points=player_total_points,
            matchup_df=matchup_df,
            weeks=weeks,
            rng=np.random.default_rng(sim + 1),
        )

        rewards = {
            team_idx: reward_for_team(season_result, team_idx, top_prizes)
            for team_idx in range(len(team_rosters))
        }

        results.append(
            {
                "simulation": sim + 1,
                "standings": season_result["standings"],
                "season_totals": season_result["season_totals"],
                "weekly_team_scores": season_result["weekly_team_scores"],
                "rewards": rewards,
            }
        )

    return results


if __name__ == "__main__":
    # Example usage: a tiny toy league for sanity-checking the logic.
    sample_matchup_df = pd.DataFrame(
        [
            {"player_name": "qb1", "week": 1, "opponent": "DAL", "star_matchup_rating": 5},
            {"player_name": "qb1", "week": 2, "opponent": "NYG", "star_matchup_rating": 2},
            {"player_name": "qb1", "week": 3, "opponent": "BYE", "star_matchup_rating": 3},
            {"player_name": "qb2", "week": 1, "opponent": "PHI", "star_matchup_rating": 1},
            {"player_name": "qb2", "week": 2, "opponent": "WAS", "star_matchup_rating": 4},
            {"player_name": "qb2", "week": 3, "opponent": "MIN", "star_matchup_rating": 3},
            {"player_name": "rb1", "week": 1, "opponent": "ATL", "star_matchup_rating": 4},
            {"player_name": "rb1", "week": 2, "opponent": "CAR", "star_matchup_rating": 3},
            {"player_name": "rb1", "week": 3, "opponent": "TB", "star_matchup_rating": 5},
        ]
    )

    player_total_points = {
        "qb1": 260,
        "qb2": 240,
        "rb1": 180,
    }

    team_rosters = [
        ["qb1", "rb1"],
        ["qb2", "rb1"],
    ]

    sim = simulate_single_season(
        team_rosters=team_rosters,
        player_total_points=player_total_points,
        matchup_df=sample_matchup_df,
        weeks=3,
    )
    print(sim["season_totals"])
    print(sim["standings"])
