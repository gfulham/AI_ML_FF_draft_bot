"""Season simulation helpers used by the draft environment.

Import this module from RL_draft_env.py and call simulate_league(...).

Pulls weekly point samples directly from sample_player_outcomes (season total
sampled per player, then split into weekly draws) so the caller no longer
needs to pre-stamp rosters with a "Sampled FP" value.
"""

import random

import pandas as pd

from sample_player_outcomes import (
    build_uniform_weekly_matchups,
    build_weekly_matchups_from_sos,
    sample_player_points,
    sample_weekly_points_for_player,
)

# Default matches a standard 2-WR-starter league; callers can pass their own starter_limits
# (e.g. from RL_draft_env.LEAGUE_CONFIGS) for leagues like the 3-WR keeper league.
STARTER_LIMITS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
FLEX_ELIGIBLE = ["RB", "WR", "TE"]


def randomize_weekly_matchups(n_weeks, n_teams):
    """Return weekly matchups by shuffling teams and pairing them up, one week at a time.
    Might play same person multiple times in a season, but fine for now"""
    if n_teams < 2:
        return []

    schedule = []
    for _ in range(n_weeks):
        teams = list(range(n_teams))
        random.shuffle(teams)

        # If there's an odd team out, it just sits out this week (no matchup).
        weekly_matchups = [(teams[i], teams[i + 1]) for i in range(0, n_teams - 1, 2)]
        schedule.append(weekly_matchups)

    return schedule


def _player_weekly_points(player, weeks, sos_df=None):
    """Sample one season's worth of weekly points for a single drafted player."""
    season_total = float(sample_player_points(pd.Series(player), n=1)[0])
    if sos_df is not None:
        weekly_matchups = build_weekly_matchups_from_sos(pd.Series(player), sos_df=sos_df, weeks=weeks)
    else:
        weekly_matchups = build_uniform_weekly_matchups(weeks=weeks, bye_week=player.get("Bye Week"))
    weekly_sims = sample_weekly_points_for_player(
        total_points=season_total,
        weekly_matchups=weekly_matchups,
        n_simulations=1,
    )
    return weekly_sims[0]  # {week_num: points, ...}


def _build_weekly_roster_points(team_roster, weeks, sos_df=None):
    """Return a list, parallel to team_roster, of each player's {week: points} dict."""
    return [_player_weekly_points(player, weeks, sos_df=sos_df) for player in team_roster]


def _optimal_lineup_score(team_roster, weekly_points, week, starter_limits=None):
    """Pick the best legal starting lineup for one week and return its total points."""
    starter_limits = starter_limits or STARTER_LIMITS
    available = [(i, player, weekly_points[i].get(week, 0.0)) for i, player in enumerate(team_roster)]
    used = set()
    total = 0.0

    for pos, limit in starter_limits.items():
        if pos == "FLEX":
            continue
        candidates = sorted(
            (c for c in available if c[1]["Position"] == pos and c[0] not in used),
            key=lambda c: c[2],
            reverse=True,
        )
        for i, _, pts in candidates[:limit]:
            total += pts
            used.add(i)

    flex_candidates = sorted(
        (c for c in available if c[1]["Position"] in FLEX_ELIGIBLE and c[0] not in used),
        key=lambda c: c[2],
        reverse=True
    )
    for i, _, pts in flex_candidates[: starter_limits["FLEX"]]:
        total += pts
        used.add(i)

    return total


def simulate_league(team_rosters, weeks=17, my_team_index=None, sos_df=None, starter_limits=None):
    """Simulate a season from drafted rosters.

    Each week, every team's optimal starting lineup is chosen from that
    week's sampled player points, and the higher-scoring lineup wins the
    matchup.

    Args:
        team_rosters: list of team rosters, where each roster is a list of player dicts.
        weeks: number of regular-season weeks to simulate.
        my_team_index: kept for caller convenience; not used in the simulation itself yet.
        sos_df: optional combined strength-of-schedule DataFrame (player_name, week,
            opponent, star_matchup_rating, opponent_rank_against_position) used to
            build real bye/matchup-strength weekly profiles per player instead of a
            uniform profile.
        starter_limits: optional per-league starting lineup requirements (e.g. from
            RL_draft_env.LEAGUE_CONFIGS["keeper"]["starter_limits"] for a 3-WR league).
            Defaults to the standard 2-WR-starter STARTER_LIMITS.

    Returns:
        dict with win counts, total points scored, and normalized rewards.
    """
    n_teams = len(team_rosters)
    if n_teams < 2:
        raise ValueError("team_rosters must contain at least 2 teams")

    matchups = randomize_weekly_matchups(weeks, n_teams)
    team_weekly_points = [_build_weekly_roster_points(roster, weeks, sos_df=sos_df) for roster in team_rosters]

    wins = {i: 0 for i in range(n_teams)}
    points_for = {i: 0.0 for i in range(n_teams)}
    for week_idx, weekly in enumerate(matchups, start=1):
        for t1, t2 in weekly:
            score1 = _optimal_lineup_score(team_rosters[t1], team_weekly_points[t1], week_idx, starter_limits=starter_limits)
            score2 = _optimal_lineup_score(team_rosters[t2], team_weekly_points[t2], week_idx, starter_limits=starter_limits)
            points_for[t1] += score1
            points_for[t2] += score2
            winner = t1 if score1 >= score2 else t2
            wins[winner] += 1

    team_rewards = {i: w / weeks for i, w in wins.items()}
    return {"standings": wins, "team_rewards": team_rewards, "points_for": points_for}


if __name__ == "__main__":
    demo_matchups = randomize_weekly_matchups(6, 4)
    print(demo_matchups)

