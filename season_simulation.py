"""Season simulation helpers used by the draft environment.

Import this module from RL_draft_env.py and call simulate_league(...).

Pulls weekly point samples directly from sample_player_outcomes (season total
sampled per player, then split into weekly draws) so the caller no longer
needs to pre-stamp rosters with a "Sampled FP" value.
"""

import random

import numpy as np
import pandas as pd

from sample_player_outcomes import (
    all_samples,
    build_uniform_weekly_matchups,
    build_weekly_matchups_from_sos,
    matchup_multiplier,
    sample_player_points,
    sample_weekly_points_for_player,
    standardize_name,
)

# Default matches a standard 2-WR-starter league; callers can pass their own starter_limits
# (e.g. from RL_draft_env.LEAGUE_CONFIGS) for leagues like the 3-WR keeper league.
STARTER_LIMITS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
FLEX_ELIGIBLE = ["RB", "WR", "TE"]
rng = np.random.default_rng()


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


def _sample_roster_season_totals(roster):
    """Fast sampling of season total points for a roster without Pandas overhead."""
    totals = []
    for p in roster:
        pname = standardize_name(str(p.get("Player", p.get("Name", ""))))
        if pname in all_samples:
            samples = all_samples[pname]
            idx = rng.integers(0, len(samples))
            totals.append(samples[idx])
        else:
            totals.append(float(p.get("points", 0.0)))
    return np.array(totals, dtype=np.float64)


def batch_sample_roster_weekly_points(team_rosters, weeks=17, sos_df=None):
    """Vectorized sampling of weekly point matrices across all teams and players."""
    n_teams = len(team_rosters)
    max_roster_size = max(len(r) for r in team_rosters)
    weekly_scores = np.zeros((n_teams, max_roster_size, weeks), dtype=np.float64)
    team_positions = []

    for t, roster in enumerate(team_rosters):
        n_players = len(roster)
        if n_players == 0:
            team_positions.append(np.array([], dtype=object))
            continue

        positions = np.array([str(p.get("Position", "")).upper() for p in roster], dtype=object)
        team_positions.append(positions)

        season_totals = _sample_roster_season_totals(roster)

        M = np.ones((n_players, weeks), dtype=np.float64)
        for i, p in enumerate(roster):
            if sos_df is not None:
                matchups = build_weekly_matchups_from_sos(pd.Series(p), sos_df=sos_df, weeks=weeks)
                for w_idx, w_data in enumerate(matchups[:weeks]):
                    if w_data.get("bye", False):
                        M[i, w_idx] = 0.0
                    else:
                        M[i, w_idx] = matchup_multiplier(w_data.get("star_rating", 3), bye_week=False)
            else:
                bye = p.get("Bye Week")
                if pd.notna(bye):
                    try:
                        bye_w = int(bye)
                        if 1 <= bye_w <= weeks:
                            M[i, bye_w - 1] = 0.0
                    except (TypeError, ValueError):
                        pass

        total_mult = M.sum(axis=1, keepdims=True)
        safe_mult = np.where(total_mult > 0, total_mult, 1.0)
        mu = np.where(total_mult > 0, (season_totals[:, np.newaxis] * M) / safe_mult, 0.0)

        sigma = np.maximum(1.0, mu * 0.35)
        raw_draws = rng.normal(mu, sigma)
        raw_draws = np.maximum(0.0, raw_draws)
        raw_draws = np.where(M == 0.0, 0.0, raw_draws)

        realized_totals = raw_draws.sum(axis=1, keepdims=True)
        scale_mask = (realized_totals > 0) & (season_totals[:, np.newaxis] > 0)
        scale = np.where(
            scale_mask,
            season_totals[:, np.newaxis] / np.where(realized_totals > 0, realized_totals, 1.0),
            1.0,
        )

        weekly_scores[t, :n_players, :] = raw_draws * scale

    return weekly_scores, team_positions


def _calculate_team_weekly_lineup_scores(roster_positions, roster_scores, starter_limits):
    """Vectorized calculation of optimal starting lineup scores across all weeks for a team."""
    n_players, weeks = roster_scores.shape
    if n_players == 0:
        return np.zeros(weeks, dtype=np.float64)

    starter_limits = starter_limits or STARTER_LIMITS
    used_mask = np.zeros((n_players, weeks), dtype=bool)
    total_scores = np.zeros(weeks, dtype=np.float64)
    week_indices = np.arange(weeks)

    for pos, limit in starter_limits.items():
        if pos == "FLEX" or limit <= 0:
            continue
        pos_indices = np.where(roster_positions == pos)[0]
        if len(pos_indices) == 0:
            continue

        n_pick = min(limit, len(pos_indices))
        pos_scores = roster_scores[pos_indices, :]
        sorted_rel_indices = np.argsort(-pos_scores, axis=0)
        top_rel_indices = sorted_rel_indices[:n_pick, :]

        top_player_indices = pos_indices[top_rel_indices]
        picked_scores = roster_scores[top_player_indices, week_indices]
        total_scores += picked_scores.sum(axis=0)
        used_mask[top_player_indices, week_indices] = True

    flex_limit = starter_limits.get("FLEX", 0)
    if flex_limit > 0:
        flex_eligible_mask = np.isin(roster_positions, FLEX_ELIGIBLE)
        flex_scores = roster_scores.copy()
        flex_scores[~flex_eligible_mask, :] = -1e9
        flex_scores[used_mask] = -1e9

        n_pick = min(flex_limit, n_players)
        sorted_flex_indices = np.argsort(-flex_scores, axis=0)
        top_flex_indices = sorted_flex_indices[:n_pick, :]

        flex_picked_scores = flex_scores[top_flex_indices, week_indices]
        flex_picked_scores = np.maximum(flex_picked_scores, 0.0)
        total_scores += flex_picked_scores.sum(axis=0)

    return total_scores


def simulate_league(
    team_rosters,
    weeks=17,
    my_team_index=None,
    sos_df=None,
    starter_limits=None,
    win_rate_reward_weight=0.5,
    points_reward_weight=0.5,
):
    """Simulate a season from drafted rosters.

    Twelve teams play a 14-week regular season. The top six teams qualify
    for playoffs: seeds 1 and 2 receive Week 15 byes, seeds 3 vs. 6 and
    4 vs. 5 play in Week 15, semifinals are in Week 16, and the championship
    is in Week 17. Week 18 is not simulated.

    Args:
        team_rosters: list of team rosters, where each roster is a list of player dicts.
        weeks: total weeks to simulate; must be 17 for the configured playoff format.
        my_team_index: kept for caller convenience; rewards are available for every team.
        sos_df: optional combined strength-of-schedule DataFrame (player_name, week,
            opponent, star_matchup_rating, opponent_rank_against_position) used to
            build real bye/matchup-strength weekly profiles per player instead of a
            uniform profile.
        starter_limits: optional per-league starting lineup requirements (e.g. from
            RL_draft_env.LEAGUE_CONFIGS["keeper"]["starter_limits"] for a 3-WR league).
            Defaults to the standard 2-WR-starter STARTER_LIMITS.
        win_rate_reward_weight: reward multiplier for regular-season win percentage.
        points_reward_weight: reward multiplier for regular-season points divided by
            the league-average regular-season points.

    Returns:
        dict with regular-season standings, season points, playoff results, and rewards.
    """
    n_teams = len(team_rosters)
    if n_teams != 12:
        raise ValueError("The playoff format requires exactly 12 teams")
    if weeks != 17:
        raise ValueError("The playoff format requires 17 weeks")

    starter_limits = starter_limits or STARTER_LIMITS
    regular_season_weeks = 14
    matchups = randomize_weekly_matchups(regular_season_weeks, n_teams)

    weekly_scores, team_positions = batch_sample_roster_weekly_points(
        team_rosters, weeks=weeks, sos_df=sos_df
    )
    team_weekly_lineup_scores = np.zeros((n_teams, weeks), dtype=np.float64)
    for t in range(n_teams):
        team_weekly_lineup_scores[t] = _calculate_team_weekly_lineup_scores(
            team_positions[t], weekly_scores[t], starter_limits=starter_limits
        )

    wins = {i: 0 for i in range(n_teams)}
    points_for = {i: 0.0 for i in range(n_teams)}
    regular_season = []
    for week_idx, weekly in enumerate(matchups, start=1):
        weekly_results = []
        for t1, t2 in weekly:
            score1 = float(team_weekly_lineup_scores[t1, week_idx - 1])
            score2 = float(team_weekly_lineup_scores[t2, week_idx - 1])
            points_for[t1] += score1
            points_for[t2] += score2
            winner = t1 if score1 >= score2 else t2
            wins[winner] += 1
            weekly_results.append({"teams": (t1, t2), "scores": (score1, score2), "winner": winner})
        regular_season.append({"week": week_idx, "matchups": weekly_results})

    regular_season_points_for = points_for.copy()
    seeds = sorted(range(n_teams), key=lambda team: (-wins[team], -points_for[team], team))

    def play_matchup(team_one, team_two, week):
        score_one = float(team_weekly_lineup_scores[team_one, week - 1])
        score_two = float(team_weekly_lineup_scores[team_two, week - 1])
        points_for[team_one] += score_one
        points_for[team_two] += score_two
        return (team_one if score_one >= score_two else team_two), score_one, score_two

    seed_one, seed_two, seed_three, seed_four, seed_five, seed_six = seeds[:6]
    winner_three_six, score_three, score_six = play_matchup(seed_three, seed_six, 15)
    winner_four_five, score_four, score_five = play_matchup(seed_four, seed_five, 15)

    semifinal_one_winner, score_one, score_four_five = play_matchup(seed_one, winner_four_five, 16)
    semifinal_two_winner, score_two, score_three_six = play_matchup(seed_two, winner_three_six, 16)
    champion, championship_score_one, championship_score_two = play_matchup(
        semifinal_one_winner, semifinal_two_winner, 17
    )
    runner_up = semifinal_two_winner if champion == semifinal_one_winner else semifinal_one_winner

    team_rewards = {i: 0.0 for i in range(n_teams)}
    team_rewards[champion] = 7.0
    team_rewards[runner_up] = 5.0
    average_regular_season_points = float(np.mean(list(regular_season_points_for.values())))
    win_rate_rewards = {team: wins[team] / regular_season_weeks for team in range(n_teams)}
    points_rewards = {
        team: regular_season_points_for[team] / average_regular_season_points
        if average_regular_season_points else 0.0
        for team in range(n_teams)
    }
    for team in range(n_teams):
        team_rewards[team] += (
            win_rate_reward_weight * win_rate_rewards[team]
            + points_reward_weight * points_rewards[team]
        )
    return {
        "standings": wins,
        "points_for": points_for,
        "regular_season_points_for": regular_season_points_for,
        "regular_season": regular_season,
        "win_rate_rewards": win_rate_rewards,
        "points_rewards": points_rewards,
        "seeds": seeds,
        "playoffs": {
            "week_15": [
                {"teams": (seed_three, seed_six), "scores": (score_three, score_six), "winner": winner_three_six},
                {"teams": (seed_four, seed_five), "scores": (score_four, score_five), "winner": winner_four_five},
            ],
            "week_16": [
                {"teams": (seed_one, winner_four_five), "scores": (score_one, score_four_five), "winner": semifinal_one_winner},
                {"teams": (seed_two, winner_three_six), "scores": (score_two, score_three_six), "winner": semifinal_two_winner},
            ],
            "champion": champion,
            "runner_up": runner_up,
            "championship_teams": (semifinal_one_winner, semifinal_two_winner),
            "championship_scores": (championship_score_one, championship_score_two),
        },
        "team_rewards": team_rewards,
    }


if __name__ == "__main__":
    demo_matchups = randomize_weekly_matchups(6, 4)
    print(demo_matchups)

