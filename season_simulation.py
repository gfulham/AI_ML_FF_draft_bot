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
POSITION_WEEKLY_VOLATILITY = {
    "QB": 0.20,
    "RB": 0.25,
    "WR": 0.27,
    "TE": 0.30,
}
INJURY_MULTIPLIERS = {
    "WR1": {"WR2": 1.15, "WR3": 1.08, "TE1": 1.08, "RB1": 1.03},
    "WR2": {"WR3": 1.10, "TE1": 1.05},
    "RB1": {"RB2": 1.30, "RB3": 1.10},
    "TE1": {"WR2": 1.05, "WR3": 1.03},
}
DEFAULT_WEEKLY_VOLATILITY = None
PLACEMENT_REWARDS = {
    1: 6.0,
    2: 4.0,
    3: 2.5,
    4: 2.0,
    5: 1.5,
    6: 1.0,
    7: 0.5,
    8: 0.4,
    9: 0.3,
    10: 0.2,
    11: 0.1,
    12: 0.0,
}
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


def batch_sample_roster_weekly_points(
    team_rosters,
    weeks=17,
    sos_df=None,
    weekly_volatility=DEFAULT_WEEKLY_VOLATILITY,
    return_projected_scores=False,
):
    """Vectorized weekly scores using position volatility or a global override."""
    n_teams = len(team_rosters)
    max_roster_size = max(len(r) for r in team_rosters) if team_rosters else 0
    weekly_scores = np.zeros((n_teams, max_roster_size, weeks), dtype=np.float64)
    projected_scores = np.zeros((n_teams, max_roster_size, weeks), dtype=np.float64)
    team_positions = []

    # 1. Sample 3-game injuries across all drafted players and track NFL team injuries per week
    # nfl_team_injuries[nfl_team][week_idx] -> list of injured team_pos strings (e.g. ["RB1"])
    from collections import defaultdict
    nfl_team_injuries = defaultdict(lambda: defaultdict(list))
    player_missed_weeks = {}

    for t, roster in enumerate(team_rosters):
        for i, p in enumerate(roster):
            raw_inj = p.get("Injury Prediction", 0.0)
            try:
                inj_prob = float(raw_inj) if pd.notna(raw_inj) else 0.0
            except (TypeError, ValueError):
                inj_prob = 0.0

            if inj_prob > 0.0 and rng.random() < inj_prob:
                start_w = int(rng.integers(0, max(1, weeks - 2)))
                missed = set(range(start_w, min(weeks, start_w + 3)))
                player_missed_weeks[(t, i)] = missed

                nfl_team = str(p.get("Team", "")).strip().upper()
                team_pos = str(p.get("Team Position", "")).strip().upper()
                if nfl_team and team_pos:
                    for w in missed:
                        nfl_team_injuries[nfl_team][w].append(team_pos)

    # 2. Build weekly projection and score matrices per team
    for t, roster in enumerate(team_rosters):
        n_players = len(roster)
        if n_players == 0:
            team_positions.append(np.array([], dtype=object))
            continue

        positions = np.array([str(p.get("Position", "")).upper() for p in roster], dtype=object)
        team_positions.append(positions)

        season_totals = _sample_roster_season_totals(roster)

        # Baseline matrix for health & SOS/bye
        M_base = np.ones((n_players, weeks), dtype=np.float64)
        for i, p in enumerate(roster):
            if sos_df is not None:
                matchups = build_weekly_matchups_from_sos(pd.Series(p), sos_df=sos_df, weeks=weeks)
                for w_idx, w_data in enumerate(matchups[:weeks]):
                    if w_data.get("bye", False):
                        M_base[i, w_idx] = 0.0
                    else:
                        M_base[i, w_idx] = matchup_multiplier(w_data.get("star_rating", 3), bye_week=False)
            else:
                bye = p.get("Bye Week")
                if pd.notna(bye):
                    try:
                        bye_w = int(bye)
                        if 1 <= bye_w <= weeks:
                            M_base[i, bye_w - 1] = 0.0
                    except (TypeError, ValueError):
                        pass

        total_base_mult = M_base.sum(axis=1, keepdims=True)
        safe_base_mult = np.where(total_base_mult > 0, total_base_mult, 1.0)

        # Build actual multiplier matrix M incorporating injuries & depth chart handcuff boosts
        M = M_base.copy()
        for i, p in enumerate(roster):
            nfl_team = str(p.get("Team", "")).strip().upper()
            team_pos = str(p.get("Team Position", "")).strip().upper()
            missed = player_missed_weeks.get((t, i), set())

            for w in range(weeks):
                if w in missed:
                    M[i, w] = 0.0
                elif nfl_team:
                    injured_pos_list = nfl_team_injuries[nfl_team].get(w, [])
                    for inj_pos in injured_pos_list:
                        boost_dict = INJURY_MULTIPLIERS.get(inj_pos, {})
                        if team_pos in boost_dict:
                            M[i, w] *= boost_dict[team_pos]

        projected_totals = np.array([float(player.get("points", 0.0)) for player in roster])
        projected_mu = np.where(total_base_mult > 0, (projected_totals[:, np.newaxis] * M) / safe_base_mult, 0.0)
        mu = np.where(total_base_mult > 0, (season_totals[:, np.newaxis] * M) / safe_base_mult, 0.0)

        if weekly_volatility is None:
            player_volatility = np.array(
                [POSITION_WEEKLY_VOLATILITY.get(position, 0.25) for position in positions],
                dtype=np.float64,
            )[:, np.newaxis]
        else:
            player_volatility = float(weekly_volatility)
        sigma = np.maximum(1.0, mu * player_volatility)
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
        projected_scores[t, :n_players, :] = projected_mu

    if return_projected_scores:
        return weekly_scores, team_positions, projected_scores
    return weekly_scores, team_positions


def _select_projected_lineup_slots(roster_positions, projected_scores, starter_limits):
    """Select each week's legal starters from projected scores, not realized outcomes."""
    n_players, weeks = projected_scores.shape
    if n_players == 0:
        return np.empty((0, weeks), dtype=object)

    starter_limits = starter_limits or STARTER_LIMITS
    used_mask = np.zeros((n_players, weeks), dtype=bool)
    slots = np.full((n_players, weeks), "BENCH", dtype=object)
    week_indices = np.arange(weeks)

    for pos, limit in starter_limits.items():
        if pos == "FLEX" or limit <= 0:
            continue
        pos_indices = np.where(roster_positions == pos)[0]
        if len(pos_indices) == 0:
            continue

        n_pick = min(limit, len(pos_indices))
        pos_scores = projected_scores[pos_indices, :].copy()
        pos_scores[pos_scores <= 0.0] = -1e9
        sorted_rel_indices = np.argsort(-pos_scores, axis=0)
        top_rel_indices = sorted_rel_indices[:n_pick, :]

        top_player_indices = pos_indices[top_rel_indices]
        top_scores = pos_scores[top_rel_indices, week_indices]
        selected_mask = top_scores > 0.0
        selected_players, selected_weeks = np.where(selected_mask)
        used_mask[top_player_indices[selected_players, selected_weeks], selected_weeks] = True
        slots[top_player_indices[selected_players, selected_weeks], selected_weeks] = pos

    flex_limit = starter_limits.get("FLEX", 0)
    if flex_limit > 0:
        flex_eligible_mask = np.isin(roster_positions, FLEX_ELIGIBLE)
        flex_scores = projected_scores.copy()
        flex_scores[~flex_eligible_mask, :] = -1e9
        flex_scores[used_mask] = -1e9
        flex_scores[flex_scores <= 0.0] = -1e9

        n_pick = min(flex_limit, n_players)
        sorted_flex_indices = np.argsort(-flex_scores, axis=0)
        top_flex_indices = sorted_flex_indices[:n_pick, :]

        top_scores = flex_scores[top_flex_indices, week_indices]
        selected_mask = top_scores > 0.0
        selected_players, selected_weeks = np.where(selected_mask)
        slots[top_flex_indices[selected_players, selected_weeks], selected_weeks] = "FLEX"

    return slots


def _select_weekly_waiver(lineup_slots, starter_limits, waiver_weekly_projections):
    """Use at most one deterministic waiver for a missing required starter each week."""
    weeks = lineup_slots.shape[1]
    waivers = [None] * weeks
    if not waiver_weekly_projections:
        return waivers

    for week_index in range(weeks):
        missing_positions = [
            position
            for position, limit in starter_limits.items()
            if position != "FLEX"
            and limit > 0
            and np.count_nonzero(lineup_slots[:, week_index] == position) < limit
        ]
        if missing_positions:
            position = max(missing_positions, key=lambda item: waiver_weekly_projections.get(item, 0.0))
            points = float(waiver_weekly_projections.get(position, 0.0))
            if points > 0.0:
                waivers[week_index] = {"position": position, "points": points}

    return waivers


def simulate_league(
    team_rosters,
    weeks=17,
    my_team_index=None,
    sos_df=None,
    starter_limits=None,
    weekly_volatility=DEFAULT_WEEKLY_VOLATILITY,
    waiver_weekly_projections=None,
):
    """Simulate a season from drafted rosters.

    Twelve teams accumulate lineup points across a 14-week regular season.
    The top six qualify for playoffs: seeds 1 and 2 receive Week 15 byes,
    seeds 3 through 6 compete for the other two spots, and the remaining four
    teams are ranked by average playoff score from Weeks 16 and 17.

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
        weekly_volatility: optional coefficient of variation global override. When
            omitted, uses POSITION_WEEKLY_VOLATILITY for each player's position.
        waiver_weekly_projections: optional replacement-level points by position.
            At most one waiver fills an otherwise empty required non-FLEX slot each week.

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
    # Retired H2H scheduling: Weeks 1-14 now rank every team solely by lineup points.
    # matchups = randomize_weekly_matchups(regular_season_weeks, n_teams)

    weekly_scores, team_positions, projected_scores = batch_sample_roster_weekly_points(
        team_rosters,
        weeks=weeks,
        sos_df=sos_df,
        weekly_volatility=weekly_volatility,
        return_projected_scores=True,
    )
    team_weekly_lineup_scores = np.zeros((n_teams, weeks), dtype=np.float64)
    team_weekly_lineup_slots = []
    team_weekly_waivers = []
    for t in range(n_teams):
        lineup_slots = _select_projected_lineup_slots(
            team_positions[t],
            projected_scores[t, :len(team_rosters[t]), :],
            starter_limits=starter_limits,
        )
        team_weekly_lineup_slots.append(lineup_slots)
        weekly_waivers = _select_weekly_waiver(
            lineup_slots,
            starter_limits,
            waiver_weekly_projections,
        )
        team_weekly_waivers.append(weekly_waivers)
        selected = lineup_slots != "BENCH"
        team_weekly_lineup_scores[t] = np.where(
            selected,
            weekly_scores[t, :len(team_rosters[t]), :],
            0.0,
        ).sum(axis=0)
        team_weekly_lineup_scores[t] += np.array(
            [waiver["points"] if waiver else 0.0 for waiver in weekly_waivers]
        )

    # Retired H2H regular season and elimination bracket:
    # wins = {i: 0 for i in range(n_teams)}
    # points_for = {i: 0.0 for i in range(n_teams)}
    # for week_idx, weekly in enumerate(matchups, start=1):
    #     for t1, t2 in weekly:
    #         score1 = float(team_weekly_lineup_scores[t1, week_idx - 1])
    #         score2 = float(team_weekly_lineup_scores[t2, week_idx - 1])
    #         points_for[t1] += score1
    #         points_for[t2] += score2
    #         wins[t1 if score1 >= score2 else t2] += 1

    regular_season_points_for = {
        team: float(team_weekly_lineup_scores[team, :regular_season_weeks].sum())
        for team in range(n_teams)
    }
    regular_season = [
        {
            "week": week,
            "team_scores": {
                team: float(team_weekly_lineup_scores[team, week - 1])
                for team in range(n_teams)
            },
        }
        for week in range(1, regular_season_weeks + 1)
    ]
    seeds = sorted(range(n_teams), key=lambda team: (-regular_season_points_for[team], team))

    seed_one, seed_two, seed_three, seed_four, seed_five, seed_six = seeds[:6]
    week_15_scores = {
        team: float(team_weekly_lineup_scores[team, 14])
        for team in (seed_three, seed_four, seed_five, seed_six)
    }
    week_15_ranking = sorted(
        week_15_scores,
        key=lambda team: (-week_15_scores[team], -regular_season_points_for[team], team),
    )
    advancing_teams = (seed_one, seed_two, *week_15_ranking[:2])
    eliminated_teams = week_15_ranking[2:]
    playoff_scores = {team: [] for team in advancing_teams}

    playoff_week_results = {}
    for week in (16, 17):
        week_scores = {
            team: float(team_weekly_lineup_scores[team, week - 1])
            for team in advancing_teams
        }
        playoff_week_results[f"week_{week}"] = week_scores
        for team, score in week_scores.items():
            playoff_scores[team].append(score)

    playoff_average_scores = {
        team: float(np.mean(scores)) for team, scores in playoff_scores.items()
    }
    playoff_ranking = sorted(
        advancing_teams,
        key=lambda team: (-playoff_average_scores[team], -regular_season_points_for[team], team),
    )
    final_placements = [*playoff_ranking, *eliminated_teams, *seeds[6:]]
    placement_by_team = {team: place for place, team in enumerate(final_placements, start=1)}
    team_rewards = {team: PLACEMENT_REWARDS[placement_by_team[team]] for team in range(n_teams)}

    # Retired reward system: champion/runner-up payouts plus H2H win rate and points bonuses.
    # team_rewards = {i: 0.0 for i in range(n_teams)}
    # team_rewards[champion] = 7.0
    # team_rewards[runner_up] = 5.0
    # team_rewards[team] += 0.5 * (wins[team] / regular_season_weeks)
    # team_rewards[team] += 0.5 * (regular_season_points_for[team] / average_regular_season_points)
    return {
        "standings": regular_season_points_for,
        "points_for": regular_season_points_for,
        "regular_season_points_for": regular_season_points_for,
        "regular_season": regular_season,
        "weekly_player_projections": projected_scores,
        "weekly_player_outcomes": weekly_scores,
        "weekly_lineup_slots": team_weekly_lineup_slots,
        "weekly_waivers": team_weekly_waivers,
        "win_rate_rewards": None,
        "points_rewards": None,
        "seeds": seeds,
        "playoffs": {
            "bye_teams": (seed_one, seed_two),
            "week_15": week_15_scores,
            "week_16": playoff_week_results["week_16"],
            "week_17": playoff_week_results["week_17"],
            "advancing_teams": advancing_teams,
            "eliminated_teams": eliminated_teams,
            "playoff_average_scores": playoff_average_scores,
        },
        "final_placements": final_placements,
        "placement_by_team": placement_by_team,
        "team_rewards": team_rewards,
    }


if __name__ == "__main__":
    demo_matchups = randomize_weekly_matchups(6, 4)
    print(demo_matchups)

