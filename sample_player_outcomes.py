# This file Creates many different samples of player outcomes based on projections
# These samples are used to simulate the draft and evaluate different strategies.
# Samples are preferred over a single point

import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob
import re

# Dictionary to hold all player samples
all_samples = {}
rng = np.random.default_rng(42)
n_samples = 1000  # Number of samples to generate for each player as a base player outcome distribution
# From here, draft a roster, and simulate through a fantasy season x times, sampling a season distribution x times with replacement
# When a new roster is drafted, it will sample a new set of outcomes from the base distribution.
# Using the same 25 samples over and over will skew some players to look better or worse than they are. 

def standardize_name(name):
    """Standardize player names to a consistent format for dictionary keys."""
    return name.strip().replace(" ", "").lower()


def normalize_player_key(name):
    """Normalize player names for cross-source matching (strip punctuation/spaces)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

BASE_DIR = Path(__file__).resolve().parent

# Same per-league data folders used by RL_draft_env.LEAGUE_CONFIGS["<league>"]["data_folder"]
LEAGUE_DATA_FOLDERS = {
    "ppr": BASE_DIR / "data" / "ppr",
    "keeper": BASE_DIR / "data" / "keeper",
    "ppr_fd": BASE_DIR / "data" / "ppr_fd",
}


def load_latest_player_data(league_config):
    """Load the latest player data CSV file for the given league based on the naming convention."""
    global all_samples
    all_samples.clear()  # <-- This line clears out old entries!
    if league_config not in LEAGUE_DATA_FOLDERS:
        raise ValueError(f"Unknown league_config '{league_config}'. Choose one of: {list(LEAGUE_DATA_FOLDERS)}")
    folder = LEAGUE_DATA_FOLDERS[league_config]
    prefix = "player_data_"
    files = sorted(Path(folder).glob(f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found with prefix '{prefix}' in {folder}")
    latest = files[-1]
    print(f"Loading latest player data from: {latest}")
    data = pd.read_csv(latest)
    df = pd.DataFrame(data)
    df = df.rename(columns={c: c.lower() for c in df.columns if c.lower() == "uncertainty"})
    # Coerce numeric columns in case of spreadsheet export errors like "#VALUE!"
    numeric_cols = [c for c in ("points", "floor", "ceiling", "sd_pts", "uncertainty") if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_rows = df[df[numeric_cols].isna().any(axis=1)]
    if not bad_rows.empty:
        print(f"Dropping {len(bad_rows)} player(s) with bad numeric data: {bad_rows['Player'].tolist()}")
    df = df.dropna(subset=numeric_cols)
    return df

def load_weekly_matchups_sos_data():
    """Load and combine weekly SOS matchup files with inferred position labels."""
    folder_path = BASE_DIR / "data"
    csv_files = sorted(folder_path.glob("*_matchups_2026_clean.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_matchups_2026_clean.csv files found in {folder_path}")

    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        pos_guess = file.name.split("_", 1)[0].upper()
        if pos_guess in {"QB", "RB", "WR", "TE"}:
            df["Position"] = pos_guess
        else:
            df["Position"] = None
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(folder_path / "combined_matchups_2026_clean.csv", index=False)
    return combined_df

def player_distribution(row):    
    """This function estimates the mean and standard deviation of a player's projected points
    based on their floor and ceiling values."""
    # mean is projected points
    mu = row["points"]

    # sigma is the std dev
    sigma = row["sd_points"]

    # uncertainty multiplier (0.85–1.15 range)
    #r = (row["uncertainty"] - 1) / 98
    #multiplier = 0.85 + 0.30 * r
    #sigma *= multiplier

    return mu, sigma

# The floor and the ceiling are the 5% and 95%ile
def sigma_from_bounds(floor, ceiling):
    z_low, z_high = -1.645, 1.645
    return (ceiling - floor) / (z_high - z_low)  # divide by 3.29

# This affects the curve by how much we trust sd_pts vs bounds
# The more cetain we are, the smaller the std dev
def blend_sigma(sd_pts, sigma_bounds, uncertainty_rank, w_hi=0.80, w_lo=0.30):
    """
    Blend σ from analyst disagreement (sd_pts) with σ implied by bounds.
    - uncertainty_rank: 1..99 (higher = more disagreement)
    - When uncertainty is LOW  -> trust sd_pts more (weight ~ w_hi)
    - When uncertainty is HIGH -> trust bounds more (weight ~ w_lo)
    """
    r = (uncertainty_rank - 1) / 98.0  # 0..1
    w = w_hi - (w_hi - w_lo) * r       # linear: 0.80 -> 0.30 as uncertainty rises
    return w * sd_pts + (1 - w) * sigma_bounds

# Fat or skinnny tails?
def dof_from_uncertainty(uncertainty_rank, dof_hi=30, dof_lo=6):
    """
    Map uncertainty to Student-t degrees of freedom.
    - High dof (~30) ~ Normal
    - Low dof (~6)  -> heavy tails
    """
    r = (uncertainty_rank - 1) / 98.0
    return dof_hi - (dof_hi - dof_lo) * r

def t_scaled_samples(mu, sigma, dof, size):
    """
    Draw from Student-t (better for uncertainty) then scale so variance = sigma^2.
    For t_d, Var = d/(d-2) (d>2). Scale by sqrt((d-2)/d).
    """
    # draw random samples from st   andard t
    z = np.random.standard_t(df=dof, size=size)
    # scale to get correct variance 
    scale = np.sqrt((dof - 2.0) / dof)
    return mu + sigma * (z * scale)

def player_distribution_params(row):
    """
    Build μ, σ_final, dof for a player from: points, floor, ceiling, sd_pts, uncertainty.
    Respects bounds; blends per uncertainty; sets tail heaviness via dof.
    """
    mu = row["points"]
    sigma_bounds = sigma_from_bounds(row["floor"], row["ceiling"])
    sigma_blended = blend_sigma(row["sd_pts"], sigma_bounds, row["uncertainty"],
                                w_hi=0.80, w_lo=0.30)

    # Optional clamps so σ doesn"t contradict bounds badly
    lower = 0.5 * sigma_bounds
    upper = 1.5 * sigma_bounds
    sigma_final = float(np.clip(sigma_blended, lower, upper))

    dof = dof_from_uncertainty(row["uncertainty"])
    return mu, sigma_final, dof
#########################################################
# CREATE SAMPLES 


def sample_player(row, n=n_samples, clip_to_bounds=True):
    """Creates n samples for a single player
    Args: 
        row: A pandas Series representing a player with required fields (points, floor, ceiling, sd_pts, uncertainty).
        n: Number of samples to generate.
        clip_to_bounds: If True, clip the samples to the player's floor and ceiling.
    """
    mu, sigma, dof = player_distribution_params(row)
    draws = t_scaled_samples(mu, sigma, dof, n)
    if clip_to_bounds:
        draws = np.clip(draws, row["floor"], row["ceiling"])
    return draws, {"mu": mu, "sigma": sigma, "dof": dof}

def create_season_player_samples(league_config="ppr"):
    """Creates samples for all players in the DataFrame and stores them in the global all_samples dictionary."""
    global all_samples
    all_samples.clear()  # Clear any existing samples

    df = load_latest_player_data(league_config=league_config)
    for _, row in df.iterrows():
        player_name = standardize_name(row["Player"])
        samples, params = sample_player(row, n=n_samples)
        all_samples[player_name] = samples
    return all_samples

def pull_player_sample(player_row, n=1):
    """Pulls in a single number from the samples for each player, which will be used to train the model."""
    player_name = standardize_name(player_row.name)
    if player_name not in all_samples:
        print(f"Missing in all_samples: {player_name}")
        print(f"First 5 keys: {list(all_samples.keys())[:5]}")
        raise KeyError(player_name)
    #else: 
        #print(f"FOUND in all_samples: {player_name}")
    #samples = all_samples[player_row['Player']]
    #samples = all_samples[player_row.name]
    samples = all_samples[player_name]
    idx = rng.choice(len(samples), size=n, replace=True)
    return samples[idx]

def export_player_samples_csv(n_samples, league_config="ppr"):
    """To view samples for accuracy.
    Creates rows of playe samples from the all_samples dictionary
    Dictionary structure: {player_name: samples_array}
    DF rows will contain: Player, Mean, Sample_1, Sample_2, ..., Sample_n
    Writes to data/<league_config>/player_samples.csv so it matches the path
    RL_draft_env.LEAGUE_CONFIGS expects for load_samples_from_csv.
    """
    rows = []

    folder = LEAGUE_DATA_FOLDERS.get(league_config, "data")
    output_csv = f"{folder}/player_samples.csv"

    for player, samples in all_samples.items():
        # Just view the first n_samples in export
        sample_values = samples[:n_samples]
        mean_val = sample_values.mean()
        row = {"Player": player, "Mean": mean_val}
        # Add sample columns: Sample_1, Sample_2, ...
        for i, val in enumerate(sample_values, 1):
            row[f"Sample_{i}"] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Exported {len(df)} players to {output_csv}")


def load_samples_from_csv(csv_path):
    """Load pre-generated season samples from a CSV file into all_samples.

    Expected format:
    - One row per player
    - A Player column
    - One or more Sample_* columns
    """
    global all_samples

    df = pd.read_csv(csv_path)
    if "Player" not in df.columns:
        raise ValueError("CSV must contain a 'Player' column")

    sample_cols = [col for col in df.columns if col.startswith("Sample_")]
    if not sample_cols:
        raise ValueError("CSV must contain at least one Sample_* column")

    all_samples.clear()
    for _, row in df.iterrows():
        player_key = standardize_name(str(row["Player"]))
        values = pd.to_numeric(row[sample_cols], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size > 0:
            all_samples[player_key] = values

    return all_samples


def sample_player_points(player_row, n=1):
    """Return n season-point draws for a player.

    This is the compatibility function used by draft/training code.
    """
    if "Player" in player_row:
        player_name = standardize_name(str(player_row["Player"]))
    else:
        player_name = standardize_name(str(player_row.name))

    if player_name in all_samples:
        samples = all_samples[player_name]
        idx = rng.choice(len(samples), size=n, replace=True)
        return samples[idx]

    # Fallback for rows with distribution columns if prebuilt samples were not loaded.
    required = {"points", "floor", "ceiling", "sd_pts", "uncertainty"}
    if required.issubset(set(player_row.index)):
        draws, _ = sample_player(player_row, n=n)
        return draws

    raise KeyError(
        f"No samples available for player '{player_name}'. "
        "Load samples via load_samples_from_csv(...) or call create_season_player_samples()."
    )

######
# Samples for total season, distributed by matches and create a weekly point projection
# 1. Need a matchup multiplier based on star rating and opponent rank
# Range on the multiplier will be .7 to 1.3
# Formula will be total points X mutlipler/games played (17). 
# This will give the weekly expected points and still add up to the total sampled season points. 
# 2. Need to distribute total season points by matchup multiplier
# 3. Need to sample weekly points for each player based on the distribution of total season points and matchup multipliers
######
def matchup_multiplier(star_rating, bye_week=False, min_mult=0.7, max_mult=1.3):
    """Create a 0.7-1.3 style multiplier from 1-5 star matchup ratings."""
    if bye_week:
        return 0.0
    if star_rating is None:
        star_rating = 3

    # Map 1..5 into min..max linearly.
    scale = (float(star_rating) - 1.0) / 4.0
    raw = min_mult + (max_mult - min_mult) * scale
    return float(np.clip(raw, min_mult, max_mult))


def build_uniform_weekly_matchups(weeks=17, bye_week=None, default_star=3):
    """Build a simple weekly matchup profile if opponent-level data is not available."""
    profile = []
    for week in range(1, weeks + 1):
        profile.append(
            {
                "week": week,
                "star_rating": default_star,
                "bye": (bye_week == week),
            }
        )
    return profile


def build_weekly_matchups_from_sos(player_row, sos_df, weeks=17, default_star=3):
    """Build weekly matchup profile for a player from SOS data.

    Falls back to a uniform profile when no SOS rows are found.
    """
    player_position = str(player_row.get("Position", "")).upper()
    candidate_names = [
        str(player_row.get("Name Edited", "")),
        str(player_row.get("Player", "")),
    ]
    candidate_keys = {normalize_player_key(name) for name in candidate_names if name}

    sos = sos_df.copy()
    sos["_player_key"] = sos["player_name"].map(normalize_player_key)
    sos["_position"] = sos.get("Position", "").astype(str).str.upper()

    filtered = sos[sos["_player_key"].isin(candidate_keys)]
    if player_position and "_position" in filtered.columns:
        filtered = filtered[(filtered["_position"] == player_position) | (filtered["_position"] == "")]

    bye_week_fallback = player_row.get("Bye Week", None)
    if pd.notna(bye_week_fallback):
        try:
            bye_week_fallback = int(bye_week_fallback)
        except (TypeError, ValueError):
            bye_week_fallback = None
    else:
        bye_week_fallback = None

    if filtered.empty:
        return build_uniform_weekly_matchups(weeks=weeks, bye_week=bye_week_fallback, default_star=default_star)

    filtered = filtered.copy()
    filtered["week"] = pd.to_numeric(filtered["week"], errors="coerce")
    filtered = filtered.dropna(subset=["week"])
    filtered["week"] = filtered["week"].astype(int)
    filtered = filtered[filtered["week"].between(1, weeks)]

    by_week = {}
    for _, row in filtered.iterrows():
        week_num = int(row["week"])
        opponent = str(row.get("opponent", "")).upper()
        is_bye = opponent == "BYE"
        star = row.get("star_matchup_rating", default_star)
        if pd.isna(star):
            star = default_star
        by_week[week_num] = {
            "week": week_num,
            "star_rating": float(star),
            "bye": bool(is_bye),
        }

    weekly_profile = []
    for week_num in range(1, weeks + 1):
        if week_num in by_week:
            weekly_profile.append(by_week[week_num])
        else:
            weekly_profile.append(
                {
                    "week": week_num,
                    "star_rating": default_star,
                    "bye": (bye_week_fallback == week_num),
                }
            )

    return weekly_profile


def distribute_season_points_by_matchups(total_points, weekly_matchups):
    """
    Distribute one season-point draw across weeks via matchup multipliers.

    Args:
        total_points (float): Sampled season total.
        weekly_matchups (list[dict]): [{week, star_rating, bye}, ...]

    Returns:
        dict[int, float]: expected weekly means summing to total_points.
    """
    multipliers = []
    for week in weekly_matchups:
        if week.get("bye", False):
            multipliers.append(0.0)
        else:
            multipliers.append(matchup_multiplier(week.get("star_rating", 3), bye_week=False))

    total_mult = sum(multipliers)
    if total_mult == 0:
        return {w["week"]: 0.0 for w in weekly_matchups}

    weekly_expected = {}
    for week_data in weekly_matchups:
        week_num = week_data["week"]
        if week_data.get("bye", False):
            weekly_expected[week_num] = 0.0
        else:
            m = matchup_multiplier(week_data.get("star_rating", 3), bye_week=False)
            weekly_expected[week_num] = float(total_points) * (m / total_mult)

    return weekly_expected


def sample_weekly_points_for_player(
    total_points,
    weekly_matchups,
    n_simulations=1000,
    weekly_volatility=0.35,
    preserve_season_total=True,
):
    """Sample weekly points from a season total.

    Returns a list of dicts, one dict per simulation:
    [{1: pts_wk1, 2: pts_wk2, ...}, ...]
    """
    weekly_expected = distribute_season_points_by_matchups(total_points, weekly_matchups)

    weekly_simulations = []
    for _ in range(n_simulations):
        weekly_draws = {}
        for week_data in weekly_matchups:
            week_num = week_data["week"]
            mu = weekly_expected.get(week_num, 0.0)

            if week_data.get("bye", False):
                weekly_draws[week_num] = 0.0
            else:
                sigma = max(1.0, mu * weekly_volatility)
                weekly_draws[week_num] = max(0.0, float(rng.normal(mu, sigma)))

        # Keep weekly variation but force each simulation to sum to the sampled season total.
        if preserve_season_total and total_points > 0:
            realized_total = sum(weekly_draws.values())
            if realized_total > 0:
                scale = float(total_points) / realized_total
                weekly_draws = {w: pts * scale for w, pts in weekly_draws.items()}

        weekly_simulations.append(weekly_draws)

    return weekly_simulations


def sample_weekly_points_from_player_row(
    player_row,
    weekly_matchups=None,
    sos_df=None,
    weeks=17,
    n_simulations=1000,
    weekly_volatility=0.35,
    preserve_season_total=True,
):
    """Generate weekly simulations for one player by first drawing season totals."""
    if weekly_matchups is None:
        if sos_df is not None:
            weekly_matchups = build_weekly_matchups_from_sos(player_row, sos_df=sos_df, weeks=weeks)
        else:
            bye_week = player_row.get("Bye Week", None)
            if pd.notna(bye_week):
                try:
                    bye_week = int(bye_week)
                except (TypeError, ValueError):
                    bye_week = None
            else:
                bye_week = None
            weekly_matchups = build_uniform_weekly_matchups(weeks=weeks, bye_week=bye_week)

    season_totals = sample_player_points(player_row, n=n_simulations)
    simulations = []
    for season_total in season_totals:
        weekly_draw = sample_weekly_points_for_player(
            total_points=float(season_total),
            weekly_matchups=weekly_matchups,
            n_simulations=1,
            weekly_volatility=weekly_volatility,
            preserve_season_total=preserve_season_total,
        )[0]
        simulations.append(weekly_draw)

    return simulations


def export_weekly_samples_to_csv(player_name, weekly_samples, output_dir=None):
    if output_dir is None:
        output_dir = BASE_DIR / "weekly_samples"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_dir) / f"{player_name}_weekly_samples.csv"

    # Creates a dataframe from the 
    df = pd.DataFrame(weekly_samples)
    df.to_csv(output_csv, index=False)
    print(f"Exported weekly samples for {player_name} to {output_csv}")
    return output_csv

def main(league_config="ppr"):
    """Create samples of a players total points over a season.
    Take total points and turn into a weekly distribution based on matchup multipliers and bye weeks.
    Save some examples to csv files to reaffirm the code is working properly"""
    create_season_player_samples(league_config=league_config)
    print(f"Total players: {len(all_samples)}")
    export_player_samples_csv(n_samples=200, league_config=league_config)

    # Build SOS-informed weekly summaries for the top 20 projected players.
    demo_df = load_latest_player_data(league_config=league_config)
    sos_df = load_weekly_matchups_sos_data()
    top20 = demo_df.sort_values("points", ascending=False).head(20).copy()

    prediction_rows = []
    summary_rows = []
    raw_rows = []
    n_simulations = 100

    for _, player_row in top20.iterrows():
        weekly_profile = build_weekly_matchups_from_sos(player_row, sos_df=sos_df, weeks=17)
        weekly_projection = distribute_season_points_by_matchups(
            total_points=float(player_row["points"]),
            weekly_matchups=weekly_profile,
        )

        prediction = {
            "Player": player_row["Player"],
            "Position": player_row["Position"],
            "Bye Week": player_row.get("Bye Week"),
            "SeasonPointsProj": player_row["points"],
        }
        for week in range(1, 18):
            prediction[f"W{week}_pred"] = float(weekly_projection.get(week, 0.0))
        prediction_rows.append(prediction)

        weekly_sims = sample_weekly_points_from_player_row(
            player_row,
            weekly_matchups=weekly_profile,
            weeks=17,
            n_simulations=n_simulations,
        )

        sim_df = pd.DataFrame(weekly_sims)
        weekly_means = sim_df.mean().to_dict()

        summary = {
            "Player": player_row["Player"],
            "Position": player_row["Position"],
            "Bye Week": player_row.get("Bye Week"),
            "SeasonPointsProj": player_row["points"],
        }
        for week in range(1, 18):
            summary[f"W{week}_mean"] = float(weekly_means.get(week, 0.0))
        summary_rows.append(summary)

        for sim_idx, week_points in enumerate(weekly_sims, start=1):
            raw = {
                "Player": player_row["Player"],
                "Position": player_row["Position"],
                "Simulation": sim_idx,
            }
            for week in range(1, 18):
                raw[f"W{week}"] = float(week_points.get(week, 0.0))
            raw_rows.append(raw)

    prediction_df = pd.DataFrame(prediction_rows)
    summary_df = pd.DataFrame(summary_rows)
    raw_df = pd.DataFrame(raw_rows)

    output_dir = BASE_DIR / "weekly_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "top20_weekly_predictions.csv"
    summary_path = output_dir / "top20_weekly_samples_summary.csv"
    raw_path = output_dir / "top20_weekly_samples_raw.csv"

    prediction_df.to_csv(prediction_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    raw_df.to_csv(raw_path, index=False)

    print("\nTop 20 weekly predictions (SOS-based expected points):")
    print(prediction_df.to_string(index=False))
    print("\nTop 20 weekly sample summary (mean points by week):")
    print(summary_df.to_string(index=False))
    print(f"\nSaved prediction CSV: {prediction_path}")
    print(f"\nSaved summary CSV: {summary_path}")
    print(f"Saved raw CSV: {raw_path}")


if __name__ == "__main__":
    import sys

    league_arg = sys.argv[1] if len(sys.argv) > 1 else "ppr"
    main(league_arg)





