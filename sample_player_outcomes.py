# This file Creates many different samples of player outcomes based on projections
# These samples are used to simulate the draft and evaluate different strategies.
# Samples are preferred over a single point

import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob

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

def load_latest_player_data():
    """Load the latest player data CSV file based on the naming convention."""
    global all_samples
    all_samples.clear()  # <-- This line clears out old entries!
    folder = "data"
    prefix = "player_data_"
    files = sorted(Path(folder).glob(f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found with prefix '{prefix}' in {folder}")
    latest = files[-1]
    print(f"Loading latest player data from: {latest}")
    data = pd.read_csv(latest)
    df = pd.DataFrame(data)
    df = df.dropna(subset=["points", "floor", "ceiling"])
    return df

def load_weekly_matchups_sos_data():
    """Load the weekly matchups CSV file and combine into one large file."""
    folder_path = Path("..") / "data"
    file_pattern = os.path.join(folder_path, "*_matchups_2026_clean.csv")
    csv_files = glob.glob(file_pattern)

    # Read csv files and combine to one df
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # save to csv
    combined_df.to_csv(folder_path / "combined_matchups_2026_clean.csv", index=False)
    
    #return df
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
    """Creates n samples for a single player"""
    mu, sigma, dof = player_distribution_params(row)
    draws = t_scaled_samples(mu, sigma, dof, n)
    if clip_to_bounds:
        draws = np.clip(draws, row["floor"], row["ceiling"])
    return draws, {"mu": mu, "sigma": sigma, "dof": dof}

def create_season_player_samples():
    """Creates samples for all players in the DataFrame and stores them in the global all_samples dictionary."""
    global all_samples
    all_samples.clear()  # Clear any existing samples

    df = load_latest_player_data() 
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

def export_player_samples_csv(n_samples):
    """Export samples to a csv file just so we can see what the samples look like."""
    rows = []
    
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    output_csv = f"player_samples_{today}.csv"

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

######
# Samples for total season, distributed by matches and create a weekly point projection
# 1. Need a matchup multiplier based on star rating and opponent rank
# Range on the multiplier will be .7 to 1.3
# Formula will be total points X mutlipler/games played (17). 
# This will give the weekly expected points and still add up to the total sampled season points. 
# 2. Need to distribute total season points by matchup multiplier
# 3. Need to sample weekly points for each player based on the distribution of total season points and matchup multipliers
######
def matchup_multiplier(star_rating):
    if star_rating is None:
        return 0.0
    """Creates a mutliple of .7-1.3 based on matchup rating of 1-5"""
    
    # star_rating is 1-5, where 5 = best matchup
    # opponent_rank vs position 1-32, e.g. rank 1 is toughest defense
    # We can invert this so low rank = easier matchup
    # EG. 1 + (5 -32) / 30 = 0.1, 1 + (5 - 1) / 30 = 1.1
    # schedule_factor = 1.0 + (5 - opponent_rank) / 30.0
    # EG 1 + .15 * (1-3) = 0.7, 1 + .15 * (5-3) = 1.3
    base = 1.0 + 0.15 * (star_rating - 3)

    return max(0.35, base)  # Ensure multiplier is not negative or too low
    
def distribute_season_points_by_matchups(total_points, weekly_matchups):
    """
    total_points: season projection from sample
    returns a dict of weekly expected points
    """
    # List of multipliers for each week, 0 for bye weeks
    # List of multipliers is used later in function to create weekly projections
    multipliers = []
    for week in weekly_matchups:
        if week.get("bye", False):
            multipliers.append(0.0)
        #else add the matchup multipler of .7 - 1.3 
        else:
            multipliers.append(
                matchup_multiplier(
                    week["star_rating"],
                    bye_week=week.get("bye", False),
                )
            )

    # Error check: if all multipliers are 0, return 0 for all weeks
    total_mult = sum(multipliers)
    if total_mult == 0:
        return {w["week"]: 0.0 for w in weekly_matchups}

    # Distribute total points by matchup multipliers using a dict containing week and expected points
    weekly_expected = {}
    for week in weekly_matchups:
        if week.get("bye", False):
            weekly_expected[week["week"]] = 0.0
        else:
            m = matchup_multiplier(week["star_rating"], bye_week=week.get("bye", False))
            weekly_expected[week["week"]] = total_points * (m / total_mult)

    return weekly_expected


def sample_weekly_points_for_player(total_points, weekly_matchups, n_simulations=1000):
    weekly_expected = distribute_season_points_by_matchups(total_points, weekly_matchups)

    season_samples = []
    for _ in range(n_simulations):
        weekly_draws = {}
        for week in weekly_matchups:
            w = week["week"]
            mu = weekly_expected.get(w, 0.0)

            if week.get("bye", False):
                weekly_draws[w] = 0.0
            else:
                # Make variance smaller for low-risk players, larger for volatile players
                sigma = max(2.0, mu * 0.35)
                weekly_draws[w] = max(0.0, np.random.normal(mu, sigma))

        season_samples.append(sum(weekly_draws.values()))

    return season_samples

def export_weekly_samples_to_csv(player_name, weekly_samples, output_dir="weekly_samples"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_dir) / f"{player_name}_weekly_samples.csv"
    df = pd.DataFrame(weekly_samples)
    df.to_csv(output_csv, index=False)
    print(f"Exported weekly samples for {player_name} to {output_csv}")

def main():
    """Create samples of a players total points over a season.
    Take total points and turn into a weekly distribution based on matchup multipliers and bye weeks.
    Save some examples to csv files to reaffirm the code is working properly"""
    create_season_player_samples()
    print(f"Total players: {len(all_samples)}")
    export_player_samples_csv(n_samples=200)


if __name__ == "__main__":
    main()    





