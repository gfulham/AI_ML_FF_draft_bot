# This script is used while in a live draft
# It pulls the current draft state from a google sheet
# and using the pre-trained q-table to make recomendations for the next pick

import pickle
from py_compile import main
import pandas as pd
from collections import defaultdict
from draft_sim_class import DraftSim
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Define nested_defaultdict if needed for unpickling
# Helps the q state load correctly
def nested_defaultdict():
    return defaultdict(nested_defaultdict)

def connect_to_google_sheet():
    # Authenticate google sheet
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(r"C:\Users\onlyu\OneDrive\Fantasy Football\\Draft_strategy\FantasyDraftBotKey.json", scope)
    client = gspread.authorize(creds)
    return client
# Authenticate google sheet
#scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
#creds = ServiceAccountCredentials.from_json_keyfile_name(r"C:\Users\onlyu\OneDrive\Fantasy Football\\Draft_strategy\FantasyDraftBotKey.json", scope)
#client = gspread.authorize(creds)
#spreadsheet_id = '1QLtApXReHc0W0mR_HauEY0545rHS6oiu77XZqKb1cQw'

def load_draft_state_from_google_sheet():
    client = connect_to_google_sheet()
    sheet = client.open("2025 Fantasy Sheets").worksheet("VBD Indexed Greg")
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    # Clean Data
    df = pd.DataFrame(data).replace("", pd.NA)
    df["ESPN ADP"] = pd.to_numeric(df["ESPN ADP"], errors="coerce")
    df["Rolling ADP"] = pd.to_numeric(df["Rolling ADP"], errors="coerce")
    df = df.dropna(subset=["ESPN ADP"])
    return df

#client = connect_to_google_sheet()
#sheet = client.open("2025 Fantasy Sheets").worksheet("VBD Indexed Greg")
#data = sheet.get_all_records()
#df = pd.DataFrame(data)

#df.head(10)

# Not usedin current state, but kept for reference
"""for col in ["Rolling Percent of VBD", "Rolling Percent of VBD Normalized", "Slice of Flex Pie"]:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .astype(float)
        )"""


def get_available_players(df):
    picked_players = df[df["Pick"].notnull()]["Player"].tolist()
    available_players = df[~df["Player"].isin(picked_players)].copy()
    return available_players

def get_rosters(df):
    rosters = [[] for _ in range(12)]  # Assuming 12 teams
    for _, row in df[df["Pick"].notnull()].iterrows():
        team_idx = int(row["Team"]) - 1  # Convert 1-12 to 0-11
        rosters[team_idx].append(row)
    return rosters

def get_current_pick(rosters):
    return sum(len(team) for team in rosters)  # Total picks made so far

def get_my_team(rosters, my_pick):
    return rosters[my_pick - 1]  # Assuming my_pick is 1-indexed

def get_legal_candidates(sim, my_team):
    available_players = sim.available_players
    legal_candidates = [
        p for p in available_players["Player"]
        if sim.can_add_player(my_team, available_players[available_players["Player"] == p].iloc[0].to_dict())
    ]
    return legal_candidates

def get_best_pick(Q, state, legal_candidates):
    # returns the legal candidate with the max q value/reward for the current state
    q_vals = {a: Q[state][a] for a in legal_candidates if a in Q[state]}
    if q_vals:
        best_pick = max(q_vals, key=q_vals.get)
        return best_pick, sorted(q_vals.items(), key=lambda x: x[1], reverse=True)
    else:
        return None, None

# load the q-table from a pickle file which stores states and their corresponding Q-values for actions
def load_q_table(file_path):
    with open(file_path, 'rb') as f:
        Q = pickle.load(f)
    return Q

def main():
    df = load_draft_state_from_google_sheet()
    available_players = get_available_players(df)
    rosters = get_rosters(df)
    current_pick = get_current_pick(rosters)

    # Provides the current state of the draft
    # State including my team, available players, and current pick
    sim = DraftSim(df, 
                   num_teams=12, 
                   num_rounds=15, 
                   my_pick=4, 
                   rosters=rosters, 
                   available_players=available_players, 
                   current_pick=current_pick)

    
    my_team = get_my_team(rosters, sim.my_pick)

    # Returns the current state of the draft based on draft_sim_class.py
    state = sim.get_state()
    # Finding which players to pull q values for
    legal_candidates = get_legal_candidates(sim, my_team)
    Q = load_q_table("q_table.pkl")
    best_pick, sorted_q_vals = get_best_pick(Q, state, legal_candidates)

    print(f"My team: {my_team}")
    print(f"Current pick number: {sim.current_pick}")
    print(f"Starting rosters: {rosters}")
    print(f"My pick: {sim.my_pick}")
    print(f"Number of teams: {sim.num_teams}")

    # Print the state for debugging
    for _ in state:
        print(f"_{_}")

    # Print the recommended pick and top candidates by Q-value  
    if best_pick:
        print("Recommended pick:", best_pick)
        print("Top candidates by Q-value:")
        print(sorted_q_vals)
    # if state has not been simulated before, add fallback or pick random 
    else:
        print("No Q-values available for this state. Consider fallback logic.")

# this way it doesn't run on import, but only when executed directly
if __name__ == "__main__":
    main()
"""not_available = df[df["Pick"].notnull()]
print(f"Not available players: {not_available['Player'].tolist()}")

picked_players = df[df["Pick"].notnull()]["Player"].tolist()
#print(f"Picked players: {picked_players}")

available_players = df[~df["Player"].isin(picked_players)].copy()
#display(available.head(20))

rosters = [[] for _ in range(12)] # Assuming 12 teams
for _, row in not_available.iterrows():
    team_idx = int(row["Team"]) - 1  # Convert 1-12 to 0-11
    rosters[team_idx].append(row)
    
current_pick = sum(len(team) for team in rosters)  # Total picks made so far

# Team rosters initialized
sim = DraftSim(df, num_teams=12, 
            num_rounds=15, 
            my_pick=4, 
            rosters=rosters,
            available_players=available_players,
            current_pick=current_pick
)

#print(starting_rosters)

# Assign picked players to their teams
for _, row in not_available.iterrows():
    team_idx = int(row["Team"]) - 1  # Convert 1-12 to 0-11
    rosters[team_idx].append(row)
    
my_team = rosters[sim.my_pick - 1]  # Assuming my_pick is 1-indexed
state = sim.get_state()

# Print statements to check code
print(f"My team: {my_team}")

print(f"Current pick: {sim.current_pick}")
print(f"Starting rosters: {rosters}")
print(sim.my_pick)
print(sim.num_teams)

for _ in state:
    print(f"_{_}")

# Get available players from your sheet
available_players = sim.available_players
legal_candidates = [
    p for p in available_players["Player"]
    if sim.can_add_player(my_team, available_players[available_players["Player"] == p].iloc[0].to_dict())
]

# Get Q-values and recommend best pick
q_vals = {a: Q[state][a] for a in legal_candidates if a in Q[state]}
if q_vals:
    best_pick = max(q_vals, key=q_vals.get)
    print("Recommended pick:", best_pick)
    print("Top candidates by Q-value:")
    print(sorted(q_vals.items(), key=lambda x: x[1], reverse=True))
else:
    print("No Q-values available for this state. Consider fallback logic.")
"""
