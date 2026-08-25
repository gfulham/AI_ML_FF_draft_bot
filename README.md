# AI / ML Draft Bot

This project is a fantasy football draft simulation and reinforcement learning experiment designed to help analyze draft decisions and give real time draft reccomendations

The core idea is to model a realistic fantasy football draft, track roster constraints, simulate player outcomes from projections, and train a Q-learning style agent to make draft picks based on state and draft context.

This repository is best understood as an experimental research project rather than a finished production product. It combines:

- fantasy football roster simulation
- rolling ADP and positional logic
- player sampling from a projection distribution
- draft-state tracking for model training
- CSV logging of draft outcomes and reward history

---

## Repository Layout

```text
AI_ML_draft_bot/
├── README.md
├── draft_sim_class.py
├── draft_helper_class.py
├── live_draft_helper.py
├── RL_draft_env.py
├── sampling_player_outcomes.py
├── test_draft_sim.py
├── data/
│   ├── all_draft_picks.csv
│   ├── full_draft_log.csv
│   ├── player_samples.csv
│   ├── reward_history.csv
│   └── sim_adp_q_agent_only.csv
├── draft_results/
│   └── historical simulation result CSVs
├── config/
│   ├── FantasyDraftBotKey.json
│   ├── google_sheets_api_info.txt
│   └── gregs_credentials.json
├── web_scraping/
│   └── web_scrape_espn_ecr_adp.ipynb
└── Notebooks/
    └── exploratory draft and RL notebooks
```

---

## Key Files

### draft_sim_class.py

This is the main draft simulation engine.

It defines the `DraftSim` class and handles:

- number of teams and rounds
- snake draft order
- roster construction per team
- positional limits and starter/flex logic
- available player pool
- rolling ADP updates as the draft progresses
- legal action filtering for available players
- draft state summaries
- reward logic tied to starter lineup value

This file is the core foundation for the RL environment and draft simulation.

### RL_draft_env.py

This file contains the reinforcement learning workflow for training a draft agent.

Key features include:

- Q-table initialization and persistence
- simulation of many draft episodes
- evaluation of draft reward using sampled player production
- exploration vs. exploitation logic with epsilon decay
- draft logs tracking each pick, player, team, and selection type
- reward history saved to CSV

This is the main training script for the model.

### draft_helper_class.py

This is a simpler helper class used for roster tracking and state construction in a lightweight way.

It is useful for:

- tracking your team's current roster
- checking if a player can be added under simple positional caps
- building a compact state representation for Q-table lookups

### live_draft_helper.py

This appears to be a support script for live draft usage or draft-board-style assistance.

It is meant to help during an active draft by interpreting current roster state and helping produce state-aware recommendations.

### sampling_player_outcomes.py

This file generates and manages player outcome distributions.

It loads projected fantasy football player projections and creates a Monte Carlo-style sample distribution for each player using uncertainty metrics like:

- points
- floor
- ceiling
- sd_pts
- uncertainty

The system approximates player variability using a Student-t style distribution and then samples outcomes for each player during draft evaluation.

This is important to help analyze risk vs reward, having too many injury prone players, or drafting ceiling vs floor players

### test_draft_sim.py

This file provides a basic sanity-check script for the draft simulator.

It verifies that:

- pick assignment works as expected
- rosters initialize correctly
- player pool data is present and valid

---

## Simulation Logic Overview

The project models a standard fantasy football draft using the following ideas:

1. Each team starts with an empty roster.
2. A snake draft order is generated across rounds.
3. The draft progresses pick by pick.
4. On each pick, the system checks which players are still available and which players can legally be added to a team based on roster rules.
5. A player is selected, removed from the available pool, and added to that team's roster.
6. The draft continues until all picks are made.
7. The system evaluates the resulting roster using player projection samples and starter lineup logic.

The draft rules include positional caps such as:

- QB: 2 total
- RB: 7 total
- WR: 7 total
- TE: 2 total

The starter constraints are also modeled:

- QB: 1 starter
- RB: 2 starters
- WR: 3 starters
- TE: 1 starter
- FLEX: 1 starter

This helps simulate a realistic lineup-building environment for reward evaluation.

---

## RL / Q-learning Approach

The project uses a reinforcement-learning-inspired draft strategy.

The training loop in `RL_draft_env.py` does roughly the following:

- initialize a Q-table for states and actions
- simulate many draft episodes
- choose actions using exploration or learned Q-values
- compare player choices against ADP/rolling ADP and legal roster constraints
- update the Q-table based on the reward received at the end of the episode
- save the Q-table and reward history for future use

The model is experimental and continues to evolve as the project develops. It is not a standard, polished fantasy optimizer, but it is a serious approach to learning from simulated draft states.

---

## Data Inputs

The project relies heavily on fantasy football player data and draft data.

### Main data sources

- player projection files in the project data folder
- draft logs in `draft_results/`
- historical results in `data/`
- Google Sheets data pulled by `gspread` for live or semi-live drafting workflows

### Important note

Several scripts reference Google Sheets, local JSON credential files, and absolute local file paths. If you clone this project to another machine, you will likely need to:

- update the Google Sheets workbook ID
- update the service account credentials path
- replace absolute file paths with your own local paths
- verify the CSV projection file names and paths

This project currently assumes a Windows local filesystem and a specific personal fantasy football setup.

---

## Setup

### Required local files

The scripts expect:

- a Google service account credential JSON file
- a fantasy football Google Sheet containing player and ADP data
- player projection sample data in the local data folder

The relevant credential paths are currently hardcoded in the scripts. You should review and update them before running the project in a new environment.

---

## Typical Workflow

### 1. Generate or load player outcome samples

Run:

```bash
python sampling_player_outcomes.py
```

This loads projection data and prepares Monte Carlo-style distributions for each player.

### 2. Run the simulation / RL trainer

Run:

```bash
python RL_draft_env.py
```

This starts the training loop that simulates multiple drafts and updates the Q-table.

### 3. Run validation checks

Run:

```bash
python test_draft_sim.py
```

This checks the draft simulation core assumptions and ensures the environment is still consistent.

---

## Output Files

The project writes and reads several output files:

- `q_table.pkl` — saved Q-table learned during training
- `reward_history.csv` — average reward history across training blocks
- `all_draft_picks.csv` — draft pick logs
- `player_samples.csv` — sampled player outcome data
- CSV files in `draft_results/` — saved simulation drafts

---

## Strengths of This Project

- realistic draft simulation with snake order and roster constraints
- probabilistic player outcome modeling based on uncertainty and projections
- rolling ADP logic integrated into the draft process
- RL training framework for draft strategy experimentation
- logs and historical outputs for iterative improvement

---

## Current Limitations

This project is still in an active experimental phase. Some important limitations to keep in mind:

- the code contains hardcoded local paths and credentials
- scripts appear to assume a very specific data pipeline and file naming convention
- some functionality is exploratory and not yet fully cleaned up or generalized
- the Q-learning workflow is a research prototype, not a production system
- there are a few old comments and draft/test code paths that may be outdated

---

## Suggested Next Improvements

If you want to continue developing this project, a good next set of improvements would be:

1. unify data loading and configuration into one settings file
2. replace hardcoded paths with environment variables or config constants
3. clean up the training flow and separate core logic from experiments
4. create a more structured package layout with modules and CLI entry points
5. add more robust tests for roster validation and draft logic
6. document the exact expected data schema for each CSV and Google Sheet
7. compare multiple strategy approaches, including pure ADP, Q-learning, and rule-based baselines

