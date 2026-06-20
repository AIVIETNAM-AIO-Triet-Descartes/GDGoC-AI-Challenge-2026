import os
import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add project root to path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agent.fsm_agent import FSMAgent

def main():
    print("Loading leaderboard to get top submission IDs...")
    lead = pd.read_excel('GDGoC AI Challenge 2026 - Leaderboard.xlsx', sheet_name='Leaderboard')
    top_teams = lead[lead['Rank'] <= 10]
    top_subs = set(top_teams['Submission ID'].unique())
    print("Top Submission IDs:", top_subs)

    json_files = glob.glob("expert_matches/*.json")
    print(f"Found {len(json_files)} match files to process.")

    all_records = []
    skipped_matches = 0
    total_steps_collected = 0

    for json_file in tqdm(json_files):
        try:
            with open(json_file, 'r') as f:
                match_data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            skipped_matches += 1
            continue

        team_ids = match_data.get("team_ids", [])
        ranks = match_data.get("ranks", [])
        history = match_data.get("history", [])

        if not team_ids or not ranks or not history:
            skipped_matches += 1
            continue

        # Find which players belong to top teams
        players_to_extract = []
        for idx, sub_id in enumerate(team_ids):
            if sub_id in top_subs:
                # Extract actions if they finished in the top 2 (rank 0 or 1)
                if idx < len(ranks) and ranks[idx] <= 1:
                    players_to_extract.append(idx)

        if not players_to_extract:
            continue

        # Process each selected player
        for p_idx in players_to_extract:
            # Instantiate agent for this player index
            agent = FSMAgent(agent_id=p_idx, logs=False, collect_data=True)
            
            # Replay history
            for s in range(len(history) - 1):
                step_data = history[s]
                next_step_data = history[s+1]
                
                # Check if player is alive at this step
                if not step_data["alive"][p_idx]:
                    continue

                # The action actually taken by the player at this step
                next_actions = next_step_data.get("actions", None)
                if not next_actions or p_idx >= len(next_actions):
                    continue
                
                actual_action = next_actions[p_idx]
                if actual_action is None or not (0 <= actual_action <= 5):
                    continue

                # Build observation (map must be np.array)
                obs = {
                    "step": step_data["step"],
                    "map": np.array(step_data["map"]),
                    "players": step_data["players"],
                    "bombs": step_data["bombs"]
                }

                # Run agent action selection to collect features
                try:
                    # Clear agent's records list to only keep the current one
                    agent.step_records = []
                    _ = agent.act(obs)
                except Exception as e:
                    # If feature extraction fails, skip this step
                    continue

                if agent.step_records:
                    record = agent.step_records[0]
                    # Modify teacher scores to prefer the actual expert action
                    t_scores = record["teacher_scores"]
                    
                    # Compute max non-infinite teacher score
                    valid_scores = [ts for ts in t_scores if ts > -500.0]
                    if valid_scores:
                        max_valid = max(valid_scores)
                    else:
                        max_valid = 0.0

                    # Boost the expert action score
                    # This ensures pairwise preference ranks the expert action first
                    t_scores[actual_action] = max(max_valid, t_scores[actual_action]) + 5.0
                    record["teacher_scores"] = t_scores
                    
                    # Ensure safety mask does not filter out the expert action
                    record["safe_mask"][actual_action] = 1
                    
                    # Set actual target and other match metadata
                    record["action_taken"] = actual_action
                    record["match_id"] = match_data.get("seed", 0)
                    record["agent_id"] = p_idx
                    record["rank"] = ranks[p_idx]
                    record["win"] = 1 if ranks[p_idx] == 0 else 0
                    record["actual_win"] = record["win"]
                    
                    # Calculate teacher margin for the expert action vs second best
                    sorted_t = sorted([t_scores[a] for a in range(6) if a != actual_action and t_scores[a] > -500.0], reverse=True)
                    if sorted_t:
                        record["teacher_margin"] = float(t_scores[actual_action] - sorted_t[0])
                    else:
                        record["teacher_margin"] = 5.0

                    all_records.append(record)
                    total_steps_collected += 1

    print(f"Processed matches. Total steps collected: {total_steps_collected}")
    if all_records:
        df = pd.DataFrame(all_records)
        output_csv = "agent/expert_utility_dataset.csv"
        df.to_csv(output_csv, index=False)
        print(f"Expert dataset successfully saved to {output_csv}")
    else:
        print("No expert steps collected!")

if __name__ == "__main__":
    main()
