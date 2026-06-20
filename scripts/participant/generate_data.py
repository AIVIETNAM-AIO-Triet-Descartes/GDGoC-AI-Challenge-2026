import os
# Force single-threaded execution to simulate production VM constraints
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

from engine.game import BomberEnv
from agent.fsm_agent import FSMAgent

def run_single_match(match_id):
    # Set seeds
    random.seed(match_id)
    np.random.seed(match_id)
    
    # Initialize environment
    env = BomberEnv(max_steps=500, seed=match_id)
    obs = env.reset(seed=match_id)
    
    # Instantiate agents
    from agent import TacticalRuleAgent, SmarterRuleAgent, GeniusRuleAgent
    opponents_classes = ["FSMAgent", "TacticalRuleAgent", "SmarterRuleAgent", "GeniusRuleAgent"]
    selected_opponents = random.choices(opponents_classes, k=3)
    
    agents = [None] * 4
    main_slot = random.randint(0, 3)
    agents[main_slot] = FSMAgent(agent_id=main_slot, logs=False, collect_data=True)
    
    class_map = {
        "FSMAgent": lambda idx: FSMAgent(agent_id=idx, logs=False, collect_data=False),
        "TacticalRuleAgent": lambda idx: TacticalRuleAgent(idx),
        "SmarterRuleAgent": lambda idx: SmarterRuleAgent(idx),
        "GeniusRuleAgent": lambda idx: GeniusRuleAgent(idx)
    }
    for i in range(4):
        if i != main_slot:
            agents[i] = class_map[selected_opponents.pop()](i)
    
    done = False
    step = 0
    prev_alive = [bool(p[2]) for p in obs["players"]]
    death_order = []
    
    while not done and step < 500:
        actions = []
        for j in range(4):
            if prev_alive[j]:
                try:
                    action = agents[j].act(obs)
                except Exception:
                    action = 0
            else:
                action = 0
            actions.append(action)
            
        obs, terminated, truncated = env.step(actions)
        done = terminated or truncated
        step += 1
        
        alive_now = [bool(p[2]) for p in obs["players"]]
        died_this_step = []
        for j in range(4):
            if prev_alive[j] and not alive_now[j]:
                died_this_step.append(j)
        if died_this_step:
            death_order.append(died_this_step)
        prev_alive = alive_now
        
    alive_final = [bool(p[2]) for p in obs["players"]]
    survivors = [j for j in range(4) if alive_final[j]]
    
    def sort_key(j):
        p = env.players[j]
        return (p.stats.get('kills', 0), p.stats.get('boxes', 0), p.stats.get('items', 0), p.stats.get('bombs', 0))
        
    survivors.sort(key=sort_key, reverse=True)
    
    ranks = [0] * 4
    current_rank = 0
    prev_stats = None
    
    for j in survivors:
        stats = sort_key(j)
        if prev_stats is not None and stats < prev_stats:
            current_rank += 1
        ranks[j] = current_rank
        prev_stats = stats
        
    if not survivors:
        current_rank = 0
    else:
        current_rank += 1
        
    for group in reversed(death_order):
        for j in group:
            ranks[j] = current_rank
        current_rank += 1
        
    # Compile matching records
    match_records = []
    rank = ranks[main_slot]
    win = 1 if rank == 0 else 0
    agent = agents[main_slot]
    for record in agent.step_records:
        record["match_id"] = match_id
        record["agent_id"] = main_slot
        record["rank"] = rank
        record["win"] = win
        record["actual_win"] = win
        match_records.append(record)
            
    return match_records

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate training dataset via parallel self-play simulations.")
    parser.add_argument("--num_matches", type=int, default=1500, help="Number of matches to simulate.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of parallel worker processes.")
    parser.add_argument("--output_path", type=str, default="agent/learned_utility_dataset.csv", help="Where to save the output CSV.")
    args = parser.parse_args()
    
    print(f"Starting dataset generation of {args.num_matches} matches using {args.num_workers} workers...")
    
    pool = mp.Pool(processes=args.num_workers)
    
    all_records = []
    # Use tqdm to show progress bar
    results = list(tqdm(pool.imap_unordered(run_single_match, range(args.num_matches)), total=args.num_matches))
    
    for match_records in results:
        all_records.extend(match_records)
        
    print(f"Completed simulation. Total steps collected: {len(all_records)}")
    
    # Create output directory if needed
    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(all_records)
    df.to_csv(out_file, index=False)
    print(f"Dataset successfully saved to {out_file}")

if __name__ == "__main__":
    main()
