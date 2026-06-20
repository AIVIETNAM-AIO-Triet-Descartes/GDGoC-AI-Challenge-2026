import os
# Force single-threaded execution to simulate production VM constraints
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import shutil
import random
import argparse
import sys
from pathlib import Path
import trueskill

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents

def run_evaluation_suite(model_name, model_json_path, num_matches=100, max_steps=500):
    agent_dir = Path("agent")
    target_model_path = agent_dir / "utility_model.json"
    
    # Setup model
    if model_json_path:
        print(f"\n[Evaluating {model_name}] Setting up model from {model_json_path}...")
        shutil.copy(model_json_path, target_model_path)
    else:
        print(f"\n[Evaluating {model_name}] No model path provided, using baseline FSM weights...")
        if target_model_path.exists():
            target_model_path.unlink()
            
    # Initialize TrueSkill
    ts_env = trueskill.TrueSkill(mu=100.0, sigma=33.333, draw_probability=0.1)
    agent_rating = ts_env.Rating()
    baseline_rating = ts_env.Rating()
    
    wins = 0
    draws = 0
    total_rank = 0
    env = BomberEnv(max_steps=max_steps)
    
    for i in range(num_matches):
        strong_baselines = ["TacticalRuleAgent", "SmarterRuleAgent", "GeniusRuleAgent"]
        opponents = random.choices(strong_baselines, k=3)
        agent_paths = ["agent/fsm_agent.py"] + opponents
        
        try:
            agents, names = make_agents(agent_paths, seed=None)
        except Exception as e:
            print(f"Failed to load agents: {e}")
            return None
            
        obs = env.reset()
        done = False
        step = 0
        
        prev_alive = [bool(p[2]) for p in obs["players"]]
        death_order = []
        
        while not done and step < max_steps:
            actions = []
            for j in range(4):
                try:
                    action = agents[j].act(obs)
                except Exception:
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
        
        # Tie-breaker sort
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
            
        if len(survivors) > 0 and ranks[0] == 0:
            if sum(1 for j in survivors if ranks[j] == 0) == 1:
                wins += 1
            else:
                draws += 1
                
        total_rank += ranks[0]
        
        # Update TrueSkill
        rating_groups = [(agent_rating,), (baseline_rating,), (baseline_rating,), (baseline_rating,)]
        new_ratings = ts_env.rate(rating_groups, ranks=ranks)
        agent_rating = new_ratings[0][0]
        
    win_rate = (wins / num_matches) * 100
    draw_rate = (draws / num_matches) * 100
    avg_rank = total_rank / num_matches
    ts_score = agent_rating.mu - 3 * agent_rating.sigma
    
    print(f"\n=== Results for {model_name} ===")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Draw Rate: {draw_rate:.1f}%")
    print(f"Average Rank: {avg_rank:.2f} (lower is better)")
    print(f"TrueSkill Score: {ts_score:.2f} (mu={agent_rating.mu:.2f}, sigma={agent_rating.sigma:.2f})")
    
    # Cleanup model file
    if target_model_path.exists():
        target_model_path.unlink()
        
    return {
        "model_name": model_name,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "avg_rank": avg_rank,
        "ts_score": ts_score
    }

def disable_pytorch_models():
    agent_dir = Path("agent")
    for p in ["farmer", "zoner", "assassin", "tie_breaker"]:
        pth = agent_dir / f"{p}_model.pth"
        if pth.exists():
            tmp = agent_dir / f"{p}_model.pth.tmp"
            if tmp.exists():
                tmp.unlink()
            pth.rename(tmp)
            
    # Disable unified model
    pth = agent_dir / "learned_utility_model.pth"
    if pth.exists():
        tmp = agent_dir / "learned_utility_model.pth.tmp"
        if tmp.exists():
            tmp.unlink()
        pth.rename(tmp)

def enable_pytorch_models():
    agent_dir = Path("agent")
    for p in ["farmer", "zoner", "assassin", "tie_breaker"]:
        tmp = agent_dir / f"{p}_model.pth.tmp"
        if tmp.exists():
            pth = agent_dir / f"{p}_model.pth"
            if pth.exists():
                pth.unlink()
            tmp.rename(pth)
            
    # Enable unified model
    tmp = agent_dir / "learned_utility_model.pth.tmp"
    if tmp.exists():
        pth = agent_dir / "learned_utility_model.pth"
        if pth.exists():
            pth.unlink()
        tmp.rename(pth)

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs learned utility models.")
    parser.add_argument("--num_matches", type=int, default=50, help="Number of matches to simulate per evaluation.")
    parser.add_argument("--rank_model_path", type=str, default="agent/rank_model.json", help="Path to trained rank model.")
    parser.add_argument("--win_model_path", type=str, default="agent/win_model.json", help="Path to trained win model.")
    parser.add_argument("--survival_model_path", type=str, default="agent/survival_model.json", help="Path to trained survival model.")
    args = parser.parse_args()
    
    # Ensure sys.path includes project root
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    results = []
    
    try:
        # 1. Disable PyTorch models for baseline and JSON evaluations
        disable_pytorch_models()
        
        # 1. Evaluate Baseline FSM
        res_base = run_evaluation_suite(
            model_name="Baseline FSM Agent",
            model_json_path=None,
            num_matches=args.num_matches
        )
        if res_base:
            results.append(res_base)
            
        # 2. Evaluate Learned Rank Regressor
        if Path(args.rank_model_path).exists():
            res_rank = run_evaluation_suite(
                model_name="Learned Utility (Rank Regressor)",
                model_json_path=args.rank_model_path,
                num_matches=args.num_matches
            )
            if res_rank:
                results.append(res_rank)
        else:
            print(f"Rank model not found at {args.rank_model_path}, skipping.")
            
        # 3. Evaluate Learned Win Classifier
        if Path(args.win_model_path).exists():
            res_win = run_evaluation_suite(
                model_name="Learned Utility (Win Classifier)",
                model_json_path=args.win_model_path,
                num_matches=args.num_matches
            )
            if res_win:
                results.append(res_win)
        else:
            print(f"Win model not found at {args.win_model_path}, skipping.")

        # 4. Evaluate Discounted Survival Regressor
        if Path(args.survival_model_path).exists():
            res_surv = run_evaluation_suite(
                model_name="Learned Utility (Survival Regressor)",
                model_json_path=args.survival_model_path,
                num_matches=args.num_matches
            )
            if res_surv:
                results.append(res_surv)
        else:
            print(f"Survival model not found at {args.survival_model_path}, skipping.")
            
        # 5. Evaluate Unified PyTorch Model
        unified_exists = (Path("agent") / "learned_utility_model.pth").exists() or (Path("agent") / "learned_utility_model.pth.tmp").exists()
        if unified_exists:
            # Enable only unified model
            tmp = Path("agent") / "learned_utility_model.pth.tmp"
            if tmp.exists():
                tmp.rename(Path("agent") / "learned_utility_model.pth")
                
            res_unified = run_evaluation_suite(
                model_name="Learned Utility (Unified PyTorch)",
                model_json_path=None,
                num_matches=args.num_matches
            )
            
            # Disable it back for other evaluations
            pth = Path("agent") / "learned_utility_model.pth"
            if pth.exists():
                pth.rename(Path("agent") / "learned_utility_model.pth.tmp")
                
            if res_unified:
                results.append(res_unified)
            
        # 6. Evaluate Phase-Specific PyTorch Models
        pytorch_exists = False
        for p in ["farmer", "zoner", "assassin", "tie_breaker"]:
            if (Path("agent") / f"{p}_model.pth").exists() or (Path("agent") / f"{p}_model.pth.tmp").exists():
                pytorch_exists = True
                break
                
        if pytorch_exists:
            # Enable phase models (temporarily disable unified to avoid overriding)
            tmp_unified = Path("agent") / "learned_utility_model.pth"
            if tmp_unified.exists():
                tmp_unified.rename(Path("agent") / "learned_utility_model.pth.tmp")
                
            # Enable phase models
            for p in ["farmer", "zoner", "assassin", "tie_breaker"]:
                tmp = Path("agent") / f"{p}_model.pth.tmp"
                if tmp.exists():
                    tmp.rename(Path("agent") / f"{p}_model.pth")
                    
            res_pytorch = run_evaluation_suite(
                model_name="Learned Utility (Phase PyTorch)",
                model_json_path=None,
                num_matches=args.num_matches
            )
            
            # Disable them back
            for p in ["farmer", "zoner", "assassin", "tie_breaker"]:
                pth = Path("agent") / f"{p}_model.pth"
                if pth.exists():
                    pth.rename(Path("agent") / f"{p}_model.pth.tmp")
                    
            if res_pytorch:
                results.append(res_pytorch)
        else:
            print("No phase-specific PyTorch models found, skipping.")
            
    finally:
        # Re-enable PyTorch models for gameplay/active agent usage
        enable_pytorch_models()
        
    print("\n\n" + "=" * 50)
    print("FINAL COMPARISON TABLE")
    print("=" * 50)
    print(f"{'Model Name':<35} | {'Win Rate':<10} | {'Avg Rank':<10} | {'TrueSkill':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['model_name']:<35} | {r['win_rate']:>7.1f}% | {r['avg_rank']:>8.2f} | {r['ts_score']:>9.2f}")
    print("=" * 75)

if __name__ == "__main__":
    main()
