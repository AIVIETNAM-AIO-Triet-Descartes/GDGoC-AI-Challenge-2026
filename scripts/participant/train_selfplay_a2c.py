import os
# Force single-threaded execution inside workers to prevent thread contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

from engine.game import BomberEnv
from agent.fsm_agent import FSMAgent
from agent.utility_model_pytorch import PhaseUtilityNet

def run_rl_match(match_id):
    # Set seeds
    random.seed(match_id)
    np.random.seed(match_id)
    
    env = BomberEnv(max_steps=500, seed=match_id)
    obs = env.reset(seed=match_id)
    
    # 50% chance of pure self-play, 50% mixed-opponents
    is_self_play = (match_id % 2 == 0)
    
    agents = [None] * 4
    
    if is_self_play:
        for i in range(4):
            agent = FSMAgent(agent_id=i, logs=False, collect_data=True)
            agent.explore = True
            agents[i] = agent
    else:
        main_slot = random.randint(0, 3)
        agent = FSMAgent(agent_id=main_slot, logs=False, collect_data=True)
        agent.explore = True
        agents[main_slot] = agent
        
        from agent import TacticalRuleAgent, SmarterRuleAgent, GeniusRuleAgent
        opponents_classes = ["FSMAgent", "TacticalRuleAgent", "SmarterRuleAgent", "GeniusRuleAgent"]
        selected_opponents = random.choices(opponents_classes, k=3)
        
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
    
    # Track stats for reward shaping
    prev_stats = []
    for j in range(4):
        p = env.players[j]
        prev_stats.append({
            'kills': p.stats.get('kills', 0),
            'boxes': p.stats.get('boxes', 0),
            'items': p.stats.get('items', 0),
            'alive': True
        })
        
    prev_alive = [True] * 4
    death_order = []
    trajectories = [[] for _ in range(4)]
    
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
            if not prev_alive[j]:
                continue
                
            collecting = getattr(agents[j], "collect_data", False)
            p = env.players[j]
            kills = p.stats.get('kills', 0)
            boxes = p.stats.get('boxes', 0)
            items = p.stats.get('items', 0)
            
            delta_kills = kills - prev_stats[j]['kills']
            delta_boxes = boxes - prev_stats[j]['boxes']
            delta_items = items - prev_stats[j]['items']
            
            prev_stats[j]['kills'] = kills
            prev_stats[j]['boxes'] = boxes
            prev_stats[j]['items'] = items
            
            if not alive_now[j]:
                died_this_step.append(j)
                prev_stats[j]['alive'] = False
            
            if collecting:
                # New shaped rewards: survival, kills, boxes, items
                r = 0.002
                r += delta_kills * 1.5
                r += delta_items * 0.05
                r += delta_boxes * 0.02
                
                # Penalty for dying
                if not alive_now[j]:
                    r -= 3.0
                    
                # Log the agent step records
                if agents[j].step_records:
                    record = agents[j].step_records[-1]
                    
                    # Fetch action features of the taken action to apply near-death & territory shape
                    action_taken = record["action_taken"]
                    feat_taken = record["action_features"][action_taken]
                    
                    # escape_margin_norm is at index 20
                    escape_margin = feat_taken[20] * 10.0
                    if escape_margin < 2.0:
                        r -= 0.05
                        
                    # threat_asymmetry is at index 30
                    threat_asymmetry = feat_taken[30]
                    if threat_asymmetry > 0.0:
                        r += 0.02
                        
                    record["reward"] = r
                    trajectories[j].append(record)
                 
        if died_this_step:
            death_order.append(died_this_step)
        prev_alive = alive_now
        
    # Calculate final ranks
    alive_final = [bool(p[2]) for p in obs["players"]]
    survivors = [j for j in range(4) if alive_final[j]]
    
    def sort_key(j):
        p = env.players[j]
        return (p.stats.get('kills', 0), p.stats.get('boxes', 0), p.stats.get('items', 0), p.stats.get('bombs', 0))
        
    survivors.sort(key=sort_key, reverse=True)
    
    ranks = [0] * 4
    current_rank = 0
    prev_s = None
    for j in survivors:
        stats = sort_key(j)
        if prev_s is not None and stats < prev_s:
            current_rank += 1
        ranks[j] = current_rank
        prev_s = stats
        
    if not survivors:
        current_rank = 0
    else:
        current_rank += 1
        
    for group in reversed(death_order):
        for j in group:
            ranks[j] = current_rank
        current_rank += 1
        
    # Match winner reward
    for j in range(4):
        if not getattr(agents[j], "collect_data", False):
            continue
        rank = ranks[j]
        if rank == 0 and trajectories[j]:
            # Win reward
            trajectories[j][-1]["reward"] += 8.0
        elif rank == 1 and trajectories[j]:
            # Top2 reward
            trajectories[j][-1]["reward"] += 2.0
            
    # Combine all trajectories
    all_trajectories = []
    for j in range(4):
        if trajectories[j]:
            all_trajectories.append(trajectories[j])
            
    return all_trajectories, ranks

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Self-Play RL Fine-Tuning (A2C-lite) for PhaseUtilityNet.")
    parser.add_argument("--model_path", type=str, default="agent/learned_utility_model.pth", help="Path to learned PyTorch model.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of policy gradient iterations.")
    parser.add_argument("--matches_per_iter", type=int, default=32, help="Number of self-play matches per iteration.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel matches.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy loss coefficient.")
    parser.add_argument("--distill_coef", type=float, default=1.0, help="Teacher KL distillation loss coefficient.")
    parser.add_argument("--critic_coef", type=float, default=0.5, help="Critic loss coefficient.")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize/load model
    model = PhaseUtilityNet(state_dim=13, action_dim=33).to(device)
    if model_path.exists():
        print(f"Loading pretrained model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    else:
        print("Starting from scratch (random critic, random/zero offsets).")
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"\nStarting self-play RL training for {args.iterations} iterations...")
    
    for it in range(args.iterations):
        # Save model so workers can load the updated policy
        torch.save(model.state_dict(), model_path)
        
        # Simulate matches
        print(f"\n[Iteration {it+1:02d}/{args.iterations:02d}] Simulating matches...")
        pool = mp.Pool(processes=args.num_workers)
        
        # Define random seeds for this iteration
        seeds = [random.randint(0, 1000000) for _ in range(args.matches_per_iter)]
        
        results = []
        ranks_summary = []
        for res, rnk in pool.map(run_rl_match, seeds):
            results.extend(res)
            ranks_summary.extend(rnk)
        pool.close()
        pool.join()
        
        # Calculate win/rank stats
        ranks_summary = np.array(ranks_summary)
        win_rate = np.mean(ranks_summary == 0)
        avg_rank = np.mean(ranks_summary) + 1.0 # 1-indexed rank
        print(f"Simulated {len(results)//4} matches. Win Rate: {win_rate:.2%}, Avg Rank: {avg_rank:.2f}")
        
        # Process trajectories to construct training batch
        states = []
        actions = []
        actions_taken = []
        teachers = []
        returns = []
        safe_masks = []
        
        for traj in results:
            # Calculate discounted returns
            rewards = [step["reward"] for step in traj]
            discounted_returns = []
            g = 0.0
            for r in reversed(rewards):
                g = r + args.gamma * g
                discounted_returns.insert(0, g)
                
            for idx, step in enumerate(traj):
                states.append(step["state_features"])
                actions.append(step["action_features"])
                actions_taken.append(step["action_taken"])
                teachers.append(step["teacher_scores"])
                safe_masks.append(step["safe_mask"])
                returns.append(discounted_returns[idx])
                
        if len(states) == 0:
            print("No steps collected, skipping update.")
            continue
            
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        actions_taken = torch.tensor(actions_taken, dtype=torch.long).to(device)
        teachers = torch.tensor(teachers, dtype=torch.float32).to(device)
        safe_masks = torch.tensor(safe_masks, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Model updates
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        offset = model(states, actions)  # [batch_size, 6]
        pred = offset                    # [batch_size, 6]
        values = model.get_value(states) # [batch_size]
        
        # Validity and safety mask
        valid_mask = (teachers > -500.0)
        mask = valid_mask & (safe_masks > 0.5)
        # Fallback if no action is safe in a step (very rare)
        any_safe = mask.any(dim=-1, keepdim=True)
        mask = torch.where(any_safe, mask, valid_mask)
        
        # Compute policy probabilities safely
        logits = pred / 0.5
        logits[~mask] = -1e9
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Identify steps where the action taken was actually valid/safe (not masked)
        action_mask = mask[range(len(actions_taken)), actions_taken]
        
        # Log probability of taken actions (safe-guarded from masked -1e9 values)
        log_prob_taken = log_probs[range(len(actions_taken)), actions_taken]
        log_prob_taken = torch.where(action_mask, log_prob_taken, torch.zeros_like(log_prob_taken))
        
        # Advantage
        advantages = returns - values.detach()
        
        # 1. Actor Loss (only optimize steps where action taken was not masked)
        loss_actor = - (advantages * log_prob_taken * action_mask.float()).sum() / (action_mask.float().sum() + 1e-9)
        
        # 2. Critic Loss
        loss_critic = (returns - values).pow(2).mean()
        
        # 3. Distillation Loss (KL divergence from teacher, safe-guarded from NaNs/large negatives)
        masked_teacher = teachers.clone()
        masked_teacher[~mask] = -1e9
        p_teacher = torch.softmax(masked_teacher / 0.5, dim=-1)
        
        log_probs_safe = torch.where(mask, log_probs, torch.zeros_like(log_probs))
        kl_distill = p_teacher * (torch.log(p_teacher + 1e-9) - log_probs_safe)
        kl_distill[~mask] = 0.0
        loss_distill = kl_distill.sum(dim=-1).mean()
        
        # 4. Entropy Loss
        p = torch.exp(log_probs)
        entropy = - torch.sum(p * log_probs, dim=-1)
        # Only compute entropy over valid/safe actions
        entropy = torch.where(any_safe.squeeze(-1), entropy, torch.zeros_like(entropy))
        loss_entropy = - entropy.mean()
        
        total_loss = (
            loss_actor +
            args.critic_coef * loss_critic +
            args.distill_coef * loss_distill +
            args.entropy_coef * loss_entropy
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"Step Rewards - Mean: {np.mean(returns.cpu().numpy()):.4f} | Std: {np.std(returns.cpu().numpy()):.4f}")
        print(f"Losses - Total: {total_loss.item():.4f} | Actor: {loss_actor.item():.4f} | Critic: {loss_critic.item():.4f} | Distill: {loss_distill.item():.4f} | Entropy: {loss_entropy.item():.4f}")
        
    # Final save
    torch.save(model.state_dict(), model_path)
    print(f"\nReinforcement learning fine-tuning complete. Model saved to {model_path}")

if __name__ == "__main__":
    main()
