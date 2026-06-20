import random
import time
import copy
from collections import deque
import numpy as np


# Neural Network definition for utility scoring
try:
    import torch
    import torch.nn as nn
    
    class PhaseUtilityNet(nn.Module):
        def __init__(self, state_dim=13, action_dim=33, hidden_dims=[128, 64]):
            super(PhaseUtilityNet, self).__init__()
            input_dim = state_dim + action_dim
            
            # Actor network
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.LayerNorm(h_dim))
                layers.append(nn.Mish())  # Mish activation
                layers.append(nn.Dropout(0.1))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.net = nn.Sequential(*layers)
            
            # Critic network (predicts state value V(s) from state features only)
            critic_layers = []
            prev_dim = state_dim
            for h_dim in hidden_dims:
                critic_layers.append(nn.Linear(prev_dim, h_dim))
                critic_layers.append(nn.LayerNorm(h_dim))
                critic_layers.append(nn.Mish())
                prev_dim = h_dim
            critic_layers.append(nn.Linear(prev_dim, 1))
            self.critic = nn.Sequential(*critic_layers)
            
        def forward(self, state, action):
            if len(state.shape) == 2:
                num_actions = action.shape[1]
                state = state.unsqueeze(1).repeat(1, num_actions, 1)
                
            x = torch.cat([state, action], dim=-1)
            batch_size, num_actions, feat_dim = x.shape
            x_flat = x.view(-1, feat_dim)
            out_flat = self.net(x_flat)
            return out_flat.view(batch_size, num_actions)

        def get_value(self, state):
            return self.critic(state).squeeze(-1)
except ImportError:
    class PhaseUtilityNet:
        def __init__(self, *args, **kwargs):
            pass


class Map:
    GRASS = 0
    WALL = 1
    BOX = 2
    ITEM_RADIUS = 3
    ITEM_CAPACITY = 4
    BOMB = 5


class FSMAgent:
    """
    State-of-the-Art (SOTA) Finite State Machine (FSM) + Utility Scoring Agent.
    
    Actions:
    0: STOP
    1: LEFT
    2: RIGHT
    3: UP
    4: DOWN
    5: PLACE_BOMB
    """

    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

    team_id = "FSMAgent"

    def __init__(self, agent_id: int, logs: bool = True, collect_data: bool = False, model_path: str = None):
        import os
        import sys
        import json
        self.agent_id = int(agent_id)
        self.collect_data = collect_data
        self.step_records = []
        self.model = None
        
        # Check environment variable and sys.argv for logging override
        env_logs = os.environ.get("AGENT_LOGS", None)
        if env_logs is not None:
            self.logs_enabled = env_logs.lower() in ("true", "1", "yes")
        else:
            self.logs_enabled = logs
            
        # Check CLI arguments for --no-logs or --logs=false
        for arg in sys.argv:
            if arg in ("--no-logs", "--logs=false", "--logs=False"):
                self.logs_enabled = False
            elif arg in ("--logs", "--logs=true", "--logs=True"):
                self.logs_enabled = True

        self.current_step = 0
        self.log_file = f"fsm_agent_{self.agent_id}.log"
        self.explore = False
        self.meta_mode = "FARM"
        
        self.pytorch_model = None
        self.pytorch_models = {}
        # PyTorch imitation model disabled: baseline FSM teacher scores
        # outperform the learned model in competitive matches (avg rank 0.67 vs 0.77)
        from pathlib import Path
        model_dir = Path(__file__).resolve().parent
        
        if model_path is None:
            default_path = model_dir / "utility_model.json"
            if default_path.exists():
                model_path = str(default_path)

        if model_path is not None and not self.pytorch_models:
            try:
                with open(model_path, "r") as f:
                    self.model = json.load(f)
                self.log(f"Successfully loaded learned utility model from {model_path}")
            except Exception as e:
                # Fallback / print warning
                print(f"Warning: Failed to load model from {model_path}: {e}", file=sys.stderr)
        
        if self.logs_enabled:
            try:
                with open(self.log_file, "w") as f:
                    f.write(f"--- FSMAgent {self.agent_id} initialized ---\n")
            except Exception as e:
                # Disabling logging if writing fails (e.g. read-only environment)
                print(f"Warning: Disabling agent logging due to error: {e}", file=sys.stderr)
                self.logs_enabled = False

    def log(self, msg: str):
        if self.logs_enabled:
            try:
                with open(self.log_file, "a") as f:
                    f.write(f"[Step {self.current_step}] {msg}\n")
            except Exception:
                pass

    def act(self, obs):
        start_time = time.perf_counter()
        
        # Episode reset detection
        players = obs.get("players", [])
        bombs = obs.get("bombs", [])
        if self.agent_id < len(players):
            my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
            my_pos = (int(my_x), int(my_y))
            is_corner = my_pos in [(1, 1), (1, 11), (11, 1), (11, 11)]
            if is_corner and bombs_left == 1 and bomb_bonus == 0 and len(bombs) == 0:
                if self.current_step > 10:
                    self.current_step = 0

        self.current_step = obs.get("step", self.current_step + 1)

        grid = obs["map"]
        players = obs["players"]
        bombs = [list(b) for b in obs["bombs"]]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            self.log("Agent is dead or invalid.")
            return 0

        # Extract player stats
        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)

        self.log(f"Position: {my_pos} | Bombs Left: {bombs_left} | Radius: {bomb_radius}")

        # Extract environment details
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        enemies = [
            (int(p[0]), int(p[1]), i, max(1, int(p[4]) + 1))
            for i, p in enumerate(players)
            if i != self.agent_id and p[2] == 1
        ]

        blocked = set(bomb_positions)
        blocked.discard(my_pos)

        # 1. Build time-layered danger map (handles chain explosions)
        danger_map = self._build_time_layered_danger_map(grid, bombs, players)

        # 2. ESCAPE OVERRIDE (Urgent safety check)
        # If standing on a tile that explodes next tick, escape immediately
        my_danger = danger_map[my_pos[0], my_pos[1]]
        if my_danger <= 2:
            self.log(f"EMERGENCY: Immediate danger detected (ticks left: {my_danger})! Executing escape BFS.")
            escape_action = self._find_best_escape(grid, my_pos, blocked, danger_map)
            elapsed = (time.perf_counter() - start_time) * 1000
            self.log(f"Escape action: {escape_action} (Time: {elapsed:.2f}ms)")
            return escape_action

        # Cache BFS safe tiles no longer needed as we compute per action

        # Meta Policy Mode Selection
        in_danger = (danger_map[my_pos[0], my_pos[1]] != 999)
        if in_danger:
            self.meta_mode = "ESCAPE"
        else:
            # Find closest active enemy
            closest_enemy_dist = 999
            for ex, ey, _, _ in enemies:
                dist = abs(my_pos[0] - ex) + abs(my_pos[1] - ey)
                if dist < closest_enemy_dist:
                    closest_enemy_dist = dist
                    
            # Check resource abundance
            items_tiles = self._item_tiles(grid)
            box_spots_tiles = self._box_bomb_spots(grid, blocked, bomb_radius)
            has_resources = (len(items_tiles) > 0 or len(box_spots_tiles) > 0)
            
            if closest_enemy_dist <= 3:
                self.meta_mode = "ASSASSIN"
            elif closest_enemy_dist <= 6:
                self.meta_mode = "PRESSURE"
            elif has_resources:
                self.meta_mode = "FARM"
            else:
                self.meta_mode = "PRESSURE"
            
        self.log(f"Meta Policy Mode: {self.meta_mode}")

        # Opponent Modeling Metrics
        closest_enemy = None
        closest_dist = 999
        for ex, ey, eb, er in enemies:
            dist = abs(my_pos[0] - ex) + abs(my_pos[1] - ey)
            if dist < closest_dist:
                closest_dist = dist
                closest_enemy = (ex, ey, eb, er)
                
        if closest_enemy is not None:
            enemy_pos = (closest_enemy[0], closest_enemy[1])
            enemy_eb = closest_enemy[2]
            enemy_er = closest_enemy[3]
            enemy_mob, enemy_safe, enemy_corridor, trap_potential, is_my_pos_enemy_bottleneck = self._get_enemy_metrics(grid, enemy_pos, blocked, danger_map, my_pos)
        else:
            enemy_mob, enemy_safe, enemy_corridor = 0, 0, 0
            trap_potential, is_my_pos_enemy_bottleneck = 0.0, 0.0
            enemy_eb, enemy_er = 0, 0

        # Determine Phase using power_score and current_step
        power_score = (bomb_radius * 1.5) + (bombs_left * 3)
        
        if self.current_step > 450:
            phase = "TIE_BREAKER"
            weights = {"item": 1.2, "box": 1.2, "kill": 1.0, "pressure": 0.1, "survive": 1.0, "mobility": 0.5, "territory": 0.3, "danger": 5.0}
        elif power_score < 10:
            phase = "FARMER"
            weights = {"item": 1.0, "box": 0.8, "kill": 0.0, "pressure": 0.1, "survive": 1.0, "mobility": 0.4, "territory": 0.2, "danger": 5.0}
        elif self.current_step < 350:
            phase = "ZONER"
            weights = {"item": 1.0, "box": 0.8, "kill": 1.0, "pressure": 0.5, "survive": 1.0, "mobility": 0.5, "territory": 0.3, "danger": 5.0}
        else:
            phase = "ASSASSIN"
            weights = {"item": 1.0, "box": 0.8, "kill": 1.5, "pressure": 0.8, "survive": 1.0, "mobility": 0.6, "territory": 0.5, "danger": 5.0}

        if danger_map[my_pos[0], my_pos[1]] != 999:
            weights["survive"] = 50.0
            self.log("DANGER: In blast zone! Overriding weights to prioritize survival.")

        self.log(f"Current Phase: {phase} | Power Score: {power_score:.2f}")

        # Compute distance maps to optimize distance lookups
        items = self._item_tiles(grid)
        item_dist_map = self._get_distance_map(grid, blocked, items)

        box_spots = self._box_bomb_spots(grid, blocked, bomb_radius)
        box_dist_map = self._get_distance_map(grid, blocked, box_spots)

        # Utility Scoring for 6 actions
        action_scores = {}
        valid_actions = []

        # Cache bomb escape logic to save CPU
        bomb_escape_valid = None
        virtual_danger_map = None

        for action in [0, 1, 2, 3, 4, 5]:
            next_pos = self._next_pos(my_pos, action)

            # --- PRE-FILTERING ---
            if action != 5 and action != 0:
                if not self._passable(grid, next_pos[0], next_pos[1]):
                    continue
                if next_pos in blocked:
                    continue

            # Special validation for Bomb action
            if action == 5:
                if bombs_left <= 0:
                    continue
                if my_pos in bomb_positions:
                    continue

                if bomb_escape_valid is None:
                    simulated_bombs = bombs + [[my_pos[0], my_pos[1], 7, bomb_radius]]
                    virtual_danger_map = self._build_time_layered_danger_map(grid, simulated_bombs, players)
                    virtual_blocked = blocked | {my_pos}
                    best_esc = self._find_best_escape(grid, my_pos, virtual_blocked, virtual_danger_map)
                    bomb_escape_valid = (best_esc != 0)

                if not bomb_escape_valid:
                    continue

            valid_actions.append(action)

        if not valid_actions:
            return 0  # Stay

        # Compute raw utility values for normalization
        raw_values = {
            a: {
                "item": 0.0,
                "box": 0.0,
                "kill": 0.0,
                "pressure": 0.0,
                "survive": 0.0,
                "mobility": 0.0,
                "territory": 0.0,
                "danger": 0.0,
                "dist_safety": 999.0
            }
            for a in valid_actions
        }

        for action in valid_actions:
            next_pos = self._next_pos(my_pos, action)
            vals = raw_values[action]

            # Item Reward
            if next_pos in item_dist_map:
                vals["item"] = 1.0 / (item_dist_map[next_pos] + 1)

            # Box Reward
            if next_pos in box_dist_map:
                vals["box"] = 1.0 / (box_dist_map[next_pos] + 1)

            # Spatial metrics (survive, mobility, territory) using combined single BFS
            if action == 5:
                action_blocked = blocked | {my_pos}
                action_danger_map = virtual_danger_map
            else:
                action_blocked = blocked
                action_danger_map = danger_map

            surv_count, mob, terr, dist_safety = self._evaluate_spatial_metrics(grid, next_pos, action_blocked, action_danger_map)
            
            if danger_map[my_pos[0], my_pos[1]] != 999:
                vals["survive"] = 1.0 / (dist_safety + 1)
            else:
                vals["survive"] = surv_count

            vals["mobility"] = mob
            vals["territory"] = terr
            vals["dist_safety"] = float(dist_safety)

            # Enemy Pressure with Line of Sight (LOS)
            pressure_sum = 0.0
            for ex, ey, _, _ in enemies:
                enemy_pos = (ex, ey)
                dist = abs(next_pos[0] - ex) + abs(next_pos[1] - ey)
                if dist <= 4:
                    if self._has_line_of_sight(grid, next_pos, enemy_pos):
                        pressure_sum += 1.5 / (dist + 1)
                    else:
                        pressure_sum += 0.5 / (dist + 1)
            vals["pressure"] = pressure_sum

            # Kill Opportunities
            if action == 5:
                kill_potential = 0.0
                for ex, ey, _, er_enemy in enemies:
                    enemy_pos = (ex, ey)
                    # Check if enemy is in blast range of our bomb
                    dx = abs(my_pos[0] - ex)
                    dy = abs(my_pos[1] - ey)
                    if (dx == 0 and dy <= bomb_radius) or (dy == 0 and dx <= bomb_radius):
                        # Run a fast escape BFS from enemy position with simulated bomb
                        sim_bombs = bombs + [[my_pos[0], my_pos[1], 7, bomb_radius]]
                        sim_danger = self._build_time_layered_danger_map(grid, sim_bombs, players)
                        sim_blocked = blocked | {my_pos}
                        enemy_escape_act = self._find_best_escape(grid, enemy_pos, sim_blocked, sim_danger)
                        if enemy_escape_act == 0:
                            # Enemy has no escape!
                            kill_potential += 10.0
                vals["kill"] = kill_potential

            # Spatial Danger Penalty
            cell_danger = action_danger_map[next_pos[0], next_pos[1]]
            if cell_danger != 999:
                vals["danger"] = 1.0 / (cell_danger + 1e-5)
            else:
                vals["danger"] = 0.0

        pre_norm_raw_values = copy.deepcopy(raw_values)

        # Normalization over valid actions
        for key in ["item", "box", "pressure", "survive", "mobility", "territory"]:
            min_val = min(raw_values[a][key] for a in valid_actions)
            max_val = max(raw_values[a][key] for a in valid_actions)
            diff = max_val - min_val
            for a in valid_actions:
                if diff > 1e-5:
                    raw_values[a][key] = (raw_values[a][key] - min_val) / diff
                else:
                    raw_values[a][key] = 0.0

        # Calculate final FSM baseline scores (teacher scores), is_safe, and features for all actions
        action_features_list = []
        teacher_scores = {}
        is_safe_list = []
        
        # Context features
        step_norm = float(self.current_step) / 500.0
        bombs_left_norm = float(bombs_left) / 10.0
        bomb_radius_norm = float(bomb_radius) / 10.0
        enemy_count_norm = float(len(enemies)) / 3.0
        power_score_norm = float(power_score) / 50.0
        
        phase_farmer = 1.0 if phase == "FARMER" else 0.0
        phase_zoner = 1.0 if phase == "ZONER" else 0.0
        phase_assassin = 1.0 if phase == "ASSASSIN" else 0.0
        phase_tie_breaker = 1.0 if phase == "TIE_BREAKER" else 0.0
        
        meta_farm = 1.0 if self.meta_mode == "FARM" else 0.0
        meta_pressure = 1.0 if self.meta_mode == "PRESSURE" else 0.0
        meta_assassin = 1.0 if self.meta_mode == "ASSASSIN" else 0.0
        meta_escape = 1.0 if self.meta_mode == "ESCAPE" else 0.0
        
        state_wide_features = [
            step_norm,
            bombs_left_norm,
            bomb_radius_norm,
            enemy_count_norm,
            phase_farmer,
            phase_zoner,
            phase_assassin,
            phase_tie_breaker,
            power_score_norm,
            meta_farm,
            meta_pressure,
            meta_assassin,
            meta_escape
        ]
        
        for a in range(6):
            is_valid = 1 if a in valid_actions else 0
            next_pos = self._next_pos(my_pos, a)
            
            if is_valid == 1:
                vals = raw_values[a]
                raw_vals = pre_norm_raw_values[a]
                dist_safety = vals["dist_safety"]
                if a == 5:
                    escape_margin = 7.0 - dist_safety
                elif danger_map[next_pos[0], next_pos[1]] != 999:
                    escape_margin = float(danger_map[next_pos[0], next_pos[1]]) - dist_safety
                else:
                    escape_margin = 999.0
            else:
                escape_margin = -999.0
                
            is_safe = 1 if (danger_map[next_pos[0], next_pos[1]] == 999 and is_valid) else 0
            if is_valid == 1 and a == 5 and escape_margin < 1.0:
                is_safe = 0
            is_safe_list.append(is_safe)
            
            if is_valid == 1:
                boxes_hit, items_hit = 0, 0
                if a == 5:
                    boxes_hit, items_hit = self._count_hits(grid, my_pos[0], my_pos[1], bomb_radius)
                
                # FSM baseline score (teacher score)
                t_score = (
                    weights["item"] * vals["item"] +
                    weights["box"] * vals["box"] +
                    weights["kill"] * vals["kill"] +
                    weights["pressure"] * vals["pressure"] +
                    weights["survive"] * vals["survive"] +
                    weights["mobility"] * vals["mobility"] +
                    weights["territory"] * vals["territory"] -
                    weights["danger"] * vals["danger"]
                )
                if a == 5:
                    # Always give base bomb bonus when safe to encourage active play
                    t_score += 1.5
                    if my_pos in box_spots:
                        t_score += boxes_hit * 4.0
                    if items_hit > 0:
                        t_score += items_hit * 1.0
                    if vals["kill"] > 0:
                        t_score += weights["kill"] * vals["kill"] * 2.0
                    if is_my_pos_enemy_bottleneck > 0.5:
                        t_score += 3.0
                if phase == "TIE_BREAKER" and a == 5:
                    t_score += 3.0
            else:
                vals = {k: 0.0 for k in ["item", "box", "kill", "pressure", "survive", "mobility", "territory", "danger"]}
                raw_vals = vals
                boxes_hit, items_hit = 0, 0
                t_score = -999.0
            
            teacher_scores[a] = t_score
            
            action_one_hot = [1.0 if i == a else 0.0 for i in range(6)]
            escape_margin_norm = float(escape_margin) / 10.0
            
            if is_valid == 1:
                enemy_safe_tiles_norm = float(enemy_safe) / 50.0
                enemy_corridor_depth_norm = float(enemy_corridor) / 4.0
                enemy_bomb_potential_norm = float(enemy_eb * enemy_er) / 20.0
                threat_asymmetry = float(vals["mobility"]) - (float(enemy_mob) / 50.0)
            else:
                enemy_safe_tiles_norm = 0.0
                enemy_corridor_depth_norm = 0.0
                enemy_bomb_potential_norm = 0.0
                threat_asymmetry = 0.0
                
            feat_a = [
                float(is_valid),
                float(is_safe),
                float(vals["item"]),
                float(vals["box"]),
                float(vals["kill"]),
                float(vals["pressure"]),
                float(vals["survive"]),
                float(vals["mobility"]),
                float(vals["territory"]),
                float(vals["danger"]),
                float(raw_vals["item"]),
                float(raw_vals["box"]),
                float(raw_vals["kill"]),
                float(raw_vals["pressure"]),
                float(raw_vals["survive"]),
                float(raw_vals["mobility"]),
                float(raw_vals["territory"]),
                float(raw_vals["danger"]),
                float(boxes_hit),
                float(items_hit),
                float(escape_margin_norm)
            ] + action_one_hot + [
                float(enemy_safe_tiles_norm),
                float(enemy_corridor_depth_norm),
                float(enemy_bomb_potential_norm),
                float(threat_asymmetry),
                float(trap_potential),
                float(is_my_pos_enemy_bottleneck)
            ]
            action_features_list.append(feat_a)

        in_danger = (danger_map[my_pos[0], my_pos[1]] != 999)
        model_scores = {}
        
        # 1. Check if unified PyTorch model is loaded
        if self.pytorch_model is not None and not in_danger:
            import torch
            try:
                with torch.no_grad():
                    state_t = torch.tensor([state_wide_features], dtype=torch.float32)
                    action_t = torch.tensor([action_features_list], dtype=torch.float32)
                    # Unified model outputs absolute utility score for all actions (Option A)
                    offset = self.pytorch_model(state_t, action_t).numpy()[0]
                for a in range(6):
                    if a in valid_actions:
                        # Safety shield: if is_safe_list[a] == 0, override to -999.0
                        if is_safe_list[a] == 0:
                            model_scores[a] = -999.0
                        else:
                            model_scores[a] = float(offset[a])
                    else:
                        model_scores[a] = -999.0
            except Exception as e:
                self.log(f"Error executing unified PyTorch model: {e}")
                
        # Compute final action scores
        candidate_scores = {}
        for a in valid_actions:
            if a in model_scores:
                score = model_scores[a]
            else:
                # Fallback options
                score = teacher_scores[a]
                if self.pytorch_models and phase in self.pytorch_models and not in_danger:
                    import torch
                    try:
                        model = self.pytorch_models[phase]
                        vals = raw_values[a]
                        raw_vals = pre_norm_raw_values[a]
                        boxes_hit, items_hit = 0, 0
                        if a == 5:
                            boxes_hit, items_hit = self._count_hits(grid, my_pos[0], my_pos[1], bomb_radius)
                        features_tensor = torch.tensor([[
                            float(vals["item"]), float(vals["box"]), float(vals["kill"]), float(vals["pressure"]), float(vals["survive"]), float(vals["mobility"]), float(vals["territory"]), float(vals["danger"]),
                            float(raw_vals["item"]), float(raw_vals["box"]), float(raw_vals["kill"]), float(raw_vals["pressure"]), float(raw_vals["survive"]), float(raw_vals["mobility"]), float(raw_vals["territory"]), float(raw_vals["danger"]),
                            float(a), float(boxes_hit), float(items_hit)
                        ]], dtype=torch.float32)
                        with torch.no_grad():
                            score = float(model(features_tensor).item())
                        score = 0.7 * score + 0.3 * teacher_scores[a]
                        score = self._adjust_action_score(a, score, vals, boxes_hit, enemies, bomb_radius, closest_dist, phase, my_pos)
                    except Exception as e:
                        self.log(f"Error in phase-specific PyTorch fallback: {e}")
                elif self.model is not None and not in_danger:
                    try:
                        vals = raw_values[a]
                        raw_vals = pre_norm_raw_values[a]
                        boxes_hit, items_hit = 0, 0
                        if a == 5:
                            boxes_hit, items_hit = self._count_hits(grid, my_pos[0], my_pos[1], bomb_radius)
                        features = {
                            "norm_item": float(vals["item"]),
                            "norm_box": float(vals["box"]),
                            "norm_kill": float(vals["kill"]),
                            "norm_pressure": float(vals["pressure"]),
                            "norm_survive": float(vals["survive"]),
                            "norm_mobility": float(vals["mobility"]),
                            "norm_territory": float(vals["territory"]),
                            "norm_danger": float(vals["danger"]),
                            "raw_item": float(raw_vals["item"]),
                            "raw_box": float(raw_vals["box"]),
                            "raw_kill": float(raw_vals["kill"]),
                            "raw_pressure": float(raw_vals["pressure"]),
                            "raw_survive": float(raw_vals["survive"]),
                            "raw_mobility": float(raw_vals["mobility"]),
                            "raw_territory": float(raw_vals["territory"]),
                            "raw_danger": float(raw_vals["danger"]),
                            "step": float(self.current_step),
                            "power_score": float(power_score),
                            "bombs_left": float(bombs_left),
                            "bomb_radius": float(bomb_radius),
                            "my_danger": float(my_danger),
                            "action": float(a),
                            "boxes_hit": float(boxes_hit),
                            "items_hit": float(items_hit),
                        }
                        score = self._predict_score(features)
                        objective = self.model.get("objective", "")
                        if "regression" in objective:
                            score = -score
                        score = 0.7 * score + 0.3 * teacher_scores[a]
                        score = self._adjust_action_score(a, score, vals, boxes_hit, enemies, bomb_radius, closest_dist, phase, my_pos)
                    except Exception as e:
                        self.log(f"Error in JSON model fallback: {e}")
            candidate_scores[a] = score

        # Identify top 3 valid actions (with score > -500)
        valid_candidates = {a: s for a, s in candidate_scores.items() if s > -500.0}
        if not valid_candidates:
            valid_candidates = candidate_scores
            
        sorted_candidates = sorted(valid_candidates.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_candidates[:3]
        
        action_scores = {}
        for a in valid_actions:
            base_score = candidate_scores[a]
            if any(a == tk[0] for tk in top_k) and base_score > -500.0:
                roll_danger_map = virtual_danger_map if (a == 5 and virtual_danger_map is not None) else danger_map
                rollout_score = self._simulate_rollout(grid, my_pos, a, roll_danger_map, bombs, enemies, bomb_radius)
                action_scores[a] = 0.5 * base_score + 0.5 * rollout_score
                self.log(f"Action {a}: base={base_score:.2f}, rollout={rollout_score:.2f} -> combined={action_scores[a]:.2f}")
            else:
                action_scores[a] = base_score
            
        if getattr(self, "explore", False) and self.pytorch_model is not None and not in_danger:
            # Stochastic action selection over valid and safe actions
            valid_and_safe = [va for va in valid_actions if is_safe_list[va] == 1]
            if not valid_and_safe:
                valid_and_safe = valid_actions
            
            scores = np.array([action_scores[va] for va in valid_and_safe])
            scores = scores - np.max(scores)  # Numerical stability
            probs = np.exp(scores / 0.5)
            probs = probs / np.sum(probs)
            best_action = int(np.random.choice(valid_and_safe, p=probs))
        else:
            best_action = max(valid_actions, key=lambda a: action_scores[a])
        
        if len(valid_actions) >= 2:
            sorted_t_scores = sorted([teacher_scores[va] for va in valid_actions], reverse=True)
            teacher_margin = float(sorted_t_scores[0] - sorted_t_scores[1])
        else:
            teacher_margin = 0.0
            
        if self.collect_data:
            safe_mask = [int(x) for x in is_safe_list]
            teacher_scores_list = [float(teacher_scores[a]) for a in range(6)]
            record = {
                "step": int(self.current_step),
                "state_features": state_wide_features,
                "actions": [0, 1, 2, 3, 4, 5],
                "action_features": action_features_list,
                "teacher_scores": teacher_scores_list,
                "safe_mask": safe_mask,
                "teacher_margin": float(teacher_margin),
                "actual_win": 0,
                "action_taken": int(best_action),
            }
            self.step_records.append(record)

        elapsed = (time.perf_counter() - start_time) * 1000
        self.log(f"Selected action: {best_action} (Time: {elapsed:.2f}ms)")
        return best_action

    ############################################################
    # HELPER ALGORITHMS
    ############################################################

    def _predict_tree(self, tree, features):
        if "leaf_value" in tree:
            return tree["leaf_value"]
        feat_name = tree["split_feature"]
        val = features.get(feat_name, 0.0)
        threshold = tree["threshold"]
        # Split decision: LightGBM uses <=
        if val <= threshold:
            return self._predict_tree(tree["left_child"], features)
        else:
            return self._predict_tree(tree["right_child"], features)

    def _predict_score(self, features):
        if not self.model:
            return 0.0
        # If it's a LightGBM model dump, it has "baseline_value" and "tree_info"
        score = self.model.get("baseline_value", 0.0)
        for tree in self.model.get("tree_info", []):
            score += self._predict_tree(tree["tree_structure"], features)
        return score

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES.get(action, (0, 0))
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and grid[x, y] in [0, 3, 4]

    def _blast_tiles(self, grid, bx, by, radius):
        if not self._in_bounds(grid, bx, by):
            return set()
        tiles = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, radius + 1):
                x = bx + dx * r
                y = by + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                cell = grid[x, y]
                if cell == Map.WALL:
                    break
                tiles.add((x, y))
                if cell == Map.BOX:
                    break
        return tiles

    def _build_time_layered_danger_map(self, grid, bombs, players):
        h, w = grid.shape
        danger_map = np.full((h, w), 999, dtype=int)

        # Parse bomb properties
        bomb_objects = []
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3]) if len(b) > 3 else -1
            radius = 2
            if 0 <= owner_id < len(players):
                radius = max(1, int(players[owner_id][4]) + 1)
            bomb_objects.append({"pos": (bx, by), "timer": timer, "radius": radius})

        # Process chain explosions (fixed-point iteration)
        changed = True
        while changed:
            changed = False
            for i in range(len(bomb_objects)):
                b1 = bomb_objects[i]
                b1_tiles = self._blast_tiles(grid, b1["pos"][0], b1["pos"][1], b1["radius"])
                for j in range(len(bomb_objects)):
                    if i == j:
                        continue
                    b2 = bomb_objects[j]
                    if b2["pos"] in b1_tiles:
                        if b2["timer"] > b1["timer"]:
                            b2["timer"] = b1["timer"]
                            changed = True

        # Construct time layers
        for b in bomb_objects:
            tiles = self._blast_tiles(grid, b["pos"][0], b["pos"][1], b["radius"])
            for tx, ty in tiles:
                danger_map[tx, ty] = min(danger_map[tx, ty], b["timer"])

        return danger_map

    def _evaluate_spatial_metrics(self, grid, start, blocked, danger_map):
        q = deque([(start, 0)])
        seen = {start}
        
        survivability_count = 0
        mobility_count = 0
        territory_count = 0
        dist_to_safety = 999
        
        while q:
            pos, d = q.popleft()
            
            if danger_map[pos[0], pos[1]] == 999:
                if dist_to_safety == 999:
                    dist_to_safety = d
                survivability_count += 1
                if d <= 3:
                    mobility_count += 1
            
            if d <= 5:
                territory_count += 1
                
            if d >= 8:
                continue
                
            for a in [1, 2, 3, 4]:
                nx, ny = pos[0] + self.MOVES[a][0], pos[1] + self.MOVES[a][1]
                if not self._passable(grid, nx, ny) or (nx, ny) in blocked:
                    continue
                if danger_map[nx, ny] <= d + 1:
                    continue
                npos = (nx, ny)
                if npos not in seen:
                    seen.add(npos)
                    q.append((npos, d + 1))
                    
        return survivability_count, mobility_count, territory_count, dist_to_safety

    def _find_best_escape(self, grid, start, blocked, danger_map):
        q = deque([(start, 0, None)])
        seen = {start}
        best_action = 0
        best_score = -999999

        while q:
            pos, d, first_action = q.popleft()

            if danger_map[pos[0], pos[1]] > d + 1:
                # Score candidate safe tiles
                open_neighs = 0
                for a in [1, 2, 3, 4]:
                    nx, ny = self._next_pos(pos, a)
                    if self._passable(grid, nx, ny) and (nx, ny) not in blocked and danger_map[nx, ny] > d + 2:
                        open_neighs += 1
                score = (open_neighs * 2) - d
                if score > best_score:
                    best_score = score
                    best_action = first_action if first_action is not None else 0

            if d >= 10:
                continue

            for a in [1, 2, 3, 4, 0]:
                if a == 0:
                    nx, ny = pos[0], pos[1]
                else:
                    nx, ny = self._next_pos(pos, a)
                    if not self._passable(grid, nx, ny) or (nx, ny) in blocked:
                        continue
                if danger_map[nx, ny] <= d + 1:
                    continue

                npos = (nx, ny)
                if npos not in seen:
                    seen.add(npos)
                    q.append((npos, d + 1, a if first_action is None else first_action))

        return best_action

    def _can_escape(self, grid, start, blocked, danger_map):
        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, d = q.popleft()
            # If we reach a completely safe tile or one that explodes far in the future
            if danger_map[pos[0], pos[1]] == 999 or danger_map[pos[0], pos[1]] > d + 5:
                return True
            if d >= 10:
                continue
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                if not self._passable(grid, nx, ny) or (nx, ny) in blocked:
                    continue
                if danger_map[nx, ny] <= d + 1:
                    continue
                npos = (nx, ny)
                if npos not in seen:
                    seen.add(npos)
                    q.append((npos, d + 1))
        return False

    def _get_distance_map(self, grid, blocked, targets):
        dist_map = {}
        if not targets:
            return dist_map
        import heapq
        q = [(0, t) for t in targets]
        for t in targets:
            dist_map[t] = 0
            
        while q:
            d, pos = heapq.heappop(q)
            if d > dist_map.get(pos, 999):
                continue
            if d >= 15:
                continue
            for a in [1, 2, 3, 4]:
                nx, ny = pos[0] + self.MOVES[a][0], pos[1] + self.MOVES[a][1]
                if not self._in_bounds(grid, nx, ny) or (nx, ny) in blocked:
                    continue
                cell = grid[nx, ny]
                if cell == 1 or cell == 2:  # WALL or BOX (non-traversable)
                    continue
                
                # Normal grass/item transition cost is 1
                nd = d + 1
                npos = (nx, ny)
                if nd < dist_map.get(npos, 999):
                    dist_map[npos] = nd
                    heapq.heappush(q, (nd, npos))
        return dist_map

    def _has_line_of_sight(self, grid, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        if x1 != x2 and y1 != y2:
            return False
        if x1 == x2:
            # Horizontal LOS
            min_y, max_y = min(y1, y2), max(y1, y2)
            for y in range(min_y + 1, max_y):
                if grid[x1, y] in [Map.WALL, Map.BOX]:
                    return False
        else:
            # Vertical LOS
            min_x, max_x = min(x1, x2), max(x1, x2)
            for x in range(min_x + 1, max_x):
                if grid[x, y1] in [Map.WALL, Map.BOX]:
                    return False
        return True

    def _item_tiles(self, grid):
        return {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if grid[x, y] in [3, 4]
        }

    def _box_bomb_spots(self, grid, blocked, bomb_radius):
        spots = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] != 2:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for step in range(1, bomb_radius + 1):
                        nx, ny = x + dx * step, y + dy * step
                        if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]):
                            break
                        if grid[nx, ny] == 1 or grid[nx, ny] == 2:  # WALL or another BOX
                            break
                        if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                            spots.add((nx, ny))
        return spots


    def _adjust_action_score(self, a, score, vals, boxes_hit, enemies, bomb_radius, closest_dist, phase, my_pos):
        if a == 5:
            enemies_hit = 0
            for ex, ey, _, _ in enemies:
                dx = abs(my_pos[0] - ex)
                dy = abs(my_pos[1] - ey)
                if (dx == 0 and dy <= bomb_radius) or (dy == 0 and dx <= bomb_radius):
                    enemies_hit += 1
            
            if boxes_hit > 0 or enemies_hit > 0 or self.current_step > 400:
                score += 3.0
                score += boxes_hit * 3.5
                score += enemies_hit * 6.0
                if closest_dist <= 3:
                    score += 2.0
            else:
                score -= 5.0
                
            if vals["kill"] > 0:
                score += 1.5 * vals["kill"]
        else:
            if vals["item"] > 0:
                score += 2.5 * vals["item"]
                
        if phase == "TIE_BREAKER" and a == 5:
            score += 2.0
            
        return score

    def _count_hits(self, grid, x, y, bomb_radius):
        boxes = 0
        items = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for step in range(1, bomb_radius + 1):
                nx, ny = x + dx * step, y + dy * step
                if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]):
                    break
                cell = grid[nx, ny]
                if cell == 1:  # WALL
                    break
                if cell == 2:  # BOX
                    boxes += 1
                    break
                if cell in [3, 4]:  # ITEM
                    items += 1
        return boxes, items
    def _get_enemy_metrics(self, grid, enemy_pos, blocked, danger_map, my_pos):
        queue = [enemy_pos]
        visited = {enemy_pos}
        safe_tiles = 0
        head = 0
        
        while head < len(queue):
            curr = queue[head]
            head += 1
            cx, cy = curr
            if danger_map[cx, cy] == 999:
                safe_tiles += 1
                
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                        if (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                            
        total_reachable = len(queue)
        cut_vertices = 0
        is_my_pos_enemy_bottleneck = 0.0
        
        if total_reachable > 1 and total_reachable <= 25:
            # Analyze bottlenecks/cut vertices for the enemy
            for bx, by in queue:
                if (bx, by) == enemy_pos:
                    continue
                
                # Run BFS from enemy_pos excluding (bx, by)
                sub_visited = {enemy_pos}
                sub_queue = deque([enemy_pos])
                reached_count = 0
                
                while sub_queue:
                    cx, cy = sub_queue.popleft()
                    reached_count += 1
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) == (bx, by):
                            continue
                        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                            if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                                if (nx, ny) not in sub_visited:
                                    sub_visited.add((nx, ny))
                                    sub_queue.append((nx, ny))
                                    
                lost_area = total_reachable - reached_count - 1
                if lost_area > 0:
                    cut_vertices += 1
                    if (bx, by) == my_pos and lost_area >= 2:
                        is_my_pos_enemy_bottleneck = 1.0
                        
        trap_potential = float(cut_vertices) / (total_reachable + 1e-5)
        
        blocked_neighbors = 0
        ex, ey = enemy_pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = ex + dx, ey + dy
            if nx < 0 or nx >= grid.shape[0] or ny < 0 or ny >= grid.shape[1]:
                blocked_neighbors += 1
            elif not self._passable(grid, nx, ny) or (nx, ny) in blocked:
                blocked_neighbors += 1
                
        return total_reachable, safe_tiles, blocked_neighbors, trap_potential, is_my_pos_enemy_bottleneck

    def _simulate_rollout(self, grid, my_pos, start_action, danger_map, bombs, enemies, bomb_radius):
        curr_pos = self._next_pos(my_pos, start_action)
        if start_action == 5:
            curr_pos = my_pos
            
        my_bomb_pos = my_pos if start_action == 5 else None
        
        # Calculate dynamic steps based on maximum bomb timer on the board
        max_bomb_timer = 8  # Default minimum steps to ensure standard bomb escape
        for b in bombs:
            max_bomb_timer = max(max_bomb_timer, int(b[2]))
        if start_action == 5:
            max_bomb_timer = max(max_bomb_timer, 7)
            
        steps = max_bomb_timer + 1
        steps = min(14, steps)  # Cap steps to avoid performance timeout
        
        path = [curr_pos]
        
        for t in range(1, steps + 1):
            orig_danger = danger_map[curr_pos[0], curr_pos[1]]
            if orig_danger == t:
                return -100.0 + t
                    
            if t == steps:
                break
                
            best_next = None
            best_score = -999.0
            
            for act in range(5):
                n_pos = self._next_pos(curr_pos, act)
                if not (0 <= n_pos[0] < grid.shape[0] and 0 <= n_pos[1] < grid.shape[1]):
                    continue
                if not self._passable(grid, n_pos[0], n_pos[1]):
                    continue
                is_bomb_blocked = False
                for bx, by, bt, br in bombs:
                    if bx == n_pos[0] and by == n_pos[1] and bt > t:
                        is_bomb_blocked = True
                        break
                if my_bomb_pos == n_pos:
                    is_bomb_blocked = True
                if is_bomb_blocked:
                    continue
                    
                n_danger = danger_map[n_pos[0], n_pos[1]]
                if n_danger == t + 1:
                    continue
                        
                score = 0.0
                if n_danger == 999:
                    score += 10.0
                else:
                    score += float(n_danger - t) * 0.5
                    
                free_neighbors = 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = n_pos[0] + dx, n_pos[1] + dy
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                        if self._passable(grid, nx, ny):
                            free_neighbors += 1
                score += free_neighbors * 0.2
                
                if score > best_score:
                    best_score = score
                    best_next = n_pos
                    
            if best_next is None:
                return -100.0 + t
            curr_pos = best_next
            path.append(curr_pos)
            
        final_score = 10.0
        end_pos = path[-1]
        free_neighbors = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = end_pos[0] + dx, end_pos[1] + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if self._passable(grid, nx, ny):
                    free_neighbors += 1
        final_score += free_neighbors * 1.5
        
        min_enemy_dist = 999
        for ex, ey, _, _ in enemies:
            d = abs(end_pos[0] - ex) + abs(end_pos[1] - ey)
            if d < min_enemy_dist:
                min_enemy_dist = d
        if min_enemy_dist < 999:
            final_score += 5.0 / (min_enemy_dist + 1)
            
        if danger_map[end_pos[0], end_pos[1]] == 999:
            final_score += 20.0
            
        return final_score

# Expose Agent for dynamically loaded runtime
Agent = FSMAgent
