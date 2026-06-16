import random
import time
from collections import deque
import numpy as np


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

    def __init__(self, agent_id: int, logs: bool = True):
        import os
        import sys
        self.agent_id = int(agent_id)
        
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

        # 3. Determine Phase using power_score and current_step
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

        # 4. Utility Scoring for 6 actions
        action_scores = {}
        valid_actions = []

        # Cache bomb escape logic to save CPU
        bomb_escape_valid = None
        virtual_danger_map = None

        for action in [0, 1, 2, 3, 4, 5]:
            next_pos = self._next_pos(my_pos, action)

            # --- PRE-FILTERING ---
            # Do not walk into wall, box, or other bombs (unless standing on it)
            if action != 5 and action != 0:
                if not self._passable(grid, next_pos[0], next_pos[1]) or (next_pos in blocked):
                    continue
            
            # Prevent stepping on a tile that is exploding next tick
            if danger_map[next_pos[0], next_pos[1]] == 1:
                continue

            # Check bomb validity
            if action == 5:
                if bombs_left == 0 or my_pos in bomb_positions:
                    continue
                # Calculate virtual danger map with the new bomb
                virtual_danger_map = self._build_time_layered_danger_map(
                    grid, bombs + [[my_pos[0], my_pos[1], 7, self.agent_id]], players
                )
                if bomb_escape_valid is None:
                    # Place bomb blocks the tile we are standing on, so we must add it to blocked
                    bomb_escape_valid = self._can_escape(grid, my_pos, blocked | {my_pos}, virtual_danger_map)
                
                self.log(f"BOMB CHECK | bombs={bombs_left} escape={bomb_escape_valid}")
                if not bomb_escape_valid:
                    self.log("Pre-filtered PLACE_BOMB: No safe escape path.")
                    continue

            # Action is valid
            valid_actions.append(action)

        if not valid_actions:
            self.log("No valid actions found. Falling back to STOP.")
            return 0

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
                "danger": 0.0
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
            
            # If our current position is dangerous, survive score must prioritize getting closer to safety (gradient)
            if danger_map[my_pos[0], my_pos[1]] != 999:
                vals["survive"] = 1.0 / (dist_safety + 1)
            else:
                vals["survive"] = surv_count

            vals["mobility"] = mob
            vals["territory"] = terr

            # Enemy Pressure with Line of Sight (LOS)
            pressure_sum = 0.0
            for ex, ey, _, _ in enemies:
                dist = abs(next_pos[0] - ex) + abs(next_pos[1] - ey)
                los = 1.0 if self._has_line_of_sight(grid, next_pos, (ex, ey)) else 0.0
                pressure_sum += (los * 5.0) + (1.0 / (dist + 1))
            vals["pressure"] = pressure_sum

            # Danger Penalty (normalized mapping: 2 -> 1.0, 3 -> 0.4, else -> 0.0)
            tile_danger = action_danger_map[next_pos[0], next_pos[1]]
            if tile_danger == 2:
                vals["danger"] = 1.0
            elif tile_danger == 3:
                vals["danger"] = 0.4

            # Kill Reward: Tactical Enemy Mobility Assessment
            close_enemies = [
                (ex, ey) for ex, ey, _, _ in enemies
                if abs(next_pos[0] - ex) + abs(next_pos[1] - ey) <= bomb_radius + 4
            ]
            if action == 5 and close_enemies:
                enemy_mob_loss = 0.0
                for ex, ey in close_enemies:
                    _, cur_mob, _, _ = self._evaluate_spatial_metrics(grid, (ex, ey), blocked, danger_map)
                    _, fut_mob, _, _ = self._evaluate_spatial_metrics(grid, (ex, ey), blocked | {my_pos}, virtual_danger_map)
                    enemy_mob_loss += max(0.0, float(cur_mob - fut_mob))
                vals["kill"] = min(1.0, enemy_mob_loss / 10.0)
        # Normalization over valid actions (kill is proportional, keep raw)
        for key in ["item", "box", "pressure", "survive", "mobility", "territory"]:
            min_val = min(raw_values[a][key] for a in valid_actions)
            max_val = max(raw_values[a][key] for a in valid_actions)
            diff = max_val - min_val
            for a in valid_actions:
                if diff > 1e-5:
                    raw_values[a][key] = (raw_values[a][key] - min_val) / diff
                else:
                    raw_values[a][key] = 0.0

        # Calculate final utility scores
        for a in valid_actions:
            vals = raw_values[a]
            score = (
                weights["item"] * vals["item"] +
                weights["box"] * vals["box"] +
                weights["kill"] * vals["kill"] +
                weights["pressure"] * vals["pressure"] +
                weights["survive"] * vals["survive"] +
                weights["mobility"] * vals["mobility"] +
                weights["territory"] * vals["territory"] -
                weights["danger"] * vals["danger"]
            )
            
            # Action 5 (PLACE_BOMB) inherently reduces mobility and territory.
            # We must add an explicit bonus to actually pull the trigger when on a target.
            if a == 5:
                boxes_hit, items_hit = self._count_hits(grid, my_pos[0], my_pos[1], bomb_radius)
                if my_pos in box_spots:
                    score += boxes_hit * 2.5
                
                if vals["kill"] > 0:
                    score += weights["kill"] * vals["kill"]
                    
                self.log(f"BOMB SCORE = {score:.2f}")
                self.log(f"BOMB VALID = {bomb_escape_valid}")
                    
            if phase == "TIE_BREAKER" and a == 5:
                # Encourage placing a bomb to break ties (Bombs Placed is 4th tie-breaker)
                score += 1.5
                
            action_scores[a] = score
            self.log(
                f"Action {a} | Score: {score:.2f} | "
                f"Item: {vals['item']:.2f}, Box: {vals['box']:.2f}, Kill: {vals['kill']:.2f}, "
                f"Pressure: {vals['pressure']:.2f}, Survive: {vals['survive']:.2f}, "
                f"Mobility: {vals['mobility']:.2f}, Territory: {vals['territory']:.2f}, Danger: {vals['danger']:.2f}"
            )

        best_action = max(valid_actions, key=lambda a: action_scores[a])
        elapsed = (time.perf_counter() - start_time) * 1000
        self.log(f"Selected action: {best_action} (Time: {elapsed:.2f}ms)")
        return best_action

    ############################################################
    # HELPER ALGORITHMS
    ############################################################

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
        q = deque([(t, 0) for t in targets])
        seen = set(targets)
        while q:
            pos, d = q.popleft()
            dist_map[pos] = d
            if d >= 15:
                continue
            for a in [1, 2, 3, 4]:
                nx, ny = pos[0] + self.MOVES[a][0], pos[1] + self.MOVES[a][1]
                if not self._passable(grid, nx, ny) or (nx, ny) in blocked:
                    continue
                npos = (nx, ny)
                if npos not in seen:
                    seen.add(npos)
                    q.append((npos, d + 1))
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
# Expose Agent for dynamically loaded runtime
Agent = FSMAgent
