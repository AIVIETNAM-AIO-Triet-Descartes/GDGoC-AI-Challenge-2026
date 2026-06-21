"""
Hybrid Agent — Game Theory + RL (DQN)

Kiến trúc (theo docs/triet_docs/plan.md):
    obs
     ├─ [B0] build_time_layered_danger_map()  → danger_map
     ├─ [B8] get_safe_actions(danger_map)      → safe_actions  (override sinh tồn)
     ├─ [B1] compute_utility_map()             → utility_map
     ├─ [B2] compute_enemy_prob_map()          → enemy_prob
     ├─ [B3] compute_bomb_attack_score()       → bomb_attack_score (scalar)
     ├─ [B4] compute_threat_zone_map()         → threat_zone_map
     ├─ [B5] encode_obs_enriched()             → map_feat (13,13,13), aux_feat (6,)
     │        DQNModel.forward()               → q_values (6,)
     ├─ [B6] get_phase_weights()               → weights (FSM phase)
     └─ [B7] expectimax_filter()               → final_action

Trạng thái: Giai đoạn 1 (B0,B8,B1,B2,B3,B4) ĐÃ implement.
Giai đoạn 2-3 (B5,B6,B7,B9) còn skeleton — sẽ điền sau.
"""

from collections import deque
from pathlib import Path

import numpy as np

SAFE_TICK = 999
GRID = 13


# ───────────────────────── Giai đoạn 1: Game Theory Layers ─────────────────────────

def _blast_tiles_list(grid, bx, by, radius):
    """[B0] Helper cho build_time_layered_danger_map.
    Trả về list tọa độ các ô bị ảnh hưởng bởi bom tại (bx, by) với bán kính radius.
    Lan theo 4 hướng, dừng lại khi gặp tường hoặc hộp (không xuyên vật cản)."""
    tiles = [(bx, by)]
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for r in range(1, radius + 1):
            nx, ny = bx + dx * r, by + dy * r
            if not (0 <= nx < GRID and 0 <= ny < GRID):
                break
            if grid[nx, ny] == 1:      # tường — chặn, không tính ô tường
                break
            tiles.append((nx, ny))
            if grid[nx, ny] == 2:      # hộp — blast tới đây rồi dừng (phá hộp)
                break
    return tiles


def build_time_layered_danger_map(obs):
    """[B0] Nền tảng sinh tồn — đầu ra được dùng bởi B1, B7, B8 và encoder B5.
    Trả về ma trận 13×13 chứa tick sớm nhất mà mỗi ô sẽ bị nổ (SAFE_TICK=999 nếu an toàn).
    Xử lý chain reaction: bom A kích bom B → tick của B được cập nhật theo A."""
    grid = obs["map"]
    players = obs["players"]
    bombs = obs["bombs"]

    danger_map = np.full((GRID, GRID), SAFE_TICK, dtype=np.int32)
    if len(bombs) == 0:
        return danger_map

    # [timer, bx, by, radius] — list mutable để ép timer khi chain reaction
    bomb_list = []
    for b in bombs:
        bx, by, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        radius = 1 + int(players[owner][4])
        bomb_list.append([timer, bx, by, radius])

    # Resolve chain reactions: bom nổ sớm chạm bom nổ muộn → ép timer muộn = sớm
    changed = True
    while changed:
        changed = False
        bomb_list.sort(key=lambda x: x[0])
        for i in range(len(bomb_list)):
            ta, ax, ay, ra = bomb_list[i]
            blast_a = set(_blast_tiles_list(grid, ax, ay, ra))
            for j in range(len(bomb_list)):
                if i == j:
                    continue
                tb, bx, by, rb = bomb_list[j]
                if (bx, by) in blast_a and tb > ta:
                    bomb_list[j][0] = ta
                    changed = True

    # Fill tick nhỏ nhất cho từng ô
    for timer, bx, by, radius in bomb_list:
        for tx, ty in _blast_tiles_list(grid, bx, by, radius):
            if timer < danger_map[tx, ty]:
                danger_map[tx, ty] = timer

    return danger_map


def _line_clear(grid, ax, ay, bx, by):
    """[B1] Helper cho compute_utility_map.
    Kiểm tra đường nhìn thẳng (line-of-sight) giữa (ax,ay) và (bx,by) — chỉ cùng hàng/cột.
    Trả về True nếu không bị tường/hộp chặn; địch không có LOS sẽ có sức hút thấp hơn."""
    if ax == bx:
        for y in range(min(ay, by) + 1, max(ay, by)):
            if grid[ax, y] in (1, 2):
                return False
        return True
    if ay == by:
        for x in range(min(ax, bx) + 1, max(ax, bx)):
            if grid[x, ay] in (1, 2):
                return False
        return True
    return False   # không thẳng hàng → coi như không có LOS


def compute_utility_map(obs, agent_id, danger_map):
    """[B1] Tạo bản đồ hấp dẫn chiến thuật cho toàn bộ grid.
    Trả về ma trận 13×13 giá trị trong [-1, 1]: item > box > địch (giảm nếu bị tường chặn),
    ô có tick<=1 bị phạt nặng. Đầu vào cho encoder B5 và expectimax B7."""
    grid = obs["map"]
    players = obs["players"]
    my_x = int(players[agent_id][0])
    my_y = int(players[agent_id][1])
    U = np.zeros((GRID, GRID), dtype=np.float32)

    BOX_BASE = 3.0
    BOX_DECAY = 2.0

    # (+) Items & Box
    for x in range(GRID):
        for y in range(GRID):
            cell = grid[x, y]
            if cell == 4:      # item capacity
                d = abs(x - my_x) + abs(y - my_y)
                U[x, y] += 5.0 / (d + 1)
            elif cell == 3:    # item radius
                d = abs(x - my_x) + abs(y - my_y)
                U[x, y] += 4.0 / (d + 1)
            elif cell == 2:    # box
                d = abs(x - my_x) + abs(y - my_y)
                U[x, y] += BOX_BASE / (d + BOX_DECAY)

    # (+) Enemy — có LOS check
    for i, p in enumerate(players):
        if i != agent_id and p[2] == 1:
            ex, ey = int(p[0]), int(p[1])
            d = abs(ex - my_x) + abs(ey - my_y)
            if _line_clear(grid, my_x, my_y, ex, ey):
                U[ex, ey] += 1.5 / (d + 1)
            else:
                U[ex, ey] += 0.4 / (d + 1)

    # (−) Blast zones — stratified penalty theo danger_map
    STRATIFIED = {1: -10.0, 2: -5.0, 3: -2.0}
    for x in range(GRID):
        for y in range(GRID):
            pen = STRATIFIED.get(int(danger_map[x, y]))
            if pen is not None:
                U[x, y] += pen

    max_abs = np.abs(U).max() + 1e-8
    return U / max_abs


def _get_passable_neighbors(grid, x, y):
    """[B2] Ô lân cận đi được (gồm đứng yên), không phải tường/hộp."""
    neighbors = []
    for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID and 0 <= ny < GRID and grid[nx, ny] not in (1, 2):
            neighbors.append((nx, ny))
    return neighbors


def _enemy_tile_score(grid, x, y):
    """[B2] Độ hấp dẫn của ô với địch — item cao, cỏ thường, còn lại 0."""
    if grid[x, y] in (3, 4):
        return 3.0
    if grid[x, y] == 0:
        return 1.0
    return 0.0


def _softmax(x, temperature=1.0):
    e = np.exp((x - x.max()) / temperature)
    return e / e.sum()


def compute_enemy_prob_map(obs, agent_id, k_steps=3):
    """[B2] Dự đoán vị trí địch trong k bước tới bằng mô hình Markov.
    Trả về ma trận 13×13 xác suất tổng=1 (heatmap tổng hợp của 3 địch còn sống).
    Đầu vào cho bomb_attack_score B3 và encoder B5; cần vectorize nếu >10ms."""
    grid = obs["map"]
    players = obs["players"]
    prob_map = np.zeros((GRID, GRID), dtype=np.float32)

    for i, p in enumerate(players):
        if i == agent_id or p[2] == 0:
            continue

        single_prob = np.zeros((GRID, GRID), dtype=np.float32)
        single_prob[int(p[0]), int(p[1])] = 1.0

        for _ in range(k_steps):
            new_prob = np.zeros((GRID, GRID), dtype=np.float32)
            for x in range(GRID):
                for y in range(GRID):
                    if single_prob[x, y] < 1e-6:
                        continue
                    neighbors = _get_passable_neighbors(grid, x, y)
                    scores = np.array([_enemy_tile_score(grid, nx, ny)
                                       for nx, ny in neighbors])
                    move_probs = _softmax(scores, temperature=1.5)
                    for (nx, ny), mp in zip(neighbors, move_probs):
                        new_prob[nx, ny] += single_prob[x, y] * mp
            single_prob = new_prob

        prob_map += single_prob

    return prob_map / (prob_map.sum() + 1e-8)


def compute_bomb_attack_score(obs, agent_id, enemy_prob_map):
    """[B3] Đánh giá lợi ích tấn công nếu đặt bom tại vị trí hiện tại.
    Trả về scalar: tổng xác suất địch trong vùng nổ (từ enemy_prob_map).
    Có guard CPU — bỏ qua tính toán nếu không có địch trong phạm vi radius+2."""
    grid = obs["map"]
    players = obs["players"]
    my_x = int(players[agent_id][0])
    my_y = int(players[agent_id][1])
    radius = 1 + int(players[agent_id][4])

    any_close = any(
        abs(int(p[0]) - my_x) + abs(int(p[1]) - my_y) <= radius + 2
        for i, p in enumerate(players)
        if i != agent_id and p[2] == 1
    )
    if not any_close:
        return 0.0

    tiles = _blast_tiles_list(grid, my_x, my_y, radius)
    return float(sum(float(enemy_prob_map[tx, ty]) for tx, ty in tiles))


def compute_threat_zone_map(obs, agent_id):
    """[B4] Tính vùng nguy hiểm do bom địch có thể đặt (không phải bom đang tồn tại).
    Trả về ma trận 13×13 trong [0, 1], weight theo radius địch / 5.
    Dùng trong expectimax B7 như hạng phạt để tránh ô địch dễ kiểm soát."""
    grid = obs["map"]
    players = obs["players"]
    threat = np.zeros((GRID, GRID), dtype=np.float32)

    for i, p in enumerate(players):
        if i == agent_id or p[2] == 0:
            continue
        ex, ey = int(p[0]), int(p[1])
        radius = 1 + int(p[4])
        weight = radius / 5.0
        for tx, ty in _blast_tiles_list(grid, ex, ey, radius):
            threat[tx, ty] += weight

    max_val = threat.max() + 1e-8
    return threat / max_val


# ───────────────────────── Giai đoạn 1 (B8): Safety Layer ─────────────────────────

# index = action: 0 STOP, 1 LEFT, 2 RIGHT, 3 UP, 4 DOWN, 5 PLACE_BOMB
_MOVE_DELTAS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]


def _can_escape_after_placing(grid, mx, my, bomb_positions, danger_soon, radius):
    """[B8] BFS check: sau khi đặt bom tại (mx,my), còn ô thoát an toàn không?"""
    my_blast = set(_blast_tiles_list(grid, mx, my, radius))
    combined = danger_soon | my_blast | {(mx, my)}
    new_bombs = bomb_positions | {(mx, my)}

    q = deque([(mx, my, 0)])
    seen = {(mx, my)}
    while q:
        x, y, depth = q.popleft()
        if (x, y) not in combined:
            return True
        if depth >= 8:
            continue
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen:
                continue
            if not (0 <= nx < GRID and 0 <= ny < GRID):
                continue
            if grid[nx, ny] in (1, 2):
                continue
            if (nx, ny) in new_bombs:
                continue
            seen.add((nx, ny))
            q.append((nx, ny, depth + 1))
    return False


def get_safe_actions(obs, agent_id, danger_map):
    """[B8] Bộ lọc sinh tồn cứng — loại bỏ mọi action đưa agent vào ô tick<=1.
    Trả về list action hợp lệ; nếu không còn action nào an toàn thì fallback [0] (STOP).
    Được gọi sớm nhất trong pipeline; nếu chỉ còn [0] thì bỏ qua RL/expectimax."""
    grid = obs["map"]
    players = obs["players"]
    bombs = obs["bombs"]
    my_x = int(players[agent_id][0])
    my_y = int(players[agent_id][1])

    bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}

    danger_soon = set()
    for x in range(GRID):
        for y in range(GRID):
            if danger_map[x, y] < SAFE_TICK:
                danger_soon.add((x, y))

    safe_actions = []
    for action in range(6):
        if action == 5:   # PLACE_BOMB
            if players[agent_id][3] > 0 and (my_x, my_y) not in bomb_positions:
                if _can_escape_after_placing(grid, my_x, my_y, bomb_positions,
                                             danger_soon, 1 + int(players[agent_id][4])):
                    safe_actions.append(5)
            continue

        dx, dy = _MOVE_DELTAS[action]
        nx, ny = my_x + dx, my_y + dy
        if not (0 <= nx < GRID and 0 <= ny < GRID):
            continue
        if grid[nx, ny] in (1, 2):
            continue
        if (nx, ny) in bomb_positions and action != 0:
            continue
        if danger_map[nx, ny] > 1:
            safe_actions.append(action)

    return safe_actions if safe_actions else [0]


# ───────────────────────── Giai đoạn 2: RL Core (TODO) ─────────────────────────

def encode_obs_enriched(obs, agent_id, utility_map, enemy_prob_map,
                        threat_zone_map, danger_map):
    """[B5] Chuyển obs thô + 4 map game-theory thành tensor đầu vào cho DQN.
    Trả về (map_feat: 13×13×13, aux_feat: 6,) — 13 channel gồm one-hot map,
    vị trí agent/địch, utility, enemy_prob, threat, danger; 6 aux là thông số agent + dist địch."""
    grid = obs["map"]
    players = obs["players"]
    H, W = grid.shape

    # CH 0-4: one-hot map (grass/wall/box/item_radius/item_capacity)
    map_channels = [(grid == v).astype(np.float32) for v in [0, 1, 2, 3, 4]]

    # CH 5: vị trí của mình
    my_x, my_y = int(players[agent_id][0]), int(players[agent_id][1])
    my_pos_ch = np.zeros((H, W), dtype=np.float32)
    if players[agent_id][2] == 1:
        my_pos_ch[my_x, my_y] = 1.0

    # CH 6-8: 3 địch (1 channel/địch, 0 nếu chết) — bỏ channel của chính mình
    enemy_channels = []
    for i in range(4):
        ch = np.zeros((H, W), dtype=np.float32)
        if i != agent_id and players[i][2] == 1:
            ch[int(players[i][0]), int(players[i][1])] = 1.0
        enemy_channels.append(ch)
    enemy_channels = [ch for idx, ch in enumerate(enemy_channels) if idx != agent_id]

    # CH 12: danger_map chuẩn hóa — 0=nguy hiểm ngay, 1=an toàn
    danger_norm = np.clip(danger_map.astype(np.float32) / 7.0, 0.0, 1.0)

    map_feat = np.stack([
        *map_channels,     # CH 0-4
        my_pos_ch,         # CH 5
        *enemy_channels,   # CH 6-8
        utility_map,       # CH 9
        enemy_prob_map,    # CH 10
        threat_zone_map,   # CH 11
        danger_norm,       # CH 12
    ], axis=0).astype(np.float32)   # (13, 13, 13)

    # Aux scalars
    bombs_left = float(players[agent_id][3]) / 5.0
    radius_bonus = float(players[agent_id][4]) / 5.0
    alive_enemies = sum(1 for i, p in enumerate(players)
                        if i != agent_id and p[2] == 1)
    num_alive = alive_enemies / 3.0

    enemy_dists = []
    for i in range(4):
        if i == agent_id:
            continue
        if players[i][2] == 0:
            enemy_dists.append(1.0)
        else:
            d = abs(int(players[i][0]) - my_x) + abs(int(players[i][1]) - my_y)
            enemy_dists.append(d / 20.0)

    aux_feat = np.array([bombs_left, radius_bonus, num_alive, *enemy_dists],
                        dtype=np.float32)

    return map_feat, aux_feat


# DQNModel ở model.py (cùng thư mục): `from model import DQNModel`.
# ReplayBuffer + TrainingAgent ở trainer.py (chỉ dùng khi train, không submit cần).


# ───────────────────────── Giai đoạn 3: Decision Layer (TODO) ─────────────────────────

def get_phase_weights(obs, agent_id, step=0):
    """[B6] Xác định pha chiến thuật hiện tại (FARMER / ZONER / ASSASSIN) theo power_score.
    Trả về dict weights {w_rl, w_gt, w_threat, w_bomb, w_survive} để expectimax biết
    cân bằng giữa RL và game-theory theo từng giai đoạn trận đấu. TODO."""
    raise NotImplementedError


def compute_survivability(obs, agent_id, danger_map, next_x, next_y, depth=3):
    """[B7] Đánh giá khả năng thoát hiểm nếu agent di chuyển tới (next_x, next_y).
    Trả về scalar: số ô an toàn (tick > 1) reach được bằng BFS trong depth bước.
    Dùng trong expectimax như bonus sinh tồn để tránh chọn action dẫn vào ngõ cụt. TODO."""
    raise NotImplementedError


def expectimax_filter(obs, agent_id, q_values, utility_map, threat_zone_map,
                      bomb_attack_score, danger_map, safe_actions, weights):
    """[B7] Tầng quyết định cuối — kết hợp DQN q-values với tín hiệu game-theory.
    Công thức: w_rl·q + w_gt·utility − w_threat·threat + w_bomb·bomb (action 5) + w_survive·survive.
    Trả về int action tốt nhất trong safe_actions theo điểm tổng hợp. TODO."""
    raise NotImplementedError


# ───────────────────────── Agent ─────────────────────────

class HybridAgent:
    """[B9] Agent chính — điều phối toàn bộ pipeline từ obs đến action.
    Nạp DQNModel từ model.pth cùng thư mục; gọi B0→B8→B1→B2→B3→B4→B5→B6→B7 theo thứ tự.
    Interface chuẩn submit: __init__(agent_id) và act(obs) -> int."""

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.step = 0
        # TODO: nạp DQN
        #   from model import DQNModel
        #   self.model = DQNModel(map_shape=(13,13,13), aux_dim=6, output_dim=6)
        #   weights = Path(__file__).parent / "model.pth"
        #   self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        #   self.model.eval()
        self.model = None

    def act(self, obs: dict) -> int:
        self.step += 1
        # TODO: pipeline B0→8→1→2→3→4→5→6→7. Tạm trả STOP để loader chạy được.
        return 0


# Alias để máy chấm tìm thấy "Agent" trực tiếp (runtime_guard ưu tiên class tên 'Agent').
Agent = HybridAgent
