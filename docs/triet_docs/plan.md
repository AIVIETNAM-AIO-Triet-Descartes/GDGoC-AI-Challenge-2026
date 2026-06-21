# Kế hoạch phát triển Agent — Bomberland AI Challenge 2026

**Hướng đi:** Hybrid (Game Theory + RL)  
**Kiến trúc:** Danger Map → Safety Layer → Utility Map → Enemy Prob Map → Bomb Attack Score → Threat Zone Map → RL Enriched State → FSM Phase Weights → Expectimax Filter → Action

```
obs
 │
 ├─► [Bước 0] build_time_layered_danger_map()  → danger_map          (13×13)   ← MỚI
 │                │
 ├─► [Bước 8] get_safe_actions(danger_map)     → safe_actions                  ← CẬP NHẬT
 │
 ├─► [Bước 1] compute_utility_map(danger_map)  → utility_map         (13×13)   ← CẬP NHẬT
 ├─► [Bước 2] compute_enemy_prob_map()         → enemy_prob          (13×13)
 │                │
 │                ├─► [Bước 3] compute_bomb_attack_score()  → bomb_attack_score (scalar) ← CẬP NHẬT
 │                └─► [Bước 4] compute_threat_zone_map()   → threat_zone_map   (13×13)
 │
 │             [Bước 5] encode_obs_enriched()
 │                │  stack 4 channels mới vào state (9→13 channels)             ← CẬP NHẬT
 │                ▼
 │             DQNModel.forward()               → q_values            (6,)
 │                │
 ├─► [Bước 6] get_phase_weights()              → weights                        ← MỚI
 │
 └─► [Bước 7] expectimax_filter(weights)       → final_action                  ← CẬP NHẬT
```

---

## Bước 0 — Time-layered Danger Map *(MỚI)*

### Mục tiêu
Thay thế cách tính `danger_now` / `danger_soon` cũ (chỉ phân 2 mức) bằng một map chính xác hơn: mỗi ô lưu **tick nào nó sẽ nổ**, kể cả **chain reaction**. Map này được dùng xuyên suốt pipeline — từ Safety Layer đến Utility Map đến Expectimax Filter.

```
danger_map[x, y] = tick sẽ nổ   (999 nếu an toàn)
```

### Cơ chế chain reaction
Sắp xếp bom theo timer tăng dần. Nếu blast của bom A chạm vào bom B → timer B bị ép bằng timer A. Lặp cho đến khi không còn thay đổi.

```
Ví dụ:
  Bom A: timer=2, Bom B: timer=6, B nằm trong blast A
  → Sau resolve: Bom B timer=2 (nổ cùng lúc với A)
```

### Code

```python
SAFE_TICK = 999

def build_time_layered_danger_map(obs):
    """
    Tính tick nổ của từng ô, kể cả chain reaction.
    Output: np.ndarray (13,13) int — giá trị = tick nổ, SAFE_TICK nếu an toàn.
    """
    grid    = obs["map"]
    players = obs["players"]
    bombs   = obs["bombs"]

    danger_map = np.full((13, 13), SAFE_TICK, dtype=np.int32)

    if len(bombs) == 0:
        return danger_map

    # Build mutable list [timer, bx, by, radius]
    bomb_list = []
    for b in bombs:
        bx, by, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        radius = 1 + int(players[owner][4])
        bomb_list.append([timer, bx, by, radius])

    # Resolve chain reactions
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

    # Fill danger_map với tick nhỏ nhất cho từng ô
    for timer, bx, by, radius in bomb_list:
        for tx, ty in _blast_tiles_list(grid, bx, by, radius):
            if timer < danger_map[tx, ty]:
                danger_map[tx, ty] = timer

    return danger_map


def _blast_tiles_list(grid, bx, by, radius):
    """Trả về list ô bị ảnh hưởng bởi bom tại (bx, by)."""
    tiles = [(bx, by)]
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        for r in range(1, radius + 1):
            nx, ny = bx + dx*r, by + dy*r
            if not (0 <= nx < 13 and 0 <= ny < 13): break
            if grid[nx, ny] == 1: break
            tiles.append((nx, ny))
            if grid[nx, ny] == 2: break
    return tiles
```

---

## Bước 1 — Tính Utility Map *(CẬP NHẬT)*

### Mục tiêu
Gán một con số thực cho mỗi ô 13×13, thể hiện mức độ "hấp dẫn" từ góc nhìn của agent. Được dùng ở 2 nơi:
- Stack vào RL như 1 input channel (Bước 5)
- Làm thành phần game theory trong `expectimax_filter()` (Bước 7)

### Thay đổi so với phiên bản cũ

| | Cũ | Mới |
|---|---|---|
| Blast penalty | `8.0 × urgency` theo timer từng bom | **Stratified penalty** theo tick từ `danger_map` |
| Enemy attraction | Hút mọi hướng | **LOS check** — giảm lực hút nếu bị tường chặn |

### Các thành phần

#### 1.1 Item & Box — hút agent lại gần

| Object | Utility |
|---|---|
| Item_Capacity (4) | `5.0 / (d + 1)` |
| Item_Radius (3) | `4.0 / (d + 1)` |
| Box (2) | `BOX_BASE / (d + BOX_DECAY)` — mặc định `3.0 / (d + 2.0)` |

#### 1.2 Enemy — hút nhẹ, có LOS check

```python
if _line_clear(grid, my_x, my_y, ex, ey):
    U[ex, ey] += 1.5 / (d + 1)   # LOS thông → hút bình thường
else:
    U[ex, ey] += 0.4 / (d + 1)   # bị tường chặn → hút yếu hơn nhiều
```

> **Lý do:** Nếu không có LOS, agent bị hút về phía địch nhưng thực ra không thể tạo áp lực gì — gây ra hành vi vô nghĩa.

#### 1.3 Blast zone — stratified penalty theo danger_map

```
tick == 1  → -10.0   (sắp nổ ngay, nguy hiểm tối đa)
tick == 2  →  -5.0   (rất nguy hiểm)
tick == 3  →  -2.0   (nguy hiểm vừa)
tick >= 4  →   0.0   (bỏ qua — còn nhiều thời gian)
```

> **Lý do bỏ mức tick ≥ 4:** Agent không nên né bom mới đặt ngay lập tức — vừa lãng phí di chuyển, vừa bỏ lỡ cơ hội tấn công.

### Helper: Line of Sight

```python
def _line_clear(grid, ax, ay, bx, by):
    """
    Kiểm tra đường thẳng từ (ax,ay) đến (bx,by) không bị tường/hộp chặn.
    Chỉ hợp lệ nếu cùng hàng hoặc cùng cột.
    """
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
```

### Code

```python
def compute_utility_map(obs, agent_id, danger_map):
    """
    Output: np.ndarray (13,13) normalized về [-1, 1]
    danger_map từ Bước 0 — dùng cho stratified penalty.
    """
    grid    = obs["map"]
    players = obs["players"]

    my_x = int(players[agent_id][0])
    my_y = int(players[agent_id][1])
    U = np.zeros((13, 13), dtype=np.float32)

    BOX_BASE  = 3.0
    BOX_DECAY = 2.0

    # (+) Items & Box
    for x in range(13):
        for y in range(13):
            if grid[x, y] == 4:
                d = abs(x - my_x) + abs(y - my_y)
                U[x, y] += 5.0 / (d + 1)
            elif grid[x, y] == 3:
                d = abs(x - my_x) + abs(y - my_y)
                U[x, y] += 4.0 / (d + 1)
            elif grid[x, y] == 2:
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

    # (−) Blast zones — stratified penalty từ danger_map
    STRATIFIED = {1: -10.0, 2: -5.0, 3: -2.0}
    for x in range(13):
        for y in range(13):
            tick = danger_map[x, y]
            if tick in STRATIFIED:
                U[x, y] += STRATIFIED[tick]

    max_abs = np.abs(U).max() + 1e-8
    return U / max_abs
```

---

## Bước 2 — Tính Enemy Probability Map (Markov Chain)

### Mục tiêu
Propagate xác suất vị trí của từng enemy qua `k_steps` bước. Output là heatmap 13×13 thể hiện xác suất địch sẽ ở ô nào. Được dùng ở 2 nơi:
- Stack vào RL như 1 input channel (Bước 5)
- Làm đầu vào cho `compute_bomb_attack_score()` (Bước 3)

> **Lưu ý:** Map này **không dùng để tránh né vật lý** — vai trò đó thuộc về `threat_zone_map` (Bước 4).

### Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `k_steps` | `3` | Số bước nhìn trước — xấp xỉ timer/2 của bom |
| `temperature` | `1.5` | Softmax temperature — cao hơn = địch khó đoán hơn |

### Code

```python
def compute_enemy_prob_map(obs, agent_id, k_steps=3):
    grid     = obs["map"]
    players  = obs["players"]
    prob_map = np.zeros((13, 13), dtype=np.float32)

    for i, p in enumerate(players):
        if i == agent_id or p[2] == 0:
            continue

        single_prob = np.zeros((13, 13), dtype=np.float32)
        single_prob[int(p[0]), int(p[1])] = 1.0

        for _ in range(k_steps):
            new_prob = np.zeros((13, 13), dtype=np.float32)
            for x in range(13):
                for y in range(13):
                    if single_prob[x, y] < 1e-6:
                        continue
                    neighbors  = _get_passable_neighbors(grid, x, y)
                    scores     = np.array([_enemy_tile_score(grid, nx, ny)
                                           for nx, ny in neighbors])
                    move_probs = _softmax(scores, temperature=1.5)
                    for (nx, ny), mp in zip(neighbors, move_probs):
                        new_prob[nx, ny] += single_prob[x, y] * mp
            single_prob = new_prob

        prob_map += single_prob

    return prob_map / (prob_map.sum() + 1e-8)


def _get_passable_neighbors(grid, x, y):
    neighbors = []
    for dx, dy in [(0,0),(0,1),(0,-1),(1,0),(-1,0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 13 and 0 <= ny < 13 and grid[nx, ny] not in (1, 2):
            neighbors.append((nx, ny))
    return neighbors


def _enemy_tile_score(grid, x, y):
    if grid[x, y] in (3, 4): return 3.0
    if grid[x, y] == 0:      return 1.0
    return 0.0


def _softmax(x, temperature=1.0):
    e = np.exp((x - x.max()) / temperature)
    return e / e.sum()
```

---

## Bước 3 — Tính Bomb Attack Score *(CẬP NHẬT)*

### Mục tiêu
Tính xác suất blast zone của bom đặt tại vị trí hiện tại sẽ hit enemy sau `k_steps` bước. Output là **1 scalar** — boost điểm cho PLACE_BOMB trong `expectimax_filter()`.

### Thay đổi: CPU Optimization

Chỉ tính full BFS nếu có ít nhất 1 enemy trong tầm `radius + 2`. Tránh tốn CPU khi không có địch gần.

```python
# Guard: nếu không có enemy nào đủ gần → bỏ qua, trả 0.0
any_close = any(
    abs(int(p[0]) - my_x) + abs(int(p[1]) - my_y) <= radius + 2
    for i, p in enumerate(players)
    if i != agent_id and p[2] == 1
)
if not any_close:
    return 0.0
```

### Code

```python
def compute_bomb_attack_score(obs, agent_id, enemy_prob_map):
    """
    Output: scalar float trong [0, ~3.0]
    CPU optimization: skip nếu không có enemy trong tầm radius + 2.
    """
    grid    = obs["map"]
    players = obs["players"]

    my_x   = int(players[agent_id][0])
    my_y   = int(players[agent_id][1])
    radius = 1 + int(players[agent_id][4])

    # CPU optimization guard
    any_close = any(
        abs(int(p[0]) - my_x) + abs(int(p[1]) - my_y) <= radius + 2
        for i, p in enumerate(players)
        if i != agent_id and p[2] == 1
    )
    if not any_close:
        return 0.0

    tiles = _blast_tiles_list(grid, my_x, my_y, radius)
    return sum(float(enemy_prob_map[tx, ty]) for tx, ty in tiles)
```

---

## Bước 4 — Tính Threat Zone Map

### Mục tiêu
Tính vùng nguy hiểm tiềm tàng nếu từng enemy đặt bom ngay tại vị trí của họ lúc này. Thể hiện nguy cơ **tương lai** dựa trên sức mạnh của địch — khác với blast zone của bom đang tồn tại (đã có trong `utility_map`).

Agent nên tránh đứng trong threat zone của địch có **radius lớn** — vì khi họ đặt bom, agent có ít thời gian escape hơn.

### Công thức

```
threat[tx, ty] += radius / 5.0   với (tx, ty) ∈ blast_tiles(enemy_pos, enemy_radius)
```

### Code

```python
def compute_threat_zone_map(obs, agent_id):
    """
    Output: np.ndarray (13,13) normalized về [0, 1]
    """
    grid    = obs["map"]
    players = obs["players"]
    threat  = np.zeros((13, 13), dtype=np.float32)

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
```

---

## Bước 5 — Encode Enriched State cho RL *(CẬP NHẬT)*

### Mục tiêu
Mở rộng state DQN baseline (9 channels, 3 aux scalars) với 4 channel từ Game Theory layer, bao gồm `danger_map` mới.

### So sánh trước / sau

| | Baseline DQN | Hybrid (v1) | Hybrid (v2 — hiện tại) |
|---|---|---|---|
| Map channels | 9 | 12 | **13** (+danger_map) |
| Aux scalars | 3 | 6 | **6** (giữ nguyên) |

### Map channels (9 → 13)

```
CH 0-4  : one-hot map (grass/wall/box/item_radius/item_capacity)
CH 5    : my_pos
CH 6-8  : 3 enemy_pos (1 channel/enemy, 0 nếu dead)
CH 9    : utility_map        (Bước 1)
CH 10   : enemy_prob_map     (Bước 2)
CH 11   : threat_zone_map    (Bước 4)
CH 12   : danger_map norm    (Bước 0) — tick/7, clip [0,1], 0=sắp nổ, 1=an toàn  ← MỚI
```

> **Lý do thêm `danger_map`:** RL cần biết mức độ khẩn cấp của từng ô — thông tin này chính xác hơn blast zone trong `utility_map` vì đã tính chain reaction.

### Aux scalars (6, giữ nguyên)

```
AUX 0 : bombs_left / 5
AUX 1 : radius_bonus / 5
AUX 2 : num_alive_enemies / 3
AUX 3 : dist_to_enemy_0 / 20
AUX 4 : dist_to_enemy_1 / 20
AUX 5 : dist_to_enemy_2 / 20
```

### Code

```python
def encode_obs_enriched(obs, agent_id,
                         utility_map, enemy_prob_map,
                         threat_zone_map, danger_map):
    """
    Output: map_feat (13, 13, 13), aux_feat (6,)
    """
    grid    = obs["map"]
    players = obs["players"]
    H, W    = grid.shape

    # CH 0-4: one-hot map
    map_channels = [(grid == v).astype(np.float32) for v in [0, 1, 2, 3, 4]]

    # CH 5: my position
    my_x, my_y = int(players[agent_id][0]), int(players[agent_id][1])
    my_pos_ch  = np.zeros((H, W), dtype=np.float32)
    if players[agent_id][2] == 1:
        my_pos_ch[my_x, my_y] = 1.0

    # CH 6-8: 3 enemy positions
    enemy_channels = []
    for i in range(4):
        ch = np.zeros((H, W), dtype=np.float32)
        if i != agent_id and players[i][2] == 1:
            ch[int(players[i][0]), int(players[i][1])] = 1.0
        enemy_channels.append(ch)
    enemy_channels = [ch for idx, ch in enumerate(enemy_channels) if idx != agent_id]

    # CH 12: danger_map normalized — 0=nguy hiểm ngay, 1=an toàn
    danger_norm = np.clip(danger_map.astype(np.float32) / 7.0, 0.0, 1.0)

    map_feat = np.stack([
        *map_channels,       # CH 0-4
        my_pos_ch,           # CH 5
        *enemy_channels,     # CH 6-8
        utility_map,         # CH 9
        enemy_prob_map,      # CH 10
        threat_zone_map,     # CH 11
        danger_norm,         # CH 12 ← MỚI
    ], axis=0).astype(np.float32)   # (13, 13, 13)

    # Aux scalars
    bombs_left    = float(players[agent_id][3]) / 5.0
    radius_bonus  = float(players[agent_id][4]) / 5.0
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
```

### Cập nhật DQNModel

```python
MAP_SHAPE = (13, 13, 13)
AUX_DIM   = 6
model = DQNModel(map_shape=MAP_SHAPE, aux_dim=AUX_DIM, output_dim=6)
```

---

## Bước 6 — FSM Phase Weights *(MỚI)*

### Mục tiêu
Thay thế weights cố định trong `expectimax_filter` bằng weights động theo giai đoạn game. Phase được xác định bởi `power_score` — phản ánh sức mạnh hiện tại của agent.

### Công thức power_score

```python
power_score = (bomb_radius * 1.5) + (bombs_left * 3)
```

Số bom quan trọng hơn radius (×3 vs ×1.5) — cần ≥2 bom để gài bẫy hiệu quả.

### Ba Phase

| Phase | Điều kiện | Trọng tâm |
|---|---|---|
| **FARMER** | `power_score < 8` | Farm item, tích lũy sức mạnh |
| **ZONER** | `power_score ≥ 8` và `step < 350` | Tạo áp lực, kết hợp tấn công |
| **ASSASSIN** | `power_score ≥ 8` và `step ≥ 350` | Tiêu diệt địch, chiếm lãnh thổ |

### Weights theo Phase

| Weight | FARMER | ZONER | ASSASSIN |
|---|---|---|---|
| `w_rl` | 0.50 | 0.45 | 0.40 |
| `w_gt` (utility) | 0.25 | 0.20 | 0.15 |
| `w_threat` | 0.10 | 0.10 | 0.05 |
| `w_bomb` (attack) | 0.05 | 0.15 | 0.25 |
| `w_survive` | 0.10 | 0.10 | 0.15 |

> **Giải thích:** FARMER ưu tiên RL + utility (farm), ASSASSIN tăng w_bomb để tích cực tấn công, w_survive tăng cuối game vì ít ô trống hơn → dễ bị kẹp hơn.

### Code

```python
def get_phase_weights(obs, agent_id, step=0):
    """
    Trả về dict weights theo FSM phase.
    step: số bước đã qua trong trận (nếu có trong obs).
    """
    players     = obs["players"]
    bomb_radius = 1 + int(players[agent_id][4])
    bombs_left  = int(players[agent_id][3])
    power_score = (bomb_radius * 1.5) + (bombs_left * 3)

    if power_score < 8:
        return {
            "phase": "FARMER",
            "w_rl": 0.50, "w_gt": 0.25, "w_threat": 0.10,
            "w_bomb": 0.05, "w_survive": 0.10,
        }
    elif step < 350:
        return {
            "phase": "ZONER",
            "w_rl": 0.45, "w_gt": 0.20, "w_threat": 0.10,
            "w_bomb": 0.15, "w_survive": 0.10,
        }
    else:
        return {
            "phase": "ASSASSIN",
            "w_rl": 0.40, "w_gt": 0.15, "w_threat": 0.05,
            "w_bomb": 0.25, "w_survive": 0.15,
        }
```

---

## Bước 7 — Expectimax Filter *(CẬP NHẬT)*

### Mục tiêu
Kết hợp Q-values từ RL với thông tin Game Theory để ra quyết định cuối cùng.

### Thay đổi so với phiên bản trước

| | v1 | v2 (hiện tại) |
|---|---|---|
| Weights | Cố định | **FSM phase weights** từ Bước 6 |
| Survivability | Không có | **Thêm** `compute_survivability()` |
| `trap_score` cũ | `-enemy_prob × 3.0` | **Đã bỏ** từ v1 |

### Công thức

```
combined[action] = w_rl      × q_norm[action]
                 + w_gt      × utility_map[next_pos]
                 + w_threat  × (-threat_zone_map[next_pos])
                 + w_bomb    × bomb_attack_score        (chỉ action 5)
                 + w_survive × survivability[next_pos]
```

### Survivability Score

Đếm số ô an toàn có thể reach được trong 3 bước từ `next_pos`. Phát hiện tình huống bị kẹp bởi 2 bom mà binary `can_escape` bỏ sót.

```python
def compute_survivability(obs, agent_id, danger_map, next_x, next_y, depth=3):
    """
    Output: float trong [0, 1] — tỉ lệ ô an toàn reach được / 20
    """
    grid = obs["map"]
    from collections import deque
    visited    = {(next_x, next_y)}
    q          = deque([(next_x, next_y, 0)])
    safe_count = 0

    while q:
        x, y, d = q.popleft()
        if danger_map[x, y] >= 4:
            safe_count += 1
        if d >= depth:
            continue
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited: continue
            if not (0 <= nx < 13 and 0 <= ny < 13): continue
            if grid[nx, ny] in (1, 2): continue
            if danger_map[nx, ny] <= 1: continue  # skip ô sắp nổ
            visited.add((nx, ny))
            q.append((nx, ny, d + 1))

    return min(safe_count / 20.0, 1.0)
```

### Code

```python
ACTION_DELTAS = {
    0: (0, 0),   # STOP
    1: (0, -1),  # LEFT
    2: (0, +1),  # RIGHT
    3: (-1, 0),  # UP
    4: (+1, 0),  # DOWN
    5: (0, 0),   # PLACE_BOMB
}

def expectimax_filter(obs, agent_id, q_values,
                      utility_map, threat_zone_map,
                      bomb_attack_score, danger_map,
                      safe_actions, weights):
    """
    Params:
        weights : dict từ get_phase_weights() — Bước 6
        danger_map : (13,13) từ Bước 0 — dùng cho survivability
    """
    players = obs["players"]
    my_x = int(players[agent_id][0])
    my_y = int(players[agent_id][1])

    q_norm = q_values - q_values.min()
    if q_norm.max() > 1e-8:
        q_norm = q_norm / q_norm.max()

    combined = np.full(6, -np.inf, dtype=np.float32)

    for action in safe_actions:
        dx, dy = ACTION_DELTAS[action]
        nx = int(np.clip(my_x + dx, 0, 12))
        ny = int(np.clip(my_y + dy, 0, 12))

        rl_score      = float(q_norm[action])
        gt_score      = float(utility_map[nx, ny])
        threat_score  = -float(threat_zone_map[nx, ny])
        bomb_score    = float(bomb_attack_score) if action == 5 else 0.0
        survive_score = compute_survivability(obs, agent_id, danger_map, nx, ny)

        combined[action] = (
            weights["w_rl"]      * rl_score
          + weights["w_gt"]      * gt_score
          + weights["w_threat"]  * threat_score
          + weights["w_bomb"]    * bomb_score
          + weights["w_survive"] * survive_score
        )

    return int(np.argmax(combined))
```

---

## Bước 8 — Safety Layer *(CẬP NHẬT)*

### Mục tiêu
Bảo hiểm sinh mạng — chạy trước mọi thứ, override tuyệt đối. Được cập nhật để dùng `danger_map` thay vì tính lại blast zone từ đầu.

### Thay đổi
Dùng `danger_map[nx, ny] <= 1` thay vì set `danger_now` tính riêng — nhất quán với Bước 0 và đã tính chain reaction.

### Code

```python
def get_safe_actions(obs, agent_id, danger_map):
    """
    Trả về list action không dẫn vào danger_map <= 1 (nổ ngay lượt tới).
    Fallback: [0] (STOP).
    """
    grid    = obs["map"]
    players = obs["players"]
    bombs   = obs["bombs"]

    my_x = int(players[agent_id][0])
    my_y = int(players[agent_id][1])

    # Tập vị trí bom (không thể bước vào)
    bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}

    # danger_soon dùng cho can_escape check (tick <= 6)
    danger_soon = set()
    for x in range(13):
        for y in range(13):
            if danger_map[x, y] < SAFE_TICK:
                danger_soon.add((x, y))

    ACTION_DELTAS = [(0,0),(0,-1),(0,1),(-1,0),(1,0),(0,0)]

    safe_actions = []
    for action in range(6):
        if action == 5:   # PLACE_BOMB
            if players[agent_id][3] > 0 and (my_x, my_y) not in bomb_positions:
                if _can_escape_after_placing(grid, my_x, my_y,
                                             bomb_positions, danger_soon,
                                             1 + int(players[agent_id][4])):
                    safe_actions.append(5)
            continue

        dx, dy = ACTION_DELTAS[action]
        nx, ny = my_x + dx, my_y + dy

        if not (0 <= nx < 13 and 0 <= ny < 13): continue
        if grid[nx, ny] in (1, 2): continue
        if (nx, ny) in bomb_positions and action != 0: continue

        # Dùng danger_map thay vì danger_now set
        if danger_map[nx, ny] > 1:
            safe_actions.append(action)

    return safe_actions if safe_actions else [0]


def _can_escape_after_placing(grid, mx, my, bomb_positions, danger_soon, radius):
    """BFS check: sau khi đặt bom tại (mx,my), có ô thoát không?"""
    from collections import deque
    my_blast  = set(_blast_tiles_list(grid, mx, my, radius))
    combined  = danger_soon | my_blast | {(mx, my)}
    new_bombs = bomb_positions | {(mx, my)}

    q    = deque([(mx, my, 0)])
    seen = {(mx, my)}
    while q:
        x, y, depth = q.popleft()
        if (x, y) not in combined:
            return True
        if depth >= 8:
            continue
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen: continue
            if not (0 <= nx < 13 and 0 <= ny < 13): continue
            if grid[nx, ny] in (1, 2): continue
            if (nx, ny) in new_bombs: continue
            seen.add((nx, ny))
            q.append((nx, ny, depth + 1))
    return False
```

---

## Bước 9 — Agent.act() — Kết nối tất cả *(CẬP NHẬT)*

```python
class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.step     = 0
        self.model    = DQNModel(map_shape=(13,13,13), aux_dim=6, output_dim=6)
        self.model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        self.model.eval()

    def act(self, obs: dict) -> int:
        self.step += 1

        # ── Bước 0: Time-layered Danger Map ─────────────────────────────────
        danger_map = build_time_layered_danger_map(obs)

        # ── Bước 8: Safety Layer ─────────────────────────────────────────────
        safe_actions = get_safe_actions(obs, self.agent_id, danger_map)
        if safe_actions == [0]:
            return 0

        # ── Bước 1 & 2: Utility + Enemy Probability ──────────────────────────
        utility_map    = compute_utility_map(obs, self.agent_id, danger_map)
        enemy_prob_map = compute_enemy_prob_map(obs, self.agent_id)

        # ── Bước 3 & 4: Bomb Attack Score + Threat Zone ───────────────────────
        bomb_attack_score = compute_bomb_attack_score(obs, self.agent_id,
                                                       enemy_prob_map)
        threat_zone_map   = compute_threat_zone_map(obs, self.agent_id)

        # ── Bước 5: Encode enriched state ────────────────────────────────────
        map_feat, aux_feat = encode_obs_enriched(
            obs, self.agent_id,
            utility_map, enemy_prob_map, threat_zone_map, danger_map
        )
        with torch.no_grad():
            map_t    = torch.tensor(map_feat[np.newaxis], dtype=torch.float32)
            aux_t    = torch.tensor(aux_feat[np.newaxis], dtype=torch.float32)
            q_values = self.model(map_t, aux_t).squeeze(0).numpy()

        # ── Bước 6: FSM Phase Weights ─────────────────────────────────────────
        weights = get_phase_weights(obs, self.agent_id, step=self.step)

        # ── Bước 7: Expectimax Filter ─────────────────────────────────────────
        action = expectimax_filter(
            obs, self.agent_id, q_values,
            utility_map, threat_zone_map,
            bomb_attack_score, danger_map,
            safe_actions, weights
        )
        return action
```

### Time budget (100ms limit)

| Bước | Ước tính |
|---|---|
| `build_time_layered_danger_map()` | ~2ms |
| `get_safe_actions()` | ~1ms |
| `compute_utility_map()` | ~3ms |
| `compute_enemy_prob_map()` k=3 | ~10ms |
| `compute_bomb_attack_score()` (với guard) | ~0.2ms |
| `compute_threat_zone_map()` | ~1ms |
| `encode_obs_enriched()` | ~1ms |
| DQN forward pass (CPU, 13 channels) | ~17ms |
| `get_phase_weights()` | ~0.1ms |
| `expectimax_filter()` + survivability | ~3ms |
| **Tổng** | **~38ms** — còn buffer 62ms |

---

## Reward Shaping *(CẬP NHẬT)*

Thêm 3 thành phần mới vào `reward.py` gốc:

```python
# ── 1. Bomb attack reward: đặt bom khi địch có khả năng đi qua vùng nổ ──────
if vừa_đặt_bom:
    enemy_prob = compute_enemy_prob_map(curr_obs, agent_id, k_steps=3)
    blast      = _blast_tiles_list(grid, my_x, my_y, radius)
    trap_value = sum(enemy_prob[tx, ty] for tx, ty in blast)
    reward    += 0.5 * trap_value

# ── 2. Threat zone penalty: đứng trong vùng nguy hiểm của địch radius lớn ───
threat_map = compute_threat_zone_map(curr_obs, agent_id)
my_threat  = float(threat_map[my_x, my_y])
reward    -= 0.08 * my_threat

# ── 3. Utility flow: thưởng nếu di chuyển đến ô utility cao hơn ─────────────
delta_utility = utility_map[curr_x, curr_y] - utility_map[prev_x, prev_y]
reward       += 0.05 * delta_utility
```

### Tổng quan tất cả reward components

| Sự kiện | Reward | Nguồn |
|---|---|---|
| Win | +2.0 | Gốc |
| Enemy kill | +1.0 | Gốc |
| Agent death | -2.0 | Gốc |
| Danger evasion | +0.12 (×1.5 khi urgency) | Gốc |
| Danger enter | -0.06 | Gốc |
| Item collection | +0.1 | Gốc |
| Plant near box | +0.05 | Gốc |
| Approach enemy | +0.02×Δdist | Gốc |
| Standing still | -0.01 | Gốc |
| Time penalty | -0.005/step | Gốc |
| **Bomb attack score** | **+0.5 × trap_value** | **MỚI** |
| **Threat zone penalty** | **-0.08 × threat_level** | **MỚI** |
| **Utility flow** | **+0.05 × Δutility** | **MỚI** |

---

*Thứ tự implement: Bước 0 → 8 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9. Reward shaping tích hợp song song với training.*
