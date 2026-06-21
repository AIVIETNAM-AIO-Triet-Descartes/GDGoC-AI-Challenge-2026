# Implementation Plan — Hybrid Agent (Game Theory + RL)

Theo dõi tiến trình triển khai theo [`plan.md`](./plan.md). Tick từng box khi xong.

**Thứ tự implement (theo plan.md):** Bước 0 → 8 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9.
Reward shaping làm song song với training.

**File đích:** module agent mới (vd `agent/hybrid_agent/agent.py` + `reward.py`),
tách khỏi `agent/dqn_agent/` để không phá baseline tham khảo.

---

## ⚠️ Phát hiện từ codebase (đọc trước khi code)

- [x] Nắm rõ: baseline `dqn_agent/agent.py` viết cho **2 người chơi** (`opp_id = 1 - user_id`),
      chỉ encode **1 địch**. Giải đấu là **4 người chơi** → phải refactor encode sang 3 địch.
- [x] Nắm rõ: `model.pth` cũ **KHÔNG nạp lại được** (shape đổi `9→13` ch, `aux 3→6`) → **train lại từ đầu**.
- [x] Tái dùng được: `ReplayBuffer`, nhánh `aux_encoder`, `forward()`, vòng `train_step` trong `dqn_agent`
      (đã copy → `hybrid_agent/model.py` + `trainer.py`).

---

## Giai đoạn 1 — Game Theory Layers (thuần numpy, test độc lập được)

> ✅ **Giai đoạn 1 đã implement + test (26/26 pass).** Hàm trong `hybrid_agent/agent.py`.

### Bước 0 — Time-layered Danger Map *(MỚI)*
- [x] `_blast_tiles_list(grid, bx, by, radius)` — list ô bị blast, dừng ở tường/hộp.
- [x] `build_time_layered_danger_map(obs)` — tick nổ từng ô, có resolve **chain reaction**.
- [x] Test: bom dây chuyền (A r4 ép B timer=6 → 2); ô an toàn = `SAFE_TICK (999)`.

### Bước 8 — Safety Layer *(CẬP NHẬT)*
- [x] `get_safe_actions(obs, agent_id, danger_map)` — loại action vào `tick <= 1`, fallback `[0]`.
- [x] `_can_escape_after_placing(...)` — BFS check còn đường thoát sau khi đặt bom.
- [x] Test: không bom → nhiều action; trap → list hợp lệ/fallback.

### Bước 1 — Utility Map *(CẬP NHẬT)*
- [x] `_line_clear(grid, ax, ay, bx, by)` — LOS check (cùng hàng/cột, không bị chặn).
- [x] `compute_utility_map(obs, agent_id, danger_map)` — item/box/enemy + stratified penalty, normalize `[-1,1]`.
- [x] Test: item > box; normalize `[-1,1]`; ô `tick=1` bị phạt âm.

### Bước 2 — Enemy Probability Map (Markov)
- [x] `_get_passable_neighbors`, `_enemy_tile_score`, `_softmax`.
- [x] `compute_enemy_prob_map(obs, agent_id, k_steps=3)` — heatmap xác suất địch, normalize tổng=1.
- [x] ⚠️ Đo thời gian — **2.06ms** (local, map trống) < 10ms → chưa cần vectorize. Đo lại trên VM khi ghép full pipeline.

### Bước 3 — Bomb Attack Score *(CẬP NHẬT)*
- [x] `compute_bomb_attack_score(obs, agent_id, enemy_prob_map)` — scalar, có **guard CPU** (skip nếu không có địch ≤ `radius+2`).

### Bước 4 — Threat Zone Map
- [x] `compute_threat_zone_map(obs, agent_id)` — vùng nổ tiềm tàng của địch, weight theo `radius/5`, normalize `[0,1]`.

---

## Giai đoạn 2 — RL Core

> ✅ **B5 encode đã implement + test (20/20 pass).** ⚠️ Forward DQNModel chưa verify local (thiếu `torch`).

### Bước 5 — Encode Enriched State *(CẬP NHẬT)*
- [x] `encode_obs_enriched(...)` → `map_feat (13,13,13)`, `aux_feat (6,)`.
- [x] CH 0-4 one-hot map, CH 5 my_pos, CH 6-8 **3 enemy_pos**, CH 9 utility, CH 10 enemy_prob, CH 11 threat, CH 12 danger_norm.
- [x] 6 aux: bombs_left/5, radius_bonus/5, num_alive_enemies/3, dist tới 3 địch /20.
- [x] Test shape + nội dung channel + aux normalize (kể cả địch chết → channel rỗng, dist=1.0).

### Cập nhật DQNModel
- [x] Copy `DQNModel` → `hybrid_agent/model.py` (default shape `(13,13,13)/6/6`).
- [x] Copy `ReplayBuffer` + `TrainingAgent` (act/train_step/target/save/load) → `hybrid_agent/trainer.py`.
- [ ] ⚠️ Kiểm tra forward (no shape mismatch) — **CHƯA chạy được**: máy local chỉ có conda `base`, không có `torch`
      (env `aic_gdgoc` trong CLAUDE.md chưa tồn tại). Test sẵn ở `scratchpad/test_stage2.py` (tự bật khi có torch).
      Phân tích tĩnh: `DQNModel` parametrized + dummy-forward tự khớp `conv_out_dim`, input `(13,13,13)/6` đúng default → khả năng mismatch ~0.

---

## Giai đoạn 3 — Decision Layer

### Bước 6 — FSM Phase Weights *(MỚI)*
- [ ] `get_phase_weights(obs, agent_id, step)` — `power_score = radius*1.5 + bombs*3`; FARMER/ZONER/ASSASSIN.

### Bước 7 — Expectimax Filter *(CẬP NHẬT)*
- [ ] `compute_survivability(obs, agent_id, danger_map, nx, ny, depth=3)` — BFS đếm ô an toàn reach được.
- [ ] `expectimax_filter(...)` — trộn `w_rl·q + w_gt·utility − w_threat·threat + w_bomb·bomb + w_survive·survive`.
- [ ] `bomb_attack_score` chỉ cộng cho action 5.

### Bước 9 — Agent.act() ghép tất cả *(CẬP NHẬT)*
- [ ] `class Agent` đúng interface submit (`__init__(agent_id)`, `act(obs)->int`).
- [ ] Nạp `model.pth` qua `Path(__file__).parent` (KHÔNG hardcode path).
- [ ] Pipeline: danger → safe (thoát sớm nếu `[0]`) → maps → encode → DQN → weights → expectimax.

---

## Giai đoạn 4 — Training

### Reward Shaping *(CẬP NHẬT `reward.py`)*
- [ ] Giữ reward gốc (win/kill/death/evasion/item/...).
- [ ] +0.5 × trap_value (bomb attack).
- [ ] −0.08 × threat_level (threat zone penalty).
- [ ] +0.05 × Δutility (utility flow).

### Train loop
- [ ] Đối chiếu `training_flow.mmd` với `TrainingAgent` thật → chỉnh khớp.
- [ ] Cấu hình self-play 4-player (hiện baseline là 2-player) — quyết định opponent pool khi train.
- [ ] Train trên Kaggle/Colab; log loss/reward/win-rate.
- [ ] Xuất `model.pth` (chỉ `state_dict`, map_location cpu).

---

## Giai đoạn 5 — Validate & Submit

- [ ] **Đo thời gian**: `python -m scripts.participant.estimate_agent_time <path> --opponents None None None` → **< 100ms/step** (mục tiêu plan.md ~38ms; chú ý CPU VM yếu hơn).
- [ ] **Chạy thử trận**: `python -m scripts.participant.run_local_match --agent_paths <path> None None None --visualize true`.
- [ ] **Ước lượng rating**: `python -m scripts.participant.estimate_rankings --agent_path <path> --num_matches 100` → so với baseline (tactical ~114.7).
- [ ] Đóng gói zip **phẳng**: `agent.py` + `model.pth` + file phụ (≤20 file, ≤100MB, tên không trùng lib).
- [ ] Submit qua form (Team ID + Token).

---

## Rủi ro cần theo dõi

- [ ] **Timeline**: deadline nộp 21/6 — train DQN tốn thời gian. Giữ `fsm_agent.py` rule-based làm **bản nộp dự phòng**.
- [ ] **Timeout 100ms**: `compute_enemy_prob_map` + DQN forward là điểm nóng — đo sớm, vectorize.
- [ ] **4-player self-play**: cần data train phản ánh đúng môi trường 4 người, không phải 1v1.

---

*Tham chiếu: [`plan.md`](./plan.md) · [`inference_flow.mmd`](./inference_flow.mmd) · [`training_flow.mmd`](./training_flow.mmd) · [`CODEBASE_MAP.md`](./CODEBASE_MAP.md)*
