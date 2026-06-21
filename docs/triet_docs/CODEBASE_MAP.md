# Bản đồ Codebase — Bomberland GDGoC AI Challenge 2026

Mô tả vai trò từng file ở mức ảnh hưởng trực tiếp đến logic, không đi sâu chi tiết
implementation. Dùng để định hướng nhanh khi sửa code.

Luồng chính:
**Đăng ký/Nộp bài** (`registration` + `ingestion`) → **Lưu trữ** (`storage`) →
**Đánh giá** (`evaluation` + `engine`) → **Xếp hạng & công bố** (`ranking` + `integrations`).

---

## `engine/` — Lõi trò chơi (nguồn sự thật về luật chơi)

| File | Vai trò |
|---|---|
| `engine/game.py` | Class `BomberEnv` — môi trường game. Vòng lặp `step()`: thu action 4 người chơi, xử lý di chuyển/va chạm, đặt bom, nổ dây chuyền, sinh item, loại agent, kiểm tra kết thúc. Định nghĩa `obs` mà agent nhận. **Mọi luật chơi gốc nằm ở đây.** |
| `engine/map.py` | Class `Map` — sinh bản đồ theo seed: tường viền cố định, tường/hộp/item ngẫu nhiên; dùng DSU đảm bảo bản đồ liên thông (không vùng cô lập). |
| `engine/player.py` | Class `Player` — trạng thái người chơi (id, vị trí, sống/chết, `bombs_left`, bonus bán kính, thống kê). `move()` xử lý di chuyển + va chạm. |
| `engine/bomb.py` | Class `Bomb` — trạng thái bom (vị trí, timer, bán kính, chủ). Giảm timer, báo nổ. |
| `engine/__init__.py` | Export `BomberEnv`, `Map`, `Player`, `Bomb`. |

---

## `agent/` — Agent baseline & template (giao diện thí sinh)

Mọi agent tuân theo hợp đồng: `class Agent` với `__init__(agent_id)` + `act(obs) -> int [0,5]`.

| File | Vai trò |
|---|---|
| `agent/random_agent.py` | Action ngẫu nhiên — baseline yếu nhất. |
| `agent/simple_rule_agent.py` | Rule cơ bản: né bom, nhặt item, đặt bom gần hộp khi có đường thoát. |
| `agent/smarter_rule_agent.py` | Nâng cấp simple: thêm truy đuổi địch cùng hàng/cột, BFS né hiểm tốt hơn. |
| `agent/box_farmer_agent.py` | Tập trung phá hộp lấy item; ưu tiên thoát hiểm trước khi đặt bom. |
| `agent/genius_rule_agent.py` | Cân bằng công/thủ, có `escape_mode`, gây sức ép địch, BFS tìm đường. |
| `agent/tactical_rule_agent.py` | Baseline mạnh nhất: chấm điểm ô an toàn, BFS, đánh giá tấn công/khai thác. |
| `agent/fsm_agent.py` | Agent FSM + utility scoring (mạnh nhất repo, không phải baseline chính thức): bản đồ nguy hiểm theo thời gian, đổi pha FARMER/ZONER/ASSASSIN, chấm điểm đa chỉ số cho 6 action. |
| `agent/error_agents.py` | Agent vi phạm luật để test sandbox: `TimeOutAgent`, `InvalidActionAgent`, `NoBombAgent`. |
| `agent/dqn_agent/agent.py` | Agent DQN tham khảo: mạng Conv2D (map) + MLP (scalar), có `TrainingAgent` (replay buffer, train) và `Agent` (nạp `.pth` để thi đấu). |
| `agent/dqn_agent/reward.py` | `compute_reward()` — hàm thưởng khi train DQN (giết/chết/né hiểm/nhặt item...). |
| `agent/dqn_agent/utils.py` | Tiện ích train: seed, lưu checkpoint, vẽ biểu đồ. |
| `agent/dqn_agent/2737502_global_step.pth` | Trọng số DQN đã train, nạp bởi `dqn_agent/agent.py`. |
| `agent/__init__.py` | Export 6 baseline agent (dùng cho preset trong match runner / scripts). |
| `agent/README.md` | Hướng dẫn thí sinh: interface, ràng buộc 100ms, cách test & nộp. |

---

## `competition/` — Hạ tầng cuộc thi

### `competition/registration/` — Đăng ký đội & nhận nộp bài (Flask)

| File | Vai trò |
|---|---|
| `registration/app.py` | Entry-point Flask. 2 endpoint: `/register` (đăng ký đội), `/submit` (nộp bài). Khởi tạo Drive service + `SubmissionStore`. |
| `registration/webhook_receiver.py` | Xử lý payload đăng ký từ Google Form: validate, sinh team ID + token, lưu DB. |
| `registration/canonical_id_generator.py` | Sinh team ID duy nhất dạng `{slug_tên_đội}_{suffix}`. |
| `registration/token_generator.py` | Sinh submission token (`os.urandom(32)`) dùng xác thực khi nộp. |
| `registration/__init__.py` | Export 3 hàm: tạo team ID, tạo token, xử lý payload đăng ký. |

### `competition/ingestion/` — Tiếp nhận & kiểm tra bài nộp

| File | Vai trò |
|---|---|
| `ingestion/collector.py` | Core intake: tải zip từ Drive, validate an toàn (chặn path traversal, check cú pháp `agent.py`), giải nén, lưu metadata. Có CLI riêng. |
| `ingestion/submission_webhook.py` | Logic endpoint `/submit`: xác thực token, kiểm quota ngày (reset 7h sáng VN), gọi xử lý bài, **trigger batch 12 trận đánh giá ngay**. |
| `ingestion/__init__.py` | Re-export hàm download/validate/extract + hằng giới hạn file. |

### `competition/evaluation/` — Chạy trận & chấm điểm

| File | Vai trò |
|---|---|
| `evaluation/match_runner.py` | Chạy 1 trận: dựng `BomberEnv`, điều phối 4 agent qua sandbox, thu history/frame, xếp hạng theo thứ tự chết + tie-break stats, lưu JSON/GIF, upload Drive. |
| `evaluation/runtime_guard.py` | Sandbox thực thi agent: nạp `agent.py` trong process con (drop quyền về `nobody`), gọi `act()` qua pipe có timeout, chặn lỗi/timeout/action sai → STOP. Có precheck validate agent. |
| `evaluation/ranking.py` | Hệ TrueSkill: cập nhật `μ/σ` theo kết quả trận, lưu DB, trả bảng xếp hạng. |
| `evaluation/pool_manager.py` | Quản lý Active Pool: đánh dấu best/recent per team, top global; quyết định agent nào được ghép trận (`is_active`). |
| `evaluation/rendering.py` | Vẽ frame trận (map/bom/player/item/nổ + panel tên agent) bằng PIL → tạo GIF. |
| `evaluation/__init__.py` | Package rỗng. |

### `competition/storage/` — Cơ sở dữ liệu

| File | Vai trò |
|---|---|
| `storage/submission_store.py` | Lớp SQLite cho team/submission/quota/match_results: tạo schema, CRUD, verify token, theo dõi quota, query batch cho leaderboard/feedback. |
| `storage/__init__.py` | Export `SubmissionStore`, `SubmissionRecord`, `TeamRecord`, `TeamLookupRecord`. |

### `competition/integrations/` — Kết nối dịch vụ ngoài

| File | Vai trò |
|---|---|
| `integrations/drive_upload.py` | Upload JSON/GIF lên Google Drive (sắp thư mục theo artifact/ngày), refresh token, locking. |
| `integrations/drive_oauth.py` | Script CLI tạo OAuth token Drive (interactive flow), lưu credentials. |
| `integrations/notifications.py` | Build/cập nhật Google Sheets leaderboard (3 tab: Leaderboard, Feedback, Logs) + gửi Discord webhook. |
| `integrations/google_workspace/google_apps_script_registration.gs` | Apps Script gắn Form đăng ký: gửi response tới `/register`, gửi email xác nhận (team ID, token, link form nộp). |
| `integrations/google_workspace/google_apps_script_submission.gs` | Apps Script gắn Form nộp bài: gửi team ID/token/file_id tới `/submit`, xử lý phản hồi. |
| `integrations/__init__.py` | Docstring package. |

### `competition/` (gốc)

| File | Vai trò |
|---|---|
| `config.py` | Cấu hình toàn cục: timezone VN, `get_vietnam_now()`, `load_env()` nạp `.env`. |
| `__init__.py` | Docstring package backend. |
| `match_20260520_112320_513617.gif` | GIF demo dùng trong README. |

---

## `scripts/` — Công cụ chạy tay

### `scripts/organizer/` — Cho ban tổ chức

| File | Vai trò |
|---|---|
| `run_evaluation.py` | Chạy đánh giá: batch cho 1 submission (12 trận) hoặc background cycle (5 trận + nghỉ 10s). Cập nhật TrueSkill, upload artifact, refresh leaderboard. Entry-point chính của worker. |
| `run_final_evaluation.py` | Chạy Grand Finals: top 8 + 1 baseline, mọi tổ hợp 4 người (`C(9,4)`), tính điểm theo hạng, ghi tab "Grand Final". |
| `calibrate_baselines.py` | Reset rating baseline về `μ=100, σ=33.33` rồi chạy hàng trăm trận song song để chuẩn hóa. |
| `reset_to_baselines.py` | Xóa submission/match non-baseline, đưa DB về trạng thái chỉ-baseline (có dry-run). |
| `collect_submissions.py` | CLI tải submission từ Drive → DB + thư mục submissions; subcommand `init-db`, `upsert-team`, `collect`. |
| `backup_db.py` | Sao lưu `competition.db` lên Drive kèm timestamp. |
| `post_daily_highlights.py` | Trích highlight 24h + top 5 leaderboard, gửi Discord. |
| `start_webhook_server.sh` | Wrapper bash khởi chạy Flask server (`registration.app`) trong conda env. |

### `scripts/participant/` — Cho thí sinh test local

| File | Vai trò |
|---|---|
| `run_local_match.py` | Chạy 1 trận 4 agent (custom/baseline), in win stats hoặc bật viewer. |
| `estimate_rankings.py` | Mô phỏng ~100 trận vs baseline, ước lượng TrueSkill (không động DB thật). |
| `estimate_agent_time.py` | Benchmark thời gian `act()`, cảnh báo nếu vượt 100ms/step. |
| `replay_viewer.py` | Xem lại match JSON bằng pygame (play/pause/tua). |
| `visualizer.py` | Viewer pygame cho episode local (render grid + sidebar stats + điều khiển phím). |

---

## `deploy/` — Triển khai VM

| File | Vai trò |
|---|---|
| `setup_vm.sh` | Setup GCP VM: cài package, Miniconda, conda env, deps, cài systemd service từ template. |
| `bomberland-web.service` | systemd unit cho Flask webhook server (registration). |
| `bomberland-worker.service` | systemd unit cho background worker chạy `run_evaluation background` liên tục. |

---

## File gốc

| File | Vai trò |
|---|---|
| `requirements.txt` | Deps: pygame, torch, trueskill, google-api-python-client, Flask, tensorflow, stable-baselines3, scipy, onnxruntime... |
| `.env.example` | Template biến môi trường: credentials Drive/Sheets, Discord webhook, timeout, port, GitHub token. |
| `.gitignore` | Bỏ qua `__pycache__`, `logs/`, `*.db`, `secrets/`, `submissions/`, `backups/`... |
| `README.md` | Tổng quan repo + quick-start. |
| `docs/COMPETITION_GUIDE.md` | Thể lệ & luật chính thức (English). |
| `CLAUDE.md` | Hướng dẫn cho Claude Code khi làm việc trong repo. |
