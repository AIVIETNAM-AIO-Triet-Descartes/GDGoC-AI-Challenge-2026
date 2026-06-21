# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

Infrastructure for the **GDGoC-HCMUS AI Challenge 2026 — Bomberland**, a multi-agent
RL/AI competition. Theme: Reinforcement Learning. 4 agents fight on a 13×13 grid
(11×11 play area): move, place bombs, break boxes, collect items, eliminate opponents.
Teams ≤ 2 people. Repo holds the game engine, evaluation worker, registration server,
baseline agents, and participant/organizer scripts.

Authoritative spec: `docs/COMPETITION_GUIDE.md`. Source docs (Vietnamese): `docs/context/`.

## Layout

- `engine/` — core game engine. `game.py` (`BomberEnv`: step resolution, explosions,
  spawning, termination), `map.py`, `bomb.py`, `player.py`. Single source of truth for
  game rules — read here before trusting prose docs.
- `agent/` — baselines + templates. `random_`, `simple_rule_`, `smarter_rule_`,
  `box_farmer_`, `genius_rule_`, `tactical_rule_agent.py`; `dqn_agent/` (RL reference,
  not an official baseline); `fsm_agent.py`, `error_agents.py`.
- `competition/` — `registration/` (Flask + webhooks), `evaluation/` (match runner,
  TrueSkill ranking, pool manager, runtime guard, rendering), `ingestion/`,
  `integrations/` (Google Drive/Sheets, Discord), `storage/`, `config.py`.
- `scripts/participant/` — local testing (`run_local_match`, `estimate_rankings`,
  `estimate_agent_time`, `replay_viewer`, `visualizer`).
- `scripts/organizer/` — `run_evaluation`, `run_final_evaluation` (Grand Finals),
  `calibrate_baselines`, `reset_to_baselines`, `backup_db`, `post_daily_highlights`.
- `deploy/` — systemd units + GCP VM setup.

## Game rules (key invariants — verify against `engine/` before editing logic)

- Actions: `0`=STOP, `1`=LEFT, `2`=RIGHT, `3`=UP, `4`=DOWN, `5`=PLACE_BOMB.
- Step order: collect actions → movement → place bombs → decrement timers →
  resolve explosions → remove agents → spawn items → check end.
- Bomb: timer 7, default radius 1, capacity 1, both cap at 5. Explosions cross 4
  directions, stop at walls + boxes (destroy box), pass through agents, last 1 step,
  chain-react. Only placer spends `bombs_left`; bomb survives owner death.
- Box drop: 30% radius / 30% capacity / 40% nothing.
- Match ends: ≤1 alive (terminated) or 500 steps (truncated). Step-500 tie-break:
  kills → boxes → items → bombs placed.
- Scoring: TrueSkill, `score = μ − 3σ`, start `μ=100, σ=33.333`.

## Agent contract (participant-facing)

`agent.py` defines `class Agent` with `__init__(self, agent_id)` and
`act(self, obs) -> int` in `[0,5]`. `obs` = `{map (13,13), players (4,5)
[row,col,alive,bombs_left,bomb_radius_bonus], bombs (N,4) [row,col,timer,owner_id]}`.
Note: engine stores players as `[x, y, ...]` internally — keep obs naming consistent
with the guide when touching `_get_obs`.

Constraints: startup ≤ 20s, `act()` ≤ 100ms/step (else defaults to STOP). No LLMs in
`Agent`. CPU-only eval, no network/file-write during match. Only libs in
`requirements.txt` (numpy, scipy, torch, tensorflow, stable-baselines3, gymnasium,
onnxruntime, stdlib).

## Working here

- Python 3.11. Env: `conda activate aic_gdgoc`; `pip install -r requirements.txt`.
- Run a local match: `python -m scripts.participant.run_local_match --agent_paths <p> None None None --visualize true`
- Estimate rating: `python -m scripts.participant.estimate_rankings --agent_path <p> --num_matches 100`
- No test suite present — validate engine/agent changes by running local matches and
  checking determinism via the match `seed`.
- This is Windows (PowerShell). README quick-start commands assume Linux/VM
  (`sudo`, `chmod`, conda paths) — translate for local dev.

## Editing docs

`docs/COMPETITION_GUIDE.md` is generated from `docs/context/*.docx`. Match-count truth
(from code, verified): submission batch on upload = **12 matches**
(`submission_webhook.py` → `run_submission_batch(n_matches=12)`); background eval cycle =
**5 matches** + 10s rest (`run_evaluation.py` `bg_cmd default=5`). Guide/README were
fixed to match. Remaining (non-build) doc gaps vs source, fix only if editing those
sections: team-size cap (≤2), Top-5 online pitching after Grand Finals, total prize
(1.5M VND), Grand-Finals corner shuffle.
