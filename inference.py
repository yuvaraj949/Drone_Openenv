"""
Baseline Inference Script — Drone Traffic Control
==================================================

Runs a complete episode using a GREEDY SHORTEST-PATH agent as the baseline.
The agent uses BFS to find the shortest route for each drone at every step,
with a simple priority rule (emergency drones move first).

Can be upgraded to an LLM or RL agent by swapping out the `act()` function.

Output format (hackathon scoring compatible):
  [START]  — episode begins
  [STEP N] — per-step summary
  [END]    — episode complete with final grader score

Usage:
  python inference.py --task easy --seed 42
  python inference.py --task medium
  python inference.py --task hard --seed 0
  python inference.py --task easy --rich
  python inference.py --task easy --visualize
  python inference.py --task easy --visualize --gif-path out.gif
"""

import argparse
from collections import deque
from typing import Dict, List, Optional

from environment.drone_env import DroneTrafficEnv
from environment.graders import grade_episode_log, grade_task
from environment.models import Action, DroneAction, DroneState, HOVER, Observation


# ---------------------------------------------------------------------------
# BFS shortest-path helper
# ---------------------------------------------------------------------------

def bfs_next_zone(
    current: str,
    destination: str,
    graph: Dict[str, List[str]],
    blocked_zones: Optional[List[str]] = None,
) -> str:
    blocked = set(blocked_zones or [])
    if current == destination:
        return current

    queue: deque = deque([(current, [current])])
    visited = {current}

    while queue:
        zone, path = queue.popleft()
        for neighbour in graph.get(zone, []):
            if neighbour in visited or neighbour in blocked:
                continue
            new_path = path + [neighbour]
            if neighbour == destination:
                return new_path[1] if len(new_path) > 1 else current
            visited.add(neighbour)
            queue.append((neighbour, new_path))

    return current


# ---------------------------------------------------------------------------
# Greedy agent
# ---------------------------------------------------------------------------

def act(obs: Observation) -> Action:
    """Greedy BFS baseline: emergency drones first, avoid claimed zones."""
    graph = obs.graph_edges
    claimed_zones: List[str] = []
    actions: List[DroneAction] = []

    sorted_drones: List[DroneState] = sorted(
        [d for d in obs.drones if not d.delivered],
        key=lambda d: (-d.priority, d.id),
    )

    for drone in sorted_drones:
        if drone.battery <= 0.0:
            actions.append(DroneAction(drone_id=drone.id, move_to=HOVER))
            continue

        next_zone = bfs_next_zone(
            current=drone.location,
            destination=drone.destination,
            graph=graph,
            blocked_zones=claimed_zones,
        )
        move = HOVER if next_zone == drone.location else next_zone
        claimed_zones.append(next_zone)
        actions.append(DroneAction(drone_id=drone.id, move_to=move))

    return Action(actions=actions)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    task: str = "easy",
    seed: Optional[int] = None,
    use_rich: bool = False,
    use_visualize: bool = False,
    gif_path: str = "episode_animation.gif",
) -> float:
    env = DroneTrafficEnv(task=task, seed=seed)
    obs = env.reset()
    done = False
    step_rewards: List[float] = []

    # ── optional renderer init ────────────────────────────────────────────
    rich_renderer = None
    animator = None

    if use_rich:
        try:
            from visualizer.terminal_vis import TerminalRenderer
            rich_renderer = TerminalRenderer(
                rows=env.cfg["rows"],
                cols=env.cfg["cols"],
                task_name=task,
            )
        except ImportError:
            print("[WARN] 'rich' not installed — run: pip install rich")

    if use_visualize:
        try:
            from visualizer.grid_vis import GridAnimator
            animator = GridAnimator(
                rows=env.cfg["rows"],
                cols=env.cfg["cols"],
                task_name=task,
                bottleneck_zones=env.cfg.get("bottleneck_zones", []),
            )
            animator.capture(obs)  # step-0 frame
        except ImportError:
            print("[WARN] matplotlib/Pillow not installed — run: pip install matplotlib Pillow")
            use_visualize = False

    # ── plain header (always — hackathon log) ─────────────────────────────
    print(f"\n[START] Task={task.upper()} | Drones={len(obs.drones)} | "
          f"MaxSteps={env.max_steps} | Seed={seed}")
    print("-" * 60)
    _print_obs_header(obs)

    # ── episode loop ──────────────────────────────────────────────────────
    while not done:
        action = act(obs)
        obs, reward, done, info = env.step(action)
        step_rewards.append(reward.total)

        print(
            f"[STEP {info['step']:>2}] "
            f"Reward={reward.total:+.2f} | "
            f"Delivered={info['delivered']}/{len(obs.drones)} | "
            f"Collisions(cum)={info['cumulative_collisions']} | "
            f"Blocked={info['blocked_zones'] or '[]'}"
        )
        if info["step"] % 5 == 0:
            bats = {d.id: f"{d.battery:.0f}%" for d in obs.drones}
            print(f"         Battery: {bats}")

        if rich_renderer is not None:
            rich_renderer.render(obs, reward, info)

        if animator is not None:
            animator.capture(obs, blocked_zones=info.get("blocked_zones", []))

    print("-" * 60)

    # ── grading ───────────────────────────────────────────────────────────
    final_score = grade_task(env.state(), env.cfg)
    ep_stats = grade_episode_log(step_rewards)

    print(f"[END] Final Score     : {final_score:.4f}")
    print(f"      Total Reward    : {ep_stats['total_reward']}")
    print(f"      Mean Step Reward: {ep_stats['mean_reward']}")
    print(f"      Collisions      : {env.state()['collisions']}")
    delivered = sum(1 for d in env.state()["drones"] if d["delivered"])
    print(f"      Delivered       : {delivered}/{len(env.state()['drones'])}")
    print()

    if rich_renderer is not None:
        rich_renderer.render_final(final_score, env.state())

    if animator is not None and animator.frame_count() > 0:
        try:
            saved = animator.save(gif_path)
            print(f"[VIS]  Animation saved → {saved}  ({animator.frame_count()} frames)")
        except Exception as exc:
            print(f"[WARN] Could not save GIF: {exc}")

    return final_score


def _print_obs_header(obs: Observation) -> None:
    print("Initial drone positions:")
    for d in obs.drones:
        tag = "[EMERGENCY]" if d.priority == 2 else "[normal]"
        print(f"  {d.id}: {d.location} -> {d.destination} {tag}")
    print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Traffic Control — Baseline Inference")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--rich", action="store_true",
                        help="Live Rich terminal rendering (pip install rich)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save episode GIF (pip install matplotlib Pillow)")
    parser.add_argument("--gif-path", default="episode_animation.gif")
    args = parser.parse_args()

    if args.all_tasks:
        scores = {}
        for t in ["easy", "medium", "hard"]:
            scores[t] = run_episode(
                task=t, seed=args.seed,
                use_rich=args.rich, use_visualize=args.visualize,
                gif_path=f"{t}_{args.gif_path}",
            )
        print("=" * 60)
        print("AGGREGATE RESULTS")
        for t, s in scores.items():
            print(f"  {t.upper():>6}: {s:.4f}")
        print(f"  {'MEAN':>6}: {sum(scores.values()) / len(scores):.4f}")
    else:
        run_episode(
            task=args.task, seed=args.seed,
            use_rich=args.rich, use_visualize=args.visualize,
            gif_path=args.gif_path,
        )
