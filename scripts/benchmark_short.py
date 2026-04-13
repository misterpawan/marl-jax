#!/usr/bin/env python3
"""Short environment and training benchmarks for quick optimization loops."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from marl.utils import helpers  # noqa: E402


DEFAULT_ENVS = (
    ("ssd", "switch"),
    ("ssd", "harvest"),
    ("ssd", "cleanup"),
    ("overcooked", "cramped_room"),
)


def _make_env(env_name: str, map_name: str):
  if env_name == "ssd":
    return helpers.make_ssd_environment(
        0,
        map_name,
        autoreset=False,
        reward_scale=1.0,
        global_observation_sharing=True,
        record=False)
  if env_name == "overcooked":
    return helpers.make_overcooked_environment(
        0,
        map_name,
        autoreset=False,
        reward_scale=1.0,
        global_observation_sharing=True,
        record=False)
  raise ValueError(f"Unsupported benchmark environment: {env_name}")


def _benchmark_env(env_name: str, map_name: str, steps: int, seed: int):
  env = _make_env(env_name, map_name)
  rng = random.Random(seed)

  reset_start = time.perf_counter()
  timestep = env.reset()
  reset_duration = time.perf_counter() - reset_start
  total_reward = 0.0
  total_step_duration = 0.0
  actual_steps = 0

  while actual_steps < steps:
    actions = [rng.randrange(env.num_actions) for _ in range(env.num_agents)]
    step_start = time.perf_counter()
    timestep = env.step(actions)
    total_step_duration += time.perf_counter() - step_start
    if timestep.reward is not None:
      total_reward += sum(float(reward) for reward in timestep.reward)
    actual_steps += 1
    if timestep.last() and actual_steps < steps:
      reset_start = time.perf_counter()
      timestep = env.reset()
      reset_duration += time.perf_counter() - reset_start

  return {
      "env_name": env_name,
      "map_name": map_name,
      "steps": actual_steps,
      "steps_per_second": actual_steps / max(total_step_duration, 1e-9),
      "mean_step_duration_sec": total_step_duration / max(1, actual_steps),
      "total_reset_duration_sec": reset_duration,
      "reward_per_step": total_reward / max(1, actual_steps),
  }


def _parse_float(row: dict[str, str], key: str):
  value = row.get(key)
  if value in (None, ""):
    return None
  return float(value)


def _read_csv_rows(path: Path):
  with path.open(newline="") as csv_file:
    return list(csv.DictReader(csv_file))


def _summarize_run(run_dir: Path):
  train_rows = _read_csv_rows(run_dir / "csv_logs" / "train.csv")
  learner_rows = _read_csv_rows(run_dir / "csv_logs" / "learner.csv")
  if not train_rows:
    raise ValueError(f"No train rows found in {run_dir}")

  steps_per_second = []
  episode_returns = []
  for row in train_rows:
    steps_value = _parse_float(row, "steps_per_second")
    if steps_value is not None:
      steps_per_second.append(steps_value)
    keys = sorted(key for key in row if key.endswith("/episode_return"))
    if not keys:
      continue
    values = [_parse_float(row, key) for key in keys]
    values = [value for value in values if value is not None]
    if values:
      episode_returns.append(sum(values) / len(values))

  summary = {
      "run_dir": str(run_dir),
      "train_rows": len(train_rows),
      "learner_rows": len(learner_rows),
      "final_train_steps": int(float(train_rows[-1]["train_steps"])),
      "mean_steps_per_second": (
          sum(steps_per_second) / len(steps_per_second) if steps_per_second else None),
      "last_steps_per_second": steps_per_second[-1] if steps_per_second else None,
      "initial_mean_episode_return": episode_returns[0] if episode_returns else None,
      "best_mean_episode_return": max(episode_returns) if episode_returns else None,
      "final_mean_episode_return": episode_returns[-1] if episode_returns else None,
      "episode_return_delta": (
          episode_returns[-1] - episode_returns[0]
          if len(episode_returns) >= 2 else None),
  }
  if learner_rows:
    summary["final_total_loss"] = _parse_float(learner_rows[-1], "total_loss")
    summary["final_critic_loss"] = _parse_float(learner_rows[-1], "critic_loss")
  return summary


def _find_latest_run(run_root: Path, env_name: str, map_name: str, algo_name: str,
                     seed: int, started_at: float):
  prefix = f"{algo_name}_{seed}_{env_name}_{map_name}_"
  candidates = [
      path for path in run_root.iterdir()
      if path.is_dir() and path.name.startswith(prefix)
      and path.stat().st_mtime >= started_at - 1.0
  ]
  if not candidates:
    raise FileNotFoundError(f"No run directory found in {run_root} for {prefix}")
  return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_training_benchmark(args):
  run_root = Path(args.exp_log_dir).resolve()
  run_root.mkdir(parents=True, exist_ok=True)
  started_at = time.time()
  command = [
      sys.executable,
      str(ROOT / "train.py"),
      f"--env_name={args.env_name}",
      f"--map_name={args.map_name}",
      f"--algo_name={args.algo_name}",
      f"--num_steps={args.num_steps}",
      f"--seed={args.seed}",
      f"--exp_log_dir={run_root}",
      f"--learner_mode={args.learner_mode}",
      f"--actor_device={args.actor_device}",
      f"--parameter_shuffle_period={args.parameter_shuffle_period}",
      f"--learner_prefetch_size={args.learner_prefetch_size}",
      "--use_tb=False",
      "--use_wandb=False",
  ]
  if args.extra_flag:
    command.extend(args.extra_flag)
  subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())
  run_dir = _find_latest_run(
      run_root, args.env_name, args.map_name, args.algo_name, args.seed,
      started_at)
  print(json.dumps(_summarize_run(run_dir), indent=2, sort_keys=True))


def main():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)

  env_parser = subparsers.add_parser("env", help="Benchmark raw environment speed.")
  env_parser.add_argument("--steps", type=int, default=2000)
  env_parser.add_argument("--seed", type=int, default=0)
  env_parser.add_argument(
      "--env",
      action="append",
      default=[],
      help="Environment in env_name:map_name form. Defaults to a fast-to-slow preset list.",
  )

  train_parser = subparsers.add_parser(
      "train", help="Run a short training benchmark and summarize the logs.")
  train_parser.add_argument("--env_name", required=True)
  train_parser.add_argument("--map_name", required=True)
  train_parser.add_argument("--algo_name", default="IMPALA")
  train_parser.add_argument("--num_steps", type=int, default=4000)
  train_parser.add_argument("--seed", type=int, default=0)
  train_parser.add_argument("--exp_log_dir", default=str(ROOT / "benchmark_runs"))
  train_parser.add_argument("--learner_mode", default="auto")
  train_parser.add_argument("--actor_device", default="auto")
  train_parser.add_argument("--parameter_shuffle_period", type=int, default=0)
  train_parser.add_argument("--learner_prefetch_size", type=int, default=2)
  train_parser.add_argument("--extra_flag", action="append", default=[])

  args = parser.parse_args()

  if args.command == "env":
    envs = []
    failures = []
    if args.env:
      for entry in args.env:
        env_name, map_name = entry.split(":", 1)
        envs.append((env_name, map_name))
    else:
      envs = list(DEFAULT_ENVS)
    results = []
    for env_name, map_name in envs:
      try:
        results.append(_benchmark_env(env_name, map_name, args.steps, args.seed))
      except Exception as exc:  # benchmark helper should keep going across envs
        failures.append({
            "env_name": env_name,
            "map_name": map_name,
            "error": str(exc),
        })
    results.sort(key=lambda item: item["steps_per_second"], reverse=True)
    print(json.dumps({
        "results": results,
        "failures": failures,
    }, indent=2, sort_keys=True))
    return

  if args.command == "train":
    _run_training_benchmark(args)
    return

  raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
  main()
