#!/usr/bin/env python3
"""JRDB Toolkit - Config-driven pipeline entry point.

Usage:
    python run.py                        # Run all enabled tasks in order
    python run.py --task filter_occluded  # Run single task (ignores enabled flag)
    python run.py --config custom.yaml   # Use different config
"""

import argparse
import importlib
import sys
from pathlib import Path


def load_config(config_path):
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg):
    """Resolve relative model paths against DATASET_ROOT."""
    root = Path(cfg["GLOBAL"]["DATASET_ROOT"])
    for key, val in cfg.get("MODELS", {}).items():
        p = Path(val)
        if not p.is_absolute():
            cfg["MODELS"][key] = str(root / val)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="JRDB Toolkit - Config-driven pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--task", type=str, default=None, help="Run a single task (ignores enabled flag)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path = args.config if args.config else script_dir / "config" / "config.yaml"

    cfg = load_config(config_path)
    cfg = resolve_paths(cfg)

    if args.task:
        task_name = args.task
        print(f"=== Running task: {task_name} ===")
        mod = importlib.import_module(f"tasks.{task_name}")
        mod.run(cfg)
    else:
        tasks = cfg.get("TASKS", {})
        enabled = [(name, tcfg) for name, tcfg in tasks.items() if tcfg.get("ENABLED", False)]
        enabled.sort(key=lambda x: x[1].get("ORDER", 999))

        if not enabled:
            print("No tasks enabled in config.")
            return

        print(f"Running {len(enabled)} enabled tasks:")
        for name, _ in enabled:
            print(f"  - {name}")
        print()

        for name, _ in enabled:
            print(f"\n{'=' * 60}")
            print(f"=== Task: {name} ===")
            print(f"{'=' * 60}")
            mod = importlib.import_module(f"tasks.{name.lower()}")
            mod.run(cfg)

    print("\nPipeline complete.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
