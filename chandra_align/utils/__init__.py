"""Shared config loading and small IO helpers. No magic numbers here — they live in config/*.yaml."""

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config at {path} is not a mapping")
    for key in ("gsd_m", "crs_strategy", "tile_size", "photometry", "matcher",
                "verification", "warp", "trust"):
        if key not in cfg:
            raise ValueError(f"config at {path} missing required key: {key}")
    return cfg


def ensure_dirs(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
