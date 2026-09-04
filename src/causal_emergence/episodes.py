"""
Centralized Episode Configuration and Masking Module
=====================================================
Single source of truth for historical market stress episodes across the CEFI pipeline.
Guarantees 100% numerical and provenance synchronization across all tables, regressions,
figures, and manuscript text.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd


def get_episodes_config_path() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, current.parents[1], current.parents[2]]:
        cfg = parent / "config" / "episodes.json"
        if cfg.exists():
            return cfg
    raise FileNotFoundError("Could not locate config/episodes.json")


def load_episodes() -> Dict[str, Dict[str, Any]]:
    """Loads the canonical dictionary of market stress episodes."""
    cfg_path = get_episodes_config_path()
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_episodes_tuples() -> Dict[str, Tuple[str, str, str]]:
    """
    Returns mapping: episode_key -> (start_date, end_date, stress_class)
    for drop-in compatibility with econometric scripts.
    """
    episodes = load_episodes()
    return {
        k: (v["start_date"], v["end_date"], v["stress_class"])
        for k, v in episodes.items()
    }


def get_crises_plot_specs() -> List[Tuple[str, str, str, str, float]]:
    """
    Returns list of tuples (label, start_date, end_date, color, alpha)
    for figure shading.
    """
    episodes = load_episodes()
    specs = []
    for k, v in episodes.items():
        specs.append((
            v.get("display_name", k),
            v["start_date"],
            v["end_date"],
            v["color"],
            v["alpha"]
        ))
    return specs


def apply_episode_dummies(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    """
    Adds binary indicators is_liquidity and is_valuation based on
    canonical date masks.
    """
    df = df.copy()
    if date_col is not None and date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
    else:
        dates = pd.to_datetime(df.index)

    df["is_liquidity"] = 0
    df["is_valuation"] = 0

    episodes = load_episodes()
    for ep_name, ep_info in episodes.items():
        mask = (dates >= ep_info["start_date"]) & (dates <= ep_info["end_date"])
        if ep_info["stress_class"] == "Systemic Liquidity":
            df.loc[mask, "is_liquidity"] = 1
        elif ep_info["stress_class"] == "Valuation Repricing":
            df.loc[mask, "is_valuation"] = 1

    return df
