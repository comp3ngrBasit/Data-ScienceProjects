"""MMA action taxonomy (Phase 3).

Every category from the specification is listed. A category is only *predictable*
once a model has been trained on enough annotated examples of it; the model
registry records which categories each model actually knows.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    group: str  # offensive | defensive | movement | unknown
    family: str  # punch | kick | knee_elbow | grappling | defense | movement | unknown
    coarse: bool = False  # extension category for data that can't be labelled more precisely


CATEGORIES: list[Category] = [
    # Offensive
    Category("left_jab", "Left Jab", "offensive", "punch"),
    Category("right_jab", "Right Jab", "offensive", "punch"),
    Category("left_hook", "Left Hook", "offensive", "punch"),
    Category("right_hook", "Right Hook", "offensive", "punch"),
    Category("left_cross", "Left Cross", "offensive", "punch"),
    Category("right_cross", "Right Cross", "offensive", "punch"),
    Category("uppercut", "Uppercut", "offensive", "punch"),
    Category("punch_unspecified", "Punch (type unspecified)", "offensive", "punch", coarse=True),
    Category("front_kick", "Front Kick", "offensive", "kick"),
    Category("low_kick", "Low Kick", "offensive", "kick"),
    Category("middle_kick", "Middle Kick", "offensive", "kick"),
    Category("high_kick", "High Kick", "offensive", "kick"),
    Category("roundhouse_kick", "Roundhouse Kick", "offensive", "kick"),
    Category("spinning_back_kick", "Spinning Back Kick", "offensive", "kick"),
    Category("knee_strike", "Knee Strike", "offensive", "knee_elbow"),
    Category("elbow_strike", "Elbow Strike", "offensive", "knee_elbow"),
    Category("takedown_attempt", "Takedown Attempt", "offensive", "grappling"),
    # Defensive
    Category("high_guard", "High Guard", "defensive", "defense"),
    Category("defensive_block", "Defensive Block", "defensive", "defense"),
    Category("parry", "Parry", "defensive", "defense"),
    Category("slip", "Slip", "defensive", "defense"),
    Category("duck", "Duck", "defensive", "defense"),
    Category("backward_evasion", "Backward Evasion", "defensive", "defense"),
    Category("sprawl", "Sprawl", "defensive", "grappling"),
    Category("takedown_defense", "Takedown Defense", "defensive", "grappling"),
    Category("counterattack", "Counterattack", "defensive", "defense"),
    # Other movements
    Category("forward_movement", "Forward Movement", "movement", "movement"),
    Category("backward_movement", "Backward Movement", "movement", "movement"),
    Category("lateral_movement", "Lateral Movement", "movement", "movement"),
    Category("stance_switch", "Stance Switching", "movement", "movement"),
    Category("feint", "Feint", "movement", "movement"),
    Category("reset_stance", "Resetting Stance", "movement", "movement"),
    Category("clinch_entry", "Clinch Entry", "movement", "grappling"),
    # Explicit "don't know"
    Category("unknown", "Unknown / Unclassified", "unknown", "unknown"),
]
BY_KEY = {c.key: c for c in CATEGORIES}
BACKGROUND = "background"  # training-only class: "no labelled action in this window"; never emitted
UNKNOWN = "unknown"

# Mapping helpers for importing external datasets
BOXINGVI_MAP_ORTHODOX = {
    "jab": "left_jab", "cross": "right_cross", "lead hook": "left_hook", "rear hook": "right_hook",
    "lead uppercut": "uppercut", "rear uppercut": "uppercut",
}
BOXINGVI_MAP_SOUTHPAW = {
    "jab": "right_jab", "cross": "left_cross", "lead hook": "right_hook", "rear hook": "left_hook",
    "lead uppercut": "uppercut", "rear uppercut": "uppercut",
}


def label(key: str) -> str:
    if key == BACKGROUND:
        return "No action"
    c = BY_KEY.get(key)
    return c.label if c else key


def group(key: str) -> str:
    c = BY_KEY.get(key)
    return c.group if c else "unknown"


def normalise_key(value: str) -> str | None:
    """Accept keys or human labels ('Left Jab', 'left-jab', 'LEFT_JAB')."""
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    if v in BY_KEY:
        return v
    for c in CATEGORIES:
        if c.label.lower().replace(" ", "_") == v:
            return c.key
    return None
