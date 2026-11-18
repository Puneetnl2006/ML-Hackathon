"""Inference and Viterbi helpers shared by the FastAPI backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS
from .train_baselines import positive_class_probabilities


def model_probabilities(model: object, candidate_features: pd.DataFrame) -> np.ndarray:
    """Score candidate rows without using route-label fields."""
    missing = sorted(set(FEATURE_COLUMNS) - set(candidate_features.columns))
    if missing:
        raise ValueError(f"Candidate features are missing columns: {missing}")
    return np.clip(
        positive_class_probabilities(model, candidate_features[FEATURE_COLUMNS]),
        1e-15,
        1 - 1e-15,
    )


def _log_transition(
    decoder: Any,
    previous_arc: int,
    next_arc: int,
    arcs: Mapping[int, tuple[int, int]],
    repeat_probability: float,
    move_probability: float,
) -> float:
    if isinstance(decoder, Mapping):
        transition = decoder.get("transition_log_prob")
        if callable(transition):
            value = transition(previous_arc, next_arc)
            if value is not None:
                return float(value)
        if isinstance(transition, np.ndarray):
            if previous_arc in arcs and next_arc in arcs:
                arc_ids = sorted(arcs)
                if previous_arc in arc_ids and next_arc in arc_ids:
                    i = arc_ids.index(previous_arc)
                    j = arc_ids.index(next_arc)
                    if i < transition.shape[0] and j < transition.shape[1]:
                        return float(np.log(max(float(transition[i, j]), 1e-15)))

    previous_end = arcs.get(previous_arc)
    next_start = arcs.get(next_arc)
    if previous_arc == next_arc:
        return float(np.log(repeat_probability))
    if previous_end is not None and next_start is not None:
        if previous_end[1] == next_start[0]:
            return float(np.log(move_probability))
    return float(np.log(1e-3))


def viterbi_route(
    candidate_frame: pd.DataFrame,
    probabilities: np.ndarray,
    arcs: Mapping[int, tuple[int, int]],
    decoder: Any = None,
) -> list[int]:
    """Decode the most likely arc sequence using candidate emissions and Viterbi."""
    if candidate_frame.empty:
        return []
    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) != len(candidate_frame):
        raise ValueError("Emissions and candidate rows must have the same length.")

    observations = list(
        candidate_frame.groupby(
            ["trip_id", "observation_index"], sort=False
        ).groups
    )
    route: list[int] = []
    previous_states: list[dict[int, int]] = []
    dp: dict[int, float] | None = None

    for observation_number, group_key in enumerate(observations):
        positions = candidate_frame.groupby(
            ["trip_id", "observation_index"], sort=False
        ).indices[group_key]
        positions = np.asarray(positions, dtype=np.int64)
        state_ids = [
            int(candidate_frame.iloc[position]["arc_id"])
            for position in positions
        ]
        emissions = probabilities[positions]

        if dp is None:
            dp = {}
            for state, emission in zip(state_ids, emissions):
                start_log_prob = 0.0
                if isinstance(decoder, Mapping):
                    start = decoder.get("start_log_prob")
                    if isinstance(start, Mapping) and state in start:
                        start_log_prob = float(start[state])
                dp[state] = start_log_prob + float(np.log(max(emission, 1e-15)))
            previous_states.append({})
            continue

        next_dp: dict[int, float] = {}
        backpointers: dict[int, int] = {}
        for state, emission in zip(state_ids, emissions):
            emission_log_prob = float(np.log(max(emission, 1e-15)))
            candidates = []
            for previous_state, previous_score in dp.items():
                transition = _log_transition(
                    decoder,
                    previous_state,
                    state,
                    arcs,
                    repeat_probability=0.35,
                    move_probability=0.65,
                )
                candidates.append((previous_score + transition, previous_state))
            best_score, best_previous = max(candidates, key=lambda item: item[0])
            next_dp[state] = best_score + emission_log_prob
            backpointers[state] = best_previous
        previous_states.append(backpointers)
        dp = next_dp

    if not dp:
        return route
    state = max(dp, key=dp.get)
    route.append(state)
    for backpointers in reversed(previous_states[1:]):
        state = backpointers[state]
        route.append(state)
    route.reverse()
    return route
