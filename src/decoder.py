"""Hidden Markov model and Viterbi decoding for PathSnap road candidates."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import TypeAlias

import numpy as np
import pandas as pd

EARTH_RADIUS_METERS = 6_371_000.0
CandidateSequence: TypeAlias = list[int | None]


def _great_circle_distance(start: np.ndarray, end: np.ndarray) -> float:
    """Return the great-circle distance between longitude/latitude pairs."""
    lon1, lat1 = np.radians(start)
    lon2, lat2 = np.radians(end)
    delta_lon, delta_lat = lon2 - lon1, lat2 - lat1
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return float(2.0 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(haversine)))


def _bearing_degrees(start: np.ndarray, end: np.ndarray) -> float:
    """Return the initial compass bearing from one geographic point to another."""
    lon1, lat1 = np.radians(start)
    lon2, lat2 = np.radians(end)
    delta_lon = lon2 - lon1
    y = np.sin(delta_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


class ViterbiDecoder:
    """Decode GPS observations into a likely sequence of directed road arcs.

    The road graph uses the PathSnap convention: node IDs are the original
    ``.nodes`` row indices, and arc IDs are the original ``.arcs`` row indices.
    Candidate rows must contain ``observation_index`` and ``arc_id``. Optional
    columns ``distance_to_segment_m``, ``heading_alignment`` and
    ``classifier_probability`` contribute to the emission likelihood.

    Missing candidate observations and large GPS jumps split the sequence into
    independent decoding runs. Disconnected candidate pairs remain possible
    with a strong penalty, allowing the decoder to return a best-effort route
    rather than fail on imperfect input.
    """

    def __init__(
        self,
        nodes: pd.DataFrame,
        arcs: pd.DataFrame,
        *,
        distance_sigma_meters: float = 30.0,
        transition_sigma_meters: float = 150.0,
        heading_weight: float = 1.5,
        classifier_probability_weight: float = 5.0,
        disconnected_log_penalty: float = 1_000.0,
        max_time_gap_seconds: float = 300.0,
        max_spatial_gap_meters: float = 5_000.0,
        max_transition_search_meters: float = 10_000.0,
        max_speed_mps: float = 60.0,
    ) -> None:
        """Create a decoder, retaining only arcs with valid endpoint nodes."""
        if distance_sigma_meters <= 0 or transition_sigma_meters <= 0:
            raise ValueError("Distance and transition scales must be positive.")
        if (
            max_time_gap_seconds <= 0
            or max_spatial_gap_meters <= 0
            or max_transition_search_meters <= 0
        ):
            raise ValueError("GPS gap thresholds must be positive.")
        if max_speed_mps <= 0:
            raise ValueError("Maximum speed must be positive.")
        if classifier_probability_weight < 0:
            raise ValueError("Classifier probability weight cannot be negative.")
        if disconnected_log_penalty <= 0:
            raise ValueError("Disconnected transition penalty must be positive.")

        self.distance_sigma_meters = distance_sigma_meters
        self.transition_sigma_meters = transition_sigma_meters
        self.heading_weight = heading_weight
        self.classifier_probability_weight = classifier_probability_weight
        self.disconnected_log_penalty = disconnected_log_penalty
        self.max_time_gap_seconds = max_time_gap_seconds
        self.max_spatial_gap_meters = max_spatial_gap_meters
        self.max_transition_search_meters = max_transition_search_meters
        self.max_speed_mps = max_speed_mps

        self.node_coordinates = {
            int(node_id): row[["longitude", "latitude"]].to_numpy(dtype=float)
            for node_id, row in nodes.iterrows()
        }
        self.arcs: dict[int, tuple[int, int, float, np.ndarray, np.ndarray]] = {}
        self.outgoing: dict[int, list[tuple[int, float]]] = defaultdict(list)

        for row_id, row in arcs.iterrows():
            arc_id = int(row.get("arc_id", row_id))
            start_node, end_node = int(row["from_node"]), int(row["to_node"])
            start = self.node_coordinates.get(start_node)
            end = self.node_coordinates.get(end_node)
            if start is None or end is None:
                continue
            length = _great_circle_distance(start, end)
            self.arcs[arc_id] = (start_node, end_node, length, start, end)
            self.outgoing[start_node].append((end_node, length))

    def decode(
        self, observations: pd.DataFrame, candidates: pd.DataFrame
    ) -> CandidateSequence:
        """Return one decoded arc ID (or ``None``) for each GPS observation.

        ``observations`` must contain longitude and latitude columns and may
        contain ``sample_time``. Candidate rows are grouped by their
        zero-based ``observation_index``; malformed or unknown arc IDs are
        ignored, while an observation with no usable candidates decodes to
        ``None``.
        """
        observation_count = len(observations)
        decoded: CandidateSequence = [None] * observation_count
        if observations.empty or candidates.empty:
            return decoded
        required_observation_columns = {"longitude", "latitude"}
        if not required_observation_columns.issubset(observations.columns):
            raise ValueError("Observations require longitude and latitude columns.")
        required_candidate_columns = {"observation_index", "arc_id"}
        if not required_candidate_columns.issubset(candidates.columns):
            raise ValueError(
                "Candidates require observation_index and arc_id columns."
            )

        indexed_candidates: dict[int, pd.DataFrame] = {}
        for index, rows in candidates.groupby("observation_index", sort=False):
            try:
                observation_index = int(index)
            except (TypeError, ValueError):
                continue
            if 0 <= observation_index < observation_count:
                valid = rows.loc[rows["arc_id"].isin(self.arcs)].copy()
                if not valid.empty:
                    indexed_candidates[observation_index] = valid

        run: list[int] = []
        for observation_index in range(observation_count):
            if observation_index not in indexed_candidates:
                self._decode_run(run, indexed_candidates, observations, decoded)
                run = []
                continue
            if run and self.is_gap_between(
                observations, run[-1], observation_index
            ):
                self._decode_run(run, indexed_candidates, observations, decoded)
                run = []
            run.append(observation_index)
        self._decode_run(run, indexed_candidates, observations, decoded)
        return decoded

    def is_transition_reachable(self, from_arc_id: int, to_arc_id: int) -> bool:
        """Return whether a directed graph path connects arcs within the search limit."""
        current = self.arcs.get(int(from_arc_id))
        following = self.arcs.get(int(to_arc_id))
        if current is None or following is None:
            return False
        if from_arc_id == to_arc_id:
            return True
        distances = self._distances_between(
            current[1],
            {following[0]},
            self.max_transition_search_meters,
        )
        return following[0] in distances

    def is_gap_between(
        self, observations: pd.DataFrame, previous_index: int, current_index: int
    ) -> bool:
        """Return whether two observations should start separate decoding runs."""
        coordinates = observations.iloc[
            [previous_index, current_index]
        ][["longitude", "latitude"]].to_numpy(dtype=float)
        if "sample_time" in observations:
            delta_time = float(
                observations["sample_time"].iloc[current_index]
                - observations["sample_time"].iloc[previous_index]
            )
        else:
            delta_time = float(current_index - previous_index)
        if np.isfinite(delta_time) and delta_time > self.max_time_gap_seconds:
            return True
        return (
            _great_circle_distance(coordinates[0], coordinates[1])
            > self.max_spatial_gap_meters
        )

    def _decode_run(
        self,
        indices: list[int],
        candidate_groups: dict[int, pd.DataFrame],
        observations: pd.DataFrame,
        decoded: CandidateSequence,
    ) -> None:
        if not indices:
            return
        state_rows = [
            [
                row
                for _, row in candidate_groups[index].iterrows()
                if int(row["arc_id"]) in self.arcs
            ]
            for index in indices
        ]
        if any(not states for states in state_rows):
            return

        coordinates = observations[
            ["longitude", "latitude"]
        ].to_numpy(dtype=float)
        times = (
            observations["sample_time"].to_numpy(dtype=float)
            if "sample_time" in observations
            else np.arange(len(observations), dtype=float)
        )
        log_scores = [
            np.asarray(
                [self._emission_log(row, index, coordinates) for row in states],
                dtype=float,
            )
            for index, states in zip(indices, state_rows)
        ]
        backpointers: list[np.ndarray] = [
            np.full(len(state_rows[0]), -1, dtype=int)
        ]

        for time_offset in range(1, len(indices)):
            previous_index, current_index = indices[time_offset - 1 : time_offset + 1]
            gps_distance = _great_circle_distance(
                coordinates[previous_index], coordinates[current_index]
            )
            delta_time = times[current_index] - times[previous_index]
            if not np.isfinite(delta_time) or delta_time <= 0:
                delta_time = 1.0

            next_scores = np.empty(len(state_rows[time_offset]), dtype=float)
            next_backpointers = np.empty(len(state_rows[time_offset]), dtype=int)
            target_nodes = {
                self.arcs[int(row["arc_id"])][0]
                for row in state_rows[time_offset]
            }
            search_limit = min(
                self.max_transition_search_meters,
                max(
                    3.0 * gps_distance + 100.0,
                    delta_time * self.max_speed_mps + 100.0,
                ),
            )
            distances_by_previous = [
                self._distances_between(
                    self.arcs[int(row["arc_id"])][1],
                    target_nodes,
                    search_limit,
                )
                for row in state_rows[time_offset - 1]
            ]
            for current_position, current_row in enumerate(state_rows[time_offset]):
                alternatives = [
                    log_scores[time_offset - 1][previous_position]
                    + self._transition_log(
                        previous_row,
                        current_row,
                        distances_by_previous[previous_position].get(
                            self.arcs[int(current_row["arc_id"])][0]
                        ),
                        gps_distance,
                        delta_time,
                        coordinates[previous_index],
                        coordinates[current_index],
                    )
                    for previous_position, previous_row in enumerate(
                        state_rows[time_offset - 1]
                    )
                ]
                best_previous = int(np.argmax(alternatives))
                next_scores[current_position] = (
                    alternatives[best_previous]
                    + log_scores[time_offset][current_position]
                )
                next_backpointers[current_position] = best_previous
            log_scores[time_offset] = next_scores
            backpointers.append(next_backpointers)

        state_position = int(np.argmax(log_scores[-1]))
        for time_offset in range(len(indices) - 1, -1, -1):
            decoded[indices[time_offset]] = int(
                state_rows[time_offset][state_position]["arc_id"]
            )
            if time_offset:
                state_position = int(backpointers[time_offset][state_position])

    def _emission_log(
        self, candidate: pd.Series, observation_index: int, coordinates: np.ndarray
    ) -> float:
        arc_id = int(candidate["arc_id"])
        _, _, _, start, end = self.arcs[arc_id]
        point = coordinates[observation_index]
        distance = candidate.get("distance_to_segment_m")
        if distance is None or not np.isfinite(distance):
            distance = self._point_segment_distance(point, start, end)
        log_probability = -0.5 * (float(distance) / self.distance_sigma_meters) ** 2

        probability = candidate.get("classifier_probability")
        if probability is None or not np.isfinite(probability):
            probability = candidate.get("probability")
        if probability is not None and np.isfinite(probability):
            log_probability += self.classifier_probability_weight * math.log(
                max(float(probability), 1e-12)
            )

        alignment = candidate.get("heading_alignment")
        if alignment is not None and np.isfinite(alignment):
            log_probability += self.heading_weight * float(np.clip(alignment, -1, 1))
        elif observation_index > 0:
            heading = _bearing_degrees(coordinates[observation_index - 1], point)
            arc_heading = _bearing_degrees(start, end)
            difference = math.radians(heading - arc_heading)
            log_probability += self.heading_weight * math.cos(difference)
        return log_probability

    def _transition_log(
        self,
        previous: pd.Series,
        current: pd.Series,
        middle_distance: float | None,
        gps_distance: float,
        delta_time: float,
        previous_point: np.ndarray,
        current_point: np.ndarray,
    ) -> float:
        previous_arc_id, current_arc_id = int(previous["arc_id"]), int(current["arc_id"])
        previous_arc, current_arc = self.arcs[previous_arc_id], self.arcs[current_arc_id]
        previous_fraction = self._arc_fraction(previous_point, previous_arc[3], previous_arc[4])
        current_fraction = self._arc_fraction(current_point, current_arc[3], current_arc[4])

        if previous_arc_id == current_arc_id and current_fraction >= previous_fraction:
            network_distance = (current_fraction - previous_fraction) * previous_arc[2]
        else:
            if middle_distance is None:
                return -self.disconnected_log_penalty
            network_distance = (
                (1.0 - previous_fraction) * previous_arc[2]
                + middle_distance
                + current_fraction * current_arc[2]
            )

        log_probability = -abs(network_distance - gps_distance) / self.transition_sigma_meters
        plausible_distance = delta_time * self.max_speed_mps
        if network_distance > plausible_distance:
            log_probability -= (
                network_distance - plausible_distance
            ) / self.transition_sigma_meters
        return log_probability

    def _distances_between(
        self, source_node: int, targets: set[int], max_distance: float
    ) -> dict[int, float]:
        """Find nearby target-node distances, stopping when all are reached."""
        remaining_targets = set(targets)
        found: dict[int, float] = {}
        if source_node in remaining_targets:
            remaining_targets.remove(source_node)
            found[source_node] = 0.0
        if not remaining_targets:
            return found
        distances = {source_node: 0.0}
        queue = [(0.0, source_node)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances[node]:
                continue
            if distance > max_distance:
                break
            if node in remaining_targets:
                found[node] = distance
                remaining_targets.remove(node)
                if not remaining_targets:
                    break
            for neighbor, edge_length in self.outgoing.get(node, []):
                next_distance = distance + edge_length
                if (
                    next_distance <= max_distance
                    and next_distance < distances.get(neighbor, float("inf"))
                ):
                    distances[neighbor] = next_distance
                    heapq.heappush(queue, (next_distance, neighbor))
        return found

    @staticmethod
    def _arc_fraction(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        reference_latitude = math.radians(float(point[1]))
        start_xy = np.array(
            [
                math.radians(float(start[0] - point[0]))
                * EARTH_RADIUS_METERS
                * math.cos(reference_latitude),
                math.radians(float(start[1] - point[1])) * EARTH_RADIUS_METERS,
            ]
        )
        end_xy = np.array(
            [
                math.radians(float(end[0] - point[0]))
                * EARTH_RADIUS_METERS
                * math.cos(reference_latitude),
                math.radians(float(end[1] - point[1])) * EARTH_RADIUS_METERS,
            ]
        )
        vector = end_xy - start_xy
        denominator = float(np.dot(vector, vector))
        if denominator == 0:
            return 0.0
        return float(np.clip(-np.dot(start_xy, vector) / denominator, 0.0, 1.0))

    @classmethod
    def _point_segment_distance(
        cls, point: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> float:
        fraction = cls._arc_fraction(point, start, end)
        reference_latitude = math.radians(float(point[1]))
        longitude_scale = EARTH_RADIUS_METERS * math.cos(reference_latitude)
        start_xy = np.array(
            [
                math.radians(float(start[0] - point[0])) * longitude_scale,
                math.radians(float(start[1] - point[1])) * EARTH_RADIUS_METERS,
            ]
        )
        end_xy = np.array(
            [
                math.radians(float(end[0] - point[0])) * longitude_scale,
                math.radians(float(end[1] - point[1])) * EARTH_RADIUS_METERS,
            ]
        )
        projection = start_xy + fraction * (end_xy - start_xy)
        return float(np.linalg.norm(projection))
