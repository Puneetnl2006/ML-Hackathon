"""Focused tests for HMM route decoding and graph edge cases."""

from __future__ import annotations

import unittest

import pandas as pd

from src.decoder import ViterbiDecoder


def small_road_network() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "longitude": [0.0, 0.001, 0.002, 0.01, 0.011],
            "latitude": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    arcs = pd.DataFrame(
        {
            "from_node": [0, 1, 3],
            "to_node": [1, 2, 4],
            "arc_id": [0, 1, 2],
        }
    )
    return nodes, arcs


class ViterbiDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        nodes, arcs = small_road_network()
        self.decoder = ViterbiDecoder(nodes, arcs)

    def test_graph_connected_sequence_overrides_independent_top_candidate(self) -> None:
        observations = pd.DataFrame(
            {
                "longitude": [0.0005, 0.0015],
                "latitude": [0.0, 0.0],
                "sample_time": [1.0, 2.0],
            }
        )
        candidates = pd.DataFrame(
            {
                "observation_index": [0, 0, 1, 1],
                "arc_id": [0, 2, 1, 2],
                "distance_to_segment_m": [0.0, 0.0, 0.0, 0.0],
                "heading_alignment": [0.0, 0.0, 0.0, 0.0],
                "classifier_probability": [0.99, 0.999, 0.99, 0.999],
            }
        )

        decoded = self.decoder.decode(observations, candidates)

        self.assertEqual(decoded, [0, 1])
        self.assertTrue(self.decoder.is_transition_reachable(0, 1))
        self.assertFalse(self.decoder.is_transition_reachable(0, 2))

    def test_missing_candidates_and_large_gaps_are_safe(self) -> None:
        observations = pd.DataFrame(
            {
                "longitude": [0.0005, 0.001, 0.0015],
                "latitude": [0.0, 0.0, 0.0],
                "sample_time": [1.0, 2.0, 1_000.0],
            }
        )
        candidates = pd.DataFrame(
            {
                "observation_index": [0, 2],
                "arc_id": [0, 1],
            }
        )

        decoded = self.decoder.decode(observations, candidates)

        self.assertEqual(decoded, [0, None, 1])
        self.assertTrue(self.decoder.is_gap_between(observations, 1, 2))
        spatial_gap = pd.DataFrame(
            {
                "longitude": [0.0, 0.1],
                "latitude": [0.0, 0.0],
                "sample_time": [1.0, 2.0],
            }
        )
        self.assertTrue(self.decoder.is_gap_between(spatial_gap, 0, 1))

    def test_disconnected_candidates_still_produce_best_effort_path(self) -> None:
        observations = pd.DataFrame(
            {
                "longitude": [0.01, 0.011],
                "latitude": [0.0, 0.0],
                "sample_time": [1.0, 2.0],
            }
        )
        candidates = pd.DataFrame(
            {"observation_index": [0, 1], "arc_id": [2, 0]}
        )

        decoded = self.decoder.decode(observations, candidates)

        self.assertEqual(decoded, [2, 0])

    def test_repeated_arc_uses_forward_progress_without_a_graph_cycle(self) -> None:
        previous = pd.Series({"arc_id": 0})
        current = pd.Series({"arc_id": 0})

        score = self.decoder._transition_log(
            previous,
            current,
            middle_distance=None,
            gps_distance=55.0,
            delta_time=1.0,
            previous_point=pd.Series([0.00025, 0.0]).to_numpy(),
            current_point=pd.Series([0.00075, 0.0]).to_numpy(),
        )

        self.assertGreater(score, -1.0)

    def test_empty_candidates_preserve_observation_count(self) -> None:
        observations = pd.DataFrame(
            {"longitude": [0.0, 0.001], "latitude": [0.0, 0.0]}
        )

        self.assertEqual(self.decoder.decode(observations, pd.DataFrame()), [None, None])


if __name__ == "__main__":
    unittest.main()
