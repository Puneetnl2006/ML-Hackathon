import unittest

import numpy as np
import pandas as pd

from src.inference import viterbi_route


class ViterbiInferenceTests(unittest.TestCase):
    def test_viterbi_decodes_contiguous_arc_sequence(self):
        candidate_frame = pd.DataFrame(
            {
                "trip_id": ["raw", "raw", "raw", "raw"],
                "observation_index": [0, 0, 1, 1],
                "arc_id": [0, 1, 1, 2],
            }
        )
        probabilities = np.array([0.1, 0.9, 0.8, 0.2])
        arcs = {0: (10, 11), 1: (11, 12), 2: (12, 13)}

        sequence = viterbi_route(candidate_frame, probabilities, arcs)

        self.assertEqual(sequence, [1, 1])

    def test_viterbi_rejects_mismatched_emissions(self):
        candidate_frame = pd.DataFrame(
            {
                "trip_id": ["raw"],
                "observation_index": [0],
                "arc_id": [0],
            }
        )

        with self.assertRaises(ValueError):
            viterbi_route(candidate_frame, np.array([0.5, 0.5]), {0: (0, 1)})


if __name__ == "__main__":
    unittest.main()
