import unittest

import pandas as pd

from run_adaptive_cross_domain_awr import ensure_target_has_no_labels


class NoLabelLeakageTests(unittest.TestCase):
    def test_online_inference_rejects_target_stage_columns(self) -> None:
        with self.assertRaises(AssertionError):
            ensure_target_has_no_labels(pd.DataFrame({"window_index": [0], "stage": [5]}))

    def test_label_free_target_frame_is_accepted(self) -> None:
        ensure_target_has_no_labels(pd.DataFrame({"window_index": [0], "BDall_xy_v2": [0.1]}))


if __name__ == "__main__":
    unittest.main()
