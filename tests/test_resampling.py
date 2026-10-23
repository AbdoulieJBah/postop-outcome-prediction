import unittest

import numpy as np

from postop.data import TARGETS, make_synthetic
from postop.resampling import random_oversample, smote


class ResamplingTest(unittest.TestCase):
    def setUp(self):
        X, y = make_synthetic("complications", seed=1)
        self.X, self.y = X.to_numpy(), y.to_numpy()

    def test_ratio_is_respected(self):
        for fn in (random_oversample, smote):
            for ratio in (0.5, 1.0):
                Xr, yr = fn(self.X, self.y, ratio, rng=0)
                self.assertAlmostEqual((yr == 1).sum() / (yr == 0).sum(), ratio, places=2)
                self.assertEqual(len(Xr), len(yr))

    def test_original_rows_are_kept(self):
        for fn in (random_oversample, smote):
            Xr, yr = fn(self.X, self.y, 1.0, rng=0)
            np.testing.assert_array_equal(Xr[: len(self.X)], self.X)
            np.testing.assert_array_equal(yr[: len(self.y)], self.y)

    def test_majority_class_untouched(self):
        for fn in (random_oversample, smote):
            _, yr = fn(self.X, self.y, 1.0, rng=0)
            self.assertEqual((yr == 0).sum(), (self.y == 0).sum())

    def test_smote_points_stay_inside_minority_bounds(self):
        Xr, _ = smote(self.X, self.y, 1.0, rng=0)
        new, pos = Xr[len(self.X):], self.X[self.y == 1]
        self.assertTrue((new >= pos.min(0) - 1e-12).all() and (new <= pos.max(0) + 1e-12).all())

    def test_no_resampling_when_already_balanced(self):
        Xr, yr = smote(self.X, self.y, 0.0, rng=0)
        self.assertEqual(len(yr), len(self.y))


class SyntheticDataTest(unittest.TestCase):
    def test_shapes_and_rates(self):
        for target, n, rate in (("complications", 1726, 0.159), ("relapses", 1250, 0.093)):
            X, y = make_synthetic(target)
            self.assertEqual(X.shape, (n, len(TARGETS[target][1])))
            self.assertAlmostEqual(y.mean(), rate, places=2)
            self.assertTrue(((X >= 0) & (X <= 1)).all().all())


if __name__ == "__main__":
    unittest.main()
